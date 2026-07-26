"""Short real-data, accumulated-gradient smoke for frozen Modus_X 1B."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import shutil
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


EXPECTED_PARAMS = 1_058_963_121


def restore_standard_checkpoint(checkpointer, path, target):
    """Support both the legacy Kaggle Orbax API and the newer args API."""
    try:
        return checkpointer.restore(path, target)
    except TypeError:
        return checkpointer.restore(path, args=ocp.args.StandardRestore(target))


def save_standard_checkpoint(checkpointer, path, state):
    """Save synchronously across Orbax releases used by Kaggle images."""
    try:
        checkpointer.save(path, state, force=True)
    except TypeError:
        checkpointer.save(path, args=ocp.args.StandardSave(state), force=True)
    wait = getattr(checkpointer, "wait_until_finished", None)
    if wait is not None:
        wait()


def keep_latest_checkpoint(checkpoint_root, latest):
    """Bound local disk use while retaining the newest completed checkpoint."""
    for candidate in checkpoint_root.glob("step_*"):
        if candidate != latest:
            shutil.rmtree(candidate)


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--code-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--target-updates", type=int, default=16)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=16)
    parser.add_argument("--scan-chunk-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260717)
    return parser.parse_args()


def load_gate(code_dir: Path):
    spec = importlib.util.spec_from_file_location(
        "full_1b_gate", code_dir / "08_full_1b_train_step_readiness.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def abstract_leaf(value, sharding, dtype):
    return jax.ShapeDtypeStruct(value.shape, dtype, sharding=sharding)


def accumulator_sharding_tree(abstract_tree, param_shardings, mesh):
    """Shard temporary FP32 gradients over data without changing parameters."""
    values, treedef = jax.tree_util.tree_flatten(abstract_tree)
    shardings = jax.tree_util.tree_leaves(param_shardings)
    outputs = []
    for value, sharding in zip(values, shardings, strict=True):
        shape = tuple(int(dim) for dim in value.shape)
        if not shape:
            outputs.append(NamedSharding(mesh, P()))
            continue
        entries = list(sharding.spec)
        entries.extend([None] * (len(shape) - len(entries)))
        candidate = next(
            (
                index
                for index, (dim, axis) in enumerate(zip(shape, entries, strict=True))
                if axis is None and dim % 4 == 0
            ),
            None,
        )
        if candidate is not None:
            entries[candidate] = "data"
        else:
            model_index = next(
                (
                    index
                    for index, (dim, axis) in enumerate(
                        zip(shape, entries, strict=True)
                    )
                    if axis == "model" and dim % 8 == 0
                ),
                None,
            )
            if model_index is not None:
                entries[model_index] = ("data", "model")
        outputs.append(NamedSharding(mesh, P(*entries)))
    return treedef.unflatten(outputs)


def main() -> None:
    a = args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    if jax.default_backend() != "tpu" or jax.device_count() != 8:
        raise RuntimeError((jax.default_backend(), jax.device_count()))
    if a.data_path.stat().st_size != 100_000_000:
        raise ValueError(f"Expected enwik8 (100M bytes), got {a.data_path}")
    if a.accumulation_steps != 4:
        raise ValueError("Frozen smoke uses four true microsteps per update")

    gate = load_gate(a.code_dir)
    cfg = gate.config(2_048)
    data_axis_size, model_axis_size = 4, 2
    batch = data_axis_size
    devices = np.array(jax.devices(), dtype=object).reshape(
        data_axis_size, model_axis_size
    )
    mesh = Mesh(devices, ("data", "model"))
    replicated = NamedSharding(mesh, P())
    data_sharding = NamedSharding(mesh, P("data", None))
    key = jax.random.key(a.seed)

    def abstract_init(init_key):
        params = gate.init_modus_x_lm(init_key, cfg)
        return jax.tree_util.tree_map(
            lambda value: value.astype(jnp.bfloat16), params
        )

    abstract_params = jax.eval_shape(abstract_init, key)
    if gate.count_params(abstract_params) != EXPECTED_PARAMS:
        raise AssertionError(gate.count_params(abstract_params))
    param_shardings, sharding_rows = gate.make_sharding_tree(abstract_params, mesh)
    accumulator_shardings = accumulator_sharding_tree(
        abstract_params, param_shardings, mesh
    )
    abstract_bf16 = jax.tree_util.tree_map(
        lambda value, sharding: abstract_leaf(value, sharding, jnp.bfloat16),
        abstract_params,
        param_shardings,
    )
    abstract_fp32 = jax.tree_util.tree_map(
        lambda value, sharding: abstract_leaf(value, sharding, jnp.float32),
        abstract_params,
        param_shardings,
    )

    checkpoint_root = a.outdir / "checkpoints"
    checkpoints = sorted(checkpoint_root.glob("step_*")) if checkpoint_root.exists() else []
    checkpointer = ocp.StandardCheckpointer()
    if checkpoints:
        latest = checkpoints[-1]
        target = {
            "params": abstract_bf16,
            "adam_m": abstract_fp32,
            "adam_v": abstract_fp32,
            "step": np.asarray(0, dtype=np.int64),
            "data_cursor": np.asarray(0, dtype=np.int64),
            "seed": np.asarray(a.seed, dtype=np.int64),
        }
        state = restore_standard_checkpoint(checkpointer, latest, target)
        params = state["params"]
        moments_m = state["adam_m"]
        moments_v = state["adam_v"]
        start_step = int(state["step"])
        data_cursor = int(state["data_cursor"])
        print("FULL_1B_REAL_RESUME", latest, "step", start_step, flush=True)
    else:
        jax.clear_caches()
        gc.collect()
        init = jax.jit(abstract_init, out_shardings=param_shardings)
        started = time.perf_counter()
        params = init(key)
        gate.block_tree(params)
        zero_fp32 = jax.jit(
            lambda tree: jax.tree_util.tree_map(
                lambda value: jnp.zeros_like(value, dtype=jnp.float32), tree
            ),
            in_shardings=(param_shardings,),
            out_shardings=param_shardings,
        )
        moments_m = zero_fp32(params)
        moments_v = zero_fp32(params)
        gate.block_tree((moments_m, moments_v))
        start_step = 0
        data_cursor = 0
        print(
            "FULL_1B_REAL_INITIALIZED",
            json.dumps({"seconds": time.perf_counter() - started}),
            flush=True,
        )

    def microstep(p, accumulator, tokens, targets):
        loss, grads = jax.value_and_grad(gate.loss_fn)(
            p, tokens, targets, cfg, a.scan_chunk_size
        )
        accumulator = jax.tree_util.tree_map(
            lambda old, grad: old + grad.astype(jnp.float32), accumulator, grads
        )
        return accumulator, loss

    compiled_microstep = jax.jit(
        microstep,
        in_shardings=(
            param_shardings,
            accumulator_shardings,
            data_sharding,
            data_sharding,
        ),
        out_shardings=(accumulator_shardings, replicated),
        donate_argnums=(1,),
    )

    def apply_update(p, m, v, accumulated_grads, step):
        beta1 = jnp.array(0.9, jnp.float32)
        beta2 = jnp.array(0.95, jnp.float32)
        step_f = step.astype(jnp.float32)
        grads = jax.tree_util.tree_map(
            lambda value: value / a.accumulation_steps, accumulated_grads
        )
        new_m = jax.tree_util.tree_map(
            lambda old, grad: beta1 * old + (1.0 - beta1) * grad, m, grads
        )
        new_v = jax.tree_util.tree_map(
            lambda old, grad: beta2 * old + (1.0 - beta2) * grad**2, v, grads
        )
        corrected_m = jax.tree_util.tree_map(
            lambda value: value / (1.0 - beta1**step_f), new_m
        )
        corrected_v = jax.tree_util.tree_map(
            lambda value: value / (1.0 - beta2**step_f), new_v
        )
        new_p = jax.tree_util.tree_map(
            lambda value, mean, variance: (
                value.astype(jnp.float32)
                - a.learning_rate
                * (
                    mean / (jnp.sqrt(variance) + 1e-8)
                    + a.weight_decay * value.astype(jnp.float32)
                )
            ).astype(jnp.bfloat16),
            p,
            corrected_m,
            corrected_v,
        )
        zeros = jax.tree_util.tree_map(jnp.zeros_like, accumulated_grads)
        return new_p, new_m, new_v, zeros

    compiled_update = jax.jit(
        apply_update,
        in_shardings=(
            param_shardings,
            param_shardings,
            param_shardings,
            accumulator_shardings,
            replicated,
        ),
        out_shardings=(
            param_shardings,
            param_shardings,
            param_shardings,
            accumulator_shardings,
        ),
        donate_argnums=(0, 1, 2, 3),
    )

    zero_fp32 = jax.jit(
        lambda tree: jax.tree_util.tree_map(
            lambda value: jnp.zeros_like(value, dtype=jnp.float32), tree
        ),
        in_shardings=(param_shardings,),
        out_shardings=accumulator_shardings,
    )
    grad_accumulator = zero_fp32(params)
    gate.block_tree(grad_accumulator)

    train = np.memmap(a.data_path, mode="r", dtype=np.uint8)[:90_000_000]
    seq_len = cfg.seq_len
    span = seq_len + 1
    rows = []
    run_started = time.perf_counter()
    processed_tokens = start_step * batch * seq_len * a.accumulation_steps

    for update_number in range(start_step + 1, a.target_updates + 1):
        micro_losses = []
        update_started = time.perf_counter()
        for _ in range(a.accumulation_steps):
            starts = [
                (data_cursor + index * span) % (len(train) - span)
                for index in range(batch)
            ]
            host = np.stack([train[start : start + span] for start in starts]).astype(
                np.int32
            )
            data_cursor = (data_cursor + batch * span) % (len(train) - span)
            tokens = jax.device_put(host[:, :-1], data_sharding)
            targets = jax.device_put(host[:, 1:], data_sharding)
            grad_accumulator, loss = compiled_microstep(
                params, grad_accumulator, tokens, targets
            )
            gate.block_tree((grad_accumulator, loss))
            micro_losses.append(float(loss))

        params, moments_m, moments_v, grad_accumulator = (
            compiled_update(
                params,
                moments_m,
                moments_v,
                grad_accumulator,
                jax.device_put(jnp.array(update_number, jnp.int32), replicated),
            )
        )
        gate.block_tree(
            (params, moments_m, moments_v, grad_accumulator)
        )
        processed_tokens += batch * seq_len * a.accumulation_steps
        elapsed = time.perf_counter() - update_started
        row = {
            "update": update_number,
            "processed_tokens": processed_tokens,
            "loss_mean": float(np.mean(micro_losses)),
            "loss_last": micro_losses[-1],
            "update_seconds": elapsed,
            "tokens_per_second": batch
            * seq_len
            * a.accumulation_steps
            / elapsed,
        }
        rows.append(row)
        print("FULL_1B_REAL_UPDATE", json.dumps(row), flush=True)

        if (
            update_number == start_step + 1
            or update_number % a.checkpoint_every == 0
            or update_number == a.target_updates
        ):
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_root / f"step_{update_number:06d}"
            state = {
                "params": params,
                "adam_m": moments_m,
                "adam_v": moments_v,
                "step": np.int64(update_number),
                "data_cursor": np.int64(data_cursor),
                "seed": np.int64(a.seed),
            }
            save_standard_checkpoint(checkpointer, checkpoint_path, state)
            keep_latest_checkpoint(checkpoint_root, checkpoint_path)
            print("FULL_1B_REAL_CHECKPOINT", checkpoint_path, flush=True)

    report = {
        "status": "PASS",
        "claim_boundary": "Real enwik8 byte-ID systems smoke; not a BPC result.",
        "parameters": EXPECTED_PARAMS,
        "mesh": [4, 2],
        "context": seq_len,
        "global_microbatch": batch,
        "accumulation_steps": a.accumulation_steps,
        "tokens_per_optimizer_update": batch * seq_len * a.accumulation_steps,
        "scan_chunk_size": a.scan_chunk_size,
        "start_update": start_step,
        "target_update": a.target_updates,
        "processed_tokens": processed_tokens,
        "data_cursor": data_cursor,
        "elapsed_seconds": time.perf_counter() - run_started,
        "rows": rows,
        "memory": gate.memory_snapshot(),
        "sharding": sharding_rows,
    }
    (a.outdir / "real_data_smoke_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("FULL_1B_REAL_DATA_SMOKE_PASS", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
