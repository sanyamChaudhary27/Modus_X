from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from models import (
    ModelConfig,
    count_params,
    lm_head_fwd,
    make_model,
    modus_x_memory_feedback_archive_layer_fwd_stateful,
)


LN2 = math.log(2.0)
EXPECTED_PARAMS = 47_437_768
TRAIN_END = 90_000_000
VALIDATION_START = 90_000_000
VALIDATION_END = 95_000_000
TEST_START = 95_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--target-characters", type=int, default=20_480_000)
    parser.add_argument("--checkpoint-every", type=int, default=1_000)
    parser.add_argument("--reset-interval", type=int, default=1_250)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def atomic_pickle(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(tree: Any) -> str:
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        value = np.asarray(leaf)
        digest.update(str(value.shape).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def block_tree(tree: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=256,
        embed_dim=512,
        hidden_dim=1536,
        ax_res=512,
        n_layers=12,
        n_heads_attn=8,
        seq_len=512,
        mamba_state_dim=512,
        vector_router=False,
        router_hidden=32,
    )


def make_initial_model(seed: int, cfg: ModelConfig):
    return make_model(
        "Modus_X_MemoryFeedbackArchive_DeepSupervision",
        jax.random.key(seed),
        cfg,
        auxiliary_layers=(6,),
        future_target_count=1,
        dropout_rate=0.0,
    )


def zero_state_single(cfg: ModelConfig, dtype: jnp.dtype):
    return (
        jnp.zeros((cfg.n_layers, cfg.ax_res, cfg.ax_res), dtype=dtype),
        jnp.zeros((cfg.n_layers, cfg.ax_res, cfg.ax_res), dtype=dtype),
        jnp.zeros((cfg.n_layers, cfg.mamba_state_dim), dtype=dtype),
    )


def stateful_deep_forward(
    params: dict[str, Any],
    token_ids: jax.Array,
    cfg: ModelConfig,
    initial_state: tuple[jax.Array, jax.Array, jax.Array],
):
    x = params["embed"][token_ids]

    def scan_layer(x_in, inputs):
        layer, current, archive, vector = inputs
        next_state, layer_output = modus_x_memory_feedback_archive_layer_fwd_stateful(
            layer, x_in, (current, archive, vector)
        )
        x_out = x_in + layer_output
        return x_out, (x_out, next_state)

    x, (layer_outputs, final_state) = lax.scan(
        scan_layer,
        x,
        (params["layers"], *initial_state),
    )
    selected = layer_outputs[jnp.array([5], dtype=jnp.int32)]
    final_logits = lm_head_fwd(params["head"], x)
    auxiliary_logits = jax.vmap(lambda hidden: lm_head_fwd(params["head"], hidden))(
        selected
    )
    if "future_heads" not in params:
        return (final_logits, auxiliary_logits), final_state
    future_logits = jax.vmap(lambda head: lm_head_fwd(head, x))(
        params["future_heads"]
    )
    auxiliary_future_logits = jax.vmap(
        lambda head: jax.vmap(lambda hidden: lm_head_fwd(head, hidden))(selected)
    )(params["future_heads"])
    return (
        final_logits,
        auxiliary_logits,
        future_logits,
        auxiliary_future_logits,
    ), final_state


def supervised_loss(outputs, targets: jax.Array) -> jax.Array:
    logits, auxiliary_logits, future_logits, _ = outputs

    def nll(values, labels):
        logp = jax.nn.log_softmax(values.astype(jnp.float32), axis=-1)
        return -jnp.take_along_axis(logp, labels[..., None], axis=-1)[..., 0]

    loss = nll(logits, targets).mean()
    future_loss = nll(future_logits[:, 0, :-1], targets[:, 1:]).mean()
    auxiliary_logp = jax.nn.log_softmax(
        auxiliary_logits.astype(jnp.float32), axis=-1
    )
    auxiliary_nll = -jnp.take_along_axis(
        auxiliary_logp,
        targets[:, None, :, None],
        axis=-1,
    )[..., 0]
    auxiliary_loss = auxiliary_nll.mean()
    return loss + 0.5 * future_loss + 0.05 * auxiliary_loss


def batch_stateful_outputs(params, states, tokens, cfg):
    return jax.vmap(
        lambda sequence, state: stateful_deep_forward(params, sequence, cfg, state),
        in_axes=(0, 0),
    )(tokens, states)


def state_summary(state) -> dict[str, Any]:
    current, archive, vector = (np.asarray(jax.device_get(value)) for value in state)
    return {
        "all_finite": bool(
            np.isfinite(current).all()
            and np.isfinite(archive).all()
            and np.isfinite(vector).all()
        ),
        "current_rms": float(np.sqrt(np.mean(np.square(current, dtype=np.float64)))),
        "archive_rms": float(np.sqrt(np.mean(np.square(archive, dtype=np.float64)))),
        "vector_rms": float(np.sqrt(np.mean(np.square(vector, dtype=np.float64)))),
        "current_max_abs": float(np.max(np.abs(current))),
        "archive_max_abs": float(np.max(np.abs(archive))),
        "vector_max_abs": float(np.max(np.abs(vector))),
    }


def training_lane_starts(seed: int, required_bytes: int) -> np.ndarray:
    lane_width = TRAIN_END // 8
    rng = np.random.default_rng(20_260_823 + seed)
    starts = []
    for lane in range(8):
        zone_start = lane * lane_width
        zone_end = (lane + 1) * lane_width
        slack = zone_end - zone_start - required_bytes - 1
        if slack <= 0:
            raise ValueError("Training stream does not fit inside its disjoint lane")
        starts.append(zone_start + int(rng.integers(0, slack + 1)))
    return np.asarray(starts, dtype=np.int64)


def ordered_stream_sha256(data: np.ndarray, starts: np.ndarray, length: int) -> str:
    digest = hashlib.sha256()
    for start in starts:
        digest.update(np.asarray(data[start : start + length + 1], dtype=np.uint8).tobytes())
    return digest.hexdigest()


def batch_at(data: np.ndarray, starts: np.ndarray, length: int):
    offsets = np.arange(length + 1, dtype=np.int64)
    chunks = data[starts[:, None] + offsets[None, :]]
    return chunks[:, :-1].astype(np.int32), chunks[:, 1:].astype(np.int32)


def validation_lanes(validation: np.ndarray):
    lane_bytes = len(validation) // 8
    segments = (lane_bytes - 1) // 512
    usable = segments * 512
    x = np.empty((segments, 8, 512), dtype=np.int32)
    y = np.empty_like(x)
    for lane in range(8):
        base = lane * lane_bytes
        values = np.asarray(validation[base : base + usable + 1], dtype=np.int32)
        x[:, lane] = values[:-1].reshape(segments, 512)
        y[:, lane] = values[1:].reshape(segments, 512)
    return x, y


def parity_check(params, canonical_forward, cfg, validation) -> dict[str, float]:
    tokens = jnp.asarray(np.asarray(validation[:512], dtype=np.int32))
    zero = zero_state_single(cfg, params["embed"].dtype)
    canonical = jax.jit(lambda p, x: canonical_forward(p, x))
    stateful = jax.jit(lambda p, x, s: stateful_deep_forward(p, x, cfg, s))
    canonical_outputs = canonical(params, tokens)
    stateful_outputs, _ = stateful(params, tokens, zero)
    errors = [
        float(jnp.max(jnp.abs(left - right)))
        for left, right in zip(
            jax.tree_util.tree_leaves(canonical_outputs),
            jax.tree_util.tree_leaves(stateful_outputs),
        )
    ]
    report = {"maximum_output_error": max(errors), "leaf_errors": errors}
    print("CONTIGUOUS_TRAINING_PARITY", json.dumps(report), flush=True)
    if report["maximum_output_error"] > 1e-5:
        raise RuntimeError(f"Stateful forward parity failed: {report}")
    return report


def make_quick_eval(cfg):
    @jax.jit
    def evaluate_batch(model_params, states, tokens, targets):
        outputs, _ = batch_stateful_outputs(model_params, states, tokens, cfg)
        logits = outputs[0]
        logp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
        return -jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0].sum()

    return evaluate_batch


def quick_reset_validation(
    params, validation, zero_state, batch_sharding, evaluate_batch
):
    starts = np.linspace(0, len(validation) - 514, 128, dtype=np.int64)
    total = 0.0
    tokens_total = 0
    for offset in range(0, len(starts), 8):
        x, y = batch_at(validation, starts[offset : offset + 8], 512)
        value = evaluate_batch(
            params,
            zero_state,
            jax.device_put(x, batch_sharding),
            jax.device_put(y, batch_sharding),
        )
        total += float(value)
        tokens_total += x.size
    return total / (tokens_total * LN2)


def make_train_step(tx, cfg):
    @jax.jit
    def train_step(params, optimizer_state, recurrent_state, tokens, targets):
        def objective(model_params):
            outputs, next_state = batch_stateful_outputs(
                model_params, recurrent_state, tokens, cfg
            )
            return supervised_loss(outputs, targets), next_state

        (loss, next_state), gradients = jax.value_and_grad(
            objective, has_aux=True
        )(params)
        updates, optimizer_state = tx.update(gradients, optimizer_state, params)
        params = optax.apply_updates(params, updates)
        next_state = jax.tree_util.tree_map(lax.stop_gradient, next_state)
        gradient_norm = optax.global_norm(gradients)
        return params, optimizer_state, next_state, loss, gradient_norm

    return train_step


def save_checkpoint(
    path,
    name,
    step,
    params,
    optimizer_state,
    recurrent_state,
    rows,
    elapsed,
    provenance,
):
    block_tree((params, optimizer_state, recurrent_state))
    atomic_pickle(
        path,
        {
            "case": name,
            "step": step,
            "params": jax.device_get(params),
            "opt_state": jax.device_get(optimizer_state),
            "recurrent_state": jax.device_get(recurrent_state),
            "rows": rows,
            "elapsed_s": elapsed,
            "provenance": provenance,
        },
    )


def train_condition(
    name,
    carry,
    initial_params,
    tx,
    train_step,
    initialize_state,
    train,
    validation,
    lane_starts,
    total_steps,
    reset_interval,
    checkpoint_every,
    outdir,
    provenance,
    batch_sharding,
    replicated,
    quick_eval,
    resume,
):
    case_dir = outdir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = case_dir / "checkpoint.pkl"
    progress = case_dir / "progress.json"
    params = initial_params
    optimizer_state = jax.device_put(tx.init(initial_params), replicated)
    recurrent_state = initialize_state()
    rows = []
    start_step = 0
    elapsed_before = 0.0
    if resume and checkpoint.exists():
        with checkpoint.open("rb") as handle:
            saved = pickle.load(handle)
        if saved["case"] != name or saved["provenance"] != provenance:
            raise RuntimeError(f"Resume provenance mismatch for {name}")
        start_step = int(saved["step"])
        params = jax.device_put(saved["params"], replicated)
        optimizer_state = jax.device_put(saved["opt_state"], replicated)
        recurrent_state = tuple(
            jax.device_put(value, sharding)
            for value, sharding in zip(saved["recurrent_state"], initialize_state.shardings)
        )
        rows = saved["rows"]
        elapsed_before = float(saved["elapsed_s"])
        print("CONTIGUOUS_TRAINING_RESUME", name, start_step, flush=True)

    started = time.perf_counter()
    for step in range(start_step + 1, total_steps + 1):
        if not carry or (step - 1) % reset_interval == 0:
            recurrent_state = initialize_state()
        starts = lane_starts + (step - 1) * 512
        x, y = batch_at(train, starts, 512)
        params, optimizer_state, next_state, loss, gradient_norm = train_step(
            params,
            optimizer_state,
            recurrent_state,
            jax.device_put(x, batch_sharding),
            jax.device_put(y, batch_sharding),
        )
        recurrent_state = next_state if carry else initialize_state()
        if step == start_step + 1 or step % 25 == 0:
            block_tree((loss, gradient_norm))
            elapsed = elapsed_before + time.perf_counter() - started
            print(
                "CONTIGUOUS_TRAINING_PROGRESS",
                json.dumps(
                    {
                        "case": name,
                        "step": step,
                        "steps": total_steps,
                        "loss": float(loss),
                        "gradient_norm": float(gradient_norm),
                        "characters_per_second": (step - start_step) * 4096
                        / max(elapsed - elapsed_before, 1e-9),
                    }
                ),
                flush=True,
            )
        if step % checkpoint_every == 0 or step == total_steps:
            elapsed = elapsed_before + time.perf_counter() - started
            sparse_bpc = quick_reset_validation(
                params,
                validation,
                initialize_state(),
                batch_sharding,
                quick_eval,
            )
            summary = state_summary(recurrent_state)
            row = {
                "case": name,
                "step": step,
                "processed_characters": step * 4096,
                "loss": float(loss),
                "gradient_norm": float(gradient_norm),
                "sparse_reset_validation_bpc": sparse_bpc,
                "elapsed_s": elapsed,
                "state": summary,
            }
            rows.append(row)
            atomic_json(progress, {"provenance": provenance, "rows": rows})
            save_checkpoint(
                checkpoint,
                name,
                step,
                params,
                optimizer_state,
                recurrent_state,
                rows,
                elapsed,
                provenance,
            )
            print("CONTIGUOUS_TRAINING_CHECKPOINT", json.dumps(row), flush=True)
            if not math.isfinite(sparse_bpc) or not summary["all_finite"]:
                raise RuntimeError(f"Non-finite training state in {name}")
    elapsed = elapsed_before + time.perf_counter() - started
    return params, {"rows": rows, "elapsed_s": elapsed, "checkpoint": str(checkpoint)}


def age_bin(index: int) -> str:
    age = index + 1
    lower = 1 << int(math.floor(math.log2(age)))
    upper = (lower << 1) - 1
    return str(lower) if lower == upper else f"{lower}-{upper}"


def evaluate_contiguous(
    name,
    params,
    carry,
    cfg,
    x_host,
    y_host,
    initialize_state,
    batch_sharding,
    eval_step,
):
    state = initialize_state()
    position_sum = np.zeros(512, dtype=np.float64)
    age = {}
    started = time.perf_counter()
    for segment in range(x_host.shape[0]):
        if not carry:
            state = initialize_state()
        state, nll = eval_step(
            params,
            state,
            jax.device_put(x_host[segment], batch_sharding),
            jax.device_put(y_host[segment], batch_sharding),
        )
        host = np.asarray(jax.device_get(nll), dtype=np.float64)
        position_sum += host
        label = age_bin(segment)
        bucket = age.setdefault(label, {"nll": 0.0, "tokens": 0})
        bucket["nll"] += float(host.sum())
        bucket["tokens"] += 8 * 512
        if segment == 0 or (segment + 1) % 256 == 0 or segment + 1 == x_host.shape[0]:
            print(
                "CONTIGUOUS_VALIDATION_PROGRESS",
                name,
                segment + 1,
                x_host.shape[0],
                flush=True,
            )
    elapsed = time.perf_counter() - started
    total_tokens = x_host.size
    report = {
        "condition": name,
        "carry": carry,
        "bpc": float(position_sum.sum() / (total_tokens * LN2)),
        "position_bands_bpc": {
            "0:31": float(position_sum[:32].sum() / (x_host.shape[0] * 8 * 32 * LN2)),
            "32:127": float(position_sum[32:128].sum() / (x_host.shape[0] * 8 * 96 * LN2)),
            "128:511": float(position_sum[128:].sum() / (x_host.shape[0] * 8 * 384 * LN2)),
        },
        "segment_age_bpc": {
            key: value["nll"] / (value["tokens"] * LN2) for key, value in age.items()
        },
        "tokens": int(total_tokens),
        "evaluation_seconds": elapsed,
        "tokens_per_second": total_tokens / elapsed,
        "final_state": state_summary(state),
    }
    print("CONTIGUOUS_VALIDATION_COMPLETE", json.dumps(report), flush=True)
    return report


def make_contiguous_eval_step(cfg):
    @jax.jit
    def eval_step(model_params, recurrent_state, tokens, targets):
        outputs, next_state = batch_stateful_outputs(
            model_params, recurrent_state, tokens, cfg
        )
        logits = outputs[0]
        logp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
        nll = -jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
        return jax.tree_util.tree_map(lax.stop_gradient, next_state), jnp.sum(
            nll, axis=0
        )

    return eval_step


def decide(training_reports, evaluation_reports):
    control_reset = evaluation_reports["reset_contiguous_reset_eval"]["bpc"]
    control_carry = evaluation_reports["reset_contiguous_carry_eval"]["bpc"]
    candidate_reset = evaluation_reports["carry_contiguous_reset_eval"]["bpc"]
    candidate_carry = evaluation_reports["carry_contiguous_carry_eval"]["bpc"]
    carry_eval_gain = control_carry - candidate_carry
    reset_eval_degradation = candidate_reset - control_reset
    control_carry_benefit = control_reset - control_carry
    candidate_carry_benefit = candidate_reset - candidate_carry
    interaction = candidate_carry_benefit - control_carry_benefit
    runtime_ratio = (
        training_reports["carry_contiguous"]["elapsed_s"]
        / training_reports["reset_contiguous"]["elapsed_s"]
    )
    finite = all(
        report["final_state"]["all_finite"] for report in evaluation_reports.values()
    )
    checks = {
        "carry_eval_gain_at_least_0p010": carry_eval_gain >= 0.010,
        "reset_eval_degradation_at_most_0p010": reset_eval_degradation <= 0.010,
        "carry_interaction_at_least_0p005": interaction >= 0.005,
        "runtime_overhead_at_most_15pct": runtime_ratio <= 1.15,
        "all_finite": finite,
    }
    passed = all(checks.values())
    return {
        "stage": "20.48M matched contiguous-training validation-only screen",
        "carry_eval_gain": carry_eval_gain,
        "reset_eval_degradation": reset_eval_degradation,
        "control_carry_benefit": control_carry_benefit,
        "candidate_carry_benefit": candidate_carry_benefit,
        "carry_training_interaction": interaction,
        "training_runtime_ratio": runtime_ratio,
        "promotion_checks": checks,
        "promotion_pass": passed,
        "next": (
            "replicate the exact 2x2 protocol across two additional seeds"
            if passed
            else "freeze this exact contiguous-training protocol without tuning"
        ),
        "test_data_read": False,
    }


def main() -> None:
    args = parse_args()
    if jax.default_backend() != "tpu" or jax.device_count() != 8:
        raise RuntimeError(
            f"Expected fresh TPU v5e-8, found {jax.default_backend()} with {jax.device_count()} devices"
        )
    if args.target_characters != 20_480_000:
        raise ValueError("The preregistered screen requires exactly 20,480,000 characters")
    if args.reset_interval != 1_250:
        raise ValueError("The preregistered carry reset interval is 1,250 segments")
    if not args.data_path.exists() or args.data_path.stat().st_size != 100_000_000:
        raise RuntimeError("Expected exact 100,000,000-byte enwik8 file")
    args.outdir.mkdir(parents=True, exist_ok=True)

    cfg = model_config()
    initial_host, canonical_forward = make_initial_model(args.seed, cfg)
    params_count = count_params(initial_host)
    if params_count != EXPECTED_PARAMS:
        raise RuntimeError(f"Parameter mismatch: {params_count} != {EXPECTED_PARAMS}")
    initial_fingerprint = tree_sha256(initial_host)
    mesh = Mesh(np.array(jax.devices(), dtype=object), ("data",))
    replicated = NamedSharding(mesh, P())
    batch_sharding = NamedSharding(mesh, P("data", None))
    state_shardings = (
        NamedSharding(mesh, P("data", None, None, None)),
        NamedSharding(mesh, P("data", None, None, None)),
        NamedSharding(mesh, P("data", None, None)),
    )
    initial_params = jax.device_put(initial_host, replicated)
    del initial_host
    dtype = initial_params["embed"].dtype

    def unsharded_state():
        single = zero_state_single(cfg, dtype)
        return tuple(jnp.broadcast_to(value, (8,) + value.shape) for value in single)

    initialize_jit = jax.jit(unsharded_state, out_shardings=state_shardings)

    class StateInitializer:
        shardings = state_shardings

        def __call__(self):
            return initialize_jit()

    initialize_state = StateInitializer()

    data = np.memmap(args.data_path, dtype=np.uint8, mode="r")
    train = data[:TRAIN_END]
    validation = data[VALIDATION_START:VALIDATION_END]
    total_steps = args.target_characters // 4096
    required_per_lane = total_steps * 512
    lane_starts = training_lane_starts(args.seed, required_per_lane)
    stream_digest = ordered_stream_sha256(train, lane_starts, required_per_lane)
    provenance = {
        "protocol": "matched_contiguous_training_47m_v1",
        "seed": args.seed,
        "params": params_count,
        "initial_parameter_sha256": initial_fingerprint,
        "dataset_path": str(args.data_path),
        "dataset_sha256": sha256(args.data_path),
        "training_lane_starts": lane_starts.tolist(),
        "ordered_training_stream_sha256": stream_digest,
        "target_characters": args.target_characters,
        "total_steps": total_steps,
        "reset_interval": args.reset_interval,
        "validation_range": [VALIDATION_START, VALIDATION_END],
        "test_range_read": False,
        "devices": [str(device) for device in jax.devices()],
    }
    atomic_json(args.outdir / "provenance.json", provenance)
    print("CONTIGUOUS_TRAINING_CONFIG", json.dumps(provenance), flush=True)
    parity = parity_check(initial_params, canonical_forward, cfg, validation)
    atomic_json(args.outdir / "parity.json", parity)

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(6e-4, weight_decay=1e-4),
    )
    train_step = make_train_step(tx, cfg)
    quick_eval = make_quick_eval(cfg)
    training_reports = {}
    trained_params = {}
    for name, carry in (("reset_contiguous", False), ("carry_contiguous", True)):
        trained_params[name], training_reports[name] = train_condition(
            name,
            carry,
            initial_params,
            tx,
            train_step,
            initialize_state,
            train,
            validation,
            lane_starts,
            total_steps,
            args.reset_interval,
            args.checkpoint_every,
            args.outdir,
            provenance,
            batch_sharding,
            replicated,
            quick_eval,
            args.resume,
        )

    x_validation, y_validation = validation_lanes(validation)
    contiguous_eval_step = make_contiguous_eval_step(cfg)
    evaluation_reports = {}
    for training_name in ("reset_contiguous", "carry_contiguous"):
        for evaluation_name, carry in (("reset_eval", False), ("carry_eval", True)):
            name = f"{training_name}_{evaluation_name}"
            evaluation_reports[name] = evaluate_contiguous(
                name,
                trained_params[training_name],
                carry,
                cfg,
                x_validation,
                y_validation,
                initialize_state,
                batch_sharding,
                contiguous_eval_step,
            )
            atomic_json(args.outdir / f"{name}.json", evaluation_reports[name])

    final_decision = decide(training_reports, evaluation_reports)
    report = {
        "status": "PASS" if final_decision["promotion_pass"] else "COMPLETE_NO_PROMOTION",
        "provenance": provenance,
        "parity": parity,
        "training": training_reports,
        "evaluation": evaluation_reports,
        "decision": final_decision,
    }
    atomic_json(args.outdir / "contiguous_training_screen.json", report)
    atomic_json(args.outdir / "decision.json", final_decision)
    print("CONTIGUOUS_TRAINING_DECISION", json.dumps(final_decision), flush=True)
    print("CONTIGUOUS_TRAINING_SCREEN_SAVED", args.outdir, flush=True)


if __name__ == "__main__":
    main()
