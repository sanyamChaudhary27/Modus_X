"""Full frozen-configuration Modus_X 1B train-step readiness gate.

This is a systems benchmark, not a quality-training run. It initializes the
exact public Modus_X parameter tree for the frozen 1B proposal with model-axis
sharding, allocates fp32 AdamW moments, and attempts two end-to-end training
updates at the requested context length.

The benchmark intentionally fails closed: an OOM, non-finite value, excessive
router saturation, or unusable projected throughput is recorded as a failed
configuration rather than hidden behind a smaller proxy.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from models import (
    ModelConfig,
    count_params,
    init_modus_x_lm,
    layer_norm,
    lm_head_fwd,
    modus_x_layer_fwd_stateful,
)


EXPECTED_PARAMS = 1_058_963_121
TOKENS_PER_ACCUMULATED_UPDATE = 32_768


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=2_048)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--data-axis-size", type=int, default=2)
    parser.add_argument("--timed-steps", type=int, default=1)
    parser.add_argument("--scan-chunk-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def config(seq_len: int) -> ModelConfig:
    return ModelConfig(
        vocab_size=50_257,
        embed_dim=1_536,
        hidden_dim=4_608,
        ax_res=1_024,
        n_layers=32,
        n_heads_attn=8,
        seq_len=seq_len,
        mamba_state_dim=1_024,
        vector_router=True,
        router_hidden=128,
    )


def path_string(path: tuple[Any, ...]) -> str:
    parts = []
    for entry in path:
        if hasattr(entry, "key"):
            parts.append(str(entry.key))
        elif hasattr(entry, "idx"):
            parts.append(str(entry.idx))
        else:
            parts.append(str(entry))
    return ".".join(parts)


def parameter_spec(path: str, shape: tuple[int, ...]) -> P:
    """Tensor-parallel layout for the exact stacked public parameter tree."""
    if not shape:
        return P()
    rank = len(shape)

    if path == "embed":
        return P(None, "model")
    if path == "head.w1":
        return P("model", None)
    if path == "head.b1":
        return P("model")
    if path == "head.w2":
        return P(None, "model")
    if path == "head.b2":
        return P()

    leaf = path.rsplit(".", 1)[-1]
    output_width_matrices = {
        "m_w_out",
        "m_proj_w",
        "s_w_gate",
        "s_proj_w",
        "r_proj",
    }
    output_width_vectors = {
        "pre_g",
        "pre_b",
        "m_b_out",
        "m_proj_b",
        "s_b_gate",
        "s_proj_b",
        "r_proj_b",
    }
    input_width_matrices = {
        "m_wk",
        "m_wq",
        "m_wv",
        "m_w_eta",
        "m_w_write",
        "m_w_ret",
        "m_w_read",
        "s_wu",
        "s_w_delta",
        "s_w_ret",
        "s_w_c",
        "r_w",
    }

    # Stacked layer arrays have the layer index as their leading dimension.
    if path.startswith("layers."):
        if leaf in output_width_matrices and rank == 3:
            return P(None, "model", None)
        if leaf in output_width_vectors and rank == 2:
            return P(None, "model")
        if leaf in input_width_matrices and rank == 3:
            return P(None, None, "model")
        return P()
    return P()


def make_sharding_tree(abstract_tree, mesh: Mesh):
    path_leaves, treedef = jax.tree_util.tree_flatten_with_path(abstract_tree)
    shardings = []
    rows = []
    for path, value in path_leaves:
        label = path_string(path)
        shape = tuple(int(dim) for dim in value.shape)
        spec = parameter_spec(label, shape)
        shardings.append(NamedSharding(mesh, spec))
        rows.append(
            {
                "path": label,
                "shape": list(shape),
                "params": int(math.prod(shape)),
                "spec": str(spec),
            }
        )
    return treedef.unflatten(shardings), rows


def tree_l2_sq(tree) -> jax.Array:
    terms = [
        jnp.sum(leaf.astype(jnp.float32) ** 2)
        for leaf in jax.tree_util.tree_leaves(tree)
    ]
    return sum(terms, jnp.array(0.0, dtype=jnp.float32))


def chunked_layer_fwd(
    layer: dict, x_seq: jax.Array, chunk_size: int
) -> jax.Array:
    """Run the exact recurrence while rematerializing within time chunks."""
    seq_len = x_seq.shape[0]
    if seq_len % chunk_size:
        raise ValueError((seq_len, chunk_size))
    chunks = x_seq.reshape(seq_len // chunk_size, chunk_size, x_seq.shape[-1])
    matrix_size = layer["m_wk"].shape[0]
    vector_size = layer["s_wu"].shape[0]
    initial_state = (
        jnp.zeros((matrix_size, matrix_size), dtype=jnp.float32),
        jnp.zeros((vector_size,), dtype=jnp.float32),
    )

    def run_chunk(state, x_chunk):
        return modus_x_layer_fwd_stateful(layer, x_chunk, state)

    _, outputs = jax.lax.scan(jax.checkpoint(run_chunk), initial_state, chunks)
    return outputs.reshape(seq_len, x_seq.shape[-1])


def loss_fn(
    params, tokens, targets, cfg: ModelConfig, scan_chunk_size: int
) -> jax.Array:
    def mixed_precision_forward(sequence):
        x = params["embed"][sequence]

        def scan_layer(x_in, layer):
            # The matrix/vector internals may promote sensitive operations to
            # fp32. The recurrent residual boundary is the explicit bf16
            # activation boundary required by lax.scan and the frozen runtime.
            x_out = x_in.astype(jnp.float32) + chunked_layer_fwd(
                layer, x_in, scan_chunk_size
            )
            return x_out.astype(jnp.bfloat16), None

        x, _ = jax.lax.scan(scan_layer, x, params["layers"])
        return lm_head_fwd(params["head"], x)

    logits = jax.vmap(mixed_precision_forward)(tokens)
    logits_fp32 = logits.astype(jnp.float32)
    target_logits = jnp.take_along_axis(
        logits_fp32, targets[..., None], axis=-1
    )[..., 0]
    return jnp.mean(jax.scipy.special.logsumexp(logits_fp32, axis=-1) - target_logits)


def layer_zero_router_stats(params, tokens) -> tuple[jax.Array, ...]:
    layer = jax.tree_util.tree_map(lambda value: value[0], params["layers"])
    x = params["embed"][tokens]
    e = layer_norm(x, layer["pre_g"], layer["pre_b"])
    hidden = jax.nn.gelu(e @ layer["r_w"].T + layer["r_b"])
    logits = hidden @ layer["r_proj"].T + layer["r_proj_b"]
    router = jax.nn.sigmoid(logits.astype(jnp.float32))
    saturation = jnp.mean((router < 0.01) | (router > 0.99))
    entropy = -jnp.mean(
        router * jnp.log(router + 1e-8)
        + (1.0 - router) * jnp.log(1.0 - router + 1e-8)
    )
    return jnp.mean(router), jnp.std(router), saturation, entropy


def memory_snapshot() -> list[dict[str, Any]]:
    def json_value(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {str(key): json_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_value(item) for item in value]
        return str(value)

    rows = []
    for device in jax.devices():
        try:
            stats = device.memory_stats() or {}
        except Exception:
            stats = {}
        rows.append({"device": str(device), "memory_stats": json_value(stats)})
    return rows


def block_tree(tree) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        leaf.block_until_ready()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    if jax.default_backend() != "tpu" or jax.device_count() != 8:
        raise RuntimeError(
            f"Expected Kaggle TPU v5e-8, got {jax.default_backend()} "
            f"with {jax.device_count()} devices"
        )
    if args.data_axis_size not in (1, 2, 4, 8):
        raise ValueError("--data-axis-size must divide the eight-device slice")
    if args.batch != args.data_axis_size:
        raise ValueError("Use one sequence per data replica for this topology gate")
    if args.seq_len not in (512, 1024, 2048):
        raise ValueError("--seq-len must be 512, 1024, or 2048")

    cfg = config(args.seq_len)
    model_axis_size = 8 // args.data_axis_size
    devices = np.array(jax.devices(), dtype=object).reshape(
        args.data_axis_size, model_axis_size
    )
    mesh = Mesh(devices, ("data", "model"))
    replicated = NamedSharding(mesh, P())
    data_sharding = NamedSharding(mesh, P("data", None))
    key = jax.random.key(args.seed)

    def abstract_init(init_key):
        params = init_modus_x_lm(init_key, cfg)
        return jax.tree_util.tree_map(lambda value: value.astype(jnp.bfloat16), params)

    abstract = jax.eval_shape(abstract_init, key)
    abstract_count = count_params(abstract)
    if abstract_count != EXPECTED_PARAMS:
        raise AssertionError((abstract_count, EXPECTED_PARAMS))
    param_shardings, sharding_rows = make_sharding_tree(abstract, mesh)

    print(
        "FULL_1B_CONFIG",
        json.dumps(
            {
                "params": abstract_count,
                "mesh": [args.data_axis_size, model_axis_size],
                "batch": args.batch,
                "seq_len": args.seq_len,
                "microbatch_tokens": args.batch * args.seq_len,
                "target_accumulated_tokens": TOKENS_PER_ACCUMULATED_UPDATE,
            }
        ),
        flush=True,
    )

    init_compiled = jax.jit(abstract_init, out_shardings=param_shardings)
    started = time.perf_counter()
    params = init_compiled(key)
    block_tree(params)
    initialization_seconds = time.perf_counter() - started
    actual_count = count_params(params)
    if actual_count != EXPECTED_PARAMS:
        raise AssertionError((actual_count, EXPECTED_PARAMS))

    # AdamW state is fp32 and follows the same model-axis partitioning.
    # Build zeros on device under the existing sharding. Creating one host zero
    # array per leaf would transiently materialize the full optimizer state on
    # the Kaggle VM and can fail before TPU memory is exercised.
    zero_moments = jax.jit(
        lambda tree: jax.tree_util.tree_map(
            lambda value: jnp.zeros_like(value, dtype=jnp.float32), tree
        ),
        in_shardings=(param_shardings,),
        out_shardings=param_shardings,
    )
    moments_m = zero_moments(params)
    moments_v = zero_moments(params)
    block_tree((moments_m, moments_v))
    memory_after_allocation = memory_snapshot()

    token_key, step_key = jax.random.split(key)
    tokens = jax.random.randint(
        token_key,
        (args.batch, args.seq_len),
        0,
        cfg.vocab_size,
        dtype=jnp.int32,
    )
    targets = jnp.roll(tokens, -1, axis=1)
    tokens = jax.device_put(tokens, data_sharding)
    targets = jax.device_put(targets, data_sharding)

    def train_step(p, m, v, token_batch, target_batch, step):
        loss, grads = jax.value_and_grad(loss_fn)(
            p, token_batch, target_batch, cfg, args.scan_chunk_size
        )
        beta1 = jnp.array(0.9, dtype=jnp.float32)
        beta2 = jnp.array(0.95, dtype=jnp.float32)
        step_f = step.astype(jnp.float32)
        new_m = jax.tree_util.tree_map(
            lambda old, grad: beta1 * old + (1.0 - beta1) * grad.astype(jnp.float32),
            m,
            grads,
        )
        new_v = jax.tree_util.tree_map(
            lambda old, grad: beta2 * old + (1.0 - beta2) * grad.astype(jnp.float32) ** 2,
            v,
            grads,
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
                - args.learning_rate
                * (
                    mean / (jnp.sqrt(variance) + 1e-8)
                    + args.weight_decay * value.astype(jnp.float32)
                )
            ).astype(jnp.bfloat16),
            p,
            corrected_m,
            corrected_v,
        )
        grad_norm = jnp.sqrt(tree_l2_sq(grads))
        return new_p, new_m, new_v, loss, grad_norm

    compiled_step = jax.jit(
        train_step,
        in_shardings=(
            param_shardings,
            param_shardings,
            param_shardings,
            data_sharding,
            data_sharding,
            replicated,
        ),
        out_shardings=(
            param_shardings,
            param_shardings,
            param_shardings,
            replicated,
            replicated,
        ),
        donate_argnums=(0, 1, 2),
    )
    router_probe = jax.jit(
        layer_zero_router_stats,
        in_shardings=(param_shardings, data_sharding),
        out_shardings=(replicated, replicated, replicated, replicated),
    )

    started = time.perf_counter()
    params, moments_m, moments_v, loss, grad_norm = compiled_step(
        params,
        moments_m,
        moments_v,
        tokens,
        targets,
        jax.device_put(jnp.array(1, dtype=jnp.int32), replicated),
    )
    block_tree((params, moments_m, moments_v, loss, grad_norm))
    compile_and_first_seconds = time.perf_counter() - started

    timings = []
    for step_number in range(2, 2 + args.timed_steps):
        started = time.perf_counter()
        params, moments_m, moments_v, loss, grad_norm = compiled_step(
            params,
            moments_m,
            moments_v,
            tokens,
            targets,
            jax.device_put(jnp.array(step_number, dtype=jnp.int32), replicated),
        )
        block_tree((params, moments_m, moments_v, loss, grad_norm))
        timings.append(time.perf_counter() - started)

    router_values = router_probe(params, tokens)
    block_tree(router_values)
    router_mean, router_std, router_saturation, router_entropy = map(
        float, router_values
    )
    loss_value = float(loss)
    grad_norm_value = float(grad_norm)
    median_step_seconds = float(np.median(timings))
    tokens_per_second = args.batch * args.seq_len / median_step_seconds
    accumulation_steps = math.ceil(
        TOKENS_PER_ACCUMULATED_UPDATE / (args.batch * args.seq_len)
    )
    projected = {
        label: token_count / tokens_per_second / 3600
        for label, token_count in {
            "100m_tokens_hours": 100_000_000,
            "2b_tokens_hours": 2_000_000_000,
            "5b_tokens_hours": 5_000_000_000,
            "20b_tokens_hours": 20_000_000_000,
        }.items()
    }
    finite = all(
        math.isfinite(value)
        for value in (loss_value, grad_norm_value, tokens_per_second)
    )
    abort_reasons = []
    if not finite:
        abort_reasons.append("non_finite_loss_gradient_or_throughput")
    if router_saturation > 0.98:
        abort_reasons.append("layer_zero_router_saturation_above_98_percent")
    if projected["100m_tokens_hours"] > 96:
        abort_reasons.append("projected_100m_systems_smoke_exceeds_96_hours")
    if compile_and_first_seconds > 3_600:
        abort_reasons.append("compile_and_first_step_exceeds_one_hour")

    report = {
        "status": "PASS" if not abort_reasons else "FAIL",
        "claim_boundary": "Full frozen 1B train-step systems gate; not quality training.",
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "mesh": {
            "shape": [args.data_axis_size, model_axis_size],
            "axes": ["data", "model"],
        },
        "config": {
            "vocab_size": cfg.vocab_size,
            "embed_dim": cfg.embed_dim,
            "hidden_dim": cfg.hidden_dim,
            "matrix_state_dim": cfg.ax_res,
            "vector_state_dim": cfg.mamba_state_dim,
            "n_layers": cfg.n_layers,
            "router_hidden": cfg.router_hidden,
            "context_length": cfg.seq_len,
            "matrix_scan_chunk_size": args.scan_chunk_size,
            "global_microbatch": args.batch,
            "microbatch_tokens": args.batch * args.seq_len,
            "gradient_accumulation_steps": accumulation_steps,
            "accumulated_tokens": accumulation_steps * args.batch * args.seq_len,
            "parameters": actual_count,
            "parameter_dtype": "bfloat16",
            "adamw_moment_dtype": "float32",
            "loss_reduction_dtype": "float32",
            "optimizer": {
                "name": "AdamW",
                "beta1": 0.9,
                "beta2": 0.95,
                "learning_rate_for_smoke": args.learning_rate,
                "weight_decay": args.weight_decay,
            },
        },
        "measurements": {
            "initialization_seconds": initialization_seconds,
            "compile_and_first_step_seconds": compile_and_first_seconds,
            "timed_step_seconds": timings,
            "median_step_seconds": median_step_seconds,
            "tokens_per_second": tokens_per_second,
            "loss": loss_value,
            "gradient_norm": grad_norm_value,
            "router_layer_zero": {
                "mean": router_mean,
                "std": router_std,
                "saturation_fraction": router_saturation,
                "binary_entropy_nats": router_entropy,
            },
        },
        "projected_single_v5e8_hours": projected,
        "abort_reasons": abort_reasons,
        "memory_after_parameter_and_optimizer_allocation": memory_after_allocation,
        "memory_after_train_step": memory_snapshot(),
        "sharding": sharding_rows,
        "scheduled_diagnostics": {
            "progress_every_optimizer_updates": 100,
            "throughput_and_memory_every_optimizer_updates": 100,
            "router_and_stream_diagnostics_every_tokens": 10_000_000,
            "held_out_evaluation_token_gates": [
                100_000_000,
                500_000_000,
                1_000_000_000,
                2_000_000_000,
                5_000_000_000,
            ],
            "local_checkpoint_minutes": 30,
            "durable_checkpoint_hours": 2,
        },
    }
    report_path = args.outdir / "full_1b_train_step_readiness_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.outdir / "full_1b_checkpoint_schema_probe.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "global_step": 1 + args.timed_steps,
                "processed_tokens": (1 + args.timed_steps)
                * args.batch
                * args.seq_len,
                "config": report["config"],
                "parameter_count": actual_count,
                "parameter_sharding_record_count": len(sharding_rows),
                "required_full_checkpoint_fields": [
                    "params",
                    "adam_m",
                    "adam_v",
                    "rng_state",
                    "global_step",
                    "processed_tokens",
                    "data_cursor",
                    "schedule_state",
                    "config",
                    "source_commit",
                    "metric_history",
                ],
                "note": (
                    "Reduced bit-identical restore is already validated. This full "
                    "gate records the exact sharded schema but deliberately does not "
                    "publish a roughly 10GB ephemeral smoke checkpoint."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("MODUS_X_FULL_1B_TRAIN_STEP_" + report["status"], flush=True)
    print(json.dumps(report, indent=2), flush=True)
    if abort_reasons:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
