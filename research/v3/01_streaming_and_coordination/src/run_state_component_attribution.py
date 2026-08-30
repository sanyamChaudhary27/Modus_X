from __future__ import annotations

import argparse
import itertools
import json
import math
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


LN2 = math.log(2.0)
STATE_NAMES = ("current", "archive", "vector")
VALIDATION_START = 90_000_000
VALIDATION_END = 95_000_000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def age_bin(index: int) -> str:
    age = index + 1
    lower = 1 << int(math.floor(math.log2(age)))
    upper = (lower << 1) - 1
    return str(lower) if lower == upper else f"{lower}-{upper}"


def mask_name(mask: tuple[bool, bool, bool]) -> str:
    selected = [name for name, enabled in zip(STATE_NAMES, mask) if enabled]
    return "none" if not selected else "_".join(selected) if len(selected) < 3 else "all"


def all_masks():
    return list(itertools.product((False, True), repeat=3))


def apply_carry_mask(next_state, zero_state, mask):
    return tuple(next_value if enabled else zero_value for next_value, zero_value, enabled in zip(next_state, zero_state, mask))


def shapley_values(bpc_by_mask: dict[str, float]) -> dict[str, float]:
    masks = all_masks()
    by_tuple = {mask: bpc_by_mask[mask_name(mask)] for mask in masks}
    values = {}
    n = len(STATE_NAMES)
    for index, component in enumerate(STATE_NAMES):
        contribution = 0.0
        for mask in masks:
            if mask[index]:
                continue
            size = sum(mask)
            weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
            with_component = list(mask)
            with_component[index] = True
            contribution += weight * (by_tuple[mask] - by_tuple[tuple(with_component)])
        values[component] = contribution
    return values


def evaluate_mask(name, params, mask, cfg, x_host, y_host, initialize_state, batch_sharding, eval_step):
    zero_state = initialize_state()
    state = zero_state
    position_sum = np.zeros(512, dtype=np.float64)
    age = {}
    started = time.perf_counter()
    for segment in range(x_host.shape[0]):
        next_state, nll = eval_step(
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
        state = apply_carry_mask(next_state, zero_state, mask)
        if segment == 0 or (segment + 1) % 256 == 0 or segment + 1 == x_host.shape[0]:
            print("STATE_ATTRIBUTION_PROGRESS", name, segment + 1, x_host.shape[0], flush=True)
    elapsed = time.perf_counter() - started
    total_tokens = int(x_host.size)
    report = {
        "condition": name,
        "mask": dict(zip(STATE_NAMES, mask)),
        "bpc": float(position_sum.sum() / (total_tokens * LN2)),
        "position_bands_bpc": {
            "0:31": float(position_sum[:32].sum() / (x_host.shape[0] * 8 * 32 * LN2)),
            "32:127": float(position_sum[32:128].sum() / (x_host.shape[0] * 8 * 96 * LN2)),
            "128:511": float(position_sum[128:].sum() / (x_host.shape[0] * 8 * 384 * LN2)),
        },
        "segment_age_bpc": {key: row["nll"] / (row["tokens"] * LN2) for key, row in age.items()},
        "tokens": total_tokens,
        "evaluation_seconds": elapsed,
        "tokens_per_second": total_tokens / elapsed,
        "all_finite": all(bool(jnp.all(jnp.isfinite(value))) for value in state),
    }
    print("STATE_ATTRIBUTION_COMPLETE", json.dumps(report), flush=True)
    return report


def main() -> None:
    args = parse_args()
    if jax.default_backend() != "tpu" or jax.device_count() != 8:
        raise RuntimeError(f"Expected fresh TPU v5e-8, found {jax.default_backend()} with {jax.device_count()} devices")
    if not args.data_path.is_file() or args.data_path.stat().st_size != 100_000_000:
        raise RuntimeError("Expected canonical 100,000,000-byte enwik8")
    sys.path.insert(0, str(args.code_root))
    import run_contiguous_training_screen as base

    cfg = base.model_config()
    mesh = Mesh(np.array(jax.devices(), dtype=object), ("data",))
    replicated = NamedSharding(mesh, P())
    batch_sharding = NamedSharding(mesh, P("data", None))
    state_shardings = (
        NamedSharding(mesh, P("data", None, None, None)),
        NamedSharding(mesh, P("data", None, None, None)),
        NamedSharding(mesh, P("data", None, None)),
    )

    def unsharded_state():
        single = base.zero_state_single(cfg, jnp.float32)
        return tuple(jnp.broadcast_to(value, (8,) + value.shape) for value in single)

    initialize_jit = jax.jit(unsharded_state, out_shardings=state_shardings)

    def initialize_state():
        return initialize_jit()

    @jax.jit
    def eval_step(model_params, recurrent_state, tokens, targets):
        outputs, next_state = base.batch_stateful_outputs(model_params, recurrent_state, tokens, cfg)
        logp = jax.nn.log_softmax(outputs[0].astype(jnp.float32), axis=-1)
        nll = -jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
        return jax.tree_util.tree_map(lax.stop_gradient, next_state), jnp.sum(nll, axis=0)

    data = np.memmap(args.data_path, dtype=np.uint8, mode="r")
    x_host, y_host = base.validation_lanes(data[VALIDATION_START:VALIDATION_END])
    args.outdir.mkdir(parents=True, exist_ok=True)
    all_reports = {}
    for training_name in ("reset_contiguous", "carry_contiguous"):
        checkpoint = args.experiment_root / training_name / "checkpoint.pkl"
        if not checkpoint.is_file():
            raise RuntimeError(f"Missing checkpoint: {checkpoint}")
        with checkpoint.open("rb") as handle:
            saved = pickle.load(handle)
        if int(saved["step"]) != 5000 or int(saved["provenance"]["seed"]) != args.seed:
            raise RuntimeError(f"Checkpoint provenance mismatch: {checkpoint}")
        params = jax.device_put(saved["params"], replicated)
        reports = {}
        for mask in all_masks():
            key = mask_name(mask)
            reports[key] = evaluate_mask(
                f"{training_name}_{key}", params, mask, cfg, x_host, y_host,
                initialize_state, batch_sharding, eval_step,
            )
            atomic_json(args.outdir / f"{training_name}_{key}.json", reports[key])
        bpc = {key: row["bpc"] for key, row in reports.items()}
        shapley = shapley_values(bpc)
        summary = {
            "bpc": bpc,
            "benefit_vs_none": {key: bpc["none"] - value for key, value in bpc.items()},
            "shapley_bpc_benefit": shapley,
            "joint_matrix_value_beyond_vector": bpc["vector"] - bpc["all"],
            "vector_value_beyond_matrices": bpc["current_archive"] - bpc["all"],
            "archive_full_context_marginal": bpc["current_vector"] - bpc["all"],
            "current_full_context_marginal": bpc["archive_vector"] - bpc["all"],
            "shapley_sum_error": sum(shapley.values()) - (bpc["none"] - bpc["all"]),
            "all_finite": all(row["all_finite"] for row in reports.values()),
        }
        all_reports[training_name] = {"evaluations": reports, "summary": summary}
        print("STATE_ATTRIBUTION_MODEL_SUMMARY", training_name, json.dumps(summary), flush=True)
        del params
    final = {
        "seed": args.seed,
        "stage": "frozen_checkpoint_full_factorial_validation_attribution",
        "models": all_reports,
        "test_data_read": False,
    }
    atomic_json(args.outdir / "state_component_attribution.json", final)
    print("STATE_COMPONENT_ATTRIBUTION_COMPLETE", json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
