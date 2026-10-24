from __future__ import annotations

import argparse
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
VALIDATION_START = 90_000_000
VALIDATION_END = 95_000_000
EXPECTED_PARAMS = 47_437_768
EXPECTED_STEP = 25_000
STATE_MASKS = {
    "reset_all": (False, False, False),
    "full_carry": (True, True, True),
    "no_archive_carry": (True, False, True),
    "archive_only_carry": (False, True, False),
}
TARGET_INTERVENTIONS = {
    "full_state": (True, True, True),
    "archive_erased": (True, False, True),
    "current_erased": (False, True, True),
    "vector_erased": (True, True, False),
    "all_erased": (False, False, False),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2)
    return parser.parse_args()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def state_with_mask(state, zero_state, mask):
    return tuple(value if keep else zero for value, zero, keep in zip(state, zero_state, mask))


def stream_eval(name, params, state0, x_host, y_host, batch_sharding, eval_step, base):
    mask = STATE_MASKS[name]
    state = state0
    nll_sum = 0.0
    started = time.perf_counter()
    for segment in range(x_host.shape[0]):
        if name == "reset_all":
            state = state0
        state, nll = eval_step(
            params,
            state,
            jax.device_put(x_host[segment], batch_sharding),
            jax.device_put(y_host[segment], batch_sharding),
        )
        nll_sum += float(np.asarray(jax.device_get(nll), dtype=np.float64).sum())
        state = state_with_mask(state, state0, mask)
        if segment == 0 or (segment + 1) % 256 == 0 or segment + 1 == x_host.shape[0]:
            print("OVER_RETENTION_STREAM_PROGRESS", name, segment + 1, x_host.shape[0], flush=True)
    elapsed = time.perf_counter() - started
    result = {
        "condition": name,
        "bpc": nll_sum / (x_host.size * LN2),
        "tokens": int(x_host.size),
        "evaluation_seconds": elapsed,
        "tokens_per_second": x_host.size / elapsed,
        "final_state": base.state_summary(state),
    }
    print("OVER_RETENTION_STREAM_COMPLETE", json.dumps(result), flush=True)
    return result


def natural_target_eval(params, state0, rows, validation, batch_sharding, advance_step, score_step, natural):
    x_host, y_host, score_host = natural.build_batch(validation, rows, False)
    state = state0
    for segment in range(x_host.shape[0] - 1):
        state = advance_step(params, state, jax.device_put(x_host[segment], batch_sharding))
    results = {}
    for name, mask in TARGET_INTERVENTIONS.items():
        intervened = state_with_mask(state, state0, mask)
        nll = score_step(
            params,
            intervened,
            jax.device_put(x_host[-1], batch_sharding),
            jax.device_put(y_host[-1], batch_sharding),
            jax.device_put(score_host[-1], batch_sharding),
        )
        per_example_bpc = np.asarray(jax.device_get(nll), dtype=np.float64) / (
            natural.CONTINUATION_BYTES * LN2
        )
        results[name] = {
            "bpc_mean": float(per_example_bpc.mean()),
            "bpc_stdev": float(per_example_bpc.std(ddof=1)),
            "per_example_bpc": per_example_bpc.tolist(),
            "all_finite": bool(np.isfinite(per_example_bpc).all()),
        }
    full = results["full_state"]["bpc_mean"]
    row = {
        "repeat_class": rows[0]["repeat_class"],
        "distance_band": rows[0]["distance_band"],
        "mean_distance_bytes": float(np.mean([item["distance_bytes"] for item in rows])),
        "conditions": results,
        "archive_target_contribution_bpc": results["archive_erased"]["bpc_mean"] - full,
        "current_target_contribution_bpc": results["current_erased"]["bpc_mean"] - full,
        "vector_target_contribution_bpc": results["vector_erased"]["bpc_mean"] - full,
        "all_state_target_contribution_bpc": results["all_erased"]["bpc_mean"] - full,
    }
    print("OVER_RETENTION_NATURAL_CELL", json.dumps(row), flush=True)
    return row


def summarize(streams, cells):
    by_name = {row["condition"]: row for row in streams}
    conflicting = [row for row in cells if row["repeat_class"] == "conflicting"]
    consistent = [row for row in cells if row["repeat_class"] == "consistent"]
    long_conflicting = [
        row for row in conflicting if row["distance_band"] in ("long", "very_long")
    ]
    finite = all(
        condition["all_finite"]
        for row in cells
        for condition in row["conditions"].values()
    ) and all(row["final_state"]["all_finite"] for row in streams)
    archive_stream_benefit = by_name["no_archive_carry"]["bpc"] - by_name["full_carry"]["bpc"]
    conflicting_mean = float(np.mean([row["archive_target_contribution_bpc"] for row in conflicting]))
    long_mean = float(np.mean([row["archive_target_contribution_bpc"] for row in long_conflicting]))
    severe = sum(row["archive_target_contribution_bpc"] < -0.01 for row in conflicting)
    checks = {
        "archive_stream_benefit_at_least_0p001": archive_stream_benefit >= 0.001,
        "conflicting_mean_archive_contribution_nonnegative": conflicting_mean >= 0.0,
        "conflicting_long_mean_archive_contribution_nonnegative": long_mean >= 0.0,
        "no_severely_harmful_conflicting_band": severe == 0,
        "all_values_finite": finite,
    }
    return {
        "stream": {
            "full_carry_benefit_vs_reset_bpc": by_name["reset_all"]["bpc"] - by_name["full_carry"]["bpc"],
            "archive_incremental_benefit_bpc": archive_stream_benefit,
            "archive_only_benefit_vs_reset_bpc": by_name["reset_all"]["bpc"] - by_name["archive_only_carry"]["bpc"],
        },
        "natural": {
            "consistent_mean_archive_contribution_bpc": float(np.mean([row["archive_target_contribution_bpc"] for row in consistent])),
            "conflicting_mean_archive_contribution_bpc": conflicting_mean,
            "conflicting_long_and_very_long_mean_archive_contribution_bpc": long_mean,
            "severely_harmful_conflicting_bands": int(severe),
        },
        "promotion_checks": checks,
        "over_retention_screen_pass": all(checks.values()),
        "next": (
            "retain segment-scale archive retention as the leading v3 candidate"
            if all(checks.values())
            else "treat update-aware retention as the next causal architecture problem"
        ),
    }


def main():
    args = parse_args()
    if jax.default_backend() != "tpu" or jax.device_count() != 8:
        raise RuntimeError(f"Expected TPU v5e-8, found {jax.default_backend()} with {jax.device_count()} devices")
    if not args.data_path.is_file() or args.data_path.stat().st_size != 100_000_000:
        raise RuntimeError("Expected canonical 100,000,000-byte enwik8")
    sys.path.insert(0, str(args.code_root))
    import models
    import run_contiguous_training_screen as base
    import run_natural_delayed_recall as natural
    from segment_scale_trainer import segment_scale_memory_feedback_stateful

    models.modus_x_memory_feedback_archive_layer_fwd_stateful = segment_scale_memory_feedback_stateful
    base.modus_x_memory_feedback_archive_layer_fwd_stateful = segment_scale_memory_feedback_stateful

    with args.checkpoint.open("rb") as handle:
        saved = pickle.load(handle)
    step = int(saved.get("step", -1))
    params = saved["params"]
    if step != EXPECTED_STEP or base.count_params(params) != EXPECTED_PARAMS:
        raise RuntimeError(f"Checkpoint mismatch: step={step}, params={base.count_params(params)}")

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

    initialize = jax.jit(unsharded_state, out_shardings=state_shardings)

    @jax.jit
    def eval_step(model_params, state, tokens, targets):
        outputs, next_state = base.batch_stateful_outputs(model_params, state, tokens, cfg)
        logp = jax.nn.log_softmax(outputs[0].astype(jnp.float32), axis=-1)
        nll = -jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
        return jax.tree_util.tree_map(lax.stop_gradient, next_state), jnp.sum(nll, axis=0)

    @jax.jit
    def advance_step(model_params, state, tokens):
        _, next_state = base.batch_stateful_outputs(model_params, state, tokens, cfg)
        return jax.tree_util.tree_map(lax.stop_gradient, next_state)

    @jax.jit
    def score_step(model_params, state, tokens, targets, score_mask):
        outputs, _ = base.batch_stateful_outputs(model_params, state, tokens, cfg)
        logp = jax.nn.log_softmax(outputs[0].astype(jnp.float32), axis=-1)
        nll = -jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
        return jnp.sum(nll * score_mask, axis=-1)

    device_params = jax.device_put(params, replicated)
    data = np.memmap(args.data_path, dtype=np.uint8, mode="r")
    validation = np.asarray(data[VALIDATION_START:VALIDATION_END], dtype=np.uint8)
    x_host, y_host = base.validation_lanes(validation)
    state0 = initialize()

    streams = [
        stream_eval(name, device_params, state0, x_host, y_host, batch_sharding, eval_step, base)
        for name in STATE_MASKS
    ]
    pairs, counts = natural.discover_pairs(validation)
    print("OVER_RETENTION_PAIR_COUNTS", json.dumps(counts), flush=True)
    cells = []
    manifest = []
    for kind in ("consistent", "conflicting"):
        for band in natural.BANDS:
            rows = pairs[(kind, band)]
            manifest.extend(rows)
            cells.append(
                natural_target_eval(
                    device_params, state0, rows, validation, batch_sharding,
                    advance_step, score_step, natural,
                )
            )

    decision = summarize(streams, cells)
    result = {
        "seed": args.seed,
        "stage": "frozen_segment_scale_over_retention_audit",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": step,
        "params": EXPECTED_PARAMS,
        "stream_evaluations": streams,
        "pair_counts": counts,
        "pair_manifest": manifest,
        "natural_cells": cells,
        "decision": decision,
        "test_data_read": False,
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.outdir / "over_retention_audit.json", result)
    print("OVER_RETENTION_DECISION", json.dumps(decision), flush=True)


if __name__ == "__main__":
    main()
