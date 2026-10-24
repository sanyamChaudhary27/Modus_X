from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


LN2 = math.log(2.0)
VALIDATION_START = 90_000_000
VALIDATION_END = 95_000_000
TRACE_AGES = (0, 1, 2, 4, 8, 16, 32, 64, 128)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--segment-scale-archive", action="store_true")
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--expected-step", type=int, default=5000)
    return parser.parse_args()


def atomic_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def summarize_trace(rows):
    indexed = {(row["repeat_class"], row["distance_band"]): row for row in rows}
    cells = []
    for key, row in indexed.items():
        initial = np.asarray(row["source_delta_rms"]["post_source"], dtype=np.float64)
        target = np.asarray(row["source_delta_rms"]["target"], dtype=np.float64)
        ratio = np.divide(target, initial, out=np.zeros_like(target), where=initial > 0)
        cells.append({
            "repeat_class": key[0],
            "distance_band": key[1],
            "post_source_delta_rms": initial.tolist(),
            "target_delta_rms": target.tolist(),
            "target_to_post_source_delta_ratio": ratio.tolist(),
            "patch_gain_bpc": row["patch_gain_bpc"],
        })
    patch_names = ("current", "archive", "matrices", "vector", "all")
    return {
        "components_order": ["current", "archive", "vector"],
        "cells": cells,
        "aggregate_mean_patch_gain_bpc": {
            name: float(np.mean([cell["patch_gain_bpc"][name] for cell in cells]))
            for name in patch_names
        },
        "interpretation_rule": {
            "write_failure": "post-source state delta is negligible",
            "retention_failure": "post-source delta is active but target delta vanishes",
            "read_use_failure": "target delta remains active but all-state causal patch has negligible NLL effect"
        },
        "claim_boundary": "State deltas show perturbation persistence, not semantic decoding."
    }


def main():
    args = parse_args()
    if jax.default_backend() != "tpu" or jax.device_count() != 8:
        raise RuntimeError(f"Expected TPU v5e-8, found {jax.default_backend()} with {jax.device_count()} devices")
    if args.data_path.stat().st_size != 100_000_000:
        raise RuntimeError("Expected canonical enwik8")
    sys.path.insert(0, str(args.code_root))
    import run_contiguous_training_screen as base
    import run_natural_delayed_recall as natural

    if args.segment_scale_archive:
        import models as model_module
        from segment_scale_trainer import segment_scale_memory_feedback_stateful

        model_module.modus_x_memory_feedback_archive_layer_fwd_stateful = (
            segment_scale_memory_feedback_stateful
        )
        base.modus_x_memory_feedback_archive_layer_fwd_stateful = (
            segment_scale_memory_feedback_stateful
        )

    if args.checkpoint_path is not None:
        checkpoint = args.checkpoint_path
    elif args.experiment_root is not None:
        checkpoint = args.experiment_root / "reset_contiguous" / "checkpoint.pkl"
    else:
        raise RuntimeError("Provide --checkpoint-path or --experiment-root")
    with checkpoint.open("rb") as handle:
        saved = pickle.load(handle)
    saved_seed = int(saved.get("provenance", {}).get("seed", args.seed))
    if int(saved["step"]) != args.expected_step or saved_seed != args.seed:
        raise RuntimeError(f"Checkpoint provenance mismatch: {checkpoint}")

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

    def batch_rms(value):
        axes = tuple(range(1, value.ndim))
        return jnp.sqrt(jnp.mean(jnp.square(value.astype(jnp.float32)), axis=axes))

    @jax.jit
    def advance_pair(model_params, original_state, removed_state, original_tokens, removed_tokens):
        _, next_original = base.batch_stateful_outputs(model_params, original_state, original_tokens, cfg)
        _, next_removed = base.batch_stateful_outputs(model_params, removed_state, removed_tokens, cfg)
        delta = jnp.stack([batch_rms(left - right) for left, right in zip(next_original, next_removed)])
        return (
            jax.tree_util.tree_map(lax.stop_gradient, next_original),
            jax.tree_util.tree_map(lax.stop_gradient, next_removed),
            delta,
        )

    @jax.jit
    def score(model_params, recurrent_state, tokens, targets, score_mask):
        outputs, _ = base.batch_stateful_outputs(model_params, recurrent_state, tokens, cfg)
        logp = jax.nn.log_softmax(outputs[0].astype(jnp.float32), axis=-1)
        nll = -jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
        return jnp.sum(nll * score_mask, axis=-1)

    params = jax.device_put(saved["params"], replicated)
    data = np.memmap(args.data_path, dtype=np.uint8, mode="r")
    validation = np.asarray(data[VALIDATION_START:VALIDATION_END], dtype=np.uint8)
    pairs, counts = natural.discover_pairs(validation)
    print("SOURCE_TRACE_PAIRS", json.dumps(counts), flush=True)
    state0 = initialize()
    rows = []

    for kind in ("consistent", "conflicting"):
        for band in natural.BANDS:
            pair_rows = pairs[(kind, band)]
            original_x, original_y, score_mask = natural.build_batch(validation, pair_rows, False)
            removed_x, removed_y, removed_score = natural.build_batch(validation, pair_rows, True)
            if not np.array_equal(original_y[-1], removed_y[-1]) or not np.array_equal(score_mask, removed_score):
                raise RuntimeError("Scored target changed under source removal")
            original_state = state0
            removed_state = state0
            per_segment_delta = []
            for segment in range(original_x.shape[0] - 1):
                original_state, removed_state, delta = advance_pair(
                    params,
                    original_state,
                    removed_state,
                    jax.device_put(original_x[segment], batch_sharding),
                    jax.device_put(removed_x[segment], batch_sharding),
                )
                per_segment_delta.append(np.asarray(jax.device_get(delta), dtype=np.float64))

            source_segments = np.asarray([
                (item["first_position"] - (item["second_position"] - item["horizon_segments"] * 512)) // 512
                for item in pair_rows
            ], dtype=np.int64)
            trace = {}
            for age in TRACE_AGES:
                values = []
                for lane, source_segment in enumerate(source_segments):
                    index = min(int(source_segment + age), len(per_segment_delta) - 1)
                    values.append(per_segment_delta[index][:, lane])
                trace[str(age)] = np.mean(np.stack(values), axis=0).tolist()
            post_source = np.mean(np.stack([
                per_segment_delta[int(source_segment)][:, lane]
                for lane, source_segment in enumerate(source_segments)
            ]), axis=0)
            target_delta = np.mean(per_segment_delta[-1], axis=1)
            trace["post_source"] = post_source.tolist()
            trace["target"] = target_delta.tolist()

            patch_states = {
                "none": removed_state,
                "current": (original_state[0], removed_state[1], removed_state[2]),
                "archive": (removed_state[0], original_state[1], removed_state[2]),
                "matrices": (original_state[0], original_state[1], removed_state[2]),
                "vector": (removed_state[0], removed_state[1], original_state[2]),
                "all": original_state,
            }
            final_tokens = jax.device_put(original_x[-1], batch_sharding)
            final_targets = jax.device_put(original_y[-1], batch_sharding)
            final_mask = jax.device_put(score_mask[-1], batch_sharding)
            bpc = {}
            for name, patch_state in patch_states.items():
                nll = np.asarray(jax.device_get(score(params, patch_state, final_tokens, final_targets, final_mask)))
                bpc[name] = float(np.mean(nll / (natural.CONTINUATION_BYTES * LN2)))
            baseline = bpc["none"]
            patch_gain = {name: baseline - value for name, value in bpc.items() if name != "none"}
            row = {
                "repeat_class": kind,
                "distance_band": band,
                "examples": natural.EXAMPLES_PER_CELL,
                "source_delta_rms": trace,
                "target_bpc": bpc,
                "patch_gain_bpc": patch_gain,
                "all_finite": bool(np.isfinite(np.asarray(list(bpc.values()))).all()),
            }
            rows.append(row)
            print("SOURCE_TRACE_CELL", json.dumps(row), flush=True)

    final = {
        "seed": args.seed,
        "stage": "frozen_source_write_retention_and_causal_patch_trace",
        "pair_counts": counts,
        "evaluations": rows,
        "summary": summarize_trace(rows),
        "test_data_read": False,
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.outdir / "source_trace.json", final)
    print("SOURCE_TRACE_DECISION", json.dumps(final["summary"]), flush=True)


if __name__ == "__main__":
    main()
