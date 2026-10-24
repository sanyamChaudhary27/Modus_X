from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


LN2 = math.log(2.0)
VALIDATION_START = 90_000_000
VALIDATION_END = 95_000_000
CUE_BYTES = 16
CONTINUATION_BYTES = 24
EXAMPLES_PER_CELL = 8
BANDS = {
    "short": (512, 2_048, 4),
    "medium": (2_049, 8_192, 16),
    "long": (8_193, 32_768, 64),
    "very_long": (32_769, 131_072, 256),
}
CONDITIONS = {
    "reset": (False, False, False),
    "full_carry": (True, True, True),
    "matrix_only_carry": (True, True, False),
    "vector_only_carry": (False, False, True),
    "source_removed_full_carry": (True, True, True),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2)
    return parser.parse_args()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def band_for_distance(distance: int):
    for name, (minimum, maximum, segments) in BANDS.items():
        if minimum <= distance <= maximum:
            return name, segments
    return None


def pair_key(row):
    return row["first_position"], row["second_position"]


def discover_pairs(validation: np.ndarray):
    wanted = {(kind, band): EXAMPLES_PER_CELL for kind in ("consistent", "conflicting") for band in BANDS}
    selected = defaultdict(list)
    selected_pairs = set()

    # Four modulo scans cover every byte position while keeping the live anchor
    # index bounded. Exact 16-byte verification prevents 8-byte hash collisions.
    for phase in range(4):
        previous = {}
        for second in range(phase, len(validation) - CUE_BYTES - CONTINUATION_BYTES, 4):
            anchor = bytes(validation[second : second + 8])
            first = previous.get(anchor)
            previous[anchor] = second
            if first is None:
                continue
            distance = second - first
            band = band_for_distance(distance)
            if band is None:
                continue
            band_name, horizon_segments = band
            if second < horizon_segments * 512 or second + 513 > len(validation):
                continue
            cue_first = validation[first : first + CUE_BYTES]
            cue_second = validation[second : second + CUE_BYTES]
            if not np.array_equal(cue_first, cue_second) or len(np.unique(cue_first)) < 4:
                continue
            continuation_first = validation[first + CUE_BYTES : first + CUE_BYTES + CONTINUATION_BYTES]
            continuation_second = validation[second + CUE_BYTES : second + CUE_BYTES + CONTINUATION_BYTES]
            kind = "consistent" if np.array_equal(continuation_first, continuation_second) else "conflicting"
            cell = (kind, band_name)
            key = (first, second)
            if len(selected[cell]) >= wanted[cell] or key in selected_pairs:
                continue
            if kind == "conflicting":
                differing = float(np.mean(continuation_first != continuation_second))
                if differing < 0.25:
                    continue
            row = {
                "repeat_class": kind,
                "distance_band": band_name,
                "horizon_segments": horizon_segments,
                "distance_bytes": distance,
                "first_position": int(first),
                "second_position": int(second),
                "cue_hex": bytes(cue_second).hex(),
                "first_continuation_hex": bytes(continuation_first).hex(),
                "second_continuation_hex": bytes(continuation_second).hex(),
            }
            selected[cell].append(row)
            selected_pairs.add(key)
        if all(len(selected[cell]) >= count for cell, count in wanted.items()):
            break

    counts = {f"{kind}_{band}": len(selected[(kind, band)]) for kind, band in wanted}
    if any(value < EXAMPLES_PER_CELL for value in counts.values()):
        raise RuntimeError(f"Insufficient distinct natural repeat pairs: {counts}")
    return {cell: rows[:EXAMPLES_PER_CELL] for cell, rows in selected.items()}, counts


def replacement_start(validation_length: int, first: int, second: int, example_index: int):
    span = CUE_BYTES + CONTINUATION_BYTES
    candidate = (first + 1_000_003 + example_index * 104_729) % (validation_length - span)
    while abs(candidate - first) < span or abs(candidate - second) < span:
        candidate = (candidate + 104_729) % (validation_length - span)
    return candidate


def build_batch(validation: np.ndarray, rows, source_removed: bool):
    horizon_segments = rows[0]["horizon_segments"]
    total_segments = horizon_segments + 1
    streams = []
    for index, row in enumerate(rows):
        second = row["second_position"]
        start = second - horizon_segments * 512
        end = second + 512 + 1
        if start < 0 or end > len(validation):
            raise RuntimeError(f"Pair cannot be framed: {row}")
        stream = np.asarray(validation[start:end], dtype=np.int32).copy()
        first_local = row["first_position"] - start
        if not 0 <= first_local < horizon_segments * 512:
            raise RuntimeError(f"First occurrence outside carried history: {row}")
        if source_removed:
            source = replacement_start(len(validation), row["first_position"], second, index)
            replacement = np.asarray(
                validation[source : source + CUE_BYTES + CONTINUATION_BYTES], dtype=np.int32
            )
            original = stream[first_local : first_local + CUE_BYTES + CONTINUATION_BYTES]
            if np.array_equal(replacement, original):
                raise RuntimeError("Natural replacement unexpectedly equals the removed source")
            stream[first_local : first_local + CUE_BYTES + CONTINUATION_BYTES] = replacement
        streams.append(stream)
    values = np.stack(streams, axis=0)
    x = values[:, :-1].reshape(EXAMPLES_PER_CELL, total_segments, 512).transpose(1, 0, 2)
    y = values[:, 1:].reshape(EXAMPLES_PER_CELL, total_segments, 512).transpose(1, 0, 2)
    score_mask = np.zeros_like(x, dtype=np.float32)
    score_mask[-1, :, CUE_BYTES - 1 : CUE_BYTES + CONTINUATION_BYTES - 1] = 1.0
    return x, y, score_mask


def evaluate_cell(params, state0, rows, condition, validation, batch_sharding, eval_step):
    mask = CONDITIONS[condition]
    source_removed = condition == "source_removed_full_carry"
    x_host, y_host, score_host = build_batch(validation, rows, source_removed)
    state = state0
    per_example_nll = np.zeros(EXAMPLES_PER_CELL, dtype=np.float64)
    started = time.perf_counter()
    for segment in range(x_host.shape[0]):
        next_state, nll = eval_step(
            params,
            state,
            jax.device_put(x_host[segment], batch_sharding),
            jax.device_put(y_host[segment], batch_sharding),
            jax.device_put(score_host[segment], batch_sharding),
        )
        per_example_nll += np.asarray(jax.device_get(nll), dtype=np.float64)
        state = tuple(
            next_value if enabled else zero_value
            for next_value, zero_value, enabled in zip(next_state, state0, mask)
        )
    elapsed = time.perf_counter() - started
    per_example_bpc = per_example_nll / (CONTINUATION_BYTES * LN2)
    row = {
        "repeat_class": rows[0]["repeat_class"],
        "distance_band": rows[0]["distance_band"],
        "condition": condition,
        "examples": EXAMPLES_PER_CELL,
        "mean_distance_bytes": float(np.mean([item["distance_bytes"] for item in rows])),
        "bpc_mean": float(np.mean(per_example_bpc)),
        "bpc_stdev": float(np.std(per_example_bpc, ddof=1)),
        "per_example_bpc": per_example_bpc.tolist(),
        "evaluation_seconds": elapsed,
        "all_finite": bool(np.isfinite(per_example_bpc).all()),
    }
    print("NATURAL_DELAYED_RECALL_EVAL", json.dumps(row), flush=True)
    return row


def summarize(evaluations):
    indexed = {
        (row["repeat_class"], row["distance_band"], row["condition"]): row
        for row in evaluations
    }
    cells = []
    for kind in ("consistent", "conflicting"):
        for band in BANDS:
            get = lambda condition: indexed[(kind, band, condition)]["bpc_mean"]
            full = get("full_carry")
            cell = {
                "repeat_class": kind,
                "distance_band": band,
                "full_carry_bpc": full,
                "carry_benefit_vs_reset": get("reset") - full,
                "source_exposure_benefit": get("source_removed_full_carry") - full,
                "matrix_causal_contribution": get("vector_only_carry") - full,
                "vector_causal_contribution": get("matrix_only_carry") - full,
            }
            cells.append(cell)
    consistent = [row for row in cells if row["repeat_class"] == "consistent"]
    conflicting = [row for row in cells if row["repeat_class"] == "conflicting"]
    return {
        "cells": cells,
        "aggregate": {
            "consistent_mean_source_exposure_benefit": float(np.mean([row["source_exposure_benefit"] for row in consistent])),
            "conflicting_mean_source_exposure_benefit": float(np.mean([row["source_exposure_benefit"] for row in conflicting])),
            "consistent_mean_matrix_contribution": float(np.mean([row["matrix_causal_contribution"] for row in consistent])),
            "consistent_mean_vector_contribution": float(np.mean([row["vector_causal_contribution"] for row in consistent])),
            "consistent_positive_exposure_bands": sum(row["source_exposure_benefit"] > 0 for row in consistent),
            "conflicting_interference_bands": sum(row["source_exposure_benefit"] < 0 for row in conflicting),
        },
        "interpretation": "Frozen causal diagnostic only; no architecture promotion is authorized by this run.",
    }


def main():
    args = parse_args()
    if jax.default_backend() != "tpu" or jax.device_count() != 8:
        raise RuntimeError(f"Expected TPU v5e-8, found {jax.default_backend()} with {jax.device_count()} devices")
    if not args.data_path.is_file() or args.data_path.stat().st_size != 100_000_000:
        raise RuntimeError("Expected canonical 100,000,000-byte enwik8")
    sys.path.insert(0, str(args.code_root))
    import run_contiguous_training_screen as base

    checkpoint = args.experiment_root / "reset_contiguous" / "checkpoint.pkl"
    with checkpoint.open("rb") as handle:
        saved = pickle.load(handle)
    if int(saved["step"]) != 5000 or int(saved["provenance"]["seed"]) != args.seed:
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

    @jax.jit
    def eval_step(model_params, recurrent_state, tokens, targets, score_mask):
        outputs, next_state = base.batch_stateful_outputs(model_params, recurrent_state, tokens, cfg)
        logp = jax.nn.log_softmax(outputs[0].astype(jnp.float32), axis=-1)
        nll = -jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
        return jax.tree_util.tree_map(lax.stop_gradient, next_state), jnp.sum(nll * score_mask, axis=-1)

    params = jax.device_put(saved["params"], replicated)
    data = np.memmap(args.data_path, dtype=np.uint8, mode="r")
    validation = np.asarray(data[VALIDATION_START:VALIDATION_END], dtype=np.uint8)
    pairs, counts = discover_pairs(validation)
    print("NATURAL_DELAYED_RECALL_PAIRS", json.dumps(counts), flush=True)

    state0 = initialize()
    args.outdir.mkdir(parents=True, exist_ok=True)
    evaluations = []
    pair_manifest = []
    for kind in ("consistent", "conflicting"):
        for band in BANDS:
            rows = pairs[(kind, band)]
            pair_manifest.extend(rows)
            for condition in CONDITIONS:
                evaluations.append(
                    evaluate_cell(params, state0, rows, condition, validation, batch_sharding, eval_step)
                )
    final = {
        "seed": args.seed,
        "stage": "frozen_natural_byte_delayed_recall_attribution",
        "checkpoint": str(checkpoint),
        "pair_counts": counts,
        "pairs": pair_manifest,
        "evaluations": evaluations,
        "summary": summarize(evaluations),
        "test_data_read": False,
    }
    atomic_json(args.outdir / "natural_delayed_recall.json", final)
    print("NATURAL_DELAYED_RECALL_DECISION", json.dumps(final["summary"]), flush=True)


if __name__ == "__main__":
    main()
