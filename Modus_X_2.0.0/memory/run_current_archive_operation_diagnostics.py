"""Diagnose CurrentArchive latest-value failures from saved paired checkpoints."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import statistics
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
MIXED_RUNNER = THIS_DIR / "run_equal_memory_mixed_update_gate.py"
MEMORY_RUNNER = THIS_DIR / "run_versioned_memory_ablation.py"
STAGE1G = THIS_DIR / "run_stage1g_versioned_memory.py"
BALANCED_KV = THIS_DIR / "balanced_kv.py"
CASES = ("balanced", "latest_heavy")
SEEDS = (17, 27, 37)
SCALAR_FIELDS = (
    "router_mean",
    "router_std",
    "memory_norm",
    "vector_norm",
    "disagreement_norm",
    "fused_norm",
    "current_read_norm",
    "history_read_norm",
    "current_history_disagreement_norm",
    "read_arbitration_current_weight",
    "read_arbitration_entropy",
)
BRANCH_FIELDS = (
    "current_branch_logits",
    "history_branch_logits",
    "memory_branch_logits",
    "vector_branch_logits",
    "fused_branch_logits",
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--n-test", type=int, default=1536)
    return parser.parse_args()


def mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def accuracy(pred: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> float:
    return 100.0 * mean((pred[mask] == labels[mask]).astype(np.float32))


def stale_rate(
    pred: np.ndarray,
    latest: np.ndarray,
    previous: np.ndarray,
    first: np.ndarray,
    mask: np.ndarray,
) -> float:
    stale = ((pred == previous) & (previous != latest)) | (
        (pred == first) & (first != latest)
    )
    return 100.0 * mean(stale[mask].astype(np.float32))


def locate_checkpoint(root: Path, case: str, seed: int) -> Path:
    expected = root / case / f"seed_{seed}" / "best_params.pkl"
    if expected.exists():
        return expected
    matches = [
        path
        for path in root.rglob("best_params.pkl")
        if path.parent.name == f"seed_{seed}" and path.parent.parent.name == case
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {case}/seed_{seed}/best_params.pkl under {root}; found {matches}"
        )
    return matches[0]


def batched_diagnostics(kv, params, fwd, aux_fn, seqs: np.ndarray, batch: int):
    fwd_batch = kv.jit(kv.vmap(fwd, in_axes=(None, 0)))
    aux_batch = kv.jit(kv.vmap(aux_fn, in_axes=(None, 0)))
    final_predictions = []
    scalars = defaultdict(list)
    branches = defaultdict(list)
    for start in range(0, len(seqs), batch):
        x = kv.jnp.asarray(seqs[start : start + batch], dtype=kv.jnp.float32)
        logits = fwd_batch(params, x)
        aux = aux_batch(params, x)
        final_predictions.append(np.asarray(kv.jnp.argmax(logits, axis=-1)))
        for field in SCALAR_FIELDS:
            scalars[field].append(np.asarray(aux[field]))
        for field in BRANCH_FIELDS:
            branches[field].append(np.asarray(kv.jnp.argmax(aux[field], axis=-1)))
    return (
        np.concatenate(final_predictions),
        {field: np.concatenate(parts) for field, parts in scalars.items()},
        {field: np.concatenate(parts) for field, parts in branches.items()},
    )


def summarize_group(
    labels: np.ndarray,
    latest: np.ndarray,
    previous: np.ndarray,
    first: np.ndarray,
    final_pred: np.ndarray,
    scalar_values: dict[str, np.ndarray],
    branch_pred: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict:
    result = {
        "examples": int(mask.sum()),
        "final_accuracy": accuracy(final_pred, labels, mask),
        "final_stale_rate": stale_rate(final_pred, latest, previous, first, mask),
    }
    for field, pred in branch_pred.items():
        name = field.removesuffix("_branch_logits")
        result[f"{name}_accuracy"] = accuracy(pred, labels, mask)
        result[f"{name}_stale_rate"] = stale_rate(
            pred, latest, previous, first, mask
        )
        result[f"final_agrees_{name}"] = 100.0 * mean(
            (final_pred[mask] == pred[mask]).astype(np.float32)
        )
    for field, values in scalar_values.items():
        selected = values[mask]
        result[f"{field}_mean"] = mean(selected)
        result[f"{field}_std"] = (
            float(np.std(selected, ddof=1)) if len(selected) > 1 else 0.0
        )
    return result


def aggregate(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["case"], row["group"])].append(row)
    output = []
    for (case, group), members in sorted(grouped.items()):
        record = {
            "case": case,
            "group": group,
            "seeds": sorted(row["seed"] for row in members),
        }
        for metric in members[0]["metrics"]:
            values = [row["metrics"][metric] for row in members]
            if any(not np.isfinite(value) for value in values):
                continue
            record[f"{metric}_mean"] = statistics.mean(values)
            record[f"{metric}_stdev"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        output.append(record)
    return output


def directional_diagnosis(rows: list[dict]) -> dict:
    latest_rows = [
        row for row in rows if row["group"] == "latest_overwritten_stale_eligible"
    ]
    comparisons = []
    for row in latest_rows:
        checkpoint = (row["case"], row["seed"])
        correct = next(
            item
            for item in rows
            if (item["case"], item["seed"]) == checkpoint
            and item["group"] == "latest_overwritten_correct"
        )
        stale = next(
            item
            for item in rows
            if (item["case"], item["seed"]) == checkpoint
            and item["group"] == "latest_overwritten_stale"
        )
        comparisons.append(
            {
                "case": row["case"],
                "seed": row["seed"],
                "stale_examples": stale["metrics"]["examples"],
                "correct_examples": correct["metrics"]["examples"],
                "router_mean_stale_minus_correct": stale["metrics"][
                    "router_mean_mean"
                ]
                - correct["metrics"]["router_mean_mean"],
                "current_norm_stale_minus_correct": stale["metrics"][
                    "current_read_norm_mean"
                ]
                - correct["metrics"]["current_read_norm_mean"],
                "history_norm_stale_minus_correct": stale["metrics"][
                    "history_read_norm_mean"
                ]
                - correct["metrics"]["history_read_norm_mean"],
                "disagreement_stale_minus_correct": stale["metrics"][
                    "current_history_disagreement_norm_mean"
                ]
                - correct["metrics"]["current_history_disagreement_norm_mean"],
                "current_branch_stale_rate": row["metrics"][
                    "current_stale_rate"
                ],
                "vector_branch_stale_rate": row["metrics"][
                    "vector_stale_rate"
                ],
                "final_stale_rate": row["metrics"]["final_stale_rate"],
            }
        )
    return {
        "comparisons": comparisons,
        "interpretation_rule": (
            "A correction is permitted only if one branch or scalar signature "
            "predicts stale failure in the same direction for all three seeds "
            "of both curricula. Otherwise freeze architecture correction."
        ),
    }


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    mixed = load_module(MIXED_RUNNER, "operation_mixed")
    runner = load_module(MEMORY_RUNNER, "operation_runner")
    stage1g = load_module(STAGE1G, "operation_stage1g")
    kv = load_module(BALANCED_KV, "operation_kv")
    protocol = argparse.Namespace(
        n_values=32,
        d_model=96,
        n_train=6144,
        n_test=args.n_test,
        batch=args.batch,
        epochs=24,
        lr=3e-4,
    )
    rows = []
    for seed in SEEDS:
        data_cfg, model_cfg = runner.make_configs(
            protocol,
            n_pairs=32,
            ax_res=64,
            overwrite_rate=0.5,
            router_bias=2.0,
            residual_scale=0.25,
        )
        root_key = kv.random.PRNGKey(seed)
        _, _, k_test, k_model = kv.random.split(root_key, 4)
        dataset = mixed.balanced_dataset(
            kv, stage1g, k_test, args.n_test, data_cfg
        )
        seqs, labels, meta = dataset
        labels = np.asarray(labels)
        latest = np.asarray(meta["latest_values"])
        previous = np.asarray(meta["previous_values"])
        first = np.asarray(meta["first_values"])
        modes = np.asarray(meta["modes"])
        overwritten = np.asarray(meta["overwritten_target"], dtype=bool)
        _, fwd, aux_fn, _ = runner.make_model_with_aux(
            "CurrentArchiveDelta", k_model, model_cfg
        )
        for case in CASES:
            checkpoint = locate_checkpoint(args.results_root, case, seed)
            with checkpoint.open("rb") as handle:
                params = pickle.load(handle)
            final_pred, scalar_values, branch_pred = batched_diagnostics(
                kv, params, fwd, aux_fn, seqs, args.batch
            )
            latest_overwritten = overwritten & (modes == "latest")
            stale_eligible = latest_overwritten & (
                (previous != latest) | (first != latest)
            )
            stale_prediction = (
                ((final_pred == previous) & (previous != latest))
                | ((final_pred == first) & (first != latest))
            )
            groups = {
                "all": np.ones(len(labels), dtype=bool),
                "clean": ~overwritten,
                "overwritten": overwritten,
                "latest_overwritten_stale_eligible": stale_eligible,
                "latest_overwritten_correct": stale_eligible
                & (final_pred == labels),
                "latest_overwritten_stale": stale_eligible & stale_prediction,
                "latest_overwritten_other_wrong": stale_eligible
                & (final_pred != labels)
                & ~stale_prediction,
            }
            for group, mask in groups.items():
                rows.append(
                    {
                        "case": case,
                        "seed": seed,
                        "group": group,
                        "metrics": summarize_group(
                            labels,
                            latest,
                            previous,
                            first,
                            final_pred,
                            scalar_values,
                            branch_pred,
                            mask,
                        ),
                    }
                )
            print(
                "OPERATION_DIAGNOSTIC_CASE",
                json.dumps(
                    {
                        "case": case,
                        "seed": seed,
                        "checkpoint": str(checkpoint),
                        "latest_overwritten_stale_rate": stale_rate(
                            final_pred,
                            latest,
                            previous,
                            first,
                            stale_eligible,
                        ),
                        "current_branch_stale_rate": stale_rate(
                            branch_pred["current_branch_logits"],
                            latest,
                            previous,
                            first,
                            stale_eligible,
                        ),
                        "vector_branch_stale_rate": stale_rate(
                            branch_pred["vector_branch_logits"],
                            latest,
                            previous,
                            first,
                            stale_eligible,
                        ),
                    }
                ),
                flush=True,
            )
    report = {
        "protocol": {
            "cases": list(CASES),
            "seeds": list(SEEDS),
            "checkpoint_source": str(args.results_root),
            "test_reconstruction": "exact seed-derived frozen balanced test",
            "model": "CurrentArchiveDelta",
            "parameters": 166791,
            "state_bytes": 16512,
            "architecture_note": (
                "Latest-role readout hard-selects current memory; diagnostics "
                "test current content, vector correction, and final fusion."
            ),
            "branch_probe_note": (
                "Branch predictions reuse the trained final classifier. They "
                "are diagnostic shared-head probes, not independently trained "
                "branch heads or causal ablations."
            ),
        },
        "rows": rows,
        "aggregate": aggregate(rows),
        "directional_diagnosis": directional_diagnosis(rows),
    }
    output = args.outdir / "current_archive_operation_diagnostics.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("OPERATION_DIAGNOSTIC_SUMMARY", json.dumps(report["aggregate"]), flush=True)
    print(
        "OPERATION_DIAGNOSTIC_DIRECTIONAL",
        json.dumps(report["directional_diagnosis"]),
        flush=True,
    )
    print("OPERATION_DIAGNOSTICS_READY", output, flush=True)


if __name__ == "__main__":
    main()
