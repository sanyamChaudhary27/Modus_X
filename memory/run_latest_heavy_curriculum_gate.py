"""Paired test of balanced versus latest-heavy CurrentArchive training."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
MIXED_RUNNER = THIS_DIR / "run_equal_memory_mixed_update_gate.py"
MEMORY_RUNNER = THIS_DIR / "run_versioned_memory_ablation.py"
STAGE1G = THIS_DIR / "run_stage1g_versioned_memory.py"
BALANCED_KV = THIS_DIR / "balanced_kv.py"
CASES = {
    "balanced": {"latest": 1 / 3, "previous": 1 / 3, "first": 1 / 3},
    "latest_heavy": {"latest": 1 / 2, "previous": 1 / 4, "first": 1 / 4},
}
ROLES = ("latest", "previous", "first")


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
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seeds", default="17,27,37")
    parser.add_argument("--matrix-width", type=int, default=64)
    parser.add_argument("--bindings", type=int, default=32)
    parser.add_argument("--overwrite-rate", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--n-train", type=int, default=6144)
    parser.add_argument("--n-validation", type=int, default=1536)
    parser.add_argument("--n-test", type=int, default=1536)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--state-dtype-bytes", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def allocate_counts(total: int, fractions: dict[str, float]) -> dict[str, int]:
    raw = {role: total * fractions[role] for role in ROLES}
    counts = {role: int(raw[role]) for role in ROLES}
    remainder = total - sum(counts.values())
    order = sorted(ROLES, key=lambda role: raw[role] - counts[role], reverse=True)
    for role in order[:remainder]:
        counts[role] += 1
    if sum(counts.values()) != total or any(count <= 0 for count in counts.values()):
        raise ValueError(f"Invalid role allocation: {counts}")
    return counts


def curriculum_dataset(kv, stage1g, key, n: int, cfg, fractions: dict[str, float]):
    if n % 2:
        raise ValueError("Clean/update-balanced dataset size must be even")
    per_group = n // 2
    role_counts = allocate_counts(per_group, fractions)
    keys = iter(kv.random.split(key, 6))
    parts = []
    manifest = []
    for target_group in ("overwritten", "clean"):
        for role in ROLES:
            count = role_counts[role]
            parts.append(
                stage1g.make_versioned_kv(
                    kv,
                    next(keys),
                    count,
                    cfg.train_len,
                    cfg,
                    query_mode=role,
                    target_mode=target_group,
                    version_tag_facts=True,
                )
            )
            manifest.append(
                {"target_group": target_group, "role": role, "examples": count}
            )
    seqs = np.concatenate([part[0] for part in parts], axis=0)
    labels = np.concatenate([part[1] for part in parts], axis=0)
    meta = {
        name: np.concatenate([np.asarray(part[2][name]) for part in parts], axis=0)
        for name in parts[0][2]
    }
    return (seqs, labels, meta), manifest


def host_tree(kv, tree):
    return kv.jax.tree_util.tree_map(lambda value: np.asarray(kv.jax.device_get(value)), tree)


def save_case_artifacts(kv, mixed, case_dir: Path, params, fwd, history, dataset, cfg) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    with (case_dir / "best_params.pkl").open("wb") as handle:
        pickle.dump(host_tree(kv, params), handle, protocol=pickle.HIGHEST_PROTOCOL)
    (case_dir / "validation_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    predictions = mixed.predict(kv, params, fwd, dataset, cfg)
    _, labels, meta = dataset
    np.savez_compressed(
        case_dir / "final_test_predictions.npz",
        predictions=np.asarray(predictions, dtype=np.int32),
        labels=np.asarray(labels, dtype=np.int32),
        latest_values=np.asarray(meta["latest_values"], dtype=np.int32),
        previous_values=np.asarray(meta["previous_values"], dtype=np.int32),
        first_values=np.asarray(meta["first_values"], dtype=np.int32),
        modes=np.asarray(meta["modes"], dtype="U16"),
        overwritten_target=np.asarray(meta["overwritten_target"], dtype=np.bool_),
        target_indices=np.asarray(meta["target_indices"], dtype=np.int32),
        overwrite_counts=np.asarray(meta["overwrite_counts"], dtype=np.int32),
    )


def aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["case"]].append(row)
    output = []
    for case, members in sorted(grouped.items()):
        result = {
            "case": case,
            "seeds": sorted(member["seed"] for member in members),
            "params_mean": statistics.mean(member["params"] for member in members),
            "state_bytes": members[0]["state_bytes"],
        }
        for metric in members[0]["metrics"]:
            values = [float(member["metrics"][metric]) for member in members]
            result[f"{metric}_mean"] = statistics.mean(values)
            result[f"{metric}_stdev"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        output.append(result)
    return output


def metric_delta(aggregates: dict[str, dict], metric: str) -> float:
    return aggregates["latest_heavy"][f"{metric}_mean"] - aggregates["balanced"][
        f"{metric}_mean"
    ]


def decision(aggregate_rows: list[dict]) -> dict:
    by_case = {row["case"]: row for row in aggregate_rows}
    latest_gain = metric_delta(by_case, "acc_latest_overwritten")
    stale_change = metric_delta(by_case, "stale_false_recall_on_latest_overwritten")
    overall_change = metric_delta(by_case, "acc_all")
    previous_change = metric_delta(by_case, "acc_previous_overwritten")
    first_change = metric_delta(by_case, "acc_first_overwritten")
    checks = {
        "latest_overwritten_gain_at_least_5pp": latest_gain >= 5.0,
        "stale_false_recall_at_most_7pct": by_case["latest_heavy"][
            "stale_false_recall_on_latest_overwritten_mean"
        ]
        <= 7.0,
        "overall_loss_at_most_2pp": overall_change >= -2.0,
        "previous_loss_at_most_5pp": previous_change >= -5.0,
        "first_loss_at_most_5pp": first_change >= -5.0,
    }
    return {
        "deltas_latest_heavy_minus_balanced": {
            "acc_latest_overwritten": latest_gain,
            "stale_false_recall_on_latest_overwritten": stale_change,
            "acc_all": overall_change,
            "acc_previous_overwritten": previous_change,
            "acc_first_overwritten": first_change,
        },
        "promotion_checks": checks,
        "training_distribution_explains_bias": all(checks.values()),
        "next": (
            "promote latest-heavy curriculum and confirm operation diagnostics"
            if all(checks.values())
            else "freeze curriculum lever; instrument current/archive arbitration"
        ),
    }


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.seeds = "17"
        args.matrix_width = 16
        args.epochs = 1
        args.n_train = 192
        args.n_validation = 96
        args.n_test = 96
        args.batch = 32
    args.outdir.mkdir(parents=True, exist_ok=True)
    mixed = load_module(MIXED_RUNNER, "latest_heavy_mixed")
    runner = load_module(MEMORY_RUNNER, "latest_heavy_runner")
    stage1g = load_module(STAGE1G, "latest_heavy_stage1g")
    kv = load_module(BALANCED_KV, "latest_heavy_kv")
    protocol = argparse.Namespace(
        n_values=32,
        d_model=96,
        n_train=args.n_train,
        n_test=args.n_test,
        batch=args.batch,
        epochs=args.epochs,
        lr=args.lr,
    )
    state_bytes = (
        2 * args.matrix_width * args.matrix_width + args.matrix_width
    ) * args.state_dtype_bytes
    rows = []
    manifests = {}
    for seed_text in args.seeds.split(","):
        seed = int(seed_text)
        data_cfg, model_cfg = runner.make_configs(
            protocol,
            n_pairs=args.bindings,
            ax_res=args.matrix_width,
            overwrite_rate=args.overwrite_rate,
            router_bias=2.0,
            residual_scale=0.25,
        )
        root_key = kv.random.PRNGKey(seed)
        k_train, k_validation, k_test, k_model = kv.random.split(root_key, 4)
        validation = mixed.balanced_dataset(
            kv, stage1g, k_validation, args.n_validation, data_cfg
        )
        test = mixed.balanced_dataset(kv, stage1g, k_test, args.n_test, data_cfg)
        for case_index, (case, fractions) in enumerate(CASES.items()):
            train, manifest = curriculum_dataset(
                kv,
                stage1g,
                kv.random.fold_in(k_train, case_index),
                args.n_train,
                data_cfg,
                fractions,
            )
            manifests[case] = manifest
            params, fwd, _, _ = runner.make_model_with_aux(
                "CurrentArchiveDelta", k_model, model_cfg
            )
            print(
                "LATEST_HEAVY_CASE",
                json.dumps(
                    {
                        "case": case,
                        "seed": seed,
                        "role_fractions": fractions,
                        "state_bytes": state_bytes,
                        "train_manifest": manifest,
                    }
                ),
                flush=True,
            )
            params, history = mixed.train_with_validation(
                kv,
                f"CurrentArchiveDelta_{case}",
                params,
                fwd,
                train,
                validation,
                data_cfg,
                seed,
            )
            metrics = mixed.evaluate(kv, params, fwd, test, data_cfg)
            row = {
                "case": case,
                "seed": seed,
                "params": runner.count_params(params),
                "state_bytes": state_bytes,
                "role_fractions": fractions,
                "metrics": metrics,
                "history": history,
            }
            rows.append(row)
            case_dir = args.outdir / case / f"seed_{seed}"
            save_case_artifacts(
                kv, mixed, case_dir, params, fwd, history, test, data_cfg
            )
            print(
                "LATEST_HEAVY_RESULT",
                json.dumps({key: value for key, value in row.items() if key != "history"}),
                flush=True,
            )
            partial_aggregate = aggregate(rows)
            (args.outdir / "progress.json").write_text(
                json.dumps({"rows": rows, "aggregate": partial_aggregate}, indent=2),
                encoding="utf-8",
            )

    aggregate_rows = aggregate(rows)
    final_decision = decision(aggregate_rows)
    report = {
        "protocol": {
            "seeds": [int(seed) for seed in args.seeds.split(",")],
            "matrix_width": args.matrix_width,
            "bindings": args.bindings,
            "overwrite_rate": args.overwrite_rate,
            "state_bytes": state_bytes,
            "train_examples": args.n_train,
            "validation_examples": args.n_validation,
            "test_examples": args.n_test,
            "checkpoint_selection": "validation_only",
            "test_distribution": "50% clean/update; equal latest/previous/first",
            "curricula": CASES,
            "train_manifests": manifests,
        },
        "rows": rows,
        "aggregate": aggregate_rows,
        "decision": final_decision,
    }
    (args.outdir / "latest_heavy_curriculum_gate.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("LATEST_HEAVY_SUMMARY", json.dumps(aggregate_rows), flush=True)
    print("LATEST_HEAVY_DECISION", json.dumps(final_decision), flush=True)
    print("LATEST_HEAVY_GATE_READY", args.outdir, flush=True)


if __name__ == "__main__":
    main()
