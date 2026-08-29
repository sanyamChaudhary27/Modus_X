"""Paired causal gate for CurrentArchive current-slot refresh."""

from __future__ import annotations

import argparse
import importlib.util
import json
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
LATEST_RUNNER = THIS_DIR / "run_latest_heavy_curriculum_gate.py"
CASES = {
    "control": "CurrentArchiveDelta",
    "latest_shadow": "CurrentArchiveLatestShadow",
}


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


def host_arrays(kv, tree):
    return [
        np.asarray(kv.jax.device_get(value))
        for value in kv.jax.tree_util.tree_leaves(tree)
    ]


def assert_paired_initialization(kv, control, candidate) -> None:
    control_leaves = host_arrays(kv, control)
    candidate_leaves = host_arrays(kv, candidate)
    if len(control_leaves) != len(candidate_leaves):
        raise AssertionError("Parameter-tree leaf count differs")
    for index, (left, right) in enumerate(zip(control_leaves, candidate_leaves)):
        if left.shape != right.shape or not np.array_equal(left, right):
            raise AssertionError(f"Initialization differs at leaf {index}")


def aggregate(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["case"]].append(row)
    output = []
    for case, members in sorted(grouped.items()):
        record = {
            "case": case,
            "model": members[0]["model"],
            "seeds": sorted(member["seed"] for member in members),
            "params_mean": statistics.mean(member["params"] for member in members),
            "state_bytes": members[0]["state_bytes"],
            "elapsed_s_mean": statistics.mean(
                member["elapsed_s"] for member in members
            ),
            "elapsed_s_stdev": statistics.stdev(
                member["elapsed_s"] for member in members
            )
            if len(members) > 1
            else 0.0,
        }
        for metric in members[0]["metrics"]:
            values = [float(member["metrics"][metric]) for member in members]
            record[f"{metric}_mean"] = statistics.mean(values)
            record[f"{metric}_stdev"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        output.append(record)
    return output


def decide(aggregate_rows: list[dict]) -> dict:
    by_case = {row["case"]: row for row in aggregate_rows}
    control = by_case["control"]
    candidate = by_case["latest_shadow"]

    def delta(metric: str) -> float:
        return candidate[f"{metric}_mean"] - control[f"{metric}_mean"]

    latest_gain = delta("acc_latest_overwritten")
    stale_change = delta("stale_false_recall_on_latest_overwritten")
    overall_change = delta("acc_all")
    previous_change = delta("acc_previous_overwritten")
    first_change = delta("acc_first_overwritten")
    runtime_ratio = candidate["elapsed_s_mean"] / control["elapsed_s_mean"]
    checks = {
        "latest_gain_at_least_3pp_or_stale_gain_at_least_2pp": (
            latest_gain >= 3.0 or stale_change <= -2.0
        ),
        "stale_false_recall_at_most_7pct": candidate[
            "stale_false_recall_on_latest_overwritten_mean"
        ]
        <= 7.0,
        "overall_loss_at_most_1pp": overall_change >= -1.0,
        "previous_loss_at_most_3pp": previous_change >= -3.0,
        "first_loss_at_most_3pp": first_change >= -3.0,
        "equal_parameters": candidate["params_mean"] == control["params_mean"],
        "equal_state_bytes": candidate["state_bytes"] == control["state_bytes"],
        "runtime_overhead_at_most_5pct": runtime_ratio <= 1.05,
    }
    return {
        "deltas_latest_shadow_minus_control": {
            "acc_latest_overwritten": latest_gain,
            "stale_false_recall_on_latest_overwritten": stale_change,
            "acc_all": overall_change,
            "acc_previous_overwritten": previous_change,
            "acc_first_overwritten": first_change,
            "runtime_ratio": runtime_ratio,
        },
        "promotion_checks": checks,
        "promotion_pass": all(checks.values()),
        "next": (
            "promote latest-shadow refresh as CurrentArchive correction"
            if all(checks.values())
            else "freeze synthetic architecture lane and package v2 evidence"
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
    mixed = load_module(MIXED_RUNNER, "refresh_mixed")
    runner = load_module(MEMORY_RUNNER, "refresh_runner")
    stage1g = load_module(STAGE1G, "refresh_stage1g")
    kv = load_module(BALANCED_KV, "refresh_kv")
    latest = load_module(LATEST_RUNNER, "refresh_latest")
    protocol = argparse.Namespace(
        n_values=32,
        d_model=96,
        n_train=args.n_train,
        n_test=args.n_test,
        batch=args.batch,
        epochs=args.epochs,
        lr=args.lr,
    )
    rows = []
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
        train, manifest = latest.curriculum_dataset(
            kv,
            stage1g,
            k_train,
            args.n_train,
            data_cfg,
            latest.CASES["latest_heavy"],
        )
        validation = mixed.balanced_dataset(
            kv, stage1g, k_validation, args.n_validation, data_cfg
        )
        test = mixed.balanced_dataset(
            kv, stage1g, k_test, args.n_test, data_cfg
        )
        initialized = {}
        functions = {}
        for case, model_name in CASES.items():
            params, fwd, _, actual_cfg = runner.make_model_with_aux(
                model_name, k_model, model_cfg
            )
            initialized[case] = params
            functions[case] = (fwd, actual_cfg)
        assert_paired_initialization(
            kv, initialized["control"], initialized["latest_shadow"]
        )
        state_bytes = (
            2 * args.matrix_width * args.matrix_width + args.matrix_width
        ) * args.state_dtype_bytes
        for case, model_name in CASES.items():
            params = initialized[case]
            fwd, actual_cfg = functions[case]
            print(
                "LATEST_SHADOW_REFRESH_CASE",
                json.dumps(
                    {
                        "case": case,
                        "model": model_name,
                        "seed": seed,
                        "params": runner.count_params(params),
                        "state_bytes": state_bytes,
                        "latest_shadow_write": actual_cfg.latest_shadow_write,
                        "paired_initialization": True,
                        "shared_training_examples": True,
                        "train_manifest": manifest,
                    }
                ),
                flush=True,
            )
            trained, history = mixed.train_with_validation(
                kv,
                model_name,
                params,
                fwd,
                train,
                validation,
                data_cfg,
                seed,
            )
            metrics = mixed.evaluate(kv, trained, fwd, test, data_cfg)
            elapsed_s = float(history[-1]["elapsed_s"])
            row = {
                "case": case,
                "model": model_name,
                "seed": seed,
                "params": runner.count_params(trained),
                "state_bytes": state_bytes,
                "elapsed_s": elapsed_s,
                "metrics": metrics,
                "history": history,
            }
            rows.append(row)
            latest.save_case_artifacts(
                kv,
                mixed,
                args.outdir / case / f"seed_{seed}",
                trained,
                fwd,
                history,
                test,
                data_cfg,
            )
            print(
                "LATEST_SHADOW_REFRESH_RESULT",
                json.dumps({key: value for key, value in row.items() if key != "history"}),
                flush=True,
            )
            (args.outdir / "progress.json").write_text(
                json.dumps({"rows": rows, "aggregate": aggregate(rows)}, indent=2),
                encoding="utf-8",
            )
    aggregate_rows = aggregate(rows)
    final_decision = decide(aggregate_rows)
    report = {
        "protocol": {
            "cases": CASES,
            "seeds": [int(seed) for seed in args.seeds.split(",")],
            "matrix_width": args.matrix_width,
            "bindings": args.bindings,
            "overwrite_rate": args.overwrite_rate,
            "state_bytes": (
                2 * args.matrix_width * args.matrix_width + args.matrix_width
            )
            * args.state_dtype_bytes,
            "train_examples": args.n_train,
            "validation_examples": args.n_validation,
            "test_examples": args.n_test,
            "curriculum": "latest_heavy_50_25_25",
            "checkpoint_selection": "validation_only",
            "paired_initialization": True,
            "shared_data_and_permutations": True,
            "only_changed_variable": "latest_shadow_write",
        },
        "rows": rows,
        "aggregate": aggregate_rows,
        "decision": final_decision,
    }
    output = args.outdir / "current_archive_latest_shadow_gate.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("LATEST_SHADOW_REFRESH_SUMMARY", json.dumps(aggregate_rows), flush=True)
    print("LATEST_SHADOW_REFRESH_DECISION", json.dumps(final_decision), flush=True)
    print("LATEST_SHADOW_REFRESH_GATE_READY", output, flush=True)


if __name__ == "__main__":
    main()
