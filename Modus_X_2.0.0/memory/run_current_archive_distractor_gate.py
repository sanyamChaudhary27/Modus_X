"""Three-seed distractor-retention gate for CurrentArchiveDelta.

The model is trained once per seed/model on the fixed Stage1G versioned-memory
protocol. Every distractor condition is evaluation-only, so robustness cannot
be attributed to retraining on the test corruption.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_RUNNER = THIS_DIR / "run_versioned_memory_ablation.py"
RUNNER = (
    LOCAL_RUNNER
    if LOCAL_RUNNER.exists()
    else REPO_ROOT / "Modus_X_2.0.0" / "experiments" / "memory" / "run_versioned_memory_ablation.py"
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
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--models", default="TwoPathLatestShadowDelta,CurrentArchiveDelta")
    p.add_argument("--seeds", default="17,27,37")
    p.add_argument("--distractor-counts", default="0,16,32,64,128,256")
    p.add_argument("--regimes", default="random,irrelevant,similar,post_update")
    p.add_argument("--bindings", type=int, default=32)
    p.add_argument("--ax-res", type=int, default=64)
    p.add_argument("--overwrite-rate", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--n-train", type=int, default=4096)
    p.add_argument("--n-test", type=int, default=1024)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--n-values", type=int, default=32)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def csv_values(text: str, cast):
    return [cast(value.strip()) for value in text.split(",") if value.strip()]


def append_distractors(
    seqs: np.ndarray,
    labels: np.ndarray,
    *,
    count: int,
    regime: str,
    cfg,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Insert distractors immediately before the final query token."""
    if regime not in {"random", "irrelevant", "similar", "post_update"}:
        raise ValueError(regime)
    n, old_len, width = seqs.shape
    if count == 0:
        return seqs.copy(), np.full((n, 0), -1, dtype=np.int32)
    rng = np.random.default_rng(seed)
    out = np.zeros((n, old_len + count, width), dtype=np.float32)
    out[:, : old_len - 1] = seqs[:, :-1]
    out[:, -1] = seqs[:, -1]
    distractor_values = np.full((n, count), -1, dtype=np.int32)

    value_offset = cfg.key_dim
    fact_marker = cfg.key_dim + cfg.n_values
    latest_marker = fact_marker + 2
    query_keys = seqs[:, -1, : cfg.key_dim]

    for row in range(n):
        for index in range(count):
            target = old_len - 1 + index
            if regime == "random":
                out[row, target] = rng.normal(0.0, 0.05, width)
                out[row, target, : cfg.key_dim + cfg.n_values + 5] = 0.0
                continue

            if regime == "similar":
                key = query_keys[row] + rng.normal(0.0, 0.12, cfg.key_dim)
            else:
                key = rng.normal(0.0, 1.0, cfg.key_dim)
                # Make irrelevant keys explicitly low-similarity to the query.
                query = query_keys[row]
                key = key - query * np.dot(key, query)
            key /= np.linalg.norm(key) + 1e-8
            value = int(rng.integers(0, cfg.n_values - 1))
            if value >= int(labels[row]):
                value += 1
            out[row, target, :] = 0.0
            out[row, target, : cfg.key_dim] = key
            out[row, target, value_offset + value] = 1.0
            out[row, target, fact_marker] = 1.0
            if regime == "post_update":
                out[row, target, latest_marker] = 1.0
            distractor_values[row, index] = value
    return out, distractor_values


def predict(kv, params, fwd_fn, seqs: np.ndarray, batch: int) -> np.ndarray:
    fwd_b = kv.jit(kv.vmap(fwd_fn, in_axes=(None, 0)))
    rows = []
    for start in range(0, len(seqs), batch):
        logits = fwd_b(
            params,
            kv.jnp.asarray(seqs[start : start + batch], dtype=kv.jnp.float32),
        )
        rows.append(np.asarray(kv.jnp.argmax(logits, axis=-1)))
    return np.concatenate(rows)


def evaluate_condition(
    kv,
    stage1g,
    params,
    fwd_fn,
    base,
    cfg,
    *,
    count: int,
    regime: str,
    seed: int,
) -> dict[str, float | int]:
    seqs, labels, meta = base
    expanded, distractor_values = append_distractors(
        np.asarray(seqs),
        np.asarray(labels),
        count=count,
        regime=regime,
        cfg=cfg,
        seed=seed,
    )
    metrics = stage1g.evaluate(kv, params, fwd_fn, expanded, labels, meta, cfg)
    pred = predict(kv, params, fwd_fn, expanded, cfg.batch)
    if count:
        distractor_confusion = 100.0 * float(
            np.mean(np.any(pred[:, None] == distractor_values, axis=1))
        )
    else:
        distractor_confusion = 0.0
    return {
        **metrics,
        "distractor_confusion": distractor_confusion,
        "sequence_length": int(expanded.shape[1]),
    }


def run_one(runner, *, model_name: str, seed: int, args) -> dict[str, object]:
    kv = runner.load_module(runner.BALANCED_KV, f"balanced_kv_distractor_{seed}_{model_name}")
    stage1g = runner.load_module(runner.STAGE1G, f"stage1g_distractor_{seed}_{model_name}")
    data_cfg, model_cfg = runner.make_configs(
        args,
        n_pairs=args.bindings,
        ax_res=args.ax_res,
        overwrite_rate=args.overwrite_rate,
        router_bias=2.0,
        residual_scale=0.25,
    )
    key = kv.random.PRNGKey(seed)
    k_train, k_eval, k_model = kv.random.split(key, 3)
    train = runner.make_train_set(kv, stage1g, k_train, data_cfg, args)
    base_eval = stage1g.make_versioned_kv(
        kv,
        k_eval,
        data_cfg.n_test,
        data_cfg.train_len,
        data_cfg,
        query_mode="mixed",
        target_mode="overwritten",
        version_tag_facts=True,
    )
    params, fwd_fn, aux_fn, actual_cfg = runner.make_model_with_aux(
        model_name, k_model, model_cfg
    )
    _, trained, history = runner.train_model(
        kv,
        model_name,
        params,
        fwd_fn,
        train,
        (base_eval[0], base_eval[1]),
        data_cfg,
    )
    conditions = {}
    for regime in args.regimes_list:
        for count in args.distractor_counts_list:
            key_name = f"{regime}_{count}"
            conditions[key_name] = evaluate_condition(
                kv,
                stage1g,
                trained,
                fwd_fn,
                base_eval,
                data_cfg,
                count=count,
                regime=regime,
                seed=seed * 10_000 + count * 17 + len(regime),
            )
            print(
                "DISTRACTOR_EVAL",
                json.dumps(
                    {
                        "model": model_name,
                        "seed": seed,
                        "condition": key_name,
                        **conditions[key_name],
                    }
                ),
                flush=True,
            )
    state_bytes = (
        (2 if model_name == "CurrentArchiveDelta" else 1)
        * args.ax_res
        * args.ax_res
        * 4
        + args.ax_res * 4
    )
    return {
        "model": model_name,
        "seed": seed,
        "params": runner.count_params(trained),
        "state_bytes_fp32": state_bytes,
        "config": actual_cfg.__dict__,
        "conditions": conditions,
        "history": history,
    }


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, float]]] = {}
    for row in rows:
        for condition, metrics in row["conditions"].items():
            grouped.setdefault((row["model"], condition), []).append(metrics)
    output = []
    metric_names = (
        "acc_all",
        "acc_latest",
        "acc_previous",
        "acc_first",
        "predicted_latest_when_not_label",
        "predicted_previous_when_not_label",
        "predicted_first_when_not_label",
        "distractor_confusion",
    )
    for (model, condition), group in grouped.items():
        row: dict[str, object] = {
            "model": model,
            "condition": condition,
            "seeds": len(group),
        }
        for metric in metric_names:
            values = [float(item[metric]) for item in group]
            row[f"{metric}_mean"] = statistics.mean(values)
            row[f"{metric}_stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(row)
    return output


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.models = "CurrentArchiveDelta"
        args.seeds = "17"
        args.distractor_counts = "0,4"
        args.regimes = "random,similar"
        args.epochs = 1
        args.n_train = 256
        args.n_test = 128
    args.models_list = csv_values(args.models, str)
    args.seeds_list = csv_values(args.seeds, int)
    args.distractor_counts_list = csv_values(args.distractor_counts, int)
    args.regimes_list = csv_values(args.regimes, str)
    args.train_curriculum = "role_balanced_overwritten"
    args.version_tag_facts = True
    args.router_biases = "2.0"
    args.residual_scales = "0.25"
    args.wrong_version_weights = "0.0"
    args.wrong_version_margin = 0.5

    args.outdir.mkdir(parents=True, exist_ok=True)
    runner = load_module(RUNNER, "current_archive_distractor_base")
    rows = []
    for model_name in args.models_list:
        for seed in args.seeds_list:
            print("DISTRACTOR_RUN", json.dumps({"model": model_name, "seed": seed}), flush=True)
            rows.append(run_one(runner, model_name=model_name, seed=seed, args=args))
            (args.outdir / "partial_results.json").write_text(
                json.dumps(rows, indent=2), encoding="utf-8"
            )
    summary = aggregate(rows)
    (args.outdir / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("CURRENT_ARCHIVE_DISTRACTOR_READY", args.outdir, flush=True)


if __name__ == "__main__":
    main()
