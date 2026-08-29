"""
Stage-1G versioned-memory diagnostics for the bounded matrix-memory lane.

Stage1F tests whether same-key updates retrieve the latest value while avoiding
stale false recall. That does not answer the harder objection: sometimes a task
needs both the updated and non-updated values.

This script tests that directly. It generates repeated facts for the same key
and asks the model for one of three versions:

- latest value;
- previous value;
- first value.

This is still controlled synthetic evidence, not a language-modeling claim.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_BALANCED_KV = Path(__file__).with_name("balanced_kv.py")
REPO_BALANCED_KV = REPO_ROOT / "Modus_X_v1.1.1" / "benchmarks" / "modus_x" / "balanced_kv.py"
BALANCED_KV = LOCAL_BALANCED_KV if LOCAL_BALANCED_KV.exists() else REPO_BALANCED_KV

MODES = ("latest", "previous", "first")


def load_balanced_kv():
    spec = importlib.util.spec_from_file_location("modus_x_balanced_kv", BALANCED_KV)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {BALANCED_KV}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="experiments/matrix_memory_capacity/results_stage1g_versioned")
    p.add_argument("--model", default="VectorLeanPM")
    p.add_argument("--seeds", default="17,27,37")
    p.add_argument("--bindings", default="16,32,64")
    p.add_argument("--ax-res", default="64,128")
    p.add_argument("--overwrite-rates", default="0.25,0.5,0.75")
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--n-train", type=int, default=4096)
    p.add_argument("--n-test", type=int, default=1024)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-values", type=int, default=32)
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--router-bias", type=float, default=None)
    p.add_argument("--residual-scale", type=float, default=None)
    return p.parse_args()


def csv_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def csv_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def mean_stdev(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def make_versioned_kv(
    kv,
    key,
    n: int,
    seq_len: int,
    cfg,
    *,
    query_mode: str = "mixed",
    target_mode: str = "any",
    version_tag_facts: bool = False,
):
    k_seed, k_keys, _, k_noise = kv.random.split(key, 4)
    rng = np.random.default_rng(int(kv.random.randint(k_seed, (), 0, 2**31 - 1)))

    seqs = np.array(kv.random.normal(k_noise, (n, seq_len, cfg.d_model)) * 0.05, dtype=np.float32)
    seqs[:, :, : cfg.key_dim + cfg.n_values + 5] = 0.0

    keys_np = np.array(kv.random.normal(k_keys, (n, cfg.n_pairs, cfg.key_dim)), dtype=np.float32)
    keys_np /= np.linalg.norm(keys_np, axis=-1, keepdims=True) + 1e-8

    value_offset = cfg.key_dim
    fact_marker = cfg.key_dim + cfg.n_values
    query_marker = fact_marker + 1
    latest_marker = fact_marker + 2
    previous_marker = fact_marker + 3
    first_marker = fact_marker + 4

    labels = np.zeros(n, dtype=np.int32)
    latest_values = np.zeros(n, dtype=np.int32)
    previous_values = np.zeros(n, dtype=np.int32)
    first_values = np.zeros(n, dtype=np.int32)
    modes = np.empty(n, dtype=object)
    overwritten_target = np.zeros(n, dtype=np.bool_)
    target_indices = np.zeros(n, dtype=np.int32)
    overwrite_counts = np.zeros(n, dtype=np.int32)

    for i in range(n):
        first_per_key = rng.permutation(cfg.n_pairs).astype(np.int32) % cfg.n_values
        current_per_key = first_per_key.copy()
        previous_per_key = first_per_key.copy()
        positions = np.sort(rng.choice(np.arange(0, seq_len - 1), size=cfg.n_pairs, replace=False))
        occupied = set()

        for j in range(cfg.n_pairs):
            pos = int(positions[j])
            occupied.add(pos)
            seqs[i, pos, :] = 0.0
            seqs[i, pos, : cfg.key_dim] = keys_np[i, j]
            seqs[i, pos, value_offset + int(first_per_key[j])] = 1.0
            seqs[i, pos, fact_marker] = 1.0
            if version_tag_facts:
                seqs[i, pos, first_marker] = 1.0

        overwrite_count = int(round(cfg.n_pairs * cfg.overwrite_rate))
        overwritten_sources: list[int] = []
        if overwrite_count:
            for source_raw in rng.choice(cfg.n_pairs, size=overwrite_count, replace=False):
                source = int(source_raw)
                available = [
                    position
                    for position in range(int(positions[source]) + 1, seq_len - 1)
                    if position not in occupied
                ]
                if not available:
                    continue
                overwrite_pos = int(rng.choice(available))
                occupied.add(overwrite_pos)
                new_value = int(rng.integers(0, cfg.n_values))
                previous_per_key[source] = current_per_key[source]
                current_per_key[source] = new_value
                overwritten_sources.append(source)

                seqs[i, overwrite_pos, :] = 0.0
                seqs[i, overwrite_pos, : cfg.key_dim] = keys_np[i, source]
                seqs[i, overwrite_pos, value_offset + new_value] = 1.0
                seqs[i, overwrite_pos, fact_marker] = 1.0
                if version_tag_facts:
                    seqs[i, overwrite_pos, latest_marker] = 1.0

        overwrite_counts[i] = len(overwritten_sources)
        overwritten_set = set(overwritten_sources)
        if target_mode == "overwritten" and overwritten_sources:
            target = int(rng.choice(overwritten_sources))
        elif target_mode == "clean":
            clean_sources = [idx for idx in range(cfg.n_pairs) if idx not in overwritten_set]
            target = int(rng.choice(clean_sources)) if clean_sources else int(rng.integers(0, cfg.n_pairs))
        else:
            target = int(rng.integers(0, cfg.n_pairs))

        mode = str(rng.choice(MODES)) if query_mode == "mixed" else query_mode
        if mode not in MODES:
            raise ValueError(f"Unknown query mode: {mode}")

        seqs[i, -1, :] = 0.0
        seqs[i, -1, : cfg.key_dim] = keys_np[i, target]
        seqs[i, -1, query_marker] = 1.0
        if mode == "latest":
            seqs[i, -1, latest_marker] = 1.0
            labels[i] = int(current_per_key[target])
        elif mode == "previous":
            seqs[i, -1, previous_marker] = 1.0
            labels[i] = int(previous_per_key[target])
        else:
            seqs[i, -1, first_marker] = 1.0
            labels[i] = int(first_per_key[target])

        latest_values[i] = int(current_per_key[target])
        previous_values[i] = int(previous_per_key[target])
        first_values[i] = int(first_per_key[target])
        modes[i] = mode
        overwritten_target[i] = target in overwritten_set
        target_indices[i] = target

    meta = {
        "latest_values": latest_values,
        "previous_values": previous_values,
        "first_values": first_values,
        "modes": modes,
        "overwritten_target": overwritten_target,
        "target_indices": target_indices,
        "overwrite_counts": overwrite_counts,
    }
    return seqs, labels, meta


def evaluate(kv, params, fwd, seqs, labels, meta, cfg):
    fwd_b = kv.jax.jit(kv.jax.vmap(fwd, in_axes=(None, 0)))
    preds = []
    for start in range(0, len(labels), cfg.batch):
        end = min(start + cfg.batch, len(labels))
        logits = fwd_b(params, kv.jnp.array(seqs[start:end], dtype=kv.jnp.float32))
        preds.append(np.array(kv.jnp.argmax(logits, axis=-1)))
    pred = np.concatenate(preds)
    labels_np = np.asarray(labels)
    overwritten = np.asarray(meta["overwritten_target"]).astype(bool)
    modes = np.asarray(meta["modes"])

    def pct(mask):
        denom = int(np.sum(mask))
        return 100.0 * float(np.sum((pred == labels_np) & mask)) / max(1, denom)

    out = {
        "acc_all": pct(np.ones_like(labels_np, dtype=bool)),
        "acc_overwritten": pct(overwritten),
        "acc_clean": pct(~overwritten),
        "overwritten_query_count": int(np.sum(overwritten)),
        "clean_query_count": int(np.sum(~overwritten)),
        "mean_overwrite_count": float(np.mean(meta["overwrite_counts"])),
    }
    for mode in MODES:
        mode_mask = modes == mode
        out[f"acc_{mode}"] = pct(mode_mask)
        out[f"acc_{mode}_overwritten"] = pct(mode_mask & overwritten)
    wrong_version = {}
    for version_name, values in (
        ("latest", np.asarray(meta["latest_values"])),
        ("previous", np.asarray(meta["previous_values"])),
        ("first", np.asarray(meta["first_values"])),
    ):
        wrong_version[f"predicted_{version_name}_when_not_label"] = 100.0 * float(
            np.sum((pred == values) & (values != labels_np))
        ) / max(1, len(labels_np))
    out.update(wrong_version)
    return out


def run_one(kv, *, seed: int, n_pairs: int, ax_res: int, overwrite_rate: float, args: argparse.Namespace):
    seq_len = max(128, 4 * n_pairs + 1)
    min_width = kv.Config().key_dim + args.n_values + 5
    cfg = kv.Config(
        d_model=max(args.d_model, min_width),
        n_values=args.n_values,
        seed=seed,
        n_pairs=n_pairs,
        ax_res=ax_res,
        vector_state=ax_res,
        train_len=seq_len,
        test_lens=(seq_len,),
        n_train=args.n_train,
        n_test=args.n_test,
        batch=args.batch,
        epochs=args.epochs,
        patience=args.epochs,
        lr=args.lr,
        overwrite_rate=overwrite_rate,
    )
    if args.router_bias is not None:
        cfg = replace(cfg, router_bias=args.router_bias)
    if args.residual_scale is not None:
        cfg = replace(cfg, residual_scale=args.residual_scale)

    key = kv.random.PRNGKey(seed)
    k_train, k_mixed, k_latest, k_prev, k_first, k_model = kv.random.split(key, 6)
    train = make_versioned_kv(kv, k_train, cfg.n_train, cfg.train_len, cfg, query_mode="mixed", target_mode="any")
    test_mixed = make_versioned_kv(kv, k_mixed, cfg.n_test, cfg.train_len, cfg, query_mode="mixed", target_mode="overwritten")
    test_latest = make_versioned_kv(kv, k_latest, cfg.n_test, cfg.train_len, cfg, query_mode="latest", target_mode="overwritten")
    test_previous = make_versioned_kv(kv, k_prev, cfg.n_test, cfg.train_len, cfg, query_mode="previous", target_mode="overwritten")
    test_first = make_versioned_kv(kv, k_first, cfg.n_test, cfg.train_len, cfg, query_mode="first", target_mode="overwritten")

    params, fwd = kv.make_model(args.model, k_model, cfg)
    best, trained, history = kv.train_model(
        args.model,
        params,
        fwd,
        (train[0], train[1]),
        (test_mixed[0], test_mixed[1]),
        cfg,
    )
    diagnostics = {
        "mixed_overwritten": evaluate(kv, trained, fwd, test_mixed[0], test_mixed[1], test_mixed[2], cfg),
        "latest_overwritten": evaluate(kv, trained, fwd, test_latest[0], test_latest[1], test_latest[2], cfg),
        "previous_overwritten": evaluate(kv, trained, fwd, test_previous[0], test_previous[1], test_previous[2], cfg),
        "first_overwritten": evaluate(kv, trained, fwd, test_first[0], test_first[1], test_first[2], cfg),
    }
    return {
        "model": args.model,
        "seed": seed,
        "n_pairs": n_pairs,
        "ax_res": ax_res,
        "load_factor": n_pairs / ax_res,
        "overwrite_rate": overwrite_rate,
        "n_values": cfg.n_values,
        "d_model": cfg.d_model,
        "params": kv.count_params(trained),
        "best_mixed_overwritten_protocol_acc": best,
        "diagnostics": diagnostics,
        "history": history,
    }


def main() -> None:
    args = parse_args()
    kv = load_balanced_kv()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ax_res in csv_ints(args.ax_res):
        for n_pairs in csv_ints(args.bindings):
            for overwrite_rate in csv_floats(args.overwrite_rates):
                for seed in csv_ints(args.seeds):
                    print(
                        "RUN_STAGE1G",
                        json.dumps(
                            {
                                "model": args.model,
                                "seed": seed,
                                "n_pairs": n_pairs,
                                "ax_res": ax_res,
                                "overwrite_rate": overwrite_rate,
                            }
                        ),
                        flush=True,
                    )
                    rows.append(
                        run_one(
                            kv,
                            seed=seed,
                            n_pairs=n_pairs,
                            ax_res=ax_res,
                            overwrite_rate=overwrite_rate,
                            args=args,
                        )
                    )
                    (outdir / "partial_results.json").write_text(json.dumps(rows, indent=2))

    grouped = {}
    for row in rows:
        key = (row["model"], row["n_pairs"], row["ax_res"], row["overwrite_rate"])
        grouped.setdefault(key, []).append(row)

    summary = []
    for (model, n_pairs, ax_res, overwrite_rate), group in grouped.items():
        record = {
            "model": model,
            "n_pairs": n_pairs,
            "ax_res": ax_res,
            "load_factor": n_pairs / ax_res,
            "overwrite_rate": overwrite_rate,
            "seeds": [row["seed"] for row in group],
        }
        for eval_name in ("mixed_overwritten", "latest_overwritten", "previous_overwritten", "first_overwritten"):
            values = [row["diagnostics"][eval_name]["acc_all"] for row in group]
            mean, stdev = mean_stdev(values)
            record[f"{eval_name}_acc_mean"] = mean
            record[f"{eval_name}_acc_stdev"] = stdev
        summary.append(record)

    (outdir / "results.json").write_text(json.dumps(rows, indent=2))
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("STAGE1G_VERSIONED_MEMORY_READY", outdir, flush=True)
    print("SUMMARY_ROWS", len(summary), json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    start = time.time()
    main()
    print("elapsed_s", time.time() - start, flush=True)
