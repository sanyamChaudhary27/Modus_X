"""Near-parameter-matched clean/update retrieval gate at equal state bytes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
FRONTIER = THIS_DIR / "run_equal_memory_frontier.py"
RUNNER = THIS_DIR / "run_versioned_memory_ablation.py"
STAGE1G = THIS_DIR / "run_stage1g_versioned_memory.py"
BALANCED_KV = THIS_DIR / "balanced_kv.py"
ROLES = ("latest", "previous", "first")
TARGET_GROUPS = ("overwritten", "clean")


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


def balanced_dataset(kv, stage1g, key, n: int, cfg):
    if n % 6:
        raise ValueError("Balanced clean/update data size must be divisible by six")
    keys = kv.random.split(key, 6)
    count = n // 6
    parts = []
    for subkey, target_group, role in zip(
        keys,
        ("overwritten", "overwritten", "overwritten", "clean", "clean", "clean"),
        ROLES + ROLES,
    ):
        parts.append(
            stage1g.make_versioned_kv(
                kv,
                subkey,
                count,
                cfg.train_len,
                cfg,
                query_mode=role,
                target_mode=target_group,
                version_tag_facts=True,
            )
        )
    seqs = np.concatenate([part[0] for part in parts], axis=0)
    labels = np.concatenate([part[1] for part in parts], axis=0)
    meta = {
        name: np.concatenate([np.asarray(part[2][name]) for part in parts], axis=0)
        for name in parts[0][2]
    }
    return seqs, labels, meta


def predict(kv, params, fwd, dataset, cfg) -> np.ndarray:
    seqs, labels, _ = dataset
    batched = kv.jit(kv.vmap(fwd, in_axes=(None, 0)))
    predictions = []
    for start in range(0, len(labels), cfg.batch):
        logits = batched(
            params, kv.jnp.asarray(seqs[start : start + cfg.batch], dtype=kv.jnp.float32)
        )
        predictions.append(np.asarray(kv.jnp.argmax(logits, axis=-1)))
    return np.concatenate(predictions)


def evaluate(kv, params, fwd, dataset, cfg) -> dict:
    _, labels, meta = dataset
    labels = np.asarray(labels)
    pred = predict(kv, params, fwd, dataset, cfg)
    overwritten = np.asarray(meta["overwritten_target"], dtype=bool)
    modes = np.asarray(meta["modes"])
    latest = np.asarray(meta["latest_values"])
    previous = np.asarray(meta["previous_values"])
    first = np.asarray(meta["first_values"])

    def accuracy(mask: np.ndarray) -> float:
        count = int(mask.sum())
        return 100.0 * float(((pred == labels) & mask).sum()) / max(1, count)

    metrics = {
        "examples": int(len(labels)),
        "clean_query_count": int((~overwritten).sum()),
        "overwritten_query_count": int(overwritten.sum()),
        "acc_all": accuracy(np.ones_like(overwritten)),
        "acc_clean": accuracy(~overwritten),
        "acc_overwritten": accuracy(overwritten),
    }
    for role in ROLES:
        role_mask = modes == role
        metrics[f"acc_{role}_clean"] = accuracy(role_mask & ~overwritten)
        metrics[f"acc_{role}_overwritten"] = accuracy(role_mask & overwritten)

    latest_overwritten = overwritten & (modes == "latest")
    stale_distinct = latest_overwritten & ((previous != latest) | (first != latest))
    stale_prediction = ((pred == previous) & (previous != latest)) | (
        (pred == first) & (first != latest)
    )
    metrics["latest_overwritten_stale_eligible"] = int(stale_distinct.sum())
    metrics["stale_false_recall_on_latest_overwritten"] = (
        100.0 * float((stale_prediction & stale_distinct).sum()) / max(1, int(stale_distinct.sum()))
    )
    return metrics


def train_with_validation(kv, name: str, params, fwd, train, validation, cfg, seed: int):
    train_seqs, train_labels, train_meta = train
    validation_labels = validation[1]
    batched = kv.jit(kv.vmap(fwd, in_axes=(None, 0)))

    def loss_fn(p, seqs, labels):
        logits = batched(p, seqs)
        return -kv.jnp.mean(
            kv.jax.nn.log_softmax(logits, axis=-1)[kv.jnp.arange(len(labels)), labels]
        )

    optimizer = kv.optax.chain(
        kv.optax.clip_by_global_norm(1.0), kv.optax.adamw(cfg.lr, weight_decay=1e-4)
    )
    optimizer_state = optimizer.init(params)

    @kv.jit
    def update(p, state, seqs, labels):
        loss, grads = kv.jax.value_and_grad(loss_fn)(p, seqs, labels)
        updates, next_state = optimizer.update(grads, state, p)
        return kv.optax.apply_updates(p, updates), next_state, loss

    best_accuracy = -1.0
    best_params = params
    history = []
    started = time.time()
    print(f"  params={kv.jax.tree_util.tree_reduce(lambda n, x: n + x.size, params, 0):,}")
    for epoch in range(cfg.epochs):
        permutation = np.random.default_rng(seed * 10_000 + epoch).permutation(len(train_labels))
        losses = []
        for start in range(0, len(train_labels), cfg.batch):
            indices = permutation[start : start + cfg.batch]
            params, optimizer_state, loss = update(
                params,
                optimizer_state,
                kv.jnp.asarray(train_seqs[indices], dtype=kv.jnp.float32),
                kv.jnp.asarray(train_labels[indices], dtype=kv.jnp.int32),
            )
            losses.append(float(loss))
        validation_pred = predict(kv, params, fwd, validation, cfg)
        validation_accuracy = 100.0 * float(
            np.mean(validation_pred == np.asarray(validation_labels))
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_params = params
        row = {
            "epoch": epoch + 1,
            "loss": statistics.mean(losses),
            "validation_accuracy": validation_accuracy,
            "best_validation_accuracy": best_accuracy,
            "elapsed_s": time.time() - started,
        }
        history.append(row)
        print("MIXED_UPDATE_PROGRESS", json.dumps({"model": name, **row}), flush=True)
    return best_params, history


def aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)
    output = []
    for model, members in sorted(grouped.items()):
        result = {
            "model": model,
            "seeds": sorted(member["seed"] for member in members),
            "params_mean": statistics.mean(member["params"] for member in members),
            "state_bytes": members[0]["state_bytes"],
            "kv_window": members[0].get("kv_window"),
        }
        for metric in members[0]["metrics"]:
            values = [float(member["metrics"][metric]) for member in members]
            result[f"{metric}_mean"] = statistics.mean(values)
            result[f"{metric}_stdev"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        output.append(result)
    return output


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
    frontier = load_module(FRONTIER, "mixed_update_frontier")
    runner = load_module(RUNNER, "mixed_update_runner")
    stage1g = load_module(STAGE1G, "mixed_update_stage1g")
    kv = load_module(BALANCED_KV, "mixed_update_kv")
    protocol = argparse.Namespace(
        n_values=32,
        d_model=96,
        n_train=args.n_train,
        n_test=args.n_test,
        batch=args.batch,
        epochs=args.epochs,
        lr=args.lr,
    )
    transformer_cfg = {
        "d_model": 64,
        "layers": 2,
        "heads": 4,
        "ffn": 512,
        "qk_mode": "tied",
    }
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
        keys = kv.random.split(kv.random.PRNGKey(seed), 5)
        train = balanced_dataset(kv, stage1g, keys[0], args.n_train, data_cfg)
        validation = balanced_dataset(
            kv, stage1g, keys[1], args.n_validation, data_cfg
        )
        test = balanced_dataset(kv, stage1g, keys[2], args.n_test, data_cfg)
        state_bytes = (
            2 * args.matrix_width * args.matrix_width + args.matrix_width
        ) * args.state_dtype_bytes
        kv_bytes_per_token = (
            transformer_cfg["layers"]
            * 2
            * transformer_cfg["d_model"]
            * args.state_dtype_bytes
        )
        kv_window = max(1, min(data_cfg.train_len - 1, state_bytes // kv_bytes_per_token))
        print(
            "MIXED_UPDATE_CASE",
            json.dumps(
                {
                    "seed": seed,
                    "matrix_width": args.matrix_width,
                    "state_bytes": state_bytes,
                    "kv_window": kv_window,
                    "train_examples": args.n_train,
                    "validation_examples": args.n_validation,
                    "test_examples": args.n_test,
                }
            ),
            flush=True,
        )

        current_params, current_fwd, _, _ = runner.make_model_with_aux(
            "CurrentArchiveDelta", keys[3], model_cfg
        )
        current_params, current_history = train_with_validation(
            kv,
            "CurrentArchiveDelta",
            current_params,
            current_fwd,
            train,
            validation,
            data_cfg,
            seed,
        )
        current_row = {
            "model": "CurrentArchiveDelta",
            "seed": seed,
            "params": runner.count_params(current_params),
            "state_bytes": state_bytes,
            "metrics": evaluate(kv, current_params, current_fwd, test, data_cfg),
            "history": current_history,
        }
        rows.append(current_row)
        print(
            "MIXED_UPDATE_RESULT",
            json.dumps({k: v for k, v in current_row.items() if k != "history"}),
            flush=True,
        )

        transformer_params = frontier.init_transformer(
            kv, keys[4], data_cfg.d_model, data_cfg.n_values, transformer_cfg
        )
        transformer_fwd = lambda p, seq, window=kv_window: frontier.transformer_forward(
            kv, p, seq, transformer_cfg, window
        )
        transformer_params, transformer_history = train_with_validation(
            kv,
            "TransformerKV",
            transformer_params,
            transformer_fwd,
            train,
            validation,
            data_cfg,
            seed,
        )
        transformer_row = {
            "model": "TransformerKV",
            "seed": seed,
            "params": runner.count_params(transformer_params),
            "state_bytes": state_bytes,
            "kv_window": kv_window,
            "metrics": evaluate(
                kv, transformer_params, transformer_fwd, test, data_cfg
            ),
            "history": transformer_history,
        }
        rows.append(transformer_row)
        print(
            "MIXED_UPDATE_RESULT",
            json.dumps({k: v for k, v in transformer_row.items() if k != "history"}),
            flush=True,
        )
        (args.outdir / "progress.json").write_text(
            json.dumps({"rows": rows, "aggregate": aggregate(rows)}, indent=2),
            encoding="utf-8",
        )

    report = {
        "protocol": {
            "seeds": [int(seed) for seed in args.seeds.split(",")],
            "matrix_width": args.matrix_width,
            "bindings": args.bindings,
            "overwrite_rate": args.overwrite_rate,
            "balanced_roles": list(ROLES),
            "balanced_target_groups": list(TARGET_GROUPS),
            "checkpoint_selection": "validation_only",
            "transformer": transformer_cfg,
        },
        "rows": rows,
        "aggregate": aggregate(rows),
    }
    (args.outdir / "mixed_update_gate.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("MIXED_UPDATE_SUMMARY", json.dumps(report["aggregate"]), flush=True)
    print("MIXED_UPDATE_GATE_READY", args.outdir, flush=True)


if __name__ == "__main__":
    main()
