from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modus_v2.models import ModelConfig, count_params, make_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="runs/assoc_recall")
    p.add_argument("--model", default="all", help="all or comma list")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--n-test", type=int, default=None)
    p.add_argument("--train-pairs", type=int, default=None)
    p.add_argument("--test-pairs", default="8,16,32,64")
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--embed-dim", type=int, default=None)
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--ax-res", type=int, default=None)
    p.add_argument("--n-layers", type=int, default=None)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def settings(quick: bool):
    if quick:
        cfg = ModelConfig(vocab_size=256, embed_dim=48, hidden_dim=128, ax_res=48, n_layers=2, seq_len=20, mamba_state_dim=64, n_heads_attn=4)
        train = {"n_train": 512, "n_test": 128, "train_pairs": 8, "batch": 32, "epochs": 2}
    else:
        cfg = ModelConfig(vocab_size=256, embed_dim=192, hidden_dim=512, ax_res=192, n_layers=4, seq_len=66, mamba_state_dim=192, n_heads_attn=8)
        train = {"n_train": 20000, "n_test": 2000, "train_pairs": 16, "batch": 128, "epochs": 60}
    return cfg, train


def make_dataset(seed: int, n: int, n_pairs: int, max_keys: int, n_values: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    key_offset = 0
    val_offset = max_keys
    query_token = max_keys + n_values
    pad_token = query_token + 1
    seq_len = 2 * n_pairs + 2
    x = np.full((n, seq_len), pad_token, dtype=np.int32)
    y = np.zeros((n,), dtype=np.int32)
    full_targets = np.full((n, seq_len), pad_token, dtype=np.int32)

    for i in range(n):
        keys = rng.choice(max_keys, size=n_pairs, replace=False)
        vals = rng.choice(n_values, size=n_pairs, replace=True)
        order = rng.permutation(n_pairs)
        pos = 0
        for j in order:
            x[i, pos] = key_offset + keys[j]
            x[i, pos + 1] = val_offset + vals[j]
            pos += 2
        target_idx = int(rng.integers(0, n_pairs))
        x[i, -2] = query_token
        x[i, -1] = key_offset + keys[target_idx]
        y[i] = val_offset + vals[target_idx]
        full_targets[i, :-1] = x[i, 1:]
        full_targets[i, -1] = y[i]
    return x, full_targets, y


def final_token_loss(params: dict, fwd_fn, x: jax.Array, y: jax.Array) -> jax.Array:
    logits = jax.vmap(lambda xi: fwd_fn(params, xi))(x)
    final_logits = logits[:, -1, :]
    logp = jax.nn.log_softmax(final_logits, axis=-1)
    return -jnp.mean(logp[jnp.arange(x.shape[0]), y])


def eval_acc(params: dict, fwd_fn, x: np.ndarray, y: np.ndarray, batch: int) -> float:
    correct = 0
    for start in range(0, len(y), batch):
        xb = jnp.array(x[start: start + batch])
        logits = jax.vmap(lambda xi: fwd_fn(params, xi))(xb)
        pred = np.array(jnp.argmax(logits[:, -1, :], axis=-1))
        correct += int(np.sum(pred == y[start: start + batch]))
    return 100.0 * correct / len(y)


def train_one(name: str, cfg: ModelConfig, train_cfg: dict, lr: float, seed: int, test_pairs: list[int]) -> dict:
    max_keys = 96
    n_values = 96
    params, fwd_fn = make_model(name, random.PRNGKey(seed), cfg)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr, weight_decay=1e-4))
    opt_state = tx.init(params)

    @jax.jit
    def update(p, st, x, y):
        loss, grads = jax.value_and_grad(lambda pp: final_token_loss(pp, fwd_fn, x, y))(p)
        updates, st = tx.update(grads, st, p)
        return optax.apply_updates(p, updates), st, loss

    x_train, _, y_train = make_dataset(seed + 1, train_cfg["n_train"], train_cfg["train_pairs"], max_keys, n_values)
    x_val, _, y_val = make_dataset(seed + 2, train_cfg["n_test"], train_cfg["train_pairs"], max_keys, n_values)
    rng = np.random.default_rng(seed + 3)
    print(f"\n=== TRAIN {name} ===")
    print(f"params={count_params(params):,}")
    best = 0.0
    rows = []
    t0 = time.time()
    for epoch in range(train_cfg["epochs"]):
        perm = rng.permutation(len(y_train))
        losses = []
        for start in range(0, len(perm) - train_cfg["batch"] + 1, train_cfg["batch"]):
            idx = perm[start: start + train_cfg["batch"]]
            params, opt_state, loss = update(params, opt_state, jnp.array(x_train[idx]), jnp.array(y_train[idx]))
            losses.append(float(loss))
        acc = eval_acc(params, fwd_fn, x_val, y_val, train_cfg["batch"])
        best = max(best, acc)
        rows.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "val_acc": acc, "best_val_acc": best, "elapsed_s": time.time() - t0})
        print(f"{name} epoch={epoch+1:03d} loss={np.mean(losses):.4f} val_acc={acc:.2f}% best={best:.2f}% elapsed={rows[-1]['elapsed_s']:.1f}s")

    length_results = {}
    for pairs in test_pairs:
        # Reuse trained params for recurrent models across longer lengths; transformer pos table is limited.
        if name == "Transformer" and pairs != train_cfg["train_pairs"]:
            length_results[str(pairs)] = None
            continue
        x_test, _, y_test = make_dataset(seed + 100 + pairs, train_cfg["n_test"], pairs, max_keys, n_values)
        length_results[str(pairs)] = eval_acc(params, fwd_fn, x_test, y_test, train_cfg["batch"])
        print(f"{name} pairs={pairs:<3} acc={length_results[str(pairs)]:.2f}%")
    return {"params": count_params(params), "best_val_acc": best, "epochs": rows, "length_acc": length_results}


def main() -> None:
    args = parse_args()
    cfg, train_cfg = settings(args.quick)
    overrides = {}
    for arg, field in [("embed_dim", "embed_dim"), ("hidden_dim", "hidden_dim"), ("ax_res", "ax_res"), ("n_layers", "n_layers")]:
        value = getattr(args, arg)
        if value is not None:
            overrides[field] = value
    if args.train_pairs is not None:
        train_cfg["train_pairs"] = args.train_pairs
    seq_len = 2 * train_cfg["train_pairs"] + 2
    if overrides or seq_len != cfg.seq_len:
        cfg = replace(cfg, seq_len=seq_len, **overrides)
    for key in ["n_train", "n_test", "batch", "epochs"]:
        value = getattr(args, key)
        if value is not None:
            train_cfg[key] = value

    os.makedirs(args.outdir, exist_ok=True)
    print("JAX devices:", jax.devices())
    print("config:", cfg)
    print("train:", train_cfg)
    names = ["Modus", "Modus_M", "Mamba", "Transformer"] if args.model == "all" else [x.strip() for x in args.model.split(",") if x.strip()]
    test_pairs = [int(x) for x in args.test_pairs.split(",") if x.strip()]
    results = {}
    for i, name in enumerate(names):
        results[name] = train_one(name, cfg, train_cfg, args.lr, args.seed + i, test_pairs)

    out = os.path.join(args.outdir, "results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
