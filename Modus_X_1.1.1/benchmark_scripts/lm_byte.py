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
from modus_v2.models import ModelConfig, count_params, lm_loss, make_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="/home/HP/enwik8")
    p.add_argument("--outdir", default="runs/lm_byte")
    p.add_argument("--model", default="all", help="all or comma list: Modus,Modus_M,Mamba,Transformer")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--steps-per-epoch", type=int, default=None)
    p.add_argument("--embed-dim", type=int, default=None)
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--ax-res", type=int, default=None)
    p.add_argument("--n-layers", type=int, default=None)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--train-bytes", type=int, default=None)
    p.add_argument("--valid-bytes", type=int, default=None)
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def base_config(quick: bool) -> tuple[ModelConfig, dict]:
    if quick:
        cfg = ModelConfig(vocab_size=256, embed_dim=48, hidden_dim=128, ax_res=48, n_layers=2, seq_len=64, mamba_state_dim=64, n_heads_attn=4)
        train = {"batch": 8, "epochs": 2, "steps_per_epoch": 8, "train_bytes": 200_000, "valid_bytes": 20_000}
    else:
        cfg = ModelConfig(vocab_size=256, embed_dim=256, hidden_dim=768, ax_res=256, n_layers=6, seq_len=512, mamba_state_dim=256, n_heads_attn=8)
        train = {"batch": 32, "epochs": 20, "steps_per_epoch": None, "train_bytes": 90_000_000, "valid_bytes": 5_000_000}
    return cfg, train


def load_data(path: str, train_bytes: int, valid_bytes: int) -> tuple[np.ndarray, np.ndarray]:
    need = train_bytes + valid_bytes + 1
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = np.frombuffer(f.read(need), dtype=np.uint8)
        if len(data) >= need:
            return data[:train_bytes], data[train_bytes: train_bytes + valid_bytes]
        print(f"WARNING: {path} has only {len(data):,} bytes; using available split.")
        split = max(1, int(len(data) * 0.9))
        return data[:split], data[split:]

    print(f"WARNING: {path} not found. Using repeated built-in smoke text.")
    text = (
        "Modus_M is a selective matrix-state language model. "
        "It writes content by delta rule and reads by normalized query. "
    ).encode("utf-8")
    data = np.frombuffer(text * max(1, need // len(text) + 1), dtype=np.uint8)[:need]
    return data[:train_bytes], data[train_bytes: train_bytes + valid_bytes]


def batch_iter(data: np.ndarray, seq_len: int, batch: int, rng: np.random.Generator):
    n = len(data) - seq_len - 1
    while True:
        starts = rng.integers(0, n, size=batch)
        xs = np.stack([data[s: s + seq_len] for s in starts]).astype(np.int32)
        ys = np.stack([data[s + 1: s + seq_len + 1] for s in starts]).astype(np.int32)
        yield xs, ys


def eval_bpc(params: dict, fwd_fn, data: np.ndarray, seq_len: int, max_chunks: int = 128) -> float:
    chunks = min(max_chunks, max(1, (len(data) - 1) // seq_len))
    total_nll, total_tokens = 0.0, 0
    for i in range(chunks):
        s = i * seq_len
        x = jnp.array(data[s: s + seq_len].astype(np.int32))
        y = jnp.array(data[s + 1: s + seq_len + 1].astype(np.int32))
        if x.shape[0] != seq_len or y.shape[0] != seq_len:
            break
        logits = fwd_fn(params, x)
        logp = jax.nn.log_softmax(logits, axis=-1)
        nll = -logp[jnp.arange(seq_len), y]
        total_nll += float(jnp.sum(nll))
        total_tokens += seq_len
    return (total_nll / max(1, total_tokens)) / np.log(2.0)


def train_one(name: str, cfg: ModelConfig, train_cfg: dict, train_data: np.ndarray, valid_data: np.ndarray, lr: float, seed: int) -> dict:
    params, fwd_fn = make_model(name, random.PRNGKey(seed), cfg)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr, weight_decay=1e-4))
    opt_state = tx.init(params)

    @jax.jit
    def update(p, st, x, y):
        loss, grads = jax.value_and_grad(lambda pp: lm_loss(pp, fwd_fn, x, y))(p)
        updates, st = tx.update(grads, st, p)
        return optax.apply_updates(p, updates), st, loss

    rng = np.random.default_rng(1000 + seed)
    batches = batch_iter(train_data, cfg.seq_len, train_cfg["batch"], rng)
    steps_per_epoch = train_cfg["steps_per_epoch"]
    if steps_per_epoch is None:
        steps_per_epoch = max(1, len(train_data) // (cfg.seq_len * train_cfg["batch"] * 50))

    print(f"\n=== TRAIN {name} ===")
    print(f"params={count_params(params):,}")
    best = float("inf")
    rows = []
    t0 = time.time()
    for epoch in range(train_cfg["epochs"]):
        last_loss = None
        for _ in range(steps_per_epoch):
            x_np, y_np = next(batches)
            params, opt_state, loss = update(params, opt_state, jnp.array(x_np), jnp.array(y_np))
            last_loss = float(loss)
        bpc = eval_bpc(params, fwd_fn, valid_data, cfg.seq_len)
        best = min(best, bpc)
        row = {"epoch": epoch + 1, "loss": last_loss, "val_bpc": bpc, "best_val_bpc": best, "elapsed_s": time.time() - t0}
        rows.append(row)
        print(f"{name} epoch={epoch+1:03d} loss={last_loss:.4f} val_bpc={bpc:.4f} best={best:.4f} elapsed={row['elapsed_s']:.1f}s")
    return {"params": count_params(params), "best_val_bpc": best, "epochs": rows}


def main() -> None:
    args = parse_args()
    cfg, train_cfg = base_config(args.quick)
    overrides = {}
    for arg, field in [("seq_len", "seq_len"), ("embed_dim", "embed_dim"), ("hidden_dim", "hidden_dim"), ("ax_res", "ax_res"), ("n_layers", "n_layers")]:
        value = getattr(args, arg)
        if value is not None:
            overrides[field] = value
    if overrides:
        cfg = replace(cfg, **overrides)
    for arg in ["batch", "epochs", "steps_per_epoch", "train_bytes", "valid_bytes"]:
        value = getattr(args, arg.replace("_", "-"), None) if False else getattr(args, arg)
        if value is not None:
            train_cfg[arg] = value

    os.makedirs(args.outdir, exist_ok=True)
    print("JAX devices:", jax.devices())
    print("config:", cfg)
    print("train:", train_cfg)
    train_data, valid_data = load_data(args.data_path, train_cfg["train_bytes"], train_cfg["valid_bytes"])
    print(f"data train={len(train_data):,} valid={len(valid_data):,}")

    names = ["Modus", "Modus_M", "Mamba", "Transformer"] if args.model == "all" else [x.strip() for x in args.model.split(",") if x.strip()]
    results = {}
    for i, name in enumerate(names):
        results[name] = train_one(name, cfg, train_cfg, train_data, valid_data, args.lr, args.seed + i)

    out = os.path.join(args.outdir, "results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSUMMARY")
    for name, res in results.items():
        print(f"{name:<14} params={res['params']:,} best_val_bpc={res['best_val_bpc']:.4f}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
