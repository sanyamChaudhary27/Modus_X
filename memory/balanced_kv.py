"""
Parameter-matched scalar-vs-vector router ablation on the recovered Modus v5
balanced continuous key-value recall protocol.

This is intentionally based on the older Zenodo Modus experiment
`v5_multihead_delta.py` rather than the tokenized `associative_recall.py` toy.
The scalar control must first reproduce the old high-recall behavior before
the vector router should be judged.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import jit, lax, random, vmap


@dataclass(frozen=True)
class Config:
    d_model: int = 96
    key_dim: int = 32
    n_values: int = 32
    n_pairs: int = 32
    train_len: int = 128
    test_lens: tuple[int, ...] = (128, 256, 512, 1024, 2048)
    n_train: int = 40000
    n_test: int = 4000
    batch: int = 256
    epochs: int = 30
    patience: int = 30
    lr: float = 3e-4
    ax_res: int = 128
    seed: int = 7
    vector_state: int = 128
    router_hidden: int = 128
    router_bias: float = 6.0
    residual_scale: float = 0.25
    overwrite_rate: float = 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="experiments/router_balanced_kv_outputs")
    p.add_argument("--models", default="ScalarPM,VectorPM")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--n-test", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--train-len", type=int, default=None)
    p.add_argument("--test-lens", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--router-bias", type=float, default=None)
    p.add_argument("--router-hidden", type=int, default=None)
    p.add_argument("--residual-scale", type=float, default=None)
    p.add_argument("--overwrite-rate", type=float, default=None)
    return p.parse_args()


def make_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    if args.quick:
        cfg = replace(cfg, n_train=4096, n_test=512, batch=128, epochs=20, test_lens=(128, 256))
    overrides = {}
    for arg, field in [
        ("epochs", "epochs"),
        ("n_train", "n_train"),
        ("n_test", "n_test"),
        ("batch", "batch"),
        ("train_len", "train_len"),
        ("seed", "seed"),
        ("router_bias", "router_bias"),
        ("router_hidden", "router_hidden"),
        ("residual_scale", "residual_scale"),
        ("overwrite_rate", "overwrite_rate"),
    ]:
        value = getattr(args, arg)
        if value is not None:
            overrides[field] = value
    if args.test_lens:
        overrides["test_lens"] = tuple(int(x) for x in args.test_lens.split(",") if x)
    return replace(cfg, **overrides) if overrides else cfg


def make_balanced_kv(key: jax.Array, n: int, seq_len: int, cfg: Config):
    k_seed, k_keys, _, k_noise = random.split(key, 4)
    rng = np.random.default_rng(int(random.randint(k_seed, (), 0, 2**31 - 1)))

    seqs = np.array(random.normal(k_noise, (n, seq_len, cfg.d_model)) * 0.05, dtype=np.float32)
    seqs[:, :, : cfg.key_dim + cfg.n_values + 2] = 0.0

    keys_np = np.array(random.normal(k_keys, (n, cfg.n_pairs, cfg.key_dim)), dtype=np.float32)
    keys_np /= np.linalg.norm(keys_np, axis=-1, keepdims=True) + 1e-8

    value_offset = cfg.key_dim
    fact_marker = cfg.key_dim + cfg.n_values
    query_marker = fact_marker + 1
    labels = np.zeros(n, dtype=np.int32)

    for i in range(n):
        perm = rng.permutation(cfg.n_pairs).astype(np.int32)
        positions = np.sort(rng.choice(np.arange(0, seq_len - 1), size=cfg.n_pairs, replace=False))
        for j in range(cfg.n_pairs):
            pos = int(positions[j])
            seqs[i, pos, :] = 0.0
            seqs[i, pos, : cfg.key_dim] = keys_np[i, j]
            seqs[i, pos, value_offset + int(perm[j])] = 1.0
            seqs[i, pos, fact_marker] = 1.0

        overwrite_count = int(round(cfg.n_pairs * cfg.overwrite_rate))
        if overwrite_count:
            overwrite_sources = rng.choice(cfg.n_pairs, size=overwrite_count, replace=False)
            occupied = set(int(position) for position in positions)
            for source in overwrite_sources:
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
                seqs[i, overwrite_pos, :] = 0.0
                seqs[i, overwrite_pos, : cfg.key_dim] = keys_np[i, source]
                seqs[i, overwrite_pos, value_offset + new_value] = 1.0
                seqs[i, overwrite_pos, fact_marker] = 1.0
                perm[source] = new_value

        target = int(rng.integers(0, cfg.n_pairs))
        seqs[i, -1, :] = 0.0
        seqs[i, -1, : cfg.key_dim] = keys_np[i, target]
        seqs[i, -1, query_marker] = 1.0
        labels[i] = int(perm[target])

    return seqs, labels


def init_head(key: jax.Array, in_dim: int, n_values: int, hidden: int) -> dict:
    k1, k2 = random.split(key)
    return {
        "w1": random.normal(k1, (hidden, in_dim)) * 0.01,
        "b1": jnp.zeros(hidden),
        "w2": random.normal(k2, (n_values, hidden)) * 0.01,
        "b2": jnp.zeros(n_values),
    }


def head_fwd(p: dict, x: jax.Array) -> jax.Array:
    return p["w2"] @ jax.nn.relu(p["w1"] @ x + p["b1"]) + p["b2"]


def init_router_delta(key: jax.Array, cfg: Config, vector_router: bool, head_hidden: int) -> dict:
    keys = random.split(key, 13)
    ax = cfg.ax_res
    d = cfg.d_model
    n = cfg.vector_state
    router_hidden = cfg.router_hidden

    wk = np.array(random.normal(keys[0], (ax, d)) * 0.001)
    wq = np.array(random.normal(keys[1], (ax, d)) * 0.001)
    wv = np.array(random.normal(keys[2], (ax, d)) * 0.001)
    read_w = np.array(random.normal(keys[3], (ax, ax)) * 0.001)

    for i in range(min(ax, cfg.key_dim)):
        wk[i, i] = 1.0
        wq[i, i] = 1.0
    val_off = cfg.key_dim
    for i in range(min(ax, cfg.n_values)):
        wv[i, val_off + i] = 1.0
        read_w[i, i] = 1.0

    router_out = ax if vector_router else 1
    return {
        "wk": jnp.array(wk),
        "wq": jnp.array(wq),
        "wv": jnp.array(wv),
        "wg": random.normal(keys[4], (1, d)) * 0.01,
        "bg": jnp.ones(1) * 4.0,
        "read_w": jnp.array(read_w),
        "read_b": jnp.zeros(ax),
        "s_wu": random.normal(keys[5], (n, d)) * 0.01,
        "s_w_delta": random.normal(keys[6], (n, d)) * 0.01,
        "s_b_delta": jnp.zeros(n),
        "s_w_ret": random.normal(keys[7], (n, d)) * 0.01,
        "s_b_ret": jnp.ones(n) * 2.0,
        "s_proj": random.normal(keys[8], (ax, n)) * 0.01,
        "router_w1": random.normal(keys[9], (router_hidden, d)) * 0.01,
        "router_b1": jnp.zeros(router_hidden),
        "router_w2": random.normal(keys[10], (router_out, router_hidden)) * 0.01,
        "router_b2": jnp.ones(router_out) * cfg.router_bias,
        "classifier": init_head(keys[11], ax, cfg.n_values, head_hidden),
    }


def model_fwd(p: dict, seq: jax.Array, cfg: Config, fusion: str = "convex") -> jax.Array:
    fact_m = cfg.key_dim + cfg.n_values

    def step(carry, x):
        h, s = carry
        k = p["wk"] @ x
        k = k / (jnp.linalg.norm(k) + 1e-8)
        val = jnp.tanh(p["wv"] @ x)
        fact_gate = x[fact_m]
        learned_gate = jax.nn.sigmoid((p["wg"] @ x + p["bg"])[0])
        gate = fact_gate * learned_gate

        old = h @ k
        h = h + gate * jnp.outer(val - old, k)

        u = jnp.tanh(p["s_wu"] @ x)
        delta = jax.nn.sigmoid(p["s_w_delta"] @ x + p["s_b_delta"])
        retain = jax.nn.sigmoid(p["s_w_ret"] @ x + p["s_b_ret"])
        s = retain * s + gate * delta * u
        return (h, s), None

    h0 = jnp.zeros((cfg.ax_res, cfg.ax_res))
    s0 = jnp.zeros(cfg.vector_state)
    (h, s), _ = lax.scan(step, (h0, s0), seq[:-1])

    query_x = seq[-1]
    q = p["wq"] @ query_x
    q = q / (jnp.linalg.norm(q) + 1e-8)
    memory_read = p["read_w"] @ (h @ q) + p["read_b"]
    vector_read = p["s_proj"] @ s

    r_hidden = jax.nn.relu(p["router_w1"] @ query_x + p["router_b1"])
    router = jax.nn.sigmoid(p["router_w2"] @ r_hidden + p["router_b2"])
    if router.shape[0] == 1:
        router = router[0]
    if fusion == "convex":
        fused = router * memory_read + (1.0 - router) * vector_read
    elif fusion == "residual":
        fused = memory_read + cfg.residual_scale * router * vector_read
    elif fusion == "delta":
        fused = memory_read + cfg.residual_scale * router * (vector_read - memory_read)
    else:
        raise ValueError(f"Unknown fusion {fusion}")
    return head_fwd(p["classifier"], fused)


def count_params(tree) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(tree) if hasattr(x, "size"))


def make_model(name: str, key: jax.Array, cfg: Config):
    if name == "VectorPM":
        params = init_router_delta(key, cfg, vector_router=True, head_hidden=128)
        fusion = "convex"
    elif name == "VectorResidualPM":
        params = init_router_delta(key, cfg, vector_router=True, head_hidden=128)
        fusion = "residual"
    elif name == "VectorDeltaPM":
        params = init_router_delta(key, cfg, vector_router=True, head_hidden=128)
        fusion = "delta"
    elif name == "VectorLeanPM":
        lean_cfg = cfg
        router_delta = (cfg.ax_res - 1) * (128 + 1)
        head_unit = cfg.ax_res + cfg.n_values + 1
        scalar_hidden = 128 + max(1, round(router_delta / head_unit))
        target_params = count_params(init_router_delta(key, cfg, vector_router=False, head_hidden=scalar_hidden))
        best_params = None
        best_hidden = 128
        best_gap = 10**18
        for hidden in range(128, 260):
            candidate = init_router_delta(key, lean_cfg, vector_router=True, head_hidden=hidden)
            gap = abs(count_params(candidate) - target_params)
            if gap < best_gap:
                best_gap = gap
                best_hidden = hidden
                best_params = candidate
        params = best_params
        fusion = "convex"
    elif name == "ScalarBase":
        params = init_router_delta(key, cfg, vector_router=False, head_hidden=128)
        fusion = "convex"
    elif name == "ScalarPM":
        # Vector router adds (ax_res - 1) * (router_hidden + 1) parameters.
        # Add approximately the same amount to the scalar classifier hidden width.
        router_delta = (cfg.ax_res - 1) * (cfg.router_hidden + 1)
        head_unit = cfg.ax_res + cfg.n_values + 1
        hidden = 128 + max(1, round(router_delta / head_unit))
        params = init_router_delta(key, cfg, vector_router=False, head_hidden=hidden)
        fusion = "convex"
    else:
        raise ValueError(f"Unknown model {name}")
    return params, lambda p, s: model_fwd(p, s, cfg, fusion=fusion)


def train_model(name: str, params: dict, fwd_fn, train, test, cfg: Config):
    tr_s, tr_l = train
    te_s, te_l = test
    fwd_b = jit(vmap(fwd_fn, in_axes=(None, 0)))

    def loss_fn(p, s, y):
        logits = fwd_b(p, s)
        logp = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.mean(logp[jnp.arange(len(y)), y])

    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(cfg.lr, weight_decay=1e-4))
    state = opt.init(params)

    @jit
    def update(p, st, s, y):
        loss, grads = jax.value_and_grad(loss_fn)(p, s, y)
        updates, st2 = opt.update(grads, st, p)
        return optax.apply_updates(p, updates), st2, loss

    def eval_acc(p) -> float:
        correct = 0
        for start in range(0, len(te_l), cfg.batch):
            end = min(start + cfg.batch, len(te_l))
            logits = fwd_b(p, jnp.array(te_s[start:end], dtype=jnp.float32))
            pred = jnp.argmax(logits, axis=-1)
            correct += int(jnp.sum(pred == jnp.array(te_l[start:end], dtype=jnp.int32)))
        return 100.0 * correct / len(te_l)

    best, best_p, pat = 0.0, params, 0
    rows = []
    t0 = time.time()
    print(f"  params={count_params(params):,}", flush=True)
    for epoch in range(cfg.epochs):
        perm = np.random.default_rng(cfg.seed * 1000 + epoch).permutation(len(tr_l))
        losses = []
        for start in range(0, len(tr_l) - cfg.batch + 1, cfg.batch):
            idx = perm[start:start + cfg.batch]
            params, state, loss = update(
                params,
                state,
                jnp.array(tr_s[idx], dtype=jnp.float32),
                jnp.array(tr_l[idx], dtype=jnp.int32),
            )
            losses.append(float(loss))
        acc = eval_acc(params)
        if acc > best:
            best, best_p, pat = acc, params, 0
        else:
            pat += 1
        row = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "acc": acc,
            "best": best,
            "elapsed_s": time.time() - t0,
        }
        rows.append(row)
        print(
            f"  [{name}] ep={epoch+1:03d} loss={row['loss']:.4f} "
            f"acc={acc:5.1f}% best={best:5.1f}% pat={pat}/{cfg.patience} "
            f"elapsed={row['elapsed_s']:6.1f}s",
            flush=True,
        )
        if pat >= cfg.patience:
            break
    return best, best_p, rows


def eval_length(name, params, fwd_fn, length, cfg, seed) -> float:
    seqs, labels = make_balanced_kv(random.PRNGKey(seed), cfg.n_test, length, cfg)
    fwd_b = jit(vmap(fwd_fn, in_axes=(None, 0)))
    correct = 0
    t0 = time.time()
    for start in range(0, len(labels), cfg.batch):
        end = min(start + cfg.batch, len(labels))
        logits = fwd_b(params, jnp.array(seqs[start:end], dtype=jnp.float32))
        pred = jnp.argmax(logits, axis=-1)
        correct += int(jnp.sum(pred == jnp.array(labels[start:end], dtype=jnp.int32)))
    acc = 100.0 * correct / len(labels)
    print(f"  {name:<10} L={length:<5} acc={acc:5.1f}% eval={time.time()-t0:5.1f}s", flush=True)
    return acc


def main() -> None:
    args = parse_args()
    cfg = make_config(args)
    os.makedirs(args.outdir, exist_ok=True)
    print("JAX devices:", jax.devices(), flush=True)
    print("config:", cfg, flush=True)

    train = make_balanced_kv(random.PRNGKey(cfg.seed + 1), cfg.n_train, cfg.train_len, cfg)
    test = make_balanced_kv(random.PRNGKey(cfg.seed + 2), cfg.n_test, cfg.train_len, cfg)
    names = [x.strip() for x in args.models.split(",") if x.strip()]

    results = {}
    trained = {}
    for i, name in enumerate(names):
        print("=" * 72, flush=True)
        print("TRAIN", name, flush=True)
        print("=" * 72, flush=True)
        params, fwd_fn = make_model(name, random.PRNGKey(100 + cfg.seed + i), cfg)
        best, best_p, rows = train_model(name, params, fwd_fn, train, test, cfg)
        results[name] = {"params": count_params(params), "train_best": best, "epochs": rows}
        trained[name] = (best_p, fwd_fn)

    print("=" * 72, flush=True)
    print("ZERO-SHOT LENGTH GENERALIZATION", flush=True)
    print("=" * 72, flush=True)
    for name, (params, fwd_fn) in trained.items():
        for length in cfg.test_lens:
            results[name][str(length)] = eval_length(name, params, fwd_fn, length, cfg, 2000 + length)

    out = os.path.join(args.outdir, "results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
