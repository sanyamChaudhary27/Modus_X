from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Callable

import jax
import jax.numpy as jnp
from jax import lax, random


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 50257
    embed_dim: int = 256
    hidden_dim: int = 768
    ax_res: int = 256
    n_layers: int = 6
    n_heads_attn: int = 8
    seq_len: int = 512
    mamba_state_dim: int = 256
    vector_router: bool = False
    router_hidden: int = 0


def count_params(tree) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(tree) if hasattr(x, "size"))


def layer_norm(x: jax.Array, g: jax.Array, b: jax.Array) -> jax.Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / jnp.sqrt(var + 1e-5) + b


def normalize(x: jax.Array) -> jax.Array:
    return x / jnp.sqrt(jnp.sum(jnp.square(x)) + 1e-8)


def init_embed(key: jax.Array, cfg: ModelConfig) -> jax.Array:
    return random.normal(key, (cfg.vocab_size, cfg.embed_dim)) * 0.02


def init_lm_head(key: jax.Array, in_dim: int, cfg: ModelConfig) -> dict:
    k1, k2 = random.split(key)
    return {
        "w1": random.normal(k1, (cfg.hidden_dim, in_dim)) * 0.02,
        "b1": jnp.zeros(cfg.hidden_dim),
        "w2": random.normal(k2, (cfg.vocab_size, cfg.hidden_dim)) * 0.02,
        "b2": jnp.zeros(cfg.vocab_size),
    }


def lm_head_fwd(p: dict, x: jax.Array) -> jax.Array:
    h = jax.nn.gelu(x @ p["w1"].T + p["b1"])
    return h @ p["w2"].T + p["b2"]


def apply_dropout(x: jax.Array, key: jax.Array | None, rate: float) -> jax.Array:
    if key is None or rate <= 0.0:
        return x
    keep = 1.0 - rate
    mask = random.bernoulli(key, keep, x.shape)
    return jnp.where(mask, x / keep, 0.0)


# ---------------------------------------------------------------------------
# Modus: v1-style matrix delta memory
# ---------------------------------------------------------------------------


def init_modus_layer(key: jax.Array, cfg: ModelConfig) -> dict:
    k1, k2, k3, k4, k5 = random.split(key, 5)
    d, r = cfg.embed_dim, cfg.ax_res
    return {
        "wk": random.normal(k1, (r, d)) * 0.02,
        "wq": random.normal(k2, (r, d)) * 0.02,
        "wv": random.normal(k3, (r, d)) * 0.02,
        "wg": random.normal(k4, (1, d)) * 0.01,
        "bg": jnp.ones(1) * 2.0,
        "ln_g": jnp.ones(r),
        "ln_b": jnp.zeros(r),
        "proj_w": random.normal(k5, (d, d + r)) * 0.02,
        "proj_b": jnp.zeros(d),
    }


@jax.checkpoint
def modus_layer_fwd(layer: dict, x_seq: jax.Array) -> jax.Array:
    r = layer["wk"].shape[0]

    def step(h, e):
        k = normalize(layer["wk"] @ e)
        q = normalize(layer["wq"] @ e)
        val = jnp.tanh(layer["wv"] @ e)
        gate = jax.nn.sigmoid((layer["wg"] @ e + layer["bg"])[0])
        old = h @ k
        h = h + gate * jnp.outer(val - old, k)
        context = layer_norm(h @ q, layer["ln_g"], layer["ln_b"])
        out = layer["proj_w"] @ jnp.concatenate([e, context]) + layer["proj_b"]
        return h, out

    _, out = lax.scan(step, jnp.zeros((r, r)), x_seq)
    return out


def init_modus_lm(key: jax.Array, cfg: ModelConfig) -> dict:
    keys = random.split(key, cfg.n_layers + 2)
    layers = [init_modus_layer(keys[i], cfg) for i in range(cfg.n_layers)]
    return {
        "embed": init_embed(keys[-2], cfg),
        "layers": jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *layers),
        "head": init_lm_head(keys[-1], cfg.embed_dim, cfg),
    }


def modus_lm_fwd(p: dict, x_ids: jax.Array, cfg: ModelConfig) -> jax.Array:
    x = p["embed"][x_ids]

    def scan_layer(x_in, layer):
        return x_in + modus_layer_fwd(layer, x_in), None

    x, _ = lax.scan(scan_layer, x, p["layers"])
    return lm_head_fwd(p["head"], x)


# ---------------------------------------------------------------------------
# Modus_M: matrix delta memory with Mamba-style selective gates
# ---------------------------------------------------------------------------


def init_modus_m_layer(key: jax.Array, cfg: ModelConfig) -> dict:
    keys = random.split(key, 12)
    d, r = cfg.embed_dim, cfg.ax_res
    return {
        "wk": random.normal(keys[0], (r, d)) * 0.02,
        "wq": random.normal(keys[1], (r, d)) * 0.02,
        "wv": random.normal(keys[2], (r, d)) * 0.02,
        "w_write": random.normal(keys[3], (1, d)) * 0.01,
        "b_write": jnp.ones(1) * 1.0,
        "w_delta": random.normal(keys[4], (1, d)) * 0.01,
        "b_delta": jnp.zeros(1),
        "w_retain": random.normal(keys[5], (1, d)) * 0.01,
        "b_retain": jnp.ones(1) * 3.0,
        "w_read": random.normal(keys[6], (r, d)) * 0.01,
        "b_read": jnp.ones(r),
        "w_out": random.normal(keys[7], (d, d)) * 0.01,
        "b_out": jnp.zeros(d),
        "w_skip": random.normal(keys[8], (d, d)) * 0.01,
        "b_skip": jnp.ones(d),
        "ln_g": jnp.ones(r),
        "ln_b": jnp.zeros(r),
        "proj_w": random.normal(keys[9], (d, d + r)) * 0.02,
        "proj_b": jnp.zeros(d),
        "pre_g": jnp.ones(d),
        "pre_b": jnp.zeros(d),
    }


@jax.checkpoint
def modus_m_layer_fwd(layer: dict, x_seq: jax.Array) -> jax.Array:
    r = layer["wk"].shape[0]

    def step(h, e_raw):
        e = layer_norm(e_raw, layer["pre_g"], layer["pre_b"])
        k = normalize(layer["wk"] @ e)
        q = normalize(layer["wq"] @ e)
        val = jnp.tanh(layer["wv"] @ e)

        write = jax.nn.sigmoid((layer["w_write"] @ e + layer["b_write"])[0])
        eta = jax.nn.sigmoid((layer["w_delta"] @ e + layer["b_delta"])[0])
        retain = jax.nn.sigmoid((layer["w_retain"] @ e + layer["b_retain"])[0])

        old = h @ k
        h = retain * h + (eta * write) * jnp.outer(val - old, k)

        read_gate = jax.nn.sigmoid(layer["w_read"] @ e + layer["b_read"])
        context = read_gate * layer_norm(h @ q, layer["ln_g"], layer["ln_b"])
        proposal = layer["proj_w"] @ jnp.concatenate([e_raw, context]) + layer["proj_b"]

        out_gate = jax.nn.sigmoid(layer["w_out"] @ e + layer["b_out"])
        skip_gate = jax.nn.sigmoid(layer["w_skip"] @ e + layer["b_skip"])
        out = skip_gate * e_raw + out_gate * proposal
        return h, out

    _, out = lax.scan(step, jnp.zeros((r, r)), x_seq)
    return out


def init_modus_m_lm(key: jax.Array, cfg: ModelConfig) -> dict:
    keys = random.split(key, cfg.n_layers + 2)
    layers = [init_modus_m_layer(keys[i], cfg) for i in range(cfg.n_layers)]
    return {
        "embed": init_embed(keys[-2], cfg),
        "layers": jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *layers),
        "head": init_lm_head(keys[-1], cfg.embed_dim, cfg),
    }


def modus_m_lm_fwd(p: dict, x_ids: jax.Array, cfg: ModelConfig) -> jax.Array:
    x = p["embed"][x_ids]

    def scan_layer(x_in, layer):
        return x_in + modus_m_layer_fwd(layer, x_in), None

    x, _ = lax.scan(scan_layer, x, p["layers"])
    return lm_head_fwd(p["head"], x)


# ---------------------------------------------------------------------------
# Modus_M2: leaner selective matrix memory, no internal residual skip
# ---------------------------------------------------------------------------


def init_modus_m2_layer(key: jax.Array, cfg: ModelConfig) -> dict:
    layer = init_modus_m_layer(key, cfg)
    return {
        **layer,
        "b_retain": jnp.ones(1) * 2.0,
        "b_write": jnp.ones(1) * 1.5,
    }


@jax.checkpoint
def modus_m2_layer_fwd(layer: dict, x_seq: jax.Array) -> jax.Array:
    r = layer["wk"].shape[0]

    def step(h, e_raw):
        e = layer_norm(e_raw, layer["pre_g"], layer["pre_b"])
        k = normalize(layer["wk"] @ e)
        q = normalize(layer["wq"] @ e)
        val = jnp.tanh(layer["wv"] @ e)

        write = jax.nn.sigmoid((layer["w_write"] @ e + layer["b_write"])[0])
        eta = jax.nn.sigmoid((layer["w_delta"] @ e + layer["b_delta"])[0])
        retain = jax.nn.sigmoid((layer["w_retain"] @ e + layer["b_retain"])[0])

        old = h @ k
        h = retain * h + (eta * write) * jnp.outer(val - old, k)

        read_gate = jax.nn.sigmoid(layer["w_read"] @ e + layer["b_read"])
        context = read_gate * layer_norm(h @ q, layer["ln_g"], layer["ln_b"])
        proposal = layer["proj_w"] @ jnp.concatenate([e_raw, context]) + layer["proj_b"]

        out_gate = jax.nn.sigmoid(layer["w_out"] @ e + layer["b_out"])
        out = out_gate * proposal
        return h, out

    _, out = lax.scan(step, jnp.zeros((r, r)), x_seq)
    return out


def init_modus_m2_lm(key: jax.Array, cfg: ModelConfig) -> dict:
    keys = random.split(key, cfg.n_layers + 2)
    layers = [init_modus_m2_layer(keys[i], cfg) for i in range(cfg.n_layers)]
    return {
        "embed": init_embed(keys[-2], cfg),
        "layers": jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *layers),
        "head": init_lm_head(keys[-1], cfg.embed_dim, cfg),
    }


def modus_m2_lm_fwd(p: dict, x_ids: jax.Array, cfg: ModelConfig) -> jax.Array:
    x = p["embed"][x_ids]

    def scan_layer(x_in, layer):
        return x_in + modus_m2_layer_fwd(layer, x_in), None

    x, _ = lax.scan(scan_layer, x, p["layers"])
    return lm_head_fwd(p["head"], x)


# ---------------------------------------------------------------------------
# Modus_M3: selective matrix memory with row-wise Mamba retention
# ---------------------------------------------------------------------------


def init_modus_m3_layer(key: jax.Array, cfg: ModelConfig) -> dict:
    layer = init_modus_m_layer(key, cfg)
    k_retain, = random.split(key, 1)
    return {
        **layer,
        "w_retain": random.normal(k_retain, (cfg.ax_res, cfg.embed_dim)) * 0.01,
        "b_retain": jnp.ones(cfg.ax_res) * 2.5,
    }


@jax.checkpoint
def modus_m3_layer_fwd(layer: dict, x_seq: jax.Array) -> jax.Array:
    r = layer["wk"].shape[0]

    def step(h, e_raw):
        e = layer_norm(e_raw, layer["pre_g"], layer["pre_b"])
        k = normalize(layer["wk"] @ e)
        q = normalize(layer["wq"] @ e)
        val = jnp.tanh(layer["wv"] @ e)

        write = jax.nn.sigmoid((layer["w_write"] @ e + layer["b_write"])[0])
        eta = jax.nn.sigmoid((layer["w_delta"] @ e + layer["b_delta"])[0])
        retain = jax.nn.sigmoid(layer["w_retain"] @ e + layer["b_retain"])

        old = h @ k
        h = retain[:, None] * h + (eta * write) * jnp.outer(val - old, k)

        read_gate = jax.nn.sigmoid(layer["w_read"] @ e + layer["b_read"])
        context = read_gate * layer_norm(h @ q, layer["ln_g"], layer["ln_b"])
        proposal = layer["proj_w"] @ jnp.concatenate([e_raw, context]) + layer["proj_b"]

        out_gate = jax.nn.sigmoid(layer["w_out"] @ e + layer["b_out"])
        skip_gate = jax.nn.sigmoid(layer["w_skip"] @ e + layer["b_skip"])
        out = skip_gate * e_raw + out_gate * proposal
        return h, out

    _, out = lax.scan(step, jnp.zeros((r, r)), x_seq)
    return out


def init_modus_m3_lm(key: jax.Array, cfg: ModelConfig) -> dict:
    keys = random.split(key, cfg.n_layers + 2)
    layers = [init_modus_m3_layer(keys[i], cfg) for i in range(cfg.n_layers)]
    return {
        "embed": init_embed(keys[-2], cfg),
        "layers": jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *layers),
        "head": init_lm_head(keys[-1], cfg.embed_dim, cfg),
    }


def modus_m3_lm_fwd(p: dict, x_ids: jax.Array, cfg: ModelConfig) -> jax.Array:
    x = p["embed"][x_ids]

    def scan_layer(x_in, layer):
        return x_in + modus_m3_layer_fwd(layer, x_in), None

    x, _ = lax.scan(scan_layer, x, p["layers"])
    return lm_head_fwd(p["head"], x)


# ---------------------------------------------------------------------------
# Mamba-ish baseline: selective vector state, no attention
# ---------------------------------------------------------------------------


def init_mamba_layer(key: jax.Array, cfg: ModelConfig) -> dict:
    keys = random.split(key, 10)
    d, n = cfg.embed_dim, cfg.mamba_state_dim
    return {
        "pre_g": jnp.ones(d),
        "pre_b": jnp.zeros(d),
        "wu": random.normal(keys[0], (n, d)) * 0.02,
        "w_delta": random.normal(keys[1], (n, d)) * 0.01,
        "b_delta": jnp.zeros(n),
        "w_retain": random.normal(keys[2], (n, d)) * 0.01,
        "b_retain": jnp.ones(n) * 2.0,
        "w_c": random.normal(keys[3], (n, d)) * 0.02,
        "w_gate": random.normal(keys[4], (d, d)) * 0.01,
        "b_gate": jnp.ones(d),
        "proj_w": random.normal(keys[5], (d, n)) * 0.02,
        "proj_b": jnp.zeros(d),
    }


@jax.checkpoint
def mamba_layer_fwd(layer: dict, x_seq: jax.Array) -> jax.Array:
    n = layer["wu"].shape[0]

    def step(state, e_raw):
        e = layer_norm(e_raw, layer["pre_g"], layer["pre_b"])
        u = jnp.tanh(layer["wu"] @ e)
        delta = jax.nn.sigmoid(layer["w_delta"] @ e + layer["b_delta"])
        retain = jax.nn.sigmoid(layer["w_retain"] @ e + layer["b_retain"])
        state = retain * state + delta * u
        c = jax.nn.sigmoid(layer["w_c"] @ e)
        y = c * state
        gate = jax.nn.sigmoid(layer["w_gate"] @ e + layer["b_gate"])
        out = gate * (layer["proj_w"] @ y + layer["proj_b"])
        return state, out

    _, out = lax.scan(step, jnp.zeros(n), x_seq)
    return out


def init_mamba_lm(key: jax.Array, cfg: ModelConfig) -> dict:
    keys = random.split(key, cfg.n_layers + 2)
    layers = [init_mamba_layer(keys[i], cfg) for i in range(cfg.n_layers)]
    return {
        "embed": init_embed(keys[-2], cfg),
        "layers": jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *layers),
        "head": init_lm_head(keys[-1], cfg.embed_dim, cfg),
    }


def mamba_lm_fwd(p: dict, x_ids: jax.Array, cfg: ModelConfig) -> jax.Array:
    x = p["embed"][x_ids]

    def scan_layer(x_in, layer):
        return x_in + mamba_layer_fwd(layer, x_in), None

    x, _ = lax.scan(scan_layer, x, p["layers"])
    return lm_head_fwd(p["head"], x)


# ---------------------------------------------------------------------------
# Transformer baseline
# ---------------------------------------------------------------------------


def init_transformer_layer(key: jax.Array, cfg: ModelConfig) -> dict:
    keys = random.split(key, 8)
    d = cfg.embed_dim
    return {
        "wq": random.normal(keys[0], (d, d)) * 0.02,
        "wk": random.normal(keys[1], (d, d)) * 0.02,
        "wv": random.normal(keys[2], (d, d)) * 0.02,
        "wo": random.normal(keys[3], (d, d)) * 0.02,
        "mlp_w1": random.normal(keys[4], (4 * d, d)) * 0.02,
        "mlp_b1": jnp.zeros(4 * d),
        "mlp_w2": random.normal(keys[5], (d, 4 * d)) * 0.02,
        "mlp_b2": jnp.zeros(d),
        "ln1_g": jnp.ones(d),
        "ln1_b": jnp.zeros(d),
        "ln2_g": jnp.ones(d),
        "ln2_b": jnp.zeros(d),
    }


def transformer_layer_fwd(layer: dict, x: jax.Array, cfg: ModelConfig) -> jax.Array:
    t, d = x.shape
    h = cfg.n_heads_attn
    dk = d // h
    xn = layer_norm(x, layer["ln1_g"], layer["ln1_b"])
    q = (xn @ layer["wq"].T).reshape(t, h, dk)
    k = (xn @ layer["wk"].T).reshape(t, h, dk)
    v = (xn @ layer["wv"].T).reshape(t, h, dk)
    scores = jnp.einsum("thd,shd->hts", q, k) / jnp.sqrt(dk)
    mask = jnp.tril(jnp.ones((t, t), dtype=bool))
    attn = jax.nn.softmax(jnp.where(mask[None], scores, -1e9), axis=-1)
    y = jnp.einsum("hts,shd->thd", attn, v).reshape(t, d)
    x = x + y @ layer["wo"].T
    xn = layer_norm(x, layer["ln2_g"], layer["ln2_b"])
    mlp = jax.nn.gelu(xn @ layer["mlp_w1"].T + layer["mlp_b1"])
    x = x + mlp @ layer["mlp_w2"].T + layer["mlp_b2"]
    return x


def init_transformer_lm(key: jax.Array, cfg: ModelConfig) -> dict:
    keys = random.split(key, cfg.n_layers + 3)
    layers = [init_transformer_layer(keys[i], cfg) for i in range(cfg.n_layers)]
    return {
        "embed": init_embed(keys[-3], cfg),
        "pos": random.normal(keys[-2], (cfg.seq_len, cfg.embed_dim)) * 0.02,
        "layers": jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *layers),
        "head": init_lm_head(keys[-1], cfg.embed_dim, cfg),
    }


def transformer_lm_fwd(p: dict, x_ids: jax.Array, cfg: ModelConfig) -> jax.Array:
    x = p["embed"][x_ids] + p["pos"][: x_ids.shape[0]]

    def scan_layer(x_in, layer):
        return transformer_layer_fwd(layer, x_in, cfg), None

    x, _ = lax.scan(scan_layer, x, p["layers"])
    return lm_head_fwd(p["head"], x)


# ---------------------------------------------------------------------------
# Modus_X: Dual-stream hybrid — Modus matrix memory || Mamba vector state
# ---------------------------------------------------------------------------
# Architecture:
#   At every token t, two independent streams process the same input e_t:
#
#   [Mamba stream]  fast local dynamics — exponential decay vector state
#     s_t = retain_s * s_{t-1} + delta_s * u_t
#     mamba_out = gate_s * (Wp_s @ (c_s * s_t))
#
#   [Modus stream]  slow content memory — delta-rule matrix state
#     H_t = retain_h * H_{t-1} + (eta*write) * outer(v - H_{t-1}k, k)
#     modus_out = out_gate * proj([e; read_gate * LN(H_t @ q)])
#
#   [Router]  input-dependent soft gate (learned, initialized 50/50)
#     r_t = sigmoid(Wr @ e_t + b_r)
#     y_t = r_t * modus_out + (1 - r_t) * mamba_out
# ---------------------------------------------------------------------------


def init_modus_x_layer(key: jax.Array, cfg: ModelConfig) -> dict:
    keys = random.split(key, 24)
    d, r, n = cfg.embed_dim, cfg.ax_res, cfg.mamba_state_dim
    router_hidden = cfg.router_hidden or d
    return {
        # ── shared pre-norm ─────────────────────────────────────────────
        "pre_g": jnp.ones(d),
        "pre_b": jnp.zeros(d),

        # ── Modus matrix stream ─────────────────────────────────────────
        "m_wk":      random.normal(keys[0], (r, d)) * 0.02,
        "m_wq":      random.normal(keys[1], (r, d)) * 0.02,
        "m_wv":      random.normal(keys[2], (r, d)) * 0.02,
        # write-rate gate (eta)
        "m_w_eta":   random.normal(keys[3], (1, d)) * 0.01,
        "m_b_eta":   jnp.zeros(1),
        # content-write gate
        "m_w_write": random.normal(keys[4], (1, d)) * 0.01,
        "m_b_write": jnp.ones(1) * 1.0,
        # matrix retention gate (scalar)
        "m_w_ret":   random.normal(keys[5], (1, d)) * 0.01,
        "m_b_ret":   jnp.ones(1) * 3.0,
        # read gate (vector, r-dim)
        "m_w_read":  random.normal(keys[6], (r, d)) * 0.01,
        "m_b_read":  jnp.ones(r),
        # output gate
        "m_w_out":   random.normal(keys[7], (d, d)) * 0.01,
        "m_b_out":   jnp.zeros(d),
        # projection [e; context] -> d
        "m_proj_w":  random.normal(keys[8], (d, d + r)) * 0.02,
        "m_proj_b":  jnp.zeros(d),
        "m_ln_g":    jnp.ones(r),
        "m_ln_b":    jnp.zeros(r),
        # Used only by Modus_X_CurrentArchive. The standard Modus_X forward
        # path ignores these parameters.
        "m_w_archive_write": random.normal(keys[20], (1, d)) * 0.01,
        "m_b_archive_write": jnp.ones(1) * -1.0,
        "m_w_archive_ret":   random.normal(keys[21], (1, d)) * 0.01,
        "m_b_archive_ret":   jnp.ones(1) * 4.0,
        "m_w_archive_mix":   random.normal(keys[22], (r, d)) * 0.01,
        "m_b_archive_mix":   jnp.zeros(r),

        # ── Mamba vector stream ─────────────────────────────────────────
        "s_wu":      random.normal(keys[9],  (n, d)) * 0.02,
        "s_w_delta": random.normal(keys[10], (n, d)) * 0.01,
        "s_b_delta": jnp.zeros(n),
        "s_w_ret":   random.normal(keys[11], (n, d)) * 0.01,
        "s_b_ret":   jnp.ones(n) * 2.0,
        "s_w_c":     random.normal(keys[12], (n, d)) * 0.02,
        "s_w_gate":  random.normal(keys[13], (d, d)) * 0.01,
        "s_b_gate":  jnp.ones(d),
        "s_proj_w":  random.normal(keys[14], (d, n)) * 0.02,
        "s_proj_b":  jnp.zeros(d),

        # ── Router: r_t = sigmoid(Wr @ e + b_r) ────────────────────────
        # initialized to 0 bias => 0.5 mix at start
        "r_w":       random.normal(keys[15], (router_hidden, d)) * 0.01,
        "r_b":       jnp.zeros(router_hidden),
        "r_proj":    random.normal(keys[16], (d if getattr(cfg, "vector_router", False) else 1, router_hidden)) * 0.01,
        "r_proj_b":  jnp.zeros(d if getattr(cfg, "vector_router", False) else 1),
    }


@jax.checkpoint
def modus_x_layer_fwd_stateful(
    layer: dict,
    x_seq: jax.Array,
    initial_state: tuple[jax.Array, jax.Array] | None = None,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
    r = layer["m_wk"].shape[0]

    def step(carry, e_raw):
        H, s = carry
        e = layer_norm(e_raw, layer["pre_g"], layer["pre_b"])

        # ── Modus matrix stream ─────────────────────────────────────────
        k = normalize(layer["m_wk"] @ e)
        q = normalize(layer["m_wq"] @ e)
        val = jnp.tanh(layer["m_wv"] @ e)

        eta    = jax.nn.sigmoid((layer["m_w_eta"]   @ e + layer["m_b_eta"])[0])
        write  = jax.nn.sigmoid((layer["m_w_write"] @ e + layer["m_b_write"])[0])
        retain = jax.nn.sigmoid((layer["m_w_ret"]   @ e + layer["m_b_ret"])[0])

        old = H @ k
        H = retain * H + (eta * write) * jnp.outer(val - old, k)

        read_gate = jax.nn.sigmoid(layer["m_w_read"] @ e + layer["m_b_read"])
        context   = read_gate * layer_norm(H @ q, layer["m_ln_g"], layer["m_ln_b"])
        proposal  = layer["m_proj_w"] @ jnp.concatenate([e_raw, context]) + layer["m_proj_b"]
        out_gate  = jax.nn.sigmoid(layer["m_w_out"] @ e + layer["m_b_out"])
        modus_out = out_gate * proposal

        # ── Mamba vector stream ─────────────────────────────────────────
        u     = jnp.tanh(layer["s_wu"] @ e)
        delta = jax.nn.sigmoid(layer["s_w_delta"] @ e + layer["s_b_delta"])
        ret_s = jax.nn.sigmoid(layer["s_w_ret"]   @ e + layer["s_b_ret"])
        s     = ret_s * s + delta * u
        c     = jax.nn.sigmoid(layer["s_w_c"] @ e)
        y_s   = c * s
        gate_s   = jax.nn.sigmoid(layer["s_w_gate"] @ e + layer["s_b_gate"])
        mamba_out = gate_s * (layer["s_proj_w"] @ y_s + layer["s_proj_b"])

        # ── Router ──────────────────────────────────────────────────────
        r_hidden = jax.nn.gelu(layer["r_w"] @ e + layer["r_b"])
        r_logits = layer["r_proj"] @ r_hidden + layer["r_proj_b"]
        router   = jax.nn.sigmoid(r_logits[0] if layer["r_proj"].shape[0] == 1 else r_logits)
        out = router * modus_out + (1.0 - router) * mamba_out

        return (H, s), out

    if initial_state is None:
        initial_state = (jnp.zeros((r, r)), jnp.zeros(layer["s_wu"].shape[0]))
    return lax.scan(step, initial_state, x_seq)


def modus_x_layer_fwd(layer: dict, x_seq: jax.Array) -> jax.Array:
    _, out = modus_x_layer_fwd_stateful(layer, x_seq)
    return out


@jax.checkpoint
def modus_x_current_archive_layer_fwd_stateful(
    layer: dict,
    x_seq: jax.Array,
    initial_state: tuple[jax.Array, jax.Array, jax.Array] | None = None,
) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
    r = layer["m_wk"].shape[0]

    def step(carry, e_raw):
        H_current, H_archive, s = carry
        e = layer_norm(e_raw, layer["pre_g"], layer["pre_b"])

        k = normalize(layer["m_wk"] @ e)
        q = normalize(layer["m_wq"] @ e)
        val = jnp.tanh(layer["m_wv"] @ e)

        eta = jax.nn.sigmoid((layer["m_w_eta"] @ e + layer["m_b_eta"])[0])
        write = jax.nn.sigmoid((layer["m_w_write"] @ e + layer["m_b_write"])[0])
        retain = jax.nn.sigmoid((layer["m_w_ret"] @ e + layer["m_b_ret"])[0])

        old_current = H_current @ k
        H_current = retain * H_current + (eta * write) * jnp.outer(val - old_current, k)

        archive_write = jax.nn.sigmoid((layer["m_w_archive_write"] @ e + layer["m_b_archive_write"])[0])
        archive_retain = jax.nn.sigmoid((layer["m_w_archive_ret"] @ e + layer["m_b_archive_ret"])[0])
        old_archive = H_archive @ k
        H_archive = archive_retain * H_archive + (eta * write * archive_write) * jnp.outer(val - old_archive, k)

        read_gate = jax.nn.sigmoid(layer["m_w_read"] @ e + layer["m_b_read"])
        current_context = layer_norm(H_current @ q, layer["m_ln_g"], layer["m_ln_b"])
        archive_context = layer_norm(H_archive @ q, layer["m_ln_g"], layer["m_ln_b"])
        archive_mix = jax.nn.sigmoid(layer["m_w_archive_mix"] @ e + layer["m_b_archive_mix"])
        context = read_gate * (archive_mix * current_context + (1.0 - archive_mix) * archive_context)
        proposal = layer["m_proj_w"] @ jnp.concatenate([e_raw, context]) + layer["m_proj_b"]
        out_gate = jax.nn.sigmoid(layer["m_w_out"] @ e + layer["m_b_out"])
        modus_out = out_gate * proposal

        u = jnp.tanh(layer["s_wu"] @ e)
        delta = jax.nn.sigmoid(layer["s_w_delta"] @ e + layer["s_b_delta"])
        ret_s = jax.nn.sigmoid(layer["s_w_ret"] @ e + layer["s_b_ret"])
        s = ret_s * s + delta * u
        c = jax.nn.sigmoid(layer["s_w_c"] @ e)
        y_s = c * s
        gate_s = jax.nn.sigmoid(layer["s_w_gate"] @ e + layer["s_b_gate"])
        mamba_out = gate_s * (layer["s_proj_w"] @ y_s + layer["s_proj_b"])

        r_hidden = jax.nn.gelu(layer["r_w"] @ e + layer["r_b"])
        r_logits = layer["r_proj"] @ r_hidden + layer["r_proj_b"]
        router = jax.nn.sigmoid(r_logits[0] if layer["r_proj"].shape[0] == 1 else r_logits)
        out = router * modus_out + (1.0 - router) * mamba_out

        return (H_current, H_archive, s), out

    if initial_state is None:
        initial_state = (
            jnp.zeros((r, r)),
            jnp.zeros((r, r)),
            jnp.zeros(layer["s_wu"].shape[0]),
        )
    return lax.scan(step, initial_state, x_seq)


def modus_x_current_archive_layer_fwd(layer: dict, x_seq: jax.Array) -> jax.Array:
    _, out = modus_x_current_archive_layer_fwd_stateful(layer, x_seq)
    return out


@jax.checkpoint
def modus_x_memory_feedback_archive_layer_fwd_stateful(
    layer: dict,
    x_seq: jax.Array,
    initial_state: tuple[jax.Array, jax.Array, jax.Array] | None = None,
) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
    """CurrentArchive with low-rank matrix-read feedback into recurrence."""
    r = layer["m_wk"].shape[0]

    def step(carry, e_raw):
        H_current, H_archive, s = carry
        e = layer_norm(e_raw, layer["pre_g"], layer["pre_b"])

        k = normalize(layer["m_wk"] @ e)
        q = normalize(layer["m_wq"] @ e)
        val = jnp.tanh(layer["m_wv"] @ e)
        eta = jax.nn.sigmoid((layer["m_w_eta"] @ e + layer["m_b_eta"])[0])
        write = jax.nn.sigmoid((layer["m_w_write"] @ e + layer["m_b_write"])[0])
        retain = jax.nn.sigmoid((layer["m_w_ret"] @ e + layer["m_b_ret"])[0])
        old_current = H_current @ k
        H_current = retain * H_current + (eta * write) * jnp.outer(val - old_current, k)

        archive_write = jax.nn.sigmoid(
            (layer["m_w_archive_write"] @ e + layer["m_b_archive_write"])[0]
        )
        archive_retain = jax.nn.sigmoid(
            (layer["m_w_archive_ret"] @ e + layer["m_b_archive_ret"])[0]
        )
        old_archive = H_archive @ k
        H_archive = archive_retain * H_archive + (eta * write * archive_write) * jnp.outer(
            val - old_archive, k
        )

        read_gate = jax.nn.sigmoid(layer["m_w_read"] @ e + layer["m_b_read"])
        current_context = layer_norm(H_current @ q, layer["m_ln_g"], layer["m_ln_b"])
        archive_context = layer_norm(H_archive @ q, layer["m_ln_g"], layer["m_ln_b"])
        archive_mix = jax.nn.sigmoid(layer["m_w_archive_mix"] @ e + layer["m_b_archive_mix"])
        context = read_gate * (
            archive_mix * current_context + (1.0 - archive_mix) * archive_context
        )
        proposal = layer["m_proj_w"] @ jnp.concatenate([e_raw, context]) + layer["m_proj_b"]
        out_gate = jax.nn.sigmoid(layer["m_w_out"] @ e + layer["m_b_out"])
        modus_out = out_gate * proposal

        feedback_gate = jax.nn.sigmoid(
            (layer["s_w_memory_feedback"] @ e + layer["s_b_memory_feedback"])[0]
        )
        compressed_memory = jnp.tanh(layer["s_memory_down"] @ context)
        memory_feedback = layer["s_memory_up"] @ compressed_memory
        e_vector = layer_norm(
            e_raw + feedback_gate * memory_feedback,
            layer["pre_g"],
            layer["pre_b"],
        )

        u = jnp.tanh(layer["s_wu"] @ e_vector)
        delta = jax.nn.sigmoid(layer["s_w_delta"] @ e_vector + layer["s_b_delta"])
        ret_s = jax.nn.sigmoid(layer["s_w_ret"] @ e_vector + layer["s_b_ret"])
        s = ret_s * s + delta * u
        c = jax.nn.sigmoid(layer["s_w_c"] @ e_vector)
        y_s = c * s
        gate_s = jax.nn.sigmoid(layer["s_w_gate"] @ e_vector + layer["s_b_gate"])
        mamba_out = gate_s * (layer["s_proj_w"] @ y_s + layer["s_proj_b"])

        r_hidden = jax.nn.gelu(layer["r_w"] @ e + layer["r_b"])
        r_logits = layer["r_proj"] @ r_hidden + layer["r_proj_b"]
        router = jax.nn.sigmoid(
            r_logits[0] if layer["r_proj"].shape[0] == 1 else r_logits
        )
        out = router * modus_out + (1.0 - router) * mamba_out
        return (H_current, H_archive, s), out

    if initial_state is None:
        initial_state = (
            jnp.zeros((r, r)),
            jnp.zeros((r, r)),
            jnp.zeros(layer["s_wu"].shape[0]),
        )
    return lax.scan(step, initial_state, x_seq)


def modus_x_memory_feedback_archive_layer_fwd(layer: dict, x_seq: jax.Array) -> jax.Array:
    _, out = modus_x_memory_feedback_archive_layer_fwd_stateful(layer, x_seq)
    return out


def modus_x_memory_feedback_archive_layer_diagnostics(
    layer: dict, x_seq: jax.Array
) -> dict[str, jax.Array]:
    r = layer["m_wk"].shape[0]

    def step(carry, e_raw):
        H_current, H_archive = carry
        e = layer_norm(e_raw, layer["pre_g"], layer["pre_b"])
        k = normalize(layer["m_wk"] @ e)
        q = normalize(layer["m_wq"] @ e)
        val = jnp.tanh(layer["m_wv"] @ e)
        eta = jax.nn.sigmoid((layer["m_w_eta"] @ e + layer["m_b_eta"])[0])
        write = jax.nn.sigmoid((layer["m_w_write"] @ e + layer["m_b_write"])[0])
        retain = jax.nn.sigmoid((layer["m_w_ret"] @ e + layer["m_b_ret"])[0])
        old_current = H_current @ k
        H_current = retain * H_current + (eta * write) * jnp.outer(val - old_current, k)
        archive_write = jax.nn.sigmoid(
            (layer["m_w_archive_write"] @ e + layer["m_b_archive_write"])[0]
        )
        archive_retain = jax.nn.sigmoid(
            (layer["m_w_archive_ret"] @ e + layer["m_b_archive_ret"])[0]
        )
        old_archive = H_archive @ k
        H_archive = archive_retain * H_archive + (eta * write * archive_write) * jnp.outer(
            val - old_archive, k
        )
        read_gate = jax.nn.sigmoid(layer["m_w_read"] @ e + layer["m_b_read"])
        current_context = layer_norm(H_current @ q, layer["m_ln_g"], layer["m_ln_b"])
        archive_context = layer_norm(H_archive @ q, layer["m_ln_g"], layer["m_ln_b"])
        archive_mix = jax.nn.sigmoid(layer["m_w_archive_mix"] @ e + layer["m_b_archive_mix"])
        context = read_gate * (
            archive_mix * current_context + (1.0 - archive_mix) * archive_context
        )
        feedback_gate = jax.nn.sigmoid(
            (layer["s_w_memory_feedback"] @ e + layer["s_b_memory_feedback"])[0]
        )
        memory_feedback = layer["s_memory_up"] @ jnp.tanh(
            layer["s_memory_down"] @ context
        )
        diagnostics = jnp.stack(
            [
                feedback_gate,
                jnp.linalg.norm(memory_feedback),
                jnp.linalg.norm(context),
                jnp.linalg.norm(feedback_gate * memory_feedback)
                / (jnp.linalg.norm(e_raw) + 1e-6),
                jnp.mean(archive_mix),
            ]
        )
        return (H_current, H_archive), diagnostics

    (_, _), values = lax.scan(
        step,
        (jnp.zeros((r, r)), jnp.zeros((r, r))),
        x_seq,
    )
    names = (
        "feedback_gate",
        "feedback_norm",
        "context_norm",
        "feedback_to_input_ratio",
        "current_mix",
    )
    return {name: values[:, index] for index, name in enumerate(names)}


def _adaptive_write_direction(
    key: jax.Array,
    usage: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Bounded diagonal inverse-frequency preconditioning for a normalized key."""
    mean_usage = jnp.mean(usage)
    scale = jnp.clip(mean_usage / (usage + 1e-6), 0.25, 4.0)
    direction = key * scale
    direction = direction / (jnp.dot(key, direction) + 1e-6)
    return direction, scale


@jax.checkpoint
def modus_x_adaptive_preconditioned_archive_layer_fwd_stateful(
    layer: dict,
    x_seq: jax.Array,
    initial_state: tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
    ] | None = None,
):
    """CurrentArchive with bounded online key-frequency preconditioning.

    The model parameters and read path are seed-paired with CurrentArchive.
    Two O(ax_res) usage traces adapt matrix write directions so frequently
    occupied key dimensions receive smaller updates and underused dimensions
    receive larger updates. The recurrent state remains fixed with sequence
    length and no MemoryFeedback parameters or path are present.
    """
    r = layer["m_wk"].shape[0]
    usage_decay = jnp.asarray(0.99, dtype=x_seq.dtype)

    def step(carry, e_raw):
        H_current, H_archive, s, current_usage, archive_usage = carry
        e = layer_norm(e_raw, layer["pre_g"], layer["pre_b"])

        k = normalize(layer["m_wk"] @ e)
        q = normalize(layer["m_wq"] @ e)
        val = jnp.tanh(layer["m_wv"] @ e)
        eta = jax.nn.sigmoid((layer["m_w_eta"] @ e + layer["m_b_eta"])[0])
        write = jax.nn.sigmoid((layer["m_w_write"] @ e + layer["m_b_write"])[0])
        retain = jax.nn.sigmoid((layer["m_w_ret"] @ e + layer["m_b_ret"])[0])

        current_usage = current_usage + (1.0 - usage_decay) * write * (
            jnp.square(k) - current_usage
        )
        current_direction, _ = _adaptive_write_direction(k, current_usage)
        old_current = H_current @ k
        H_current = retain * H_current + (eta * write) * jnp.outer(
            val - old_current,
            current_direction,
        )

        archive_write = jax.nn.sigmoid(
            (layer["m_w_archive_write"] @ e + layer["m_b_archive_write"])[0]
        )
        archive_retain = jax.nn.sigmoid(
            (layer["m_w_archive_ret"] @ e + layer["m_b_archive_ret"])[0]
        )
        archive_strength = write * archive_write
        archive_usage = archive_usage + (1.0 - usage_decay) * archive_strength * (
            jnp.square(k) - archive_usage
        )
        archive_direction, _ = _adaptive_write_direction(k, archive_usage)
        old_archive = H_archive @ k
        H_archive = archive_retain * H_archive + (eta * archive_strength) * jnp.outer(
            val - old_archive,
            archive_direction,
        )

        read_gate = jax.nn.sigmoid(layer["m_w_read"] @ e + layer["m_b_read"])
        current_context = layer_norm(H_current @ q, layer["m_ln_g"], layer["m_ln_b"])
        archive_context = layer_norm(H_archive @ q, layer["m_ln_g"], layer["m_ln_b"])
        archive_mix = jax.nn.sigmoid(layer["m_w_archive_mix"] @ e + layer["m_b_archive_mix"])
        context = read_gate * (
            archive_mix * current_context + (1.0 - archive_mix) * archive_context
        )
        proposal = layer["m_proj_w"] @ jnp.concatenate([e_raw, context]) + layer["m_proj_b"]
        out_gate = jax.nn.sigmoid(layer["m_w_out"] @ e + layer["m_b_out"])
        modus_out = out_gate * proposal

        u = jnp.tanh(layer["s_wu"] @ e)
        delta = jax.nn.sigmoid(layer["s_w_delta"] @ e + layer["s_b_delta"])
        ret_s = jax.nn.sigmoid(layer["s_w_ret"] @ e + layer["s_b_ret"])
        s = ret_s * s + delta * u
        c = jax.nn.sigmoid(layer["s_w_c"] @ e)
        y_s = c * s
        gate_s = jax.nn.sigmoid(layer["s_w_gate"] @ e + layer["s_b_gate"])
        mamba_out = gate_s * (layer["s_proj_w"] @ y_s + layer["s_proj_b"])

        r_hidden = jax.nn.gelu(layer["r_w"] @ e + layer["r_b"])
        r_logits = layer["r_proj"] @ r_hidden + layer["r_proj_b"]
        router = jax.nn.sigmoid(
            r_logits[0] if layer["r_proj"].shape[0] == 1 else r_logits
        )
        out = router * modus_out + (1.0 - router) * mamba_out
        return (
            H_current,
            H_archive,
            s,
            current_usage,
            archive_usage,
        ), out

    if initial_state is None:
        initial_usage = jnp.ones(r, dtype=x_seq.dtype) / r
        initial_state = (
            jnp.zeros((r, r), dtype=x_seq.dtype),
            jnp.zeros((r, r), dtype=x_seq.dtype),
            jnp.zeros(layer["s_wu"].shape[0], dtype=x_seq.dtype),
            initial_usage,
            initial_usage,
        )
    return lax.scan(step, initial_state, x_seq)


def modus_x_adaptive_preconditioned_archive_layer_fwd(
    layer: dict,
    x_seq: jax.Array,
) -> jax.Array:
    _, out = modus_x_adaptive_preconditioned_archive_layer_fwd_stateful(layer, x_seq)
    return out


def modus_x_adaptive_preconditioned_archive_layer_diagnostics(
    layer: dict,
    x_seq: jax.Array,
) -> dict[str, jax.Array]:
    r = layer["m_wk"].shape[0]
    usage_decay = jnp.asarray(0.99, dtype=x_seq.dtype)

    def step(carry, e_raw):
        H_current, H_archive, current_usage, archive_usage = carry
        e = layer_norm(e_raw, layer["pre_g"], layer["pre_b"])
        k = normalize(layer["m_wk"] @ e)
        val = jnp.tanh(layer["m_wv"] @ e)
        eta = jax.nn.sigmoid((layer["m_w_eta"] @ e + layer["m_b_eta"])[0])
        write = jax.nn.sigmoid((layer["m_w_write"] @ e + layer["m_b_write"])[0])
        retain = jax.nn.sigmoid((layer["m_w_ret"] @ e + layer["m_b_ret"])[0])
        current_usage = current_usage + (1.0 - usage_decay) * write * (
            jnp.square(k) - current_usage
        )
        current_direction, current_scale = _adaptive_write_direction(k, current_usage)
        old_current = H_current @ k
        current_residual = val - old_current
        H_current = retain * H_current + (eta * write) * jnp.outer(
            current_residual,
            current_direction,
        )

        archive_write = jax.nn.sigmoid(
            (layer["m_w_archive_write"] @ e + layer["m_b_archive_write"])[0]
        )
        archive_retain = jax.nn.sigmoid(
            (layer["m_w_archive_ret"] @ e + layer["m_b_archive_ret"])[0]
        )
        archive_strength = write * archive_write
        archive_usage = archive_usage + (1.0 - usage_decay) * archive_strength * (
            jnp.square(k) - archive_usage
        )
        archive_direction, archive_scale = _adaptive_write_direction(k, archive_usage)
        old_archive = H_archive @ k
        H_archive = archive_retain * H_archive + (eta * archive_strength) * jnp.outer(
            val - old_archive,
            archive_direction,
        )
        current_cv = jnp.std(current_usage) / (jnp.mean(current_usage) + 1e-6)
        archive_cv = jnp.std(archive_usage) / (jnp.mean(archive_usage) + 1e-6)
        diagnostics = jnp.stack(
            [
                write,
                archive_strength,
                jnp.linalg.norm(current_direction),
                jnp.linalg.norm(archive_direction),
                current_cv,
                archive_cv,
                jnp.max(current_scale) / (jnp.min(current_scale) + 1e-6),
                jnp.max(archive_scale) / (jnp.min(archive_scale) + 1e-6),
                jnp.sqrt(jnp.mean(jnp.square(current_residual)) + 1e-6),
            ]
        )
        return (H_current, H_archive, current_usage, archive_usage), diagnostics

    initial_usage = jnp.ones(r, dtype=x_seq.dtype) / r
    (_, _, _, _), values = lax.scan(
        step,
        (
            jnp.zeros((r, r), dtype=x_seq.dtype),
            jnp.zeros((r, r), dtype=x_seq.dtype),
            initial_usage,
            initial_usage,
        ),
        x_seq,
    )
    names = (
        "write",
        "archive_write_strength",
        "current_update_direction_norm",
        "archive_update_direction_norm",
        "current_usage_cv",
        "archive_usage_cv",
        "current_preconditioner_spread",
        "archive_preconditioner_spread",
        "current_residual_rms",
    )
    return {name: values[:, index] for index, name in enumerate(names)}


ATTENTION_TO_WRITE_WINDOW = 64
ATTENTION_TO_WRITE_HEADS = 4
ATTENTION_TO_WRITE_WIDTH = 64
ATTENTION_TO_WRITE_LAYER_STRIDE = 4


def init_attention_to_write_module(key: jax.Array, cfg: ModelConfig) -> dict:
    if ATTENTION_TO_WRITE_WIDTH % ATTENTION_TO_WRITE_HEADS:
        raise ValueError("Attention width must divide the number of heads")
    keys = random.split(key, 5)
    d_head = ATTENTION_TO_WRITE_WIDTH // ATTENTION_TO_WRITE_HEADS
    return {
        "wq": random.normal(
            keys[0],
            (ATTENTION_TO_WRITE_HEADS, cfg.embed_dim, d_head),
        ) * 0.02,
        "wk": random.normal(
            keys[1],
            (ATTENTION_TO_WRITE_HEADS, cfg.embed_dim, d_head),
        ) * 0.02,
        "wv": random.normal(
            keys[2],
            (ATTENTION_TO_WRITE_HEADS, cfg.embed_dim, d_head),
        ) * 0.02,
        "to_value": random.normal(
            keys[3],
            (cfg.ax_res, ATTENTION_TO_WRITE_WIDTH),
        ) * 0.01,
        "to_write": random.normal(keys[4], (1, ATTENTION_TO_WRITE_WIDTH)) * 0.01,
        "b_write": jnp.zeros(1),
    }


def bounded_causal_attention_to_write_context(
    attention: dict,
    layer: dict,
    x_seq: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return fixed-window causal attention context and attention weights."""
    normalized = jax.vmap(
        lambda x: layer_norm(x, layer["pre_g"], layer["pre_b"])
    )(x_seq)
    query = jnp.einsum("td,hdk->htk", normalized, attention["wq"])
    key = jnp.einsum("td,hdk->htk", normalized, attention["wk"])
    value = jnp.einsum("td,hdk->htk", normalized, attention["wv"])
    scores = jnp.einsum("htd,hsd->hts", query, key) / jnp.sqrt(
        float(ATTENTION_TO_WRITE_WIDTH // ATTENTION_TO_WRITE_HEADS)
    )
    seq_len = x_seq.shape[0]
    positions = jnp.arange(seq_len)
    causal = positions[None, :] <= positions[:, None]
    local = positions[None, :] >= (
        positions[:, None] - ATTENTION_TO_WRITE_WINDOW + 1
    )
    scores = jnp.where((causal & local)[None, :, :], scores, -1e9)
    weights = jax.nn.softmax(scores, axis=-1)
    attended = jnp.einsum("hts,hsd->htd", weights, value)
    context = jnp.transpose(attended, (1, 0, 2)).reshape(
        seq_len,
        ATTENTION_TO_WRITE_WIDTH,
    )
    return context, weights


@jax.checkpoint
def modus_x_attention_to_write_archive_layer_fwd_stateful(
    layer: dict,
    attention: dict,
    x_seq: jax.Array,
    attention_context: jax.Array,
    initial_state: tuple[jax.Array, jax.Array, jax.Array] | None = None,
):
    """CurrentArchive whose matrix writes receive bounded local context."""
    r = layer["m_wk"].shape[0]

    def step(carry, inputs):
        H_current, H_archive, s = carry
        e_raw, local_context = inputs
        e = layer_norm(e_raw, layer["pre_g"], layer["pre_b"])
        k = normalize(layer["m_wk"] @ e)
        q = normalize(layer["m_wq"] @ e)
        base_value = layer["m_wv"] @ e
        # Fixed 0.25 injection prevents the write controller from becoming a
        # post-read expert or silently collapsing to zero.
        value_adjustment = attention["to_value"] @ local_context
        val = jnp.tanh(base_value + 0.25 * value_adjustment)
        eta = jax.nn.sigmoid((layer["m_w_eta"] @ e + layer["m_b_eta"])[0])
        write_adjustment = (attention["to_write"] @ local_context + attention["b_write"])[0]
        write = jax.nn.sigmoid(
            (layer["m_w_write"] @ e + layer["m_b_write"])[0]
            + 0.25 * write_adjustment
        )
        retain = jax.nn.sigmoid((layer["m_w_ret"] @ e + layer["m_b_ret"])[0])
        old_current = H_current @ k
        H_current = retain * H_current + (eta * write) * jnp.outer(
            val - old_current,
            k,
        )

        archive_write = jax.nn.sigmoid(
            (layer["m_w_archive_write"] @ e + layer["m_b_archive_write"])[0]
        )
        archive_retain = jax.nn.sigmoid(
            (layer["m_w_archive_ret"] @ e + layer["m_b_archive_ret"])[0]
        )
        old_archive = H_archive @ k
        H_archive = archive_retain * H_archive + (
            eta * write * archive_write
        ) * jnp.outer(val - old_archive, k)

        read_gate = jax.nn.sigmoid(layer["m_w_read"] @ e + layer["m_b_read"])
        current_context = layer_norm(H_current @ q, layer["m_ln_g"], layer["m_ln_b"])
        archive_context = layer_norm(H_archive @ q, layer["m_ln_g"], layer["m_ln_b"])
        archive_mix = jax.nn.sigmoid(layer["m_w_archive_mix"] @ e + layer["m_b_archive_mix"])
        context = read_gate * (
            archive_mix * current_context + (1.0 - archive_mix) * archive_context
        )
        proposal = layer["m_proj_w"] @ jnp.concatenate([e_raw, context]) + layer["m_proj_b"]
        out_gate = jax.nn.sigmoid(layer["m_w_out"] @ e + layer["m_b_out"])
        modus_out = out_gate * proposal

        u = jnp.tanh(layer["s_wu"] @ e)
        delta = jax.nn.sigmoid(layer["s_w_delta"] @ e + layer["s_b_delta"])
        ret_s = jax.nn.sigmoid(layer["s_w_ret"] @ e + layer["s_b_ret"])
        s = ret_s * s + delta * u
        c = jax.nn.sigmoid(layer["s_w_c"] @ e)
        y_s = c * s
        gate_s = jax.nn.sigmoid(layer["s_w_gate"] @ e + layer["s_b_gate"])
        mamba_out = gate_s * (layer["s_proj_w"] @ y_s + layer["s_proj_b"])
        r_hidden = jax.nn.gelu(layer["r_w"] @ e + layer["r_b"])
        r_logits = layer["r_proj"] @ r_hidden + layer["r_proj_b"]
        router = jax.nn.sigmoid(
            r_logits[0] if layer["r_proj"].shape[0] == 1 else r_logits
        )
        out = router * modus_out + (1.0 - router) * mamba_out
        return (H_current, H_archive, s), out

    if initial_state is None:
        initial_state = (
            jnp.zeros((r, r), dtype=x_seq.dtype),
            jnp.zeros((r, r), dtype=x_seq.dtype),
            jnp.zeros(layer["s_wu"].shape[0], dtype=x_seq.dtype),
        )
    return lax.scan(step, initial_state, (x_seq, attention_context))


def modus_x_attention_to_write_archive_layer_fwd(
    layer: dict,
    attention: dict,
    x_seq: jax.Array,
) -> jax.Array:
    attention_context, _ = bounded_causal_attention_to_write_context(
        attention,
        layer,
        x_seq,
    )
    _, out = modus_x_attention_to_write_archive_layer_fwd_stateful(
        layer,
        attention,
        x_seq,
        attention_context,
    )
    return out


@jax.checkpoint
def modus_x_feedback_attention_to_write_archive_layer_fwd_stateful(
    layer: dict,
    attention: dict,
    x_seq: jax.Array,
    attention_context: jax.Array,
    initial_state: tuple[jax.Array, jax.Array, jax.Array] | None = None,
):
    """MemoryFeedbackArchive with bounded attention applied only to writes."""
    r = layer["m_wk"].shape[0]

    def step(carry, inputs):
        H_current, H_archive, s = carry
        e_raw, local_context = inputs
        e = layer_norm(e_raw, layer["pre_g"], layer["pre_b"])
        k = normalize(layer["m_wk"] @ e)
        q = normalize(layer["m_wq"] @ e)
        value_adjustment = attention["to_value"] @ local_context
        val = jnp.tanh(layer["m_wv"] @ e + 0.25 * value_adjustment)
        eta = jax.nn.sigmoid((layer["m_w_eta"] @ e + layer["m_b_eta"])[0])
        write_adjustment = (
            attention["to_write"] @ local_context + attention["b_write"]
        )[0]
        write = jax.nn.sigmoid(
            (layer["m_w_write"] @ e + layer["m_b_write"])[0]
            + 0.25 * write_adjustment
        )
        retain = jax.nn.sigmoid((layer["m_w_ret"] @ e + layer["m_b_ret"])[0])
        old_current = H_current @ k
        H_current = retain * H_current + (eta * write) * jnp.outer(
            val - old_current,
            k,
        )
        archive_write = jax.nn.sigmoid(
            (layer["m_w_archive_write"] @ e + layer["m_b_archive_write"])[0]
        )
        archive_retain = jax.nn.sigmoid(
            (layer["m_w_archive_ret"] @ e + layer["m_b_archive_ret"])[0]
        )
        old_archive = H_archive @ k
        H_archive = archive_retain * H_archive + (
            eta * write * archive_write
        ) * jnp.outer(val - old_archive, k)

        read_gate = jax.nn.sigmoid(layer["m_w_read"] @ e + layer["m_b_read"])
        current_context = layer_norm(H_current @ q, layer["m_ln_g"], layer["m_ln_b"])
        archive_context = layer_norm(H_archive @ q, layer["m_ln_g"], layer["m_ln_b"])
        archive_mix = jax.nn.sigmoid(layer["m_w_archive_mix"] @ e + layer["m_b_archive_mix"])
        context = read_gate * (
            archive_mix * current_context + (1.0 - archive_mix) * archive_context
        )
        proposal = layer["m_proj_w"] @ jnp.concatenate([e_raw, context]) + layer["m_proj_b"]
        out_gate = jax.nn.sigmoid(layer["m_w_out"] @ e + layer["m_b_out"])
        modus_out = out_gate * proposal

        feedback_gate = jax.nn.sigmoid(
            (layer["s_w_memory_feedback"] @ e + layer["s_b_memory_feedback"])[0]
        )
        memory_feedback = layer["s_memory_up"] @ jnp.tanh(
            layer["s_memory_down"] @ context
        )
        e_vector = layer_norm(
            e_raw + feedback_gate * memory_feedback,
            layer["pre_g"],
            layer["pre_b"],
        )
        u = jnp.tanh(layer["s_wu"] @ e_vector)
        delta = jax.nn.sigmoid(layer["s_w_delta"] @ e_vector + layer["s_b_delta"])
        ret_s = jax.nn.sigmoid(layer["s_w_ret"] @ e_vector + layer["s_b_ret"])
        s = ret_s * s + delta * u
        c = jax.nn.sigmoid(layer["s_w_c"] @ e_vector)
        y_s = c * s
        gate_s = jax.nn.sigmoid(
            layer["s_w_gate"] @ e_vector + layer["s_b_gate"]
        )
        mamba_out = gate_s * (layer["s_proj_w"] @ y_s + layer["s_proj_b"])
        r_hidden = jax.nn.gelu(layer["r_w"] @ e + layer["r_b"])
        r_logits = layer["r_proj"] @ r_hidden + layer["r_proj_b"]
        router = jax.nn.sigmoid(
            r_logits[0] if layer["r_proj"].shape[0] == 1 else r_logits
        )
        out = router * modus_out + (1.0 - router) * mamba_out
        return (H_current, H_archive, s), out

    if initial_state is None:
        initial_state = (
            jnp.zeros((r, r), dtype=x_seq.dtype),
            jnp.zeros((r, r), dtype=x_seq.dtype),
            jnp.zeros(layer["s_wu"].shape[0], dtype=x_seq.dtype),
        )
    return lax.scan(step, initial_state, (x_seq, attention_context))


def modus_x_feedback_attention_to_write_archive_layer_fwd(
    layer: dict,
    attention: dict,
    x_seq: jax.Array,
) -> jax.Array:
    attention_context, _ = bounded_causal_attention_to_write_context(
        attention,
        layer,
        x_seq,
    )
    _, out = modus_x_feedback_attention_to_write_archive_layer_fwd_stateful(
        layer,
        attention,
        x_seq,
        attention_context,
    )
    return out


@jax.checkpoint
def modus_x_displaced_archive_layer_fwd_stateful(
    layer: dict,
    x_seq: jax.Array,
    initial_state: tuple[jax.Array, jax.Array, jax.Array] | None = None,
) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
    """Current/archive memory that transfers displaced current values.

    Unlike CurrentArchive, the archive is not given a slower copy of every new
    value. A learned conflict gate observes the token and memory residual, then
    writes the value retrieved from current memory into the archive before the
    current matrix receives the replacement.
    """
    r = layer["m_wk"].shape[0]

    def step(carry, e_raw):
        H_current, H_archive, s = carry
        e = layer_norm(e_raw, layer["pre_g"], layer["pre_b"])

        k = normalize(layer["m_wk"] @ e)
        q = normalize(layer["m_wq"] @ e)
        val = jnp.tanh(layer["m_wv"] @ e)

        eta = jax.nn.sigmoid((layer["m_w_eta"] @ e + layer["m_b_eta"])[0])
        write = jax.nn.sigmoid((layer["m_w_write"] @ e + layer["m_b_write"])[0])
        retain = jax.nn.sigmoid((layer["m_w_ret"] @ e + layer["m_b_ret"])[0])

        old_current = H_current @ k
        old_archive = H_archive @ k
        residual = val - old_current
        current_rms = jnp.sqrt(jnp.mean(jnp.square(old_current)) + 1e-6)
        residual_rms = jnp.sqrt(jnp.mean(jnp.square(residual)) + 1e-6)
        # Epsilon must be inside sqrt. norm(x) has an undefined gradient at
        # x=0 even when epsilon is added to the final denominator.
        val_norm = jnp.sqrt(jnp.sum(jnp.square(val)) + 1e-6)
        current_norm = jnp.sqrt(jnp.sum(jnp.square(old_current)) + 1e-6)
        alignment = jnp.sum(val * old_current) / (val_norm * current_norm)
        archive_disagreement = jnp.sqrt(
            jnp.mean(jnp.square(old_current - old_archive)) + 1e-6
        )
        conflict_stats = jnp.stack(
            [current_rms, residual_rms, alignment, archive_disagreement]
        )
        conflict = jax.nn.sigmoid(
            (layer["m_w_archive_conflict"] @ e)[0]
            + (layer["m_w_archive_conflict_stats"] @ conflict_stats)[0]
            + layer["m_b_archive_conflict"][0]
        )

        archive_retain = jax.nn.sigmoid(
            (layer["m_w_archive_ret"] @ e + layer["m_b_archive_ret"])[0]
        )
        transfer = eta * write * conflict
        H_archive = archive_retain * H_archive + transfer * jnp.outer(
            old_current - old_archive, k
        )
        H_current = retain * H_current + (eta * write) * jnp.outer(residual, k)

        read_gate = jax.nn.sigmoid(layer["m_w_read"] @ e + layer["m_b_read"])
        current_context = layer_norm(H_current @ q, layer["m_ln_g"], layer["m_ln_b"])
        archive_context = layer_norm(H_archive @ q, layer["m_ln_g"], layer["m_ln_b"])
        archive_mix = jax.nn.sigmoid(layer["m_w_archive_mix"] @ e + layer["m_b_archive_mix"])
        context = read_gate * (
            archive_mix * current_context + (1.0 - archive_mix) * archive_context
        )
        proposal = layer["m_proj_w"] @ jnp.concatenate([e_raw, context]) + layer["m_proj_b"]
        out_gate = jax.nn.sigmoid(layer["m_w_out"] @ e + layer["m_b_out"])
        modus_out = out_gate * proposal

        u = jnp.tanh(layer["s_wu"] @ e)
        delta = jax.nn.sigmoid(layer["s_w_delta"] @ e + layer["s_b_delta"])
        ret_s = jax.nn.sigmoid(layer["s_w_ret"] @ e + layer["s_b_ret"])
        s = ret_s * s + delta * u
        c = jax.nn.sigmoid(layer["s_w_c"] @ e)
        y_s = c * s
        gate_s = jax.nn.sigmoid(layer["s_w_gate"] @ e + layer["s_b_gate"])
        mamba_out = gate_s * (layer["s_proj_w"] @ y_s + layer["s_proj_b"])

        r_hidden = jax.nn.gelu(layer["r_w"] @ e + layer["r_b"])
        r_logits = layer["r_proj"] @ r_hidden + layer["r_proj_b"]
        router = jax.nn.sigmoid(
            r_logits[0] if layer["r_proj"].shape[0] == 1 else r_logits
        )
        out = router * modus_out + (1.0 - router) * mamba_out

        return (H_current, H_archive, s), out

    if initial_state is None:
        initial_state = (
            jnp.zeros((r, r)),
            jnp.zeros((r, r)),
            jnp.zeros(layer["s_wu"].shape[0]),
        )
    return lax.scan(step, initial_state, x_seq)


def modus_x_displaced_archive_layer_fwd(layer: dict, x_seq: jax.Array) -> jax.Array:
    _, out = modus_x_displaced_archive_layer_fwd_stateful(layer, x_seq)
    return out


def modus_x_displaced_archive_layer_diagnostics(
    layer: dict, x_seq: jax.Array
) -> dict[str, jax.Array]:
    """Replay one layer's memory path and expose operation statistics."""
    r = layer["m_wk"].shape[0]

    def step(carry, e_raw):
        H_current, H_archive = carry
        e = layer_norm(e_raw, layer["pre_g"], layer["pre_b"])
        k = normalize(layer["m_wk"] @ e)
        val = jnp.tanh(layer["m_wv"] @ e)
        eta = jax.nn.sigmoid((layer["m_w_eta"] @ e + layer["m_b_eta"])[0])
        write = jax.nn.sigmoid((layer["m_w_write"] @ e + layer["m_b_write"])[0])
        retain = jax.nn.sigmoid((layer["m_w_ret"] @ e + layer["m_b_ret"])[0])
        archive_retain = jax.nn.sigmoid(
            (layer["m_w_archive_ret"] @ e + layer["m_b_archive_ret"])[0]
        )
        old_current = H_current @ k
        old_archive = H_archive @ k
        residual = val - old_current
        current_rms = jnp.sqrt(jnp.mean(jnp.square(old_current)) + 1e-6)
        residual_rms = jnp.sqrt(jnp.mean(jnp.square(residual)) + 1e-6)
        val_norm = jnp.sqrt(jnp.sum(jnp.square(val)) + 1e-6)
        current_norm = jnp.sqrt(jnp.sum(jnp.square(old_current)) + 1e-6)
        alignment = jnp.sum(val * old_current) / (val_norm * current_norm)
        archive_disagreement = jnp.sqrt(
            jnp.mean(jnp.square(old_current - old_archive)) + 1e-6
        )
        stats = jnp.stack([current_rms, residual_rms, alignment, archive_disagreement])
        conflict = jax.nn.sigmoid(
            (layer["m_w_archive_conflict"] @ e)[0]
            + (layer["m_w_archive_conflict_stats"] @ stats)[0]
            + layer["m_b_archive_conflict"][0]
        )
        transfer = eta * write * conflict
        H_archive = archive_retain * H_archive + transfer * jnp.outer(
            old_current - old_archive, k
        )
        H_current = retain * H_current + (eta * write) * jnp.outer(residual, k)
        archive_mix = jax.nn.sigmoid(layer["m_w_archive_mix"] @ e + layer["m_b_archive_mix"])
        diagnostics = jnp.stack(
            [
                conflict,
                transfer,
                jnp.mean(archive_mix),
                current_rms,
                residual_rms,
                archive_disagreement,
                retain,
                archive_retain,
            ]
        )
        return (H_current, H_archive), diagnostics

    (_, _), values = lax.scan(
        step,
        (jnp.zeros((r, r)), jnp.zeros((r, r))),
        x_seq,
    )
    names = (
        "conflict",
        "transfer",
        "current_mix",
        "current_rms",
        "residual_rms",
        "archive_disagreement",
        "current_retention",
        "archive_retention",
    )
    return {name: values[:, index] for index, name in enumerate(names)}


def init_modus_x_lm(key: jax.Array, cfg: ModelConfig) -> dict:
    keys = random.split(key, cfg.n_layers + 2)
    layers = [init_modus_x_layer(keys[i], cfg) for i in range(cfg.n_layers)]
    return {
        "embed":  init_embed(keys[-2], cfg),
        "layers": jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *layers),
        "head":   init_lm_head(keys[-1], cfg.embed_dim, cfg),
    }


def init_modus_x_displaced_archive_lm(key: jax.Array, cfg: ModelConfig) -> dict:
    """Initialize a seed-paired CurrentArchive extension.

    Existing parameters use exactly the same layer keys as CurrentArchive.
    New conflict-controller keys are derived with fold_in so adding them does
    not perturb the initialization of the control model.
    """
    keys = random.split(key, cfg.n_layers + 2)
    layers = []
    for layer_index in range(cfg.n_layers):
        layer_key = keys[layer_index]
        layer = init_modus_x_layer(layer_key, cfg)
        conflict_keys = random.split(random.fold_in(layer_key, 0xD15A), 2)
        layer.update(
            {
                "m_w_archive_conflict": random.normal(
                    conflict_keys[0], (1, cfg.embed_dim)
                )
                * 0.01,
                "m_w_archive_conflict_stats": random.normal(
                    conflict_keys[1], (1, 4)
                )
                * 0.01,
                "m_b_archive_conflict": jnp.ones(1) * -2.0,
            }
        )
        # The archive begins mostly empty. Favor current memory initially and
        # let training increase historical recall when it becomes useful.
        layer["m_b_archive_mix"] = jnp.ones_like(layer["m_b_archive_mix"]) * 2.0
        layers.append(layer)
    return {
        "embed": init_embed(keys[-2], cfg),
        "layers": jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *layers),
        "head": init_lm_head(keys[-1], cfg.embed_dim, cfg),
    }


def init_modus_x_memory_feedback_archive_lm(key: jax.Array, cfg: ModelConfig) -> dict:
    """Seed-paired CurrentArchive plus a rank-32 matrix-to-vector bridge."""
    keys = random.split(key, cfg.n_layers + 2)
    feedback_rank = min(32, cfg.ax_res, cfg.mamba_state_dim)
    layers = []
    for layer_index in range(cfg.n_layers):
        layer_key = keys[layer_index]
        layer = init_modus_x_layer(layer_key, cfg)
        feedback_keys = random.split(random.fold_in(layer_key, 0xF33D), 3)
        layer.update(
            {
                "s_memory_down": random.normal(
                    feedback_keys[0], (feedback_rank, cfg.ax_res)
                )
                * 0.02,
                "s_memory_up": random.normal(
                    feedback_keys[1], (cfg.embed_dim, feedback_rank)
                )
                * 0.02,
                "s_w_memory_feedback": random.normal(
                    feedback_keys[2], (1, cfg.embed_dim)
                )
                * 0.01,
                "s_b_memory_feedback": jnp.ones(1) * -2.0,
            }
        )
        layers.append(layer)
    return {
        "embed": init_embed(keys[-2], cfg),
        "layers": jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *layers),
        "head": init_lm_head(keys[-1], cfg.embed_dim, cfg),
    }


def init_modus_x_attention_to_write_archive_lm(
    key: jax.Array,
    cfg: ModelConfig,
) -> dict:
    """Seed-paired CurrentArchive plus three bounded write controllers."""
    params = init_modus_x_lm(key, cfg)
    module_count = cfg.n_layers // ATTENTION_TO_WRITE_LAYER_STRIDE
    attention_keys = random.split(random.fold_in(key, 0xA77E), module_count)
    modules = [
        init_attention_to_write_module(attention_key, cfg)
        for attention_key in attention_keys
    ]
    params["attention_to_write"] = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs),
        *modules,
    )
    return params


def init_modus_x_feedback_attention_to_write_archive_lm(
    key: jax.Array,
    cfg: ModelConfig,
) -> dict:
    """Exact MemoryFeedback initialization plus bounded write controllers."""
    params = init_modus_x_memory_feedback_archive_lm(key, cfg)
    module_count = cfg.n_layers // ATTENTION_TO_WRITE_LAYER_STRIDE
    attention_keys = random.split(random.fold_in(key, 0xA77E), module_count)
    modules = [
        init_attention_to_write_module(attention_key, cfg)
        for attention_key in attention_keys
    ]
    params["attention_to_write"] = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs),
        *modules,
    )
    return params


def add_future_heads(params: dict, key: jax.Array, cfg: ModelConfig, count: int) -> dict:
    if count <= 0:
        return params
    keys = random.split(key, count)
    heads = [init_lm_head(keys[i], cfg.embed_dim, cfg) for i in range(count)]
    return {
        **params,
        "future_heads": jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *heads),
    }


def modus_x_lm_fwd(p: dict, x_ids: jax.Array, cfg: ModelConfig) -> jax.Array:
    x = p["embed"][x_ids]

    def scan_layer(x_in, layer):
        return x_in + modus_x_layer_fwd(layer, x_in), None

    x, _ = lax.scan(scan_layer, x, p["layers"])
    return lm_head_fwd(p["head"], x)


def modus_x_current_archive_lm_fwd(p: dict, x_ids: jax.Array, cfg: ModelConfig) -> jax.Array:
    x = p["embed"][x_ids]

    def scan_layer(x_in, layer):
        return x_in + modus_x_current_archive_layer_fwd(layer, x_in), None

    x, _ = lax.scan(scan_layer, x, p["layers"])
    return lm_head_fwd(p["head"], x)


def modus_x_memory_feedback_archive_lm_fwd(
    p: dict, x_ids: jax.Array, cfg: ModelConfig
) -> jax.Array:
    x = p["embed"][x_ids]

    def scan_layer(x_in, layer):
        return x_in + modus_x_memory_feedback_archive_layer_fwd(layer, x_in), None

    x, _ = lax.scan(scan_layer, x, p["layers"])
    return lm_head_fwd(p["head"], x)


def modus_x_adaptive_preconditioned_archive_lm_fwd(
    p: dict,
    x_ids: jax.Array,
    cfg: ModelConfig,
) -> jax.Array:
    x = p["embed"][x_ids]

    def scan_layer(x_in, layer):
        layer_out = modus_x_adaptive_preconditioned_archive_layer_fwd(layer, x_in)
        return x_in + layer_out, None

    x, _ = lax.scan(scan_layer, x, p["layers"])
    return lm_head_fwd(p["head"], x)


def modus_x_attention_to_write_archive_lm_fwd(
    p: dict,
    x_ids: jax.Array,
    cfg: ModelConfig,
) -> jax.Array:
    x = p["embed"][x_ids]

    def scan_layer(x_in, inputs):
        layer_index, layer = inputs
        attention_slot = layer_index // ATTENTION_TO_WRITE_LAYER_STRIDE
        attention = jax.tree_util.tree_map(
            lambda value: value[attention_slot],
            p["attention_to_write"],
        )

        def attention_layer(_):
            return x_in + modus_x_attention_to_write_archive_layer_fwd(
                layer,
                attention,
                x_in,
            )

        def recurrent_layer(_):
            return x_in + modus_x_current_archive_layer_fwd(layer, x_in)

        uses_attention = (
            (layer_index + 1) % ATTENTION_TO_WRITE_LAYER_STRIDE == 0
        )
        x_out = lax.cond(uses_attention, attention_layer, recurrent_layer, operand=None)
        return x_out, None

    x, _ = lax.scan(
        scan_layer,
        x,
        (jnp.arange(cfg.n_layers), p["layers"]),
    )
    return lm_head_fwd(p["head"], x)


def modus_x_feedback_attention_to_write_archive_lm_fwd(
    p: dict,
    x_ids: jax.Array,
    cfg: ModelConfig,
) -> jax.Array:
    x = p["embed"][x_ids]

    def scan_layer(x_in, inputs):
        layer_index, layer = inputs
        attention_slot = layer_index // ATTENTION_TO_WRITE_LAYER_STRIDE
        attention = jax.tree_util.tree_map(
            lambda value: value[attention_slot],
            p["attention_to_write"],
        )

        def combined_layer(_):
            return x_in + modus_x_feedback_attention_to_write_archive_layer_fwd(
                layer,
                attention,
                x_in,
            )

        def feedback_layer(_):
            return x_in + modus_x_memory_feedback_archive_layer_fwd(layer, x_in)

        uses_attention = (
            (layer_index + 1) % ATTENTION_TO_WRITE_LAYER_STRIDE == 0
        )
        x_out = lax.cond(uses_attention, combined_layer, feedback_layer, operand=None)
        return x_out, None

    x, _ = lax.scan(
        scan_layer,
        x,
        (jnp.arange(cfg.n_layers), p["layers"]),
    )
    return lm_head_fwd(p["head"], x)


def modus_x_attention_to_write_archive_diagnostics(
    p: dict,
    x_ids: jax.Array,
    cfg: ModelConfig,
) -> dict[str, jax.Array]:
    x = p["embed"][x_ids]
    positions = jnp.arange(x_ids.shape[0])

    def scan_layer(x_in, inputs):
        layer_index, layer = inputs
        attention_slot = layer_index // ATTENTION_TO_WRITE_LAYER_STRIDE
        attention = jax.tree_util.tree_map(
            lambda value: value[attention_slot],
            p["attention_to_write"],
        )

        def attention_layer(_):
            local_context, weights = bounded_causal_attention_to_write_context(
                attention,
                layer,
                x_in,
            )
            _, layer_out = modus_x_attention_to_write_archive_layer_fwd_stateful(
                layer,
                attention,
                x_in,
                local_context,
            )
            value_adjustment = jax.vmap(
                lambda context: attention["to_value"] @ context
            )(local_context)
            write_adjustment = jax.vmap(
                lambda context: (attention["to_write"] @ context)[0]
            )(local_context)
            entropy = -jnp.mean(
                jnp.sum(weights * jnp.log(weights + 1e-8), axis=-1),
                axis=0,
            )
            distance = positions[:, None] - positions[None, :]
            mean_distance = jnp.mean(
                jnp.sum(weights * distance[None, :, :], axis=-1),
                axis=0,
            )
            diagnostics = jnp.stack(
                [
                    entropy,
                    mean_distance,
                    jnp.linalg.norm(value_adjustment, axis=-1),
                    jnp.abs(write_adjustment),
                    jnp.linalg.norm(local_context, axis=-1),
                ],
                axis=-1,
            )
            return x_in + layer_out, diagnostics

        def recurrent_layer(_):
            layer_out = modus_x_current_archive_layer_fwd(layer, x_in)
            return x_in + layer_out, jnp.zeros(
                (x_in.shape[0], 5),
                dtype=x_in.dtype,
            )

        uses_attention = (
            (layer_index + 1) % ATTENTION_TO_WRITE_LAYER_STRIDE == 0
        )
        return lax.cond(uses_attention, attention_layer, recurrent_layer, operand=None)

    _, diagnostics = lax.scan(
        scan_layer,
        x,
        (jnp.arange(cfg.n_layers), p["layers"]),
    )
    diagnostics = diagnostics[ATTENTION_TO_WRITE_LAYER_STRIDE - 1 :: ATTENTION_TO_WRITE_LAYER_STRIDE]
    names = (
        "attention_entropy",
        "attention_mean_distance",
        "value_adjustment_norm",
        "write_adjustment_abs",
        "attention_context_norm",
    )
    return {name: diagnostics[:, :, index] for index, name in enumerate(names)}


def modus_x_adaptive_preconditioned_archive_diagnostics(
    p: dict,
    x_ids: jax.Array,
    cfg: ModelConfig,
) -> dict[str, jax.Array]:
    x = p["embed"][x_ids]

    def scan_layer(x_in, layer):
        diagnostics = modus_x_adaptive_preconditioned_archive_layer_diagnostics(
            layer,
            x_in,
        )
        layer_out = modus_x_adaptive_preconditioned_archive_layer_fwd(layer, x_in)
        return x_in + layer_out, diagnostics

    _, diagnostics = lax.scan(scan_layer, x, p["layers"])
    return diagnostics


def modus_x_memory_feedback_archive_diagnostics(
    p: dict, x_ids: jax.Array, cfg: ModelConfig
) -> dict[str, jax.Array]:
    x = p["embed"][x_ids]

    def scan_layer(x_in, layer):
        diagnostics = modus_x_memory_feedback_archive_layer_diagnostics(layer, x_in)
        x_out = x_in + modus_x_memory_feedback_archive_layer_fwd(layer, x_in)
        return x_out, diagnostics

    _, diagnostics = lax.scan(scan_layer, x, p["layers"])
    return diagnostics


def modus_x_displaced_archive_lm_fwd(p: dict, x_ids: jax.Array, cfg: ModelConfig) -> jax.Array:
    x = p["embed"][x_ids]

    def scan_layer(x_in, layer):
        return x_in + modus_x_displaced_archive_layer_fwd(layer, x_in), None

    x, _ = lax.scan(scan_layer, x, p["layers"])
    return lm_head_fwd(p["head"], x)


def modus_x_displaced_archive_diagnostics(
    p: dict, x_ids: jax.Array, cfg: ModelConfig
) -> dict[str, jax.Array]:
    """Return per-layer, per-token memory-operation diagnostics."""
    x = p["embed"][x_ids]

    def scan_layer(x_in, layer):
        diagnostics = modus_x_displaced_archive_layer_diagnostics(layer, x_in)
        x_out = x_in + modus_x_displaced_archive_layer_fwd(layer, x_in)
        return x_out, diagnostics

    _, diagnostics = lax.scan(scan_layer, x, p["layers"])
    return diagnostics


def modus_x_lm_fwd_deep_supervision(
    p: dict,
    x_ids: jax.Array,
    cfg: ModelConfig,
    auxiliary_layers: tuple[int, ...] | None = None,
    dropout_key: jax.Array | None = None,
    dropout_rate: float = 0.0,
):
    x = p["embed"][x_ids]

    def scan_layer(x_in, layer):
        x_out = x_in + modus_x_layer_fwd(layer, x_in)
        return x_out, x_out

    x, layer_outputs = lax.scan(scan_layer, x, p["layers"])
    if auxiliary_layers is None:
        auxiliary_layers = (cfg.n_layers // 2,)
    layer_indexes = jnp.array([layer - 1 for layer in auxiliary_layers], dtype=jnp.int32)
    selected_outputs = layer_outputs[layer_indexes]
    if dropout_key is not None and dropout_rate > 0.0:
        final_key, aux_key = random.split(dropout_key)
        x_for_head = apply_dropout(x, final_key, dropout_rate)
        selected_for_heads = apply_dropout(selected_outputs, aux_key, dropout_rate)
    else:
        x_for_head = x
        selected_for_heads = selected_outputs
    final_logits = lm_head_fwd(p["head"], x_for_head)
    auxiliary_logits = jax.vmap(lambda h: lm_head_fwd(p["head"], h))(selected_for_heads)
    if "future_heads" in p:
        future_logits = jax.vmap(lambda head: lm_head_fwd(head, x_for_head))(p["future_heads"])
        auxiliary_future_logits = jax.vmap(
            lambda head: jax.vmap(lambda h: lm_head_fwd(head, h))(selected_for_heads)
        )(p["future_heads"])
        return final_logits, auxiliary_logits, future_logits, auxiliary_future_logits
    return final_logits, auxiliary_logits


def modus_x_current_archive_lm_fwd_deep_supervision(
    p: dict,
    x_ids: jax.Array,
    cfg: ModelConfig,
    auxiliary_layers: tuple[int, ...] | None = None,
    dropout_key: jax.Array | None = None,
    dropout_rate: float = 0.0,
):
    x = p["embed"][x_ids]

    def scan_layer(x_in, layer):
        x_out = x_in + modus_x_current_archive_layer_fwd(layer, x_in)
        return x_out, x_out

    x, layer_outputs = lax.scan(scan_layer, x, p["layers"])
    if auxiliary_layers is None:
        auxiliary_layers = (cfg.n_layers // 2,)
    layer_indexes = jnp.array([layer - 1 for layer in auxiliary_layers], dtype=jnp.int32)
    selected_outputs = layer_outputs[layer_indexes]
    if dropout_key is not None and dropout_rate > 0.0:
        final_key, aux_key = random.split(dropout_key)
        x_for_head = apply_dropout(x, final_key, dropout_rate)
        selected_for_heads = apply_dropout(selected_outputs, aux_key, dropout_rate)
    else:
        x_for_head = x
        selected_for_heads = selected_outputs
    final_logits = lm_head_fwd(p["head"], x_for_head)
    auxiliary_logits = jax.vmap(lambda h: lm_head_fwd(p["head"], h))(selected_for_heads)
    if "future_heads" in p:
        future_logits = jax.vmap(lambda head: lm_head_fwd(head, x_for_head))(p["future_heads"])
        auxiliary_future_logits = jax.vmap(
            lambda head: jax.vmap(lambda h: lm_head_fwd(head, h))(selected_for_heads)
        )(p["future_heads"])
        return final_logits, auxiliary_logits, future_logits, auxiliary_future_logits
    return final_logits, auxiliary_logits


def modus_x_memory_feedback_archive_lm_fwd_deep_supervision(
    p: dict,
    x_ids: jax.Array,
    cfg: ModelConfig,
    auxiliary_layers: tuple[int, ...] | None = None,
    dropout_key: jax.Array | None = None,
    dropout_rate: float = 0.0,
):
    x = p["embed"][x_ids]

    def scan_layer(x_in, layer):
        x_out = x_in + modus_x_memory_feedback_archive_layer_fwd(layer, x_in)
        return x_out, x_out

    x, layer_outputs = lax.scan(scan_layer, x, p["layers"])
    if auxiliary_layers is None:
        auxiliary_layers = (cfg.n_layers // 2,)
    layer_indexes = jnp.array([layer - 1 for layer in auxiliary_layers], dtype=jnp.int32)
    selected_outputs = layer_outputs[layer_indexes]
    if dropout_key is not None and dropout_rate > 0.0:
        final_key, aux_key = random.split(dropout_key)
        x_for_head = apply_dropout(x, final_key, dropout_rate)
        selected_for_heads = apply_dropout(selected_outputs, aux_key, dropout_rate)
    else:
        x_for_head = x
        selected_for_heads = selected_outputs
    final_logits = lm_head_fwd(p["head"], x_for_head)
    auxiliary_logits = jax.vmap(lambda h: lm_head_fwd(p["head"], h))(selected_for_heads)
    if "future_heads" in p:
        future_logits = jax.vmap(lambda head: lm_head_fwd(head, x_for_head))(p["future_heads"])
        auxiliary_future_logits = jax.vmap(
            lambda head: jax.vmap(lambda h: lm_head_fwd(head, h))(selected_for_heads)
        )(p["future_heads"])
        return final_logits, auxiliary_logits, future_logits, auxiliary_future_logits
    return final_logits, auxiliary_logits


def modus_x_adaptive_preconditioned_archive_lm_fwd_deep_supervision(
    p: dict,
    x_ids: jax.Array,
    cfg: ModelConfig,
    auxiliary_layers: tuple[int, ...] | None = None,
    dropout_key: jax.Array | None = None,
    dropout_rate: float = 0.0,
):
    x = p["embed"][x_ids]

    def scan_layer(x_in, layer):
        layer_out = modus_x_adaptive_preconditioned_archive_layer_fwd(layer, x_in)
        x_out = x_in + layer_out
        return x_out, x_out

    x, layer_outputs = lax.scan(scan_layer, x, p["layers"])
    if auxiliary_layers is None:
        auxiliary_layers = (cfg.n_layers // 2,)
    layer_indexes = jnp.array([layer - 1 for layer in auxiliary_layers], dtype=jnp.int32)
    selected_outputs = layer_outputs[layer_indexes]
    if dropout_key is not None and dropout_rate > 0.0:
        final_key, aux_key = random.split(dropout_key)
        x_for_head = apply_dropout(x, final_key, dropout_rate)
        selected_for_heads = apply_dropout(selected_outputs, aux_key, dropout_rate)
    else:
        x_for_head = x
        selected_for_heads = selected_outputs
    final_logits = lm_head_fwd(p["head"], x_for_head)
    auxiliary_logits = jax.vmap(lambda h: lm_head_fwd(p["head"], h))(
        selected_for_heads
    )
    if "future_heads" in p:
        future_logits = jax.vmap(lambda head: lm_head_fwd(head, x_for_head))(
            p["future_heads"]
        )
        auxiliary_future_logits = jax.vmap(
            lambda head: jax.vmap(lambda h: lm_head_fwd(head, h))(
                selected_for_heads
            )
        )(p["future_heads"])
        return final_logits, auxiliary_logits, future_logits, auxiliary_future_logits
    return final_logits, auxiliary_logits


def modus_x_attention_to_write_archive_lm_fwd_deep_supervision(
    p: dict,
    x_ids: jax.Array,
    cfg: ModelConfig,
    auxiliary_layers: tuple[int, ...] | None = None,
    dropout_key: jax.Array | None = None,
    dropout_rate: float = 0.0,
):
    x = p["embed"][x_ids]

    def scan_layer(x_in, inputs):
        layer_index, layer = inputs
        attention_slot = layer_index // ATTENTION_TO_WRITE_LAYER_STRIDE
        attention = jax.tree_util.tree_map(
            lambda value: value[attention_slot],
            p["attention_to_write"],
        )

        def attention_layer(_):
            return x_in + modus_x_attention_to_write_archive_layer_fwd(
                layer,
                attention,
                x_in,
            )

        def recurrent_layer(_):
            return x_in + modus_x_current_archive_layer_fwd(layer, x_in)

        uses_attention = (
            (layer_index + 1) % ATTENTION_TO_WRITE_LAYER_STRIDE == 0
        )
        x_out = lax.cond(uses_attention, attention_layer, recurrent_layer, operand=None)
        return x_out, x_out

    x, layer_outputs = lax.scan(
        scan_layer,
        x,
        (jnp.arange(cfg.n_layers), p["layers"]),
    )
    if auxiliary_layers is None:
        auxiliary_layers = (cfg.n_layers // 2,)
    layer_indexes = jnp.array(
        [layer - 1 for layer in auxiliary_layers],
        dtype=jnp.int32,
    )
    selected_outputs = layer_outputs[layer_indexes]
    if dropout_key is not None and dropout_rate > 0.0:
        final_key, aux_key = random.split(dropout_key)
        x_for_head = apply_dropout(x, final_key, dropout_rate)
        selected_for_heads = apply_dropout(selected_outputs, aux_key, dropout_rate)
    else:
        x_for_head = x
        selected_for_heads = selected_outputs
    final_logits = lm_head_fwd(p["head"], x_for_head)
    auxiliary_logits = jax.vmap(lambda h: lm_head_fwd(p["head"], h))(
        selected_for_heads
    )
    if "future_heads" in p:
        future_logits = jax.vmap(lambda head: lm_head_fwd(head, x_for_head))(
            p["future_heads"]
        )
        auxiliary_future_logits = jax.vmap(
            lambda head: jax.vmap(lambda h: lm_head_fwd(head, h))(
                selected_for_heads
            )
        )(p["future_heads"])
        return final_logits, auxiliary_logits, future_logits, auxiliary_future_logits
    return final_logits, auxiliary_logits


def modus_x_feedback_attention_to_write_archive_lm_fwd_deep_supervision(
    p: dict,
    x_ids: jax.Array,
    cfg: ModelConfig,
    auxiliary_layers: tuple[int, ...] | None = None,
    dropout_key: jax.Array | None = None,
    dropout_rate: float = 0.0,
):
    x = p["embed"][x_ids]

    def scan_layer(x_in, inputs):
        layer_index, layer = inputs
        attention_slot = layer_index // ATTENTION_TO_WRITE_LAYER_STRIDE
        attention = jax.tree_util.tree_map(
            lambda value: value[attention_slot],
            p["attention_to_write"],
        )

        def combined_layer(_):
            return x_in + modus_x_feedback_attention_to_write_archive_layer_fwd(
                layer,
                attention,
                x_in,
            )

        def feedback_layer(_):
            return x_in + modus_x_memory_feedback_archive_layer_fwd(layer, x_in)

        uses_attention = (
            (layer_index + 1) % ATTENTION_TO_WRITE_LAYER_STRIDE == 0
        )
        x_out = lax.cond(uses_attention, combined_layer, feedback_layer, operand=None)
        return x_out, x_out

    x, layer_outputs = lax.scan(
        scan_layer,
        x,
        (jnp.arange(cfg.n_layers), p["layers"]),
    )
    if auxiliary_layers is None:
        auxiliary_layers = (cfg.n_layers // 2,)
    layer_indexes = jnp.array(
        [layer - 1 for layer in auxiliary_layers],
        dtype=jnp.int32,
    )
    selected_outputs = layer_outputs[layer_indexes]
    if dropout_key is not None and dropout_rate > 0.0:
        final_key, aux_key = random.split(dropout_key)
        x_for_head = apply_dropout(x, final_key, dropout_rate)
        selected_for_heads = apply_dropout(selected_outputs, aux_key, dropout_rate)
    else:
        x_for_head = x
        selected_for_heads = selected_outputs
    final_logits = lm_head_fwd(p["head"], x_for_head)
    auxiliary_logits = jax.vmap(lambda h: lm_head_fwd(p["head"], h))(
        selected_for_heads
    )
    if "future_heads" in p:
        future_logits = jax.vmap(lambda head: lm_head_fwd(head, x_for_head))(
            p["future_heads"]
        )
        auxiliary_future_logits = jax.vmap(
            lambda head: jax.vmap(lambda h: lm_head_fwd(head, h))(
                selected_for_heads
            )
        )(p["future_heads"])
        return final_logits, auxiliary_logits, future_logits, auxiliary_future_logits
    return final_logits, auxiliary_logits


def modus_x_displaced_archive_lm_fwd_deep_supervision(
    p: dict,
    x_ids: jax.Array,
    cfg: ModelConfig,
    auxiliary_layers: tuple[int, ...] | None = None,
    dropout_key: jax.Array | None = None,
    dropout_rate: float = 0.0,
):
    x = p["embed"][x_ids]

    def scan_layer(x_in, layer):
        x_out = x_in + modus_x_displaced_archive_layer_fwd(layer, x_in)
        return x_out, x_out

    x, layer_outputs = lax.scan(scan_layer, x, p["layers"])
    if auxiliary_layers is None:
        auxiliary_layers = (cfg.n_layers // 2,)
    layer_indexes = jnp.array([layer - 1 for layer in auxiliary_layers], dtype=jnp.int32)
    selected_outputs = layer_outputs[layer_indexes]
    if dropout_key is not None and dropout_rate > 0.0:
        final_key, aux_key = random.split(dropout_key)
        x_for_head = apply_dropout(x, final_key, dropout_rate)
        selected_for_heads = apply_dropout(selected_outputs, aux_key, dropout_rate)
    else:
        x_for_head = x
        selected_for_heads = selected_outputs
    final_logits = lm_head_fwd(p["head"], x_for_head)
    auxiliary_logits = jax.vmap(lambda h: lm_head_fwd(p["head"], h))(selected_for_heads)
    if "future_heads" in p:
        future_logits = jax.vmap(lambda head: lm_head_fwd(head, x_for_head))(p["future_heads"])
        auxiliary_future_logits = jax.vmap(
            lambda head: jax.vmap(lambda h: lm_head_fwd(head, h))(selected_for_heads)
        )(p["future_heads"])
        return final_logits, auxiliary_logits, future_logits, auxiliary_future_logits
    return final_logits, auxiliary_logits


def init_modus_x_state(cfg: ModelConfig) -> tuple[jax.Array, jax.Array]:
    return (
        jnp.zeros((cfg.n_layers, cfg.ax_res, cfg.ax_res)),
        jnp.zeros((cfg.n_layers, cfg.mamba_state_dim)),
    )


def modus_x_lm_fwd_stateful(
    p: dict,
    x_ids: jax.Array,
    cfg: ModelConfig,
    initial_state: tuple[jax.Array, jax.Array] | None = None,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    x = p["embed"][x_ids]
    if initial_state is None:
        initial_state = init_modus_x_state(cfg)

    def scan_layer(x_in, inputs):
        layer, H, s = inputs
        (next_H, next_s), layer_out = modus_x_layer_fwd_stateful(layer, x_in, (H, s))
        return x_in + layer_out, (next_H, next_s)

    x, final_state = lax.scan(scan_layer, x, (p["layers"], *initial_state))
    return lm_head_fwd(p["head"], x), final_state


MODEL_REGISTRY: dict[str, tuple[Callable, Callable]] = {
    "Modus": (init_modus_lm, modus_lm_fwd),
    "Modus_M": (init_modus_m_lm, modus_m_lm_fwd),
    "Modus_M2": (init_modus_m2_lm, modus_m2_lm_fwd),
    "Modus_M3": (init_modus_m3_lm, modus_m3_lm_fwd),
    "Modus_X": (init_modus_x_lm, modus_x_lm_fwd),
    "Modus_X_CurrentArchive": (init_modus_x_lm, modus_x_current_archive_lm_fwd),
    "Modus_X_MemoryFeedbackArchive": (
        init_modus_x_memory_feedback_archive_lm,
        modus_x_memory_feedback_archive_lm_fwd,
    ),
    "Modus_X_AdaptivePreconditionedArchive": (
        init_modus_x_lm,
        modus_x_adaptive_preconditioned_archive_lm_fwd,
    ),
    "Modus_X_AttentionToWriteArchive": (
        init_modus_x_attention_to_write_archive_lm,
        modus_x_attention_to_write_archive_lm_fwd,
    ),
    "Modus_X_FeedbackAttentionToWriteArchive": (
        init_modus_x_feedback_attention_to_write_archive_lm,
        modus_x_feedback_attention_to_write_archive_lm_fwd,
    ),
    "Modus_X_DisplacedArchive": (
        init_modus_x_displaced_archive_lm,
        modus_x_displaced_archive_lm_fwd,
    ),
    "Mamba": (init_mamba_lm, mamba_lm_fwd),
    "Transformer": (init_transformer_lm, transformer_lm_fwd),
}


def make_model(
    name: str,
    key: jax.Array,
    cfg: ModelConfig,
    auxiliary_layers: tuple[int, ...] | None = None,
    future_target_count: int = 0,
    dropout_rate: float = 0.0,
) -> tuple[dict, Callable]:
    if name == "Modus_X_Scalar":
        cfg = replace(cfg, vector_router=False)
        name = "Modus_X"
    elif name == "Modus_X_Vector":
        cfg = replace(cfg, vector_router=True)
        name = "Modus_X"
    elif name == "Modus_X_Scalar_PM":
        cfg = replace(
            cfg,
            vector_router=False,
            router_hidden=cfg.embed_dim,
        )
        name = "Modus_X"
    elif name == "Modus_X_Scalar_Lean":
        cfg = replace(
            cfg,
            vector_router=False,
            router_hidden=cfg.router_hidden or max(8, cfg.embed_dim // 16),
        )
        name = "Modus_X"
    elif name == "Modus_X_Vector_PM":
        cfg = replace(cfg, vector_router=True)
        name = "Modus_X"
    elif name == "Modus_X_Vector_Lean":
        cfg = replace(
            cfg,
            vector_router=True,
            router_hidden=cfg.router_hidden or max(8, cfg.embed_dim // 16),
        )
        name = "Modus_X"
    elif name == "Modus_X_CurrentArchive":
        cfg = replace(
            cfg,
            vector_router=True,
            router_hidden=cfg.router_hidden or max(8, cfg.embed_dim // 16),
        )
    elif name == "Modus_X_MemoryFeedbackArchive":
        cfg = replace(
            cfg,
            vector_router=True,
            router_hidden=cfg.router_hidden or max(8, cfg.embed_dim // 16),
        )
    elif name == "Modus_X_AdaptivePreconditionedArchive":
        cfg = replace(
            cfg,
            vector_router=True,
            router_hidden=cfg.router_hidden or max(8, cfg.embed_dim // 16),
        )
    elif name == "Modus_X_AttentionToWriteArchive":
        cfg = replace(
            cfg,
            vector_router=True,
            router_hidden=cfg.router_hidden or max(8, cfg.embed_dim // 16),
        )
    elif name == "Modus_X_FeedbackAttentionToWriteArchive":
        cfg = replace(
            cfg,
            vector_router=True,
            router_hidden=cfg.router_hidden or max(8, cfg.embed_dim // 16),
        )
    elif name == "Modus_X_DisplacedArchive":
        cfg = replace(
            cfg,
            vector_router=True,
            router_hidden=cfg.router_hidden or max(8, cfg.embed_dim // 16),
        )
    deep_supervision = name.endswith("_DeepSupervision")
    if deep_supervision:
        name = name.removesuffix("_DeepSupervision")
        if name == "Modus_X_Scalar_PM":
            cfg = replace(
                cfg,
                vector_router=False,
                router_hidden=cfg.embed_dim,
            )
            name = "Modus_X"
        elif name == "Modus_X_Scalar_Lean":
            cfg = replace(
                cfg,
                vector_router=False,
                router_hidden=cfg.router_hidden or max(8, cfg.embed_dim // 16),
            )
            name = "Modus_X"
        elif name == "Modus_X_Vector_Lean":
            cfg = replace(
                cfg,
                vector_router=True,
                router_hidden=cfg.router_hidden or max(8, cfg.embed_dim // 16),
            )
            name = "Modus_X"
        elif name == "Modus_X_CurrentArchive":
            cfg = replace(
                cfg,
                vector_router=True,
                router_hidden=cfg.router_hidden or max(8, cfg.embed_dim // 16),
            )
        elif name == "Modus_X_MemoryFeedbackArchive":
            cfg = replace(
                cfg,
                vector_router=True,
                router_hidden=cfg.router_hidden or max(8, cfg.embed_dim // 16),
            )
        elif name == "Modus_X_AdaptivePreconditionedArchive":
            cfg = replace(
                cfg,
                vector_router=True,
                router_hidden=cfg.router_hidden or max(8, cfg.embed_dim // 16),
            )
        elif name == "Modus_X_AttentionToWriteArchive":
            cfg = replace(
                cfg,
                vector_router=True,
                router_hidden=cfg.router_hidden or max(8, cfg.embed_dim // 16),
            )
        elif name == "Modus_X_FeedbackAttentionToWriteArchive":
            cfg = replace(
                cfg,
                vector_router=True,
                router_hidden=cfg.router_hidden or max(8, cfg.embed_dim // 16),
            )
        elif name == "Modus_X_DisplacedArchive":
            cfg = replace(
                cfg,
                vector_router=True,
                router_hidden=cfg.router_hidden or max(8, cfg.embed_dim // 16),
            )
    if name not in MODEL_REGISTRY:
        options = sorted([*MODEL_REGISTRY, "Modus_X_Scalar", "Modus_X_Vector", "Modus_X_Scalar_PM", "Modus_X_Scalar_Lean", "Modus_X_Vector_PM", "Modus_X_Vector_Lean"])
        raise ValueError(f"Unknown model {name}. Options: {options}")
    init_fn, fwd_fn = MODEL_REGISTRY[name]
    if deep_supervision:
        if name not in (
            "Modus_X",
            "Modus_X_CurrentArchive",
            "Modus_X_DisplacedArchive",
            "Modus_X_MemoryFeedbackArchive",
            "Modus_X_AdaptivePreconditionedArchive",
            "Modus_X_AttentionToWriteArchive",
            "Modus_X_FeedbackAttentionToWriteArchive",
        ):
            raise ValueError("Deep supervision is currently implemented only for Modus_X variants.")
        keys = random.split(key, 2)
        params = init_fn(keys[0], cfg)
        params = add_future_heads(params, keys[1], cfg, future_target_count)
        if name == "Modus_X_CurrentArchive":
            return params, lambda p, x, dropout_key=None: modus_x_current_archive_lm_fwd_deep_supervision(
                p,
                x,
                cfg,
                auxiliary_layers,
                dropout_key,
                dropout_rate,
            )
        if name == "Modus_X_MemoryFeedbackArchive":
            return params, lambda p, x, dropout_key=None: modus_x_memory_feedback_archive_lm_fwd_deep_supervision(
                p,
                x,
                cfg,
                auxiliary_layers,
                dropout_key,
                dropout_rate,
            )
        if name == "Modus_X_AdaptivePreconditionedArchive":
            return params, lambda p, x, dropout_key=None: modus_x_adaptive_preconditioned_archive_lm_fwd_deep_supervision(
                p,
                x,
                cfg,
                auxiliary_layers,
                dropout_key,
                dropout_rate,
            )
        if name == "Modus_X_AttentionToWriteArchive":
            return params, lambda p, x, dropout_key=None: modus_x_attention_to_write_archive_lm_fwd_deep_supervision(
                p,
                x,
                cfg,
                auxiliary_layers,
                dropout_key,
                dropout_rate,
            )
        if name == "Modus_X_FeedbackAttentionToWriteArchive":
            return params, lambda p, x, dropout_key=None: modus_x_feedback_attention_to_write_archive_lm_fwd_deep_supervision(
                p,
                x,
                cfg,
                auxiliary_layers,
                dropout_key,
                dropout_rate,
            )
        if name == "Modus_X_DisplacedArchive":
            return params, lambda p, x, dropout_key=None: modus_x_displaced_archive_lm_fwd_deep_supervision(
                p,
                x,
                cfg,
                auxiliary_layers,
                dropout_key,
                dropout_rate,
            )
        return params, lambda p, x, dropout_key=None: modus_x_lm_fwd_deep_supervision(
            p,
            x,
            cfg,
            auxiliary_layers,
            dropout_key,
            dropout_rate,
        )
    keys = random.split(key, 2)
    params = init_fn(keys[0], cfg)
    params = add_future_heads(params, keys[1], cfg, future_target_count)
    return params, lambda p, x: fwd_fn(p, x, cfg)


def lm_loss(
    params: dict,
    fwd_fn: Callable,
    x: jax.Array,
    y: jax.Array,
    auxiliary_weight: float = 0.3,
) -> jax.Array:
    outputs = jax.vmap(lambda xi: fwd_fn(params, xi))(x)
    logits, auxiliary_logits = outputs if isinstance(outputs, tuple) else (outputs, None)
    logp = jax.nn.log_softmax(logits, axis=-1)
    b, t = x.shape
    nll = -logp[jnp.arange(b)[:, None], jnp.arange(t)[None, :], y]
    loss = jnp.mean(nll)
    if auxiliary_logits is not None:
        aux_logp = jax.nn.log_softmax(auxiliary_logits, axis=-1)
        aux_nll = -jnp.take_along_axis(
            aux_logp,
            y[:, None, :, None],
            axis=-1,
        )[..., 0]
        loss = loss + auxiliary_weight * jnp.mean(aux_nll)
    return loss
