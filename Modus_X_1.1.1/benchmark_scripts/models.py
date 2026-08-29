from __future__ import annotations

from dataclasses import dataclass
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


def count_params(tree) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(tree) if hasattr(x, "size"))


def layer_norm(x: jax.Array, g: jax.Array, b: jax.Array) -> jax.Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / jnp.sqrt(var + 1e-5) + b


def normalize(x: jax.Array) -> jax.Array:
    return x / (jnp.linalg.norm(x) + 1e-8)


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
    keys = random.split(key, 20)
    d, r, n = cfg.embed_dim, cfg.ax_res, cfg.mamba_state_dim
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
        "r_w":       random.normal(keys[15], (d, d)) * 0.01,
        "r_b":       jnp.zeros(d),
        "r_proj":    random.normal(keys[16], (d if getattr(cfg, "vector_router", False) else 1, d)) * 0.01,
        "r_proj_b":  jnp.zeros(d if getattr(cfg, "vector_router", False) else 1),
    }


@jax.checkpoint
def modus_x_layer_fwd(layer: dict, x_seq: jax.Array) -> jax.Array:
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

    H0 = jnp.zeros((r, r))
    s0 = jnp.zeros(layer["s_wu"].shape[0])
    _, out = lax.scan(step, (H0, s0), x_seq)
    return out


def init_modus_x_lm(key: jax.Array, cfg: ModelConfig) -> dict:
    keys = random.split(key, cfg.n_layers + 2)
    layers = [init_modus_x_layer(keys[i], cfg) for i in range(cfg.n_layers)]
    return {
        "embed":  init_embed(keys[-2], cfg),
        "layers": jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *layers),
        "head":   init_lm_head(keys[-1], cfg.embed_dim, cfg),
    }


def modus_x_lm_fwd(p: dict, x_ids: jax.Array, cfg: ModelConfig) -> jax.Array:
    x = p["embed"][x_ids]

    def scan_layer(x_in, layer):
        return x_in + modus_x_layer_fwd(layer, x_in), None

    x, _ = lax.scan(scan_layer, x, p["layers"])
    return lm_head_fwd(p["head"], x)


MODEL_REGISTRY: dict[str, tuple[Callable, Callable]] = {
    "Modus": (init_modus_lm, modus_lm_fwd),
    "Modus_M": (init_modus_m_lm, modus_m_lm_fwd),
    "Modus_M2": (init_modus_m2_lm, modus_m2_lm_fwd),
    "Modus_M3": (init_modus_m3_lm, modus_m3_lm_fwd),
    "Modus_X": (init_modus_x_lm, modus_x_lm_fwd),
    "Mamba": (init_mamba_lm, mamba_lm_fwd),
    "Transformer": (init_transformer_lm, transformer_lm_fwd),
}


def make_model(name: str, key: jax.Array, cfg: ModelConfig) -> tuple[dict, Callable]:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name}. Options: {sorted(MODEL_REGISTRY)}")
    init_fn, fwd_fn = MODEL_REGISTRY[name]
    return init_fn(key, cfg), lambda p, x: fwd_fn(p, x, cfg)


def lm_loss(params: dict, fwd_fn: Callable, x: jax.Array, y: jax.Array) -> jax.Array:
    logits = jax.vmap(lambda xi: fwd_fn(params, xi))(x)
    logp = jax.nn.log_softmax(logits, axis=-1)
    b, t = x.shape
    nll = -logp[jnp.arange(b)[:, None], jnp.arange(t)[None, :], y]
    return jnp.mean(nll)
