"""
Experimental Modus_X 2.1.0 coordinated dual-memory components.

This module is intentionally small and isolated. It is not the published
Modus_X v1 implementation. It exists to test whether tighter coordination
between vector recurrence and matrix memory fixes the bounded-memory failures
seen in Stage1F/Stage1G.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import lax, random


@dataclass(frozen=True)
class CoordinatedMemoryConfig:
    d_model: int = 96
    key_dim: int = 32
    n_values: int = 32
    ax_res: int = 128
    vector_state: int = 128
    router_hidden: int = 128
    head_hidden: int = 128
    router_bias: float = 2.0
    residual_scale: float = 0.25
    vector_write_control: bool = True
    post_read_router: bool = True
    memory_feedback: bool = False
    version_aware_keys: bool = False
    disciplined_delta: bool = False
    vector_channel_gates: bool = False
    write_error_feedback: bool = False
    role_key_strength: float = 1.0
    operation_aware_router: bool = False
    latest_shadow_write: bool = False
    split_readout_heads: bool = False
    two_path_read: bool = False
    learned_read_arbitration: bool = False
    local_attention: bool = False
    local_attention_window: int = 64
    local_attention_heads: int = 4
    hard_latest_attention: bool = False
    current_archive_delta: bool = False


def count_params(tree) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(tree) if hasattr(x, "size"))


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


def init_params(key: jax.Array, cfg: CoordinatedMemoryConfig, *, vector_router: bool = True) -> dict:
    keys = random.split(key, 40)
    ax = cfg.ax_res
    d = cfg.d_model
    n = cfg.vector_state
    router_hidden = cfg.router_hidden

    wk = random.normal(keys[0], (ax, d)) * 0.001
    wq = random.normal(keys[1], (ax, d)) * 0.001
    wv = random.normal(keys[2], (ax, d)) * 0.001
    read_w = random.normal(keys[3], (ax, ax)) * 0.001
    role_wk = jnp.zeros((ax, 3))
    role_wq = jnp.zeros((ax, 3))

    # Bias the synthetic key/value slots so the model starts with the right
    # inductive bias instead of discovering identity projections from scratch.
    wk = wk.at[: min(ax, cfg.key_dim), : min(d, cfg.key_dim)].set(
        jnp.eye(min(ax, cfg.key_dim), min(d, cfg.key_dim))
    )
    wq = wq.at[: min(ax, cfg.key_dim), : min(d, cfg.key_dim)].set(
        jnp.eye(min(ax, cfg.key_dim), min(d, cfg.key_dim))
    )
    for i in range(min(ax, cfg.n_values)):
        if cfg.key_dim + i < d:
            wv = wv.at[i, cfg.key_dim + i].set(1.0)
        read_w = read_w.at[i, i].set(1.0)

    # Role order is [latest, previous, first]. Facts can be tagged as first or
    # latest. Queries can ask for latest, previous, or first. In the simple
    # one-update protocol, previous maps to the first-version address.
    if ax > cfg.key_dim + 1:
        latest_axis = cfg.key_dim
        first_axis = cfg.key_dim + 1
        role_wk = role_wk.at[latest_axis, 0].set(cfg.role_key_strength)
        role_wk = role_wk.at[first_axis, 2].set(cfg.role_key_strength)
        role_wq = role_wq.at[latest_axis, 0].set(cfg.role_key_strength)
        role_wq = role_wq.at[first_axis, 1].set(cfg.role_key_strength)
        role_wq = role_wq.at[first_axis, 2].set(cfg.role_key_strength)

    router_out = ax if vector_router else 1
    router_in = d + (3 * ax if cfg.post_read_router else 0)
    attn_wq = random.normal(keys[23], (cfg.local_attention_heads, d, d // cfg.local_attention_heads)) * 0.01
    attn_wk = random.normal(keys[24], (cfg.local_attention_heads, d, d // cfg.local_attention_heads)) * 0.01
    attn_wv = random.normal(keys[25], (cfg.local_attention_heads, d, d // cfg.local_attention_heads)) * 0.01
    attn_wo = random.normal(keys[26], (cfg.local_attention_heads, d // cfg.local_attention_heads, ax)) * 0.01
    # Give the bounded attention lane a fair short-term precision prior: one
    # head can compare synthetic keys and copy nearby value slots into the same
    # ax-space used by matrix memory. Training can still override this.
    attn_width = min(d // cfg.local_attention_heads, cfg.key_dim, ax)
    attn_wq = attn_wq.at[0, :attn_width, :attn_width].set(jnp.eye(attn_width))
    attn_wk = attn_wk.at[0, :attn_width, :attn_width].set(jnp.eye(attn_width))
    value_width = min(d // cfg.local_attention_heads, cfg.n_values, ax)
    for i in range(value_width):
        attn_wv = attn_wv.at[0, cfg.key_dim + i, i].set(1.0)
        attn_wo = attn_wo.at[0, i, i].set(1.0)

    params = {
        "wk": wk,
        "wq": wq,
        "wv": wv,
        "role_wk": role_wk,
        "role_wq": role_wq,
        "fact_w": random.normal(keys[4], (1, d)) * 0.01,
        "fact_b": jnp.ones(1) * 4.0,
        "write_x_w": random.normal(keys[5], (1, d)) * 0.01,
        "write_s_w": random.normal(keys[6], (1, n)) * 0.01,
        "write_b": jnp.zeros(1),
        "write_vec_x": random.normal(keys[15], (ax, d)) * 0.01,
        "write_vec_s": random.normal(keys[16], (ax, n)) * 0.01,
        "write_vec_b": jnp.zeros(ax),
        "erase_vec_x": random.normal(keys[17], (ax, d)) * 0.01,
        "erase_vec_s": random.normal(keys[18], (ax, n)) * 0.01,
        "erase_vec_b": jnp.zeros(ax),
        "write_error_to_s": random.normal(keys[19], (n, ax)) * 0.01,
        "read_w": read_w,
        "read_b": jnp.zeros(ax),
        "s_wu": random.normal(keys[7], (n, d)) * 0.01,
        "s_w_delta": random.normal(keys[8], (n, d)) * 0.01,
        "s_b_delta": jnp.zeros(n),
        "s_w_ret": random.normal(keys[9], (n, d)) * 0.01,
        "s_b_ret": jnp.ones(n) * 2.0,
        "s_proj": random.normal(keys[10], (ax, n)) * 0.01,
        "mem_to_s": random.normal(keys[11], (n, ax)) * 0.01,
        "router_w1": random.normal(keys[12], (router_hidden, router_in)) * 0.01,
        "router_b1": jnp.zeros(router_hidden),
        "router_w2": random.normal(keys[13], (router_out, router_hidden)) * 0.01,
        "router_b2": jnp.ones(router_out) * cfg.router_bias,
        "router_role_w": random.normal(keys[20], (router_out, 3)) * 0.01,
        "arb_w": random.normal(keys[22], (1, 3 + 4 * ax)) * 0.01,
        "arb_b": jnp.zeros(1),
        "attn_wq": attn_wq,
        "attn_wk": attn_wk,
        "attn_wv": attn_wv,
        "attn_wo": attn_wo,
        "attn_gate_w": random.normal(keys[27], (1, d + ax)) * 0.01,
        "attn_gate_b": jnp.zeros(1),
        "classifier": init_head(keys[14], ax, cfg.n_values, cfg.head_hidden),
    }
    if cfg.split_readout_heads:
        params["history_classifier"] = init_head(keys[21], ax, cfg.n_values, cfg.head_hidden)
    return params


def role_vector(x: jax.Array, cfg: CoordinatedMemoryConfig) -> jax.Array:
    fact_m = cfg.key_dim + cfg.n_values
    latest_marker = fact_m + 2
    previous_marker = fact_m + 3
    first_marker = fact_m + 4
    # The model config can be used on older tasks that do not allocate the
    # version-marker slots. In that case, return zeros and behave like v1.
    if cfg.d_model <= first_marker:
        return jnp.zeros(3)
    return jnp.array([x[latest_marker], x[previous_marker], x[first_marker]])


def address_key(p: dict, cfg: CoordinatedMemoryConfig, x: jax.Array, *, kind: str) -> jax.Array:
    base = p["wk"] @ x if kind == "write" else p["wq"] @ x
    if cfg.version_aware_keys:
        role = role_vector(x, cfg)
        base = base + (p["role_wk"] if kind == "write" else p["role_wq"]) @ role
    return base / (jnp.linalg.norm(base) + 1e-8)


def latest_shadow_key(p: dict, cfg: CoordinatedMemoryConfig, x: jax.Array) -> jax.Array:
    """Address the latest slot even for first-version facts.

    Stage1G writes first facts with a first marker and overwrites with a latest
    marker. Without a shadow write, clean keys have no latest-slot value, which
    makes "latest" ambiguous during training. This key gives first facts a
    consistent current/latest slot that later overwrites can update.
    """
    fact_m = cfg.key_dim + cfg.n_values
    latest_marker = fact_m + 2
    first_marker = fact_m + 4
    if cfg.d_model <= first_marker:
        return address_key(p, cfg, x, kind="write")
    x_latest = x.at[first_marker].set(0.0).at[latest_marker].set(1.0)
    return address_key(p, cfg, x_latest, kind="write")


def query_with_role(cfg: CoordinatedMemoryConfig, x: jax.Array, role_name: str) -> jax.Array:
    """Force a query into a current/latest or historical address lane."""
    fact_m = cfg.key_dim + cfg.n_values
    latest_marker = fact_m + 2
    previous_marker = fact_m + 3
    first_marker = fact_m + 4
    if cfg.d_model <= first_marker:
        return x
    x_role = x.at[latest_marker].set(0.0).at[previous_marker].set(0.0).at[first_marker].set(0.0)
    if role_name == "latest":
        return x_role.at[latest_marker].set(1.0)
    if role_name == "history":
        # In the one-update Stage1G protocol, previous and first share the
        # durable historical slot. Later long-history variants can split this.
        return x_role.at[first_marker].set(1.0)
    raise ValueError(f"Unknown query role: {role_name}")


def step_state(p: dict, cfg: CoordinatedMemoryConfig, carry, x):
    h, s = carry
    fact_m = cfg.key_dim + cfg.n_values

    k = address_key(p, cfg, x, kind="write")
    val = jnp.tanh(p["wv"] @ x)
    if cfg.current_archive_delta:
        h_current = h[0]
        h_archive = h[1]
        old = h_current @ k
    else:
        h_current = h
        old = h @ k

    u = jnp.tanh(p["s_wu"] @ x)
    delta = jax.nn.sigmoid(p["s_w_delta"] @ x + p["s_b_delta"])
    retain = jax.nn.sigmoid(p["s_w_ret"] @ x + p["s_b_ret"])
    s_next = retain * s + x[fact_m] * delta * u
    if cfg.write_error_feedback:
        s_next = s_next + x[fact_m] * cfg.residual_scale * jnp.tanh(p["write_error_to_s"] @ (val - old))

    fact_gate = x[fact_m] * jax.nn.sigmoid((p["fact_w"] @ x + p["fact_b"])[0])
    if cfg.vector_write_control:
        write_control = jax.nn.sigmoid((p["write_x_w"] @ x + p["write_s_w"] @ s + p["write_b"])[0])
    else:
        write_control = 1.0
    gate = fact_gate * write_control

    if cfg.disciplined_delta:
        if cfg.vector_channel_gates:
            write_vec = jax.nn.sigmoid(p["write_vec_x"] @ x + p["write_vec_s"] @ s + p["write_vec_b"])
            erase_vec = jax.nn.sigmoid(p["erase_vec_x"] @ x + p["erase_vec_s"] @ s + p["erase_vec_b"])
        else:
            write_vec = jnp.ones(cfg.ax_res)
            erase_vec = jnp.ones(cfg.ax_res)
        h_next = h_current - gate * jnp.outer(erase_vec * old, k) + gate * jnp.outer(write_vec * val, k)
    else:
        h_next = h_current + gate * jnp.outer(val - old, k)

    if cfg.current_archive_delta:
        first_marker = fact_m + 4
        latest_marker = fact_m + 2
        first_fact = x[fact_m] * (x[first_marker] if cfg.d_model > first_marker else 1.0)
        latest_fact = x[fact_m] * (x[latest_marker] if cfg.d_model > latest_marker else 0.0)
        archive_write = jnp.clip(first_fact + (1.0 - latest_fact) * (1.0 - first_fact), 0.0, 1.0)
        old_archive = h_archive @ k
        if cfg.disciplined_delta:
            h_archive_next = h_archive - archive_write * gate * jnp.outer(erase_vec * old_archive, k)
            h_archive_next = h_archive_next + archive_write * gate * jnp.outer(write_vec * val, k)
        else:
            h_archive_next = h_archive + archive_write * gate * jnp.outer(val - old_archive, k)
        h_next = jnp.stack([h_next, h_archive_next])

    if cfg.latest_shadow_write:
        fact_m = cfg.key_dim + cfg.n_values
        first_marker = fact_m + 4
        first_fact = x[fact_m] * (x[first_marker] if cfg.d_model > first_marker else 0.0)
        k_latest = latest_shadow_key(p, cfg, x)
        h_for_shadow = h_next[0] if cfg.current_archive_delta else h_next
        old_latest = h_for_shadow @ k_latest
        if cfg.disciplined_delta:
            h_shadow = h_for_shadow - gate * jnp.outer(erase_vec * old_latest, k_latest) + gate * jnp.outer(write_vec * val, k_latest)
        else:
            h_shadow = h_for_shadow + gate * jnp.outer(val - old_latest, k_latest)
        if cfg.current_archive_delta:
            h_next = h_next.at[0].set(h_for_shadow + first_fact * (h_shadow - h_for_shadow))
        else:
            h_next = h_next + first_fact * (h_shadow - h_next)
    return (h_next, s_next), None


def local_attention_query_read(p: dict, cfg: CoordinatedMemoryConfig, seq: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Bounded causal local-attention read for the final query token.

    This is intentionally a short-term precision path. It only attends from the
    final query position to earlier tokens inside a fixed window; it is not a
    growing global KV cache.
    """
    local_seq = seq[:-1]
    window = min(cfg.local_attention_window, local_seq.shape[0])
    local = local_seq[-window:]
    query = seq[-1]
    d_head = cfg.d_model // cfg.local_attention_heads
    q = jnp.einsum("d,hdk->hk", query, p["attn_wq"])
    k = jnp.einsum("td,hdk->htk", local, p["attn_wk"])
    v = jnp.einsum("td,hdk->htk", local, p["attn_wv"])
    scores = jnp.einsum("hk,htk->ht", q, k) / jnp.sqrt(float(d_head))
    weights = jax.nn.softmax(scores, axis=-1)
    attended = jnp.einsum("ht,htk->hk", weights, v)
    read = jnp.einsum("hk,hka->a", attended, p["attn_wo"])
    entropy = -jnp.mean(jnp.sum(weights * jnp.log(weights + 1e-8), axis=-1))
    return read, entropy


def model_readout(p: dict, seq: jax.Array, cfg: CoordinatedMemoryConfig, *, fusion: str = "coordinated"):
    if cfg.current_archive_delta:
        h0 = jnp.zeros((2, cfg.ax_res, cfg.ax_res))
    else:
        h0 = jnp.zeros((cfg.ax_res, cfg.ax_res))
    s0 = jnp.zeros(cfg.vector_state)
    (h, s), _ = lax.scan(lambda carry, x: step_state(p, cfg, carry, x), (h0, s0), seq[:-1])

    query_x = seq[-1]
    q = address_key(p, cfg, query_x, kind="query")
    h_read = h[0] if cfg.current_archive_delta else h
    memory_read = p["read_w"] @ (h_read @ q) + p["read_b"]
    current_read = memory_read
    history_read = memory_read
    if cfg.current_archive_delta:
        q_current = address_key(p, cfg, query_with_role(cfg, query_x, "latest"), kind="query")
        q_history = address_key(p, cfg, query_with_role(cfg, query_x, "history"), kind="query")
        current_read = p["read_w"] @ (h[0] @ q_current) + p["read_b"]
        history_read = p["read_w"] @ (h[1] @ q_history) + p["read_b"]
        role = role_vector(query_x, cfg)
        latest_weight = role[0]
        history_weight = jnp.maximum(role[1], role[2])
        fallback_weight = 1.0 - jnp.maximum(latest_weight, history_weight)
        current_weight = latest_weight + fallback_weight
        memory_read = current_weight * current_read + history_weight * history_read
    elif cfg.two_path_read:
        q_current = address_key(p, cfg, query_with_role(cfg, query_x, "latest"), kind="query")
        q_history = address_key(p, cfg, query_with_role(cfg, query_x, "history"), kind="query")
        current_read = p["read_w"] @ (h @ q_current) + p["read_b"]
        history_read = p["read_w"] @ (h @ q_history) + p["read_b"]
        if cfg.learned_read_arbitration:
            role = role_vector(query_x, cfg)
            arb_input = jnp.concatenate(
                [
                    role,
                    current_read,
                    history_read,
                    current_read - history_read,
                    jnp.abs(current_read - history_read),
                ]
            )
            current_weight = jax.nn.sigmoid((p["arb_w"] @ arb_input + p["arb_b"])[0])
            memory_read = current_weight * current_read + (1.0 - current_weight) * history_read
        else:
            role = role_vector(query_x, cfg)
            latest_weight = role[0]
            history_weight = jnp.maximum(role[1], role[2])
            fallback_weight = 1.0 - jnp.maximum(latest_weight, history_weight)
            current_weight = latest_weight + fallback_weight
            memory_read = current_weight * current_read + history_weight * history_read
    else:
        current_weight = jnp.array(1.0)
    vector_read = p["s_proj"] @ s

    if cfg.memory_feedback:
        s = s + cfg.residual_scale * jnp.tanh(p["mem_to_s"] @ memory_read)
        vector_read = p["s_proj"] @ s

    if cfg.post_read_router:
        router_input = jnp.concatenate([query_x, memory_read, vector_read, jnp.abs(memory_read - vector_read)])
    else:
        router_input = query_x

    r_hidden = jax.nn.relu(p["router_w1"] @ router_input + p["router_b1"])
    router_logits = p["router_w2"] @ r_hidden + p["router_b2"]
    if cfg.operation_aware_router:
        router_logits = router_logits + p["router_role_w"] @ role_vector(query_x, cfg)
    router = jax.nn.sigmoid(router_logits)
    if router.shape[0] == 1:
        router = router[0]

    if fusion == "late":
        fused = router * memory_read + (1.0 - router) * vector_read
    elif fusion == "residual":
        fused = memory_read + cfg.residual_scale * router * vector_read
    elif fusion == "coordinated":
        # Memory remains the durable base; vector state corrects/conditions it.
        fused = memory_read + cfg.residual_scale * router * (vector_read - memory_read)
    else:
        raise ValueError(f"Unknown fusion: {fusion}")
    if cfg.local_attention:
        attn_read, attn_entropy = local_attention_query_read(p, cfg, seq)
        attn_gate_input = jnp.concatenate([query_x, attn_read])
        attn_gate = jax.nn.sigmoid((p["attn_gate_w"] @ attn_gate_input + p["attn_gate_b"])[0])
        if cfg.hard_latest_attention:
            latest_role = role_vector(query_x, cfg)[0]
            fused = (1.0 - latest_role) * fused + latest_role * attn_read
            attn_gate = latest_role
        else:
            fused = fused + cfg.residual_scale * attn_gate * (attn_read - fused)
    else:
        attn_read = jnp.zeros(cfg.ax_res)
        attn_entropy = jnp.array(0.0)
        attn_gate = jnp.array(0.0)
    latest_logits = head_fwd(p["classifier"], fused)
    if cfg.split_readout_heads:
        history_logits = head_fwd(p["history_classifier"], fused)
        role = role_vector(query_x, cfg)
        latest_weight = role[0]
        history_weight = jnp.maximum(role[1], role[2])
        fallback_weight = 1.0 - jnp.maximum(latest_weight, history_weight)
        logits = (latest_weight + fallback_weight) * latest_logits + history_weight * history_logits
    else:
        logits = latest_logits
    aux = {
        "router_mean": jnp.mean(router),
        "router_std": jnp.std(router),
        "memory_norm": jnp.linalg.norm(memory_read),
        "vector_norm": jnp.linalg.norm(vector_read),
        "disagreement_norm": jnp.linalg.norm(memory_read - vector_read),
        "fused_norm": jnp.linalg.norm(fused),
        "current_read_norm": jnp.linalg.norm(current_read),
        "history_read_norm": jnp.linalg.norm(history_read),
        "current_history_disagreement_norm": jnp.linalg.norm(current_read - history_read),
        "read_arbitration_current_weight": current_weight,
        "read_arbitration_entropy": -(
            jnp.clip(current_weight, 1e-6, 1.0 - 1e-6) * jnp.log(jnp.clip(current_weight, 1e-6, 1.0 - 1e-6))
            + (1.0 - jnp.clip(current_weight, 1e-6, 1.0 - 1e-6))
            * jnp.log(1.0 - jnp.clip(current_weight, 1e-6, 1.0 - 1e-6))
        ),
        "attention_norm": jnp.linalg.norm(attn_read),
        "attention_gate": attn_gate,
        "attention_entropy": attn_entropy,
        # Diagnostic-only branch probes. They reuse the trained classifier and
        # do not affect the model output or training path.
        "current_branch_logits": head_fwd(p["classifier"], current_read),
        "history_branch_logits": head_fwd(p["classifier"], history_read),
        "memory_branch_logits": head_fwd(p["classifier"], memory_read),
        "vector_branch_logits": head_fwd(p["classifier"], vector_read),
        "fused_branch_logits": latest_logits,
    }
    return logits, aux


def model_fwd(p: dict, seq: jax.Array, cfg: CoordinatedMemoryConfig, *, fusion: str = "coordinated") -> jax.Array:
    logits, _ = model_readout(p, seq, cfg, fusion=fusion)
    return logits


def model_aux(p: dict, seq: jax.Array, cfg: CoordinatedMemoryConfig, *, fusion: str = "coordinated") -> dict:
    _, aux = model_readout(p, seq, cfg, fusion=fusion)
    return aux


def with_head_hidden(cfg: CoordinatedMemoryConfig, head_hidden: int) -> CoordinatedMemoryConfig:
    return CoordinatedMemoryConfig(**{**cfg.__dict__, "head_hidden": head_hidden})


def make_model(name: str, key: jax.Array, cfg: CoordinatedMemoryConfig):
    if name == "LateFusionV1Control":
        control_cfg = CoordinatedMemoryConfig(
            **{
                **cfg.__dict__,
                "vector_write_control": False,
                "post_read_router": False,
                "memory_feedback": False,
            }
        )
        params = init_params(key, control_cfg, vector_router=True)
        return params, lambda p, seq: model_fwd(p, seq, control_cfg, fusion="late"), control_cfg
    if name == "LateFusionPMControl":
        target_cfg = CoordinatedMemoryConfig(
            **{
                **cfg.__dict__,
                "version_aware_keys": True,
                "vector_write_control": True,
                "post_read_router": True,
                "disciplined_delta": True,
                "vector_channel_gates": True,
                "write_error_feedback": True,
            }
        )
        target_params = init_params(key, target_cfg, vector_router=True)
        control_base = CoordinatedMemoryConfig(
            **{
                **cfg.__dict__,
                "vector_write_control": False,
                "post_read_router": False,
                "memory_feedback": False,
                "version_aware_keys": False,
                "disciplined_delta": False,
                "vector_channel_gates": False,
                "write_error_feedback": False,
            }
        )
        best_cfg = control_base
        best_params = init_params(key, control_base, vector_router=True)
        best_gap = abs(count_params(best_params) - count_params(target_params))
        for hidden in range(control_base.head_hidden, 768):
            candidate_cfg = with_head_hidden(control_base, hidden)
            candidate = init_params(key, candidate_cfg, vector_router=True)
            gap = abs(count_params(candidate) - count_params(target_params))
            if gap < best_gap:
                best_gap = gap
                best_cfg = candidate_cfg
                best_params = candidate
        return best_params, lambda p, seq: model_fwd(p, seq, best_cfg, fusion="late"), best_cfg
    if name == "CoordinatedWrite":
        run_cfg = CoordinatedMemoryConfig(
            **{**cfg.__dict__, "vector_write_control": True, "post_read_router": False, "memory_feedback": False}
        )
        params = init_params(key, run_cfg, vector_router=True)
        return params, lambda p, seq: model_fwd(p, seq, run_cfg, fusion="late"), run_cfg
    if name == "PostReadRouter":
        run_cfg = CoordinatedMemoryConfig(
            **{**cfg.__dict__, "vector_write_control": False, "post_read_router": True, "memory_feedback": False}
        )
        params = init_params(key, run_cfg, vector_router=True)
        return params, lambda p, seq: model_fwd(p, seq, run_cfg, fusion="coordinated"), run_cfg
    if name == "CoordinatedDualMemory":
        params = init_params(key, cfg, vector_router=True)
        return params, lambda p, seq: model_fwd(p, seq, cfg, fusion="coordinated"), cfg
    if name == "CoordinatedDualMemoryFeedback":
        run_cfg = CoordinatedMemoryConfig(**{**cfg.__dict__, "memory_feedback": True})
        params = init_params(key, run_cfg, vector_router=True)
        return params, lambda p, seq: model_fwd(p, seq, run_cfg, fusion="coordinated"), run_cfg
    if name == "VersionAwareDelta":
        run_cfg = CoordinatedMemoryConfig(
            **{
                **cfg.__dict__,
                "version_aware_keys": True,
                "vector_write_control": True,
                "post_read_router": True,
                "disciplined_delta": False,
            }
        )
        params = init_params(key, run_cfg, vector_router=True)
        return params, lambda p, seq: model_fwd(p, seq, run_cfg, fusion="coordinated"), run_cfg
    if name == "DisciplinedDeltaMemory":
        run_cfg = CoordinatedMemoryConfig(
            **{
                **cfg.__dict__,
                "version_aware_keys": True,
                "vector_write_control": True,
                "post_read_router": True,
                "disciplined_delta": True,
                "vector_channel_gates": True,
                "write_error_feedback": True,
            }
        )
        params = init_params(key, run_cfg, vector_router=True)
        return params, lambda p, seq: model_fwd(p, seq, run_cfg, fusion="coordinated"), run_cfg
    if name == "OperationAwareDisciplinedDelta":
        run_cfg = CoordinatedMemoryConfig(
            **{
                **cfg.__dict__,
                "version_aware_keys": True,
                "vector_write_control": True,
                "post_read_router": True,
                "disciplined_delta": True,
                "vector_channel_gates": True,
                "write_error_feedback": True,
                "operation_aware_router": True,
            }
        )
        params = init_params(key, run_cfg, vector_router=True)
        return params, lambda p, seq: model_fwd(p, seq, run_cfg, fusion="coordinated"), run_cfg
    if name == "LatestShadowDisciplinedDelta":
        run_cfg = CoordinatedMemoryConfig(
            **{
                **cfg.__dict__,
                "version_aware_keys": True,
                "vector_write_control": True,
                "post_read_router": True,
                "disciplined_delta": True,
                "vector_channel_gates": True,
                "write_error_feedback": True,
                "latest_shadow_write": True,
            }
        )
        params = init_params(key, run_cfg, vector_router=True)
        return params, lambda p, seq: model_fwd(p, seq, run_cfg, fusion="coordinated"), run_cfg
    if name == "SplitHeadLatestShadowDelta":
        run_cfg = CoordinatedMemoryConfig(
            **{
                **cfg.__dict__,
                "version_aware_keys": True,
                "vector_write_control": True,
                "post_read_router": True,
                "disciplined_delta": True,
                "vector_channel_gates": True,
                "write_error_feedback": True,
                "latest_shadow_write": True,
                "split_readout_heads": True,
            }
        )
        params = init_params(key, run_cfg, vector_router=True)
        return params, lambda p, seq: model_fwd(p, seq, run_cfg, fusion="coordinated"), run_cfg
    if name == "TwoPathLatestShadowDelta":
        run_cfg = CoordinatedMemoryConfig(
            **{
                **cfg.__dict__,
                "version_aware_keys": True,
                "vector_write_control": True,
                "post_read_router": True,
                "disciplined_delta": True,
                "vector_channel_gates": True,
                "write_error_feedback": True,
                "latest_shadow_write": True,
                "two_path_read": True,
            }
        )
        params = init_params(key, run_cfg, vector_router=True)
        return params, lambda p, seq: model_fwd(p, seq, run_cfg, fusion="coordinated"), run_cfg
    if name == "TwoPathSplitHeadLatestShadowDelta":
        run_cfg = CoordinatedMemoryConfig(
            **{
                **cfg.__dict__,
                "version_aware_keys": True,
                "vector_write_control": True,
                "post_read_router": True,
                "disciplined_delta": True,
                "vector_channel_gates": True,
                "write_error_feedback": True,
                "latest_shadow_write": True,
                "two_path_read": True,
                "split_readout_heads": True,
            }
        )
        params = init_params(key, run_cfg, vector_router=True)
        return params, lambda p, seq: model_fwd(p, seq, run_cfg, fusion="coordinated"), run_cfg
    if name == "LearnedArbitrationLatestShadowDelta":
        run_cfg = CoordinatedMemoryConfig(
            **{
                **cfg.__dict__,
                "version_aware_keys": True,
                "vector_write_control": True,
                "post_read_router": True,
                "disciplined_delta": True,
                "vector_channel_gates": True,
                "write_error_feedback": True,
                "latest_shadow_write": True,
                "two_path_read": True,
                "learned_read_arbitration": True,
            }
        )
        params = init_params(key, run_cfg, vector_router=True)
        return params, lambda p, seq: model_fwd(p, seq, run_cfg, fusion="coordinated"), run_cfg
    if name == "LocalAttentionHybrid":
        run_cfg = CoordinatedMemoryConfig(
            **{
                **cfg.__dict__,
                "vector_write_control": False,
                "post_read_router": False,
                "memory_feedback": False,
                "local_attention": True,
            }
        )
        params = init_params(key, run_cfg, vector_router=True)
        return params, lambda p, seq: model_fwd(p, seq, run_cfg, fusion="late"), run_cfg
    if name == "LocalAttentionDisciplinedDelta":
        run_cfg = CoordinatedMemoryConfig(
            **{
                **cfg.__dict__,
                "version_aware_keys": True,
                "vector_write_control": True,
                "post_read_router": True,
                "disciplined_delta": True,
                "vector_channel_gates": True,
                "write_error_feedback": True,
                "latest_shadow_write": True,
                "two_path_read": True,
                "local_attention": True,
            }
        )
        params = init_params(key, run_cfg, vector_router=True)
        return params, lambda p, seq: model_fwd(p, seq, run_cfg, fusion="coordinated"), run_cfg
    if name == "HardLatestAttentionDelta":
        run_cfg = CoordinatedMemoryConfig(
            **{
                **cfg.__dict__,
                "version_aware_keys": True,
                "vector_write_control": True,
                "post_read_router": True,
                "disciplined_delta": True,
                "vector_channel_gates": True,
                "write_error_feedback": True,
                "latest_shadow_write": True,
                "two_path_read": True,
                "local_attention": True,
                "local_attention_window": max(cfg.local_attention_window, 512),
                "hard_latest_attention": True,
            }
        )
        params = init_params(key, run_cfg, vector_router=True)
        return params, lambda p, seq: model_fwd(p, seq, run_cfg, fusion="coordinated"), run_cfg
    if name == "CurrentArchiveDelta":
        run_cfg = CoordinatedMemoryConfig(
            **{
                **cfg.__dict__,
                "version_aware_keys": True,
                "vector_write_control": True,
                "post_read_router": True,
                "disciplined_delta": True,
                "vector_channel_gates": True,
                "write_error_feedback": True,
                "current_archive_delta": True,
            }
        )
        params = init_params(key, run_cfg, vector_router=True)
        return params, lambda p, seq: model_fwd(p, seq, run_cfg, fusion="coordinated"), run_cfg
    if name == "CurrentArchiveLatestShadow":
        run_cfg = CoordinatedMemoryConfig(
            **{
                **cfg.__dict__,
                "version_aware_keys": True,
                "vector_write_control": True,
                "post_read_router": True,
                "disciplined_delta": True,
                "vector_channel_gates": True,
                "write_error_feedback": True,
                "current_archive_delta": True,
                "latest_shadow_write": True,
            }
        )
        params = init_params(key, run_cfg, vector_router=True)
        return params, lambda p, seq: model_fwd(p, seq, run_cfg, fusion="coordinated"), run_cfg
    raise ValueError(f"Unknown Modus_X 2.1.0 model: {name}")


def make_model_with_aux(name: str, key: jax.Array, cfg: CoordinatedMemoryConfig):
    params, fwd, actual_cfg = make_model(name, key, cfg)
    if name in ("LateFusionV1Control", "LateFusionPMControl", "CoordinatedWrite", "LocalAttentionHybrid"):
        fusion = "late"
    else:
        fusion = "coordinated"
    return params, fwd, lambda p, seq: model_aux(p, seq, actual_cfg, fusion=fusion), actual_cfg
