from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


# ============================================================================
# PROJECT PATHS / IMPORTS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LANGUAGE_DIR = PROJECT_ROOT / "language"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LANGUAGE_DIR))

from models import (  # noqa: E402
    ModelConfig,
    count_params,
    layer_norm,
    lm_head_fwd,
    make_model,
    normalize,
)


# ============================================================================
# EXPERIMENT CONSTANTS
# ============================================================================

MODEL_NAME = "Modus_X_MemoryFeedbackArchive_DeepSupervision"

CANONICAL_PARAMS = 47_437_768

CANONICAL = {
    "vocab_size": 256,
    "embed_dim": 512,
    "n_layers": 12,
    "n_heads": 8,
    "seq_len": 512,
    "router_hidden": 32,
    "vector_router": False,
    "dropout": 0.0,
    "aux_layers": (6,),
    "future_target_count": 1,
}

CANDIDATES = {
    "C1_q128_k128": {
        "hidden_dim": 1664,
        "mamba_state_dim": 608,
        "ax_res": 512,
        "q_rank": 128,
        "k_rank": 128,
        "expected_params": 47_440_328,
    },
    "C2_q64_k64": {
        "hidden_dim": 2048,
        "mamba_state_dim": 640,
        "ax_res": 512,
        "q_rank": 64,
        "k_rank": 64,
        "expected_params": 47_441_864,
    },
    "C3_q128_k64": {
        "hidden_dim": 1536,
        "mamba_state_dim": 640,
        "ax_res": 512,
        "q_rank": 128,
        "k_rank": 64,
        "expected_params": 47_440_840,
    },
}

GPU_REFERENCES = {
    4_096_000: 2.506,
    20_480_000: 1.8638,
    40_960_000: 1.6918,
}


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PB1-C FINAL low-rank Q/K trainer with 32-token truncated BPTT"
    )

    p.add_argument(
        "--candidate",
        required=True,
        choices=tuple(CANDIDATES.keys()),
    )
    p.add_argument("--data-path", required=True)
    p.add_argument("--outdir", required=True)

    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--target-chars", type=int, default=10_000_000)
    p.add_argument("--stop-chars", type=int, default=None)
    p.add_argument("--checkpoint-chars", type=int, default=4_096_000)

    p.add_argument("--eval-chunks", type=int, default=128)
    p.add_argument("--eval-batch", type=int, default=8)

    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--auxiliary-weight", type=float, default=0.05)
    p.add_argument("--future-target-weight", type=float, default=0.5)

    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--schedule", choices=("constant", "warmup_cosine"), default="constant")
    p.add_argument("--end-lr-ratio", type=float, default=0.05)

    p.add_argument("--input-seq-len", type=int, default=512)
    p.add_argument("--loss-tail", type=int, default=512)
    p.add_argument("--train-chunk-len", type=int, default=32)

    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--resume", action="store_true")

    return p.parse_args()


# ============================================================================
# I/O
# ============================================================================

def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def atomic_pickle(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def block_tree(tree: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


# ============================================================================
# DATA
# ============================================================================

def load_split(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(data) < 100_000_000:
        raise ValueError(
            f"Expected at least 100,000,000 bytes for the canonical enwik8 split; "
            f"found {len(data):,}."
        )
    return (
        data[:90_000_000],
        data[90_000_000:95_000_000],
        data[95_000_000:100_000_000],
    )


def batch_at(
    data: np.ndarray,
    starts: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.arange(seq_len + 1)
    chunks = data[starts[:, None] + offsets[None, :]]
    return (
        chunks[:, :-1].astype(np.int32),
        chunks[:, 1:].astype(np.int32),
    )


# ============================================================================
# MODEL CONFIG
# ============================================================================

def build_config(candidate: Dict[str, int]) -> ModelConfig:
    return ModelConfig(
        vocab_size=CANONICAL["vocab_size"],
        embed_dim=CANONICAL["embed_dim"],
        hidden_dim=candidate["hidden_dim"],
        ax_res=candidate["ax_res"],
        n_layers=CANONICAL["n_layers"],
        n_heads_attn=CANONICAL["n_heads"],
        seq_len=CANONICAL["seq_len"],
        mamba_state_dim=candidate["mamba_state_dim"],
        vector_router=CANONICAL["vector_router"],
        router_hidden=CANONICAL["router_hidden"],
    )


# ============================================================================
# LOW-RANK INITIALIZATION
# ============================================================================

def factorize_weight(
    weight: jax.Array,
    rank: int,
) -> tuple[jax.Array, jax.Array]:
    """
    Factor layer-stacked W [L, out, in] as:
        W ~= A @ B
        A [L, out, rank]
        B [L, rank, in]

    Balanced SVD:
        A = U_r sqrt(S_r)
        B = sqrt(S_r) V_r^T
    """
    weight = jnp.asarray(weight, dtype=jnp.float32)

    if weight.ndim != 3:
        raise ValueError(f"Expected [layers,out,in], got {weight.shape}")

    layers = []
    b_layers = []

    for layer_index in range(weight.shape[0]):
        w = weight[layer_index]
        u, s, vh = jnp.linalg.svd(w, full_matrices=False)

        if rank >= min(w.shape):
            raise ValueError(
                f"Rank {rank} is not a strict factorization of matrix {w.shape}"
            )

        sr = s[:rank]
        root = jnp.sqrt(jnp.maximum(sr, 0.0))

        a = u[:, :rank] * root[None, :]
        b = root[:, None] * vh[:rank, :]

        layers.append(a)
        b_layers.append(b)

    return jnp.stack(layers), jnp.stack(b_layers)


def convert_dense_to_factorized(
    dense_params: Dict[str, Any],
    q_rank: int,
    k_rank: int,
) -> tuple[Dict[str, Any], Dict[str, int]]:
    layers = dense_params["layers"]

    q_a, q_b = factorize_weight(layers["m_wq"], q_rank)
    k_a, k_b = factorize_weight(layers["m_wk"], k_rank)

    new_layers = dict(layers)

    new_layers.pop("m_wq")
    new_layers.pop("m_wk")

    new_layers["m_wq_A"] = q_a
    new_layers["m_wq_B"] = q_b
    new_layers["m_wk_A"] = k_a
    new_layers["m_wk_B"] = k_b

    factorized = dict(dense_params)
    factorized["layers"] = new_layers

    q_dense = int(np.prod(layers["m_wq"].shape))
    k_dense = int(np.prod(layers["m_wk"].shape))

    q_factor = int(np.prod(q_a.shape) + np.prod(q_b.shape))
    k_factor = int(np.prod(k_a.shape) + np.prod(k_b.shape))

    stats = {
        "q_dense_params": q_dense,
        "q_factor_params": q_factor,
        "q_savings": q_dense - q_factor,
        "k_dense_params": k_dense,
        "k_factor_params": k_factor,
        "k_savings": k_dense - k_factor,
        "qk_savings": (q_dense - q_factor) + (k_dense - k_factor),
    }

    return factorized, stats


# ============================================================================
# TRUE FACTORIZED LAYER FORWARD
# ============================================================================

def factorized_memory_feedback_archive_layer_fwd(
    layer: Dict[str, Any],
    x_seq: jax.Array,
) -> jax.Array:
    """
    Exact MemoryFeedbackArchive layer computation, except:

        m_wq @ e -> m_wq_A @ (m_wq_B @ e)
        m_wk @ e -> m_wk_A @ (m_wk_B @ e)

    No dense Q/K reconstruction occurs.
    """
    r = layer["m_wk_A"].shape[0]

    def step(carry, e_raw):
        H_current, H_archive, s = carry

        e = layer_norm(
            e_raw,
            layer["pre_g"],
            layer["pre_b"],
        )

        # TRUE LOW-RANK Q/K COMPUTATION.
        k = normalize(
            layer["m_wk_A"] @ (layer["m_wk_B"] @ e)
        )
        q = normalize(
            layer["m_wq_A"] @ (layer["m_wq_B"] @ e)
        )

        val = jnp.tanh(layer["m_wv"] @ e)

        eta = jax.nn.sigmoid(
            (layer["m_w_eta"] @ e + layer["m_b_eta"])[0]
        )
        write = jax.nn.sigmoid(
            (layer["m_w_write"] @ e + layer["m_b_write"])[0]
        )
        retain = jax.nn.sigmoid(
            (layer["m_w_ret"] @ e + layer["m_b_ret"])[0]
        )

        old_current = H_current @ k
        H_current = (
            retain * H_current
            + (eta * write)
            * jnp.outer(val - old_current, k)
        )

        archive_write = jax.nn.sigmoid(
            (layer["m_w_archive_write"] @ e + layer["m_b_archive_write"])[0]
        )
        archive_retain = jax.nn.sigmoid(
            (layer["m_w_archive_ret"] @ e + layer["m_b_archive_ret"])[0]
        )

        old_archive = H_archive @ k
        H_archive = (
            archive_retain * H_archive
            + (eta * write * archive_write)
            * jnp.outer(val - old_archive, k)
        )

        read_gate = jax.nn.sigmoid(
            layer["m_w_read"] @ e + layer["m_b_read"]
        )

        current_context = layer_norm(
            H_current @ q,
            layer["m_ln_g"],
            layer["m_ln_b"],
        )
        archive_context = layer_norm(
            H_archive @ q,
            layer["m_ln_g"],
            layer["m_ln_b"],
        )

        archive_mix = jax.nn.sigmoid(
            layer["m_w_archive_mix"] @ e
            + layer["m_b_archive_mix"]
        )

        context = read_gate * (
            archive_mix * current_context
            + (1.0 - archive_mix) * archive_context
        )

        proposal = (
            layer["m_proj_w"]
            @ jnp.concatenate([e_raw, context])
            + layer["m_proj_b"]
        )

        out_gate = jax.nn.sigmoid(
            layer["m_w_out"] @ e + layer["m_b_out"]
        )

        modus_out = out_gate * proposal

        feedback_gate = jax.nn.sigmoid(
            (
                layer["s_w_memory_feedback"] @ e
                + layer["s_b_memory_feedback"]
            )[0]
        )

        compressed_memory = jnp.tanh(
            layer["s_memory_down"] @ context
        )
        memory_feedback = (
            layer["s_memory_up"] @ compressed_memory
        )

        e_vector = layer_norm(
            e_raw + feedback_gate * memory_feedback,
            layer["pre_g"],
            layer["pre_b"],
        )

        u = jnp.tanh(layer["s_wu"] @ e_vector)

        delta = jax.nn.sigmoid(
            layer["s_w_delta"] @ e_vector
            + layer["s_b_delta"]
        )

        ret_s = jax.nn.sigmoid(
            layer["s_w_ret"] @ e_vector
            + layer["s_b_ret"]
        )

        s = ret_s * s + delta * u

        c = jax.nn.sigmoid(
            layer["s_w_c"] @ e_vector
        )

        y_s = c * s

        gate_s = jax.nn.sigmoid(
            layer["s_w_gate"] @ e_vector
            + layer["s_b_gate"]
        )

        mamba_out = gate_s * (
            layer["s_proj_w"] @ y_s
            + layer["s_proj_b"]
        )

        r_hidden = jax.nn.gelu(
            layer["r_w"] @ e + layer["r_b"]
        )

        r_logits = (
            layer["r_proj"] @ r_hidden
            + layer["r_proj_b"]
        )

        router = jax.nn.sigmoid(
            r_logits[0]
            if layer["r_proj"].shape[0] == 1
            else r_logits
        )

        out = (
            router * modus_out
            + (1.0 - router) * mamba_out
        )

        return (H_current, H_archive, s), out

    initial_state = (
        jnp.zeros((r, r), dtype=jnp.float32),
        jnp.zeros((r, r), dtype=jnp.float32),
        jnp.zeros(
            layer["s_wu"].shape[0],
            dtype=jnp.float32,
        ),
    )

    # TPU-v5e memory-safe recurrent differentiation.
    #
    # The recurrent carry contains TWO 512x512 float32 matrices. Full
    # reverse-mode differentiation through all 512 tokens exceeded the
    # TPU-v5e-1 HBM budget in PB1-C.
    #
    # PB1-C v4 therefore uses EXPLICIT 32-token truncated BPTT:
    #   * forward recurrence remains continuous across all 512 tokens;
    #   * each 32-token chunk receives gradients through its own recurrence;
    #   * the recurrent carry is stop_gradient'ed at every chunk boundary.
    #
    # This is intentionally a different training protocol from full
    # 512-token BPTT and is recorded as such in the experiment metadata.
    RECURRENT_CHUNK = 32
    if x_seq.shape[0] % RECURRENT_CHUNK != 0:
        raise ValueError(
            f"Sequence length {x_seq.shape[0]} must be divisible by "
            f"RECURRENT_CHUNK={RECURRENT_CHUNK}."
        )

    n_chunks = x_seq.shape[0] // RECURRENT_CHUNK
    chunks = x_seq.reshape(
        (n_chunks, RECURRENT_CHUNK) + x_seq.shape[1:]
    )

    def run_chunk(carry, xs):
        carry_out, ys = jax.lax.scan(
            step,
            carry,
            xs,
        )

        # Truncate the gradient at the 32-token boundary while preserving
        # the actual recurrent state values for the next forward chunk.
        carry_out = jax.tree_util.tree_map(
            jax.lax.stop_gradient,
            carry_out,
        )
        return carry_out, ys

    checkpointed_chunk = jax.checkpoint(run_chunk)

    final_state, chunk_outputs = jax.lax.scan(
        checkpointed_chunk,
        initial_state,
        chunks,
    )
    del final_state

    out = chunk_outputs.reshape(x_seq.shape)

    return out


# ============================================================================
# TRUE FACTORIZED DEEP SUPERVISION FORWARD
# ============================================================================

def factorized_memory_feedback_archive_layer_fwd_stateful(
    layer: Dict[str, Any],
    x_seq: jax.Array,
    initial_state: tuple | None = None,
) -> tuple[jax.Array, tuple]:
    """
    Exact MemoryFeedbackArchive layer computation, except:

        m_wq @ e -> m_wq_A @ (m_wq_B @ e)
        m_wk @ e -> m_wk_A @ (m_wk_B @ e)

    No dense Q/K reconstruction occurs.
    """
    r = layer["m_wk_A"].shape[0]

    def step(carry, e_raw):
        H_current, H_archive, s = carry

        e = layer_norm(
            e_raw,
            layer["pre_g"],
            layer["pre_b"],
        )

        # TRUE LOW-RANK Q/K COMPUTATION.
        k = normalize(
            layer["m_wk_A"] @ (layer["m_wk_B"] @ e)
        )
        q = normalize(
            layer["m_wq_A"] @ (layer["m_wq_B"] @ e)
        )

        val = jnp.tanh(layer["m_wv"] @ e)

        eta = jax.nn.sigmoid(
            (layer["m_w_eta"] @ e + layer["m_b_eta"])[0]
        )
        write = jax.nn.sigmoid(
            (layer["m_w_write"] @ e + layer["m_b_write"])[0]
        )
        retain = jax.nn.sigmoid(
            (layer["m_w_ret"] @ e + layer["m_b_ret"])[0]
        )

        old_current = H_current @ k
        H_current = (
            retain * H_current
            + (eta * write)
            * jnp.outer(val - old_current, k)
        )

        archive_write = jax.nn.sigmoid(
            (layer["m_w_archive_write"] @ e + layer["m_b_archive_write"])[0]
        )
        archive_retain = jax.nn.sigmoid(
            (layer["m_w_archive_ret"] @ e + layer["m_b_archive_ret"])[0]
        )

        old_archive = H_archive @ k
        H_archive = (
            archive_retain * H_archive
            + (eta * write * archive_write)
            * jnp.outer(val - old_archive, k)
        )

        read_gate = jax.nn.sigmoid(
            layer["m_w_read"] @ e + layer["m_b_read"]
        )

        current_context = layer_norm(
            H_current @ q,
            layer["m_ln_g"],
            layer["m_ln_b"],
        )
        archive_context = layer_norm(
            H_archive @ q,
            layer["m_ln_g"],
            layer["m_ln_b"],
        )

        archive_mix = jax.nn.sigmoid(
            layer["m_w_archive_mix"] @ e
            + layer["m_b_archive_mix"]
        )

        context = read_gate * (
            archive_mix * current_context
            + (1.0 - archive_mix) * archive_context
        )

        proposal = (
            layer["m_proj_w"]
            @ jnp.concatenate([e_raw, context])
            + layer["m_proj_b"]
        )

        out_gate = jax.nn.sigmoid(
            layer["m_w_out"] @ e + layer["m_b_out"]
        )

        modus_out = out_gate * proposal

        feedback_gate = jax.nn.sigmoid(
            (
                layer["s_w_memory_feedback"] @ e
                + layer["s_b_memory_feedback"]
            )[0]
        )

        compressed_memory = jnp.tanh(
            layer["s_memory_down"] @ context
        )
        memory_feedback = (
            layer["s_memory_up"] @ compressed_memory
        )

        e_vector = layer_norm(
            e_raw + feedback_gate * memory_feedback,
            layer["pre_g"],
            layer["pre_b"],
        )

        u = jnp.tanh(layer["s_wu"] @ e_vector)

        delta = jax.nn.sigmoid(
            layer["s_w_delta"] @ e_vector
            + layer["s_b_delta"]
        )

        ret_s = jax.nn.sigmoid(
            layer["s_w_ret"] @ e_vector
            + layer["s_b_ret"]
        )

        s = ret_s * s + delta * u

        c = jax.nn.sigmoid(
            layer["s_w_c"] @ e_vector
        )

        y_s = c * s

        gate_s = jax.nn.sigmoid(
            layer["s_w_gate"] @ e_vector
            + layer["s_b_gate"]
        )

        mamba_out = gate_s * (
            layer["s_proj_w"] @ y_s
            + layer["s_proj_b"]
        )

        r_hidden = jax.nn.gelu(
            layer["r_w"] @ e + layer["r_b"]
        )

        r_logits = (
            layer["r_proj"] @ r_hidden
            + layer["r_proj_b"]
        )

        router = jax.nn.sigmoid(
            r_logits[0]
            if layer["r_proj"].shape[0] == 1
            else r_logits
        )

        out = (
            router * modus_out
            + (1.0 - router) * mamba_out
        )

        return (H_current, H_archive, s), out

    if initial_state is None:
        initial_state = (
            jnp.zeros((r, r), dtype=jnp.float32),
            jnp.zeros((r, r), dtype=jnp.float32),
            jnp.zeros(
                layer["s_wu"].shape[0],
                dtype=jnp.float32,
            ),
        )

    # v5: one optimizer update is exactly one 32-token recurrent chunk.
    # There is no outer 512-token differentiated scan. The returned carry
    # is detached by the caller before the next optimizer update, giving a
    # true 32-token gradient horizon while preserving continuous forward
    # state values across updates.
    final_state, outputs = jax.lax.scan(
        step,
        initial_state,
        x_seq,
    )

    return outputs, final_state


# ============================================================================
# STATEFUL TRAINING FORWARD (v5)
# ============================================================================

def factorized_deep_supervision_fwd_stateful(
    p: Dict[str, Any],
    x_ids: jax.Array,
    cfg: ModelConfig,
    auxiliary_layers: tuple[int, ...] = (6,),
    dropout_key: jax.Array | None = None,
    dropout_rate: float = 0.0,
    recurrent_states: tuple | None = None,
):
    x = p["embed"][x_ids]

    if recurrent_states is None:
        recurrent_states = tuple(
            (
                jnp.zeros((p["layers"]["m_wk_A"][i].shape[0], p["layers"]["m_wk_A"][i].shape[0]), dtype=jnp.float32),
                jnp.zeros((p["layers"]["m_wk_A"][i].shape[0], p["layers"]["m_wk_A"][i].shape[0]), dtype=jnp.float32),
                jnp.zeros(p["layers"]["s_wu"][i].shape[0], dtype=jnp.float32),
            )
            for i in range(len(p["layers"]["m_wk_A"]))
        )

    new_states = []
    layer_outputs_list = []
    # p["layers"] is a dictionary of stacked parameter arrays; its length
    # is the number of parameter families, not the number of model layers.
    # Use a known layer-stacked parameter to determine the 12 layer count.
    for layer_index in range(len(p["layers"]["m_wk_A"])):
        layer = jax.tree_util.tree_map(
            lambda a, i=layer_index: a[i],
            p["layers"],
        )
        layer_out, new_state = jax.checkpoint(
            factorized_memory_feedback_archive_layer_fwd_stateful
        )(layer, x, recurrent_states[layer_index])
        x = x + layer_out
        new_states.append(new_state)
        layer_outputs_list.append(x)

    layer_outputs = jnp.stack(layer_outputs_list, axis=0)

    layer_indexes = jnp.array(
        [layer - 1 for layer in auxiliary_layers],
        dtype=jnp.int32,
    )

    selected_outputs = layer_outputs[layer_indexes]

    if dropout_key is not None and dropout_rate > 0.0:
        final_key, aux_key = jax.random.split(dropout_key)
        x_for_head = _apply_dropout(
            x,
            final_key,
            dropout_rate,
        )
        selected_for_heads = _apply_dropout(
            selected_outputs,
            aux_key,
            dropout_rate,
        )
    else:
        x_for_head = x
        selected_for_heads = selected_outputs

    final_logits = lm_head_fwd(
        p["head"],
        x_for_head,
    )

    auxiliary_logits = jax.vmap(
        lambda h: lm_head_fwd(p["head"], h)
    )(selected_for_heads)

    if "future_heads" in p:
        future_logits = jax.vmap(
            lambda head: lm_head_fwd(
                head,
                x_for_head,
            )
        )(p["future_heads"])

        auxiliary_future_logits = jax.vmap(
            lambda head: jax.vmap(
                lambda h: lm_head_fwd(head, h)
            )(selected_for_heads)
        )(p["future_heads"])

        return (
            final_logits,
            auxiliary_logits,
            future_logits,
            auxiliary_future_logits,
        ), tuple(new_states)

    return (final_logits, auxiliary_logits, None, None), tuple(new_states)


def _apply_dropout(
    x: jax.Array,
    key: jax.Array,
    rate: float,
) -> jax.Array:
    keep = 1.0 - rate
    mask = jax.random.bernoulli(
        key,
        keep,
        x.shape,
    )
    return jnp.where(mask, x / keep, 0.0)


# ============================================================================
# LOSS
# ============================================================================

def token_loss(
    logits: jax.Array,
    targets: jax.Array,
) -> jax.Array:
    logp = jax.nn.log_softmax(
        logits.astype(jnp.float32),
        axis=-1,
    )
    return -jnp.take_along_axis(
        logp,
        targets[..., None],
        axis=-1,
    )[..., 0]


def loss_fn(
    params,
    fwd_fn,
    x,
    y,
    auxiliary_weight: float,
    loss_tail: int,
    future_targets: tuple[int, ...],
    future_target_weight: float,
):
    # Training on TPU-v5e-1 uses batch=1. Avoid vmapping a recurrent
    # matrix-state model over a singleton batch: vmap makes the recurrent
    # carry explicitly batched and can force XLA to materialize
    # [time,batch,512,512] temporaries. The single-sequence path is
    # mathematically identical for batch=1.
    if x.shape[0] == 1:
        outputs_single = fwd_fn(params, x[0])
        if not isinstance(outputs_single, tuple) or len(outputs_single) != 4:
            raise RuntimeError("Expected four DeepSupervision outputs.")
        logits_s, auxiliary_s, future_s, auxiliary_future_s = outputs_single

        logits = logits_s[None, ...]
        auxiliary_logits = (
            None if auxiliary_s is None else auxiliary_s[None, ...]
        )
        future_logits = (
            None if future_s is None else future_s[None, ...]
        )
        auxiliary_future_logits = (
            None
            if auxiliary_future_s is None
            else auxiliary_future_s[None, ...]
        )
        outputs = (
            logits,
            auxiliary_logits,
            future_logits,
            auxiliary_future_logits,
        )
    else:
        outputs = jax.vmap(
            lambda sequence: fwd_fn(params, sequence)
        )(x)
        logits, auxiliary_logits, future_logits, auxiliary_future_logits = outputs

    if not isinstance(outputs, tuple) or len(outputs) != 4:
        raise RuntimeError(
            "Expected four DeepSupervision outputs."
        )

    full_y = y

    logits_tail = logits[:, -loss_tail:]
    y_tail = y[:, -loss_tail:]

    loss = token_loss(
        logits_tail,
        y_tail,
    ).mean()

    for head_index, offset in enumerate(future_targets):
        usable = future_logits[
            :,
            head_index,
            : -(offset - 1),
        ]
        target = full_y[:, offset - 1:]

        usable = usable[:, -loss_tail:]
        target = target[:, -loss_tail:]

        loss = loss + future_target_weight * token_loss(
            usable,
            target,
        ).mean()

    if auxiliary_logits is not None:
        if auxiliary_logits.ndim == logits.ndim + 1:
            aux = auxiliary_logits[:, :, -loss_tail:]
            aux_targets = y_tail[:, None, :, None]

            aux_logp = jax.nn.log_softmax(
                aux.astype(jnp.float32),
                axis=-1,
            )

            aux_losses = -jnp.take_along_axis(
                aux_logp,
                aux_targets,
                axis=-1,
            )[..., 0]

            aux_loss = aux_losses.mean()
        else:
            aux = auxiliary_logits[:, -loss_tail:]
            aux_loss = token_loss(
                aux,
                y_tail,
            ).mean()

        loss = loss + auxiliary_weight * aux_loss

        if auxiliary_future_logits is not None:
            aux_future_total = 0.0

            for head_index, offset in enumerate(future_targets):
                usable = auxiliary_future_logits[
                    :,
                    head_index,
                    :,
                    : -(offset - 1),
                ]
                target = full_y[:, offset - 1:]

                usable = usable[:, :, -loss_tail:]
                target = target[:, -loss_tail:]

                aux_future_logp = jax.nn.log_softmax(
                    usable.astype(jnp.float32),
                    axis=-1,
                )

                aux_future_losses = -jnp.take_along_axis(
                    aux_future_logp,
                    target[:, None, :, None],
                    axis=-1,
                )[..., 0]

                aux_future_total = (
                    aux_future_total
                    + aux_future_losses.mean()
                )

            loss = (
                loss
                + future_target_weight
                * aux_future_total
            )

    return loss


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(
    params,
    fwd_fn,
    data,
    seq_len,
    chunks,
    batch_size,
    batch_sharding,
    loss_tail,
):
    @jax.jit
    def eval_batch(x, y):
        return loss_fn(
            params,
            fwd_fn,
            x,
            y,
            0.0,
            loss_tail,
            (),
            0.0,
        )

    losses = []

    max_start = len(data) - seq_len - 1
    starts = np.linspace(
        0,
        max_start,
        chunks,
        dtype=np.int64,
    )

    for offset in range(0, chunks, batch_size):
        selected = starts[
            offset : offset + batch_size
        ]

        if len(selected) < batch_size:
            selected = np.pad(
                selected,
                (0, batch_size - len(selected)),
                mode="edge",
            )

        x, y = batch_at(
            data,
            selected,
            seq_len,
        )

        value = eval_batch(
            jax.device_put(
                x,
                batch_sharding,
            ),
            jax.device_put(
                y,
                batch_sharding,
            ),
        )

        losses.append(float(value))

    return float(
        np.mean(losses) / math.log(2.0)
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    args = parse_args()

    candidate = CANDIDATES[args.candidate]

    print("=" * 80)
    print("MODUS_X PB1-C — FINAL LOW-RANK Q/K TRAINING")
    print("=" * 80)
    print(f"Project root                 : {PROJECT_ROOT}")
    print(f"Language dir                 : {LANGUAGE_DIR}")
    print(f"Candidate                    : {args.candidate}")
    print(f"Hidden dimension             : {candidate['hidden_dim']}")
    print(f"Mamba state dimension        : {candidate['mamba_state_dim']}")
    print(f"ax_res                       : {candidate['ax_res']}")
    print(f"Q rank                       : {candidate['q_rank']}")
    print(f"K rank                       : {candidate['k_rank']}")
    print(f"Expected parameters          : {candidate['expected_params']:,}")
    print(f"Canonical budget             : {CANONICAL_PARAMS:,}")
    print(f"Batch                        : {args.batch}")
    print(f"Sequence length              : {args.input_seq_len}")
    print(f"Target characters            : {args.target_chars:,}")
    print(f"Validation chunks            : {args.eval_chunks}")
    print(f"Seed                         : {args.seed}")
    print("Q/K computation              : TRUE A @ (B @ e)")
    print("Training protocol            : 32-token truncated BPTT; continuous forward state")
    print("Owner files modified         : NO")
    print()

    if jax.default_backend() != "tpu":
        raise RuntimeError(
            f"Expected TPU backend, found {jax.default_backend()}. "
            "PB1-C is the TPU training screen."
        )

    if args.batch % jax.device_count():
        raise ValueError(
            "Global batch must divide TPU device count."
        )

    if args.eval_batch % jax.device_count():
        raise ValueError(
            "Evaluation batch must divide TPU device count."
        )

    if args.input_seq_len != CANONICAL["seq_len"]:
        raise ValueError(
            "PB1-C requires sequence length 512."
        )

    if args.loss_tail != args.input_seq_len:
        raise ValueError(
            "PB1-C requires loss_tail == input sequence length."
        )

    if args.target_chars <= 0:
        raise ValueError("--target-chars must be positive.")

    if args.eval_chunks <= 0:
        raise ValueError("--eval-chunks must be positive.")

    if args.eval_batch <= 0:
        raise ValueError("--eval-batch must be positive.")

    outdir = Path(args.outdir)
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = outdir / "checkpoint.pkl"
    progress_path = outdir / "progress.json"
    config_path = outdir / "config.json"

    # ------------------------------------------------------------------------
    # MODEL INITIALIZATION
    # ------------------------------------------------------------------------

    cfg = build_config(candidate)

    dense_params, _official_fwd = make_model(
        MODEL_NAME,
        jax.random.key(args.seed),
        cfg,
        auxiliary_layers=CANONICAL["aux_layers"],
        future_target_count=CANONICAL["future_target_count"],
        dropout_rate=0.0,
    )

    dense_count = int(count_params(dense_params))

    print("=" * 80)
    print("DENSE DONOR INITIALIZATION")
    print("=" * 80)
    print(f"Dense donor parameters       : {dense_count:,}")

    factorized_params, factor_stats = convert_dense_to_factorized(
        dense_params,
        candidate["q_rank"],
        candidate["k_rank"],
    )

    measured_count = int(
        count_params(factorized_params)
    )

    expected_count = candidate["expected_params"]

    print(f"Factorized parameters        : {measured_count:,}")
    print(f"PB1-A expected parameters    : {expected_count:,}")
    print(f"Budget error                 : {measured_count - CANONICAL_PARAMS:+,}")
    print(f"Q/K savings                  : {factor_stats['qk_savings']:,}")

    if measured_count != expected_count:
        raise RuntimeError(
            f"PB1-A count mismatch for {args.candidate}: "
            f"expected {expected_count:,}, "
            f"found {measured_count:,}."
        )

    if abs(measured_count - CANONICAL_PARAMS) > 10_000:
        raise RuntimeError(
            "Candidate is unexpectedly far from the fixed budget."
        )

    print("Factorized parameter count   : PASS")

    # ------------------------------------------------------------------------
    # TRUE FACTORIZED FORWARD
    # ------------------------------------------------------------------------

    def fwd_fn(p, x):
        outputs, _ = factorized_deep_supervision_fwd_stateful(
            p,
            x,
            cfg,
            tuple(CANONICAL["aux_layers"]),
            None,
            0.0,
            None,
        )
        return outputs

    # Smoke-test the exact training forward before placing on TPU.
    smoke_rng = np.random.default_rng(
        10_000 + args.seed
    )
    smoke_x = jnp.asarray(
        smoke_rng.integers(
            0,
            CANONICAL["vocab_size"],
            size=(args.input_seq_len,),
            dtype=np.int32,
        )
    )

    smoke_outputs = fwd_fn(
        factorized_params,
        smoke_x,
    )

    if not isinstance(smoke_outputs, tuple) or len(smoke_outputs) != 4:
        raise RuntimeError(
            "Factorized training forward did not return four outputs."
        )

    for output in smoke_outputs:
        if not np.all(np.isfinite(np.asarray(output))):
            raise RuntimeError(
                "Non-finite value found in factorized smoke forward."
            )

    print("True factorized forward      : PASS")
    print("DeepSupervision outputs      : PASS")

    # ------------------------------------------------------------------------
    # OPTIMIZER
    # ------------------------------------------------------------------------

    if args.batch != 1:
        raise ValueError("PB1-C v5 requires --batch 1 because recurrent state is carried between optimizer updates.")
    if args.train_chunk_len != 32:
        raise ValueError("PB1-C v5 requires --train-chunk-len 32 for the declared 32-token BPTT protocol.")

    chars_per_step = args.train_chunk_len

    total_steps = math.ceil(
        args.target_chars / chars_per_step
    )

    stop_steps = (
        total_steps
        if args.stop_chars is None
        else min(
            total_steps,
            math.ceil(
                args.stop_chars / chars_per_step
            ),
        )
    )

    if stop_steps <= 0:
        raise ValueError(
            "--stop-chars must produce at least one step."
        )

    checkpoint_steps = max(
        1,
        round(
            args.checkpoint_chars
            / chars_per_step
        ),
    )

    if args.schedule == "warmup_cosine":
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=args.lr * 0.05,
            peak_value=args.lr,
            warmup_steps=args.warmup_steps,
            decay_steps=total_steps,
            end_value=args.lr * args.end_lr_ratio,
        )
    else:
        schedule = args.lr

    optimizer = optax.adamw(
        schedule,
        weight_decay=args.weight_decay,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optimizer,
    )

    opt_state = tx.init(
        factorized_params
    )

    # ------------------------------------------------------------------------
    # TPU SHARDING
    # ------------------------------------------------------------------------

    mesh = Mesh(
        np.array(
            jax.devices(),
            dtype=object,
        ),
        ("data",),
    )

    batch_sharding = NamedSharding(
        mesh,
        P("data", None),
    )

    replicated = NamedSharding(
        mesh,
        P(),
    )

    factorized_params = jax.device_put(
        factorized_params,
        replicated,
    )

    opt_state = jax.device_put(
        opt_state,
        replicated,
    )

    recurrent_states = tuple(
        (
            jnp.zeros((factorized_params["layers"]["m_wk_A"][i].shape[0], factorized_params["layers"]["m_wk_A"][i].shape[0]), dtype=jnp.float32),
            jnp.zeros((factorized_params["layers"]["m_wk_A"][i].shape[0], factorized_params["layers"]["m_wk_A"][i].shape[0]), dtype=jnp.float32),
            jnp.zeros(factorized_params["layers"]["s_wu"][i].shape[0], dtype=jnp.float32),
        )
        for i in range(len(factorized_params["layers"]["m_wk_A"]))
    )
    recurrent_states = jax.device_put(recurrent_states, replicated)
    if "resume_recurrent_states" in locals() and resume_recurrent_states is not None:
        recurrent_states = jax.device_put(resume_recurrent_states, replicated)

    # ------------------------------------------------------------------------
    # TRAINING UPDATE
    # ------------------------------------------------------------------------

    def stateful_loss_from_outputs(outputs, y):
        logits, auxiliary_logits, future_logits, auxiliary_future_logits = outputs
        logits_tail = logits[-min(args.train_chunk_len, logits.shape[0]):]
        y_tail = y[-min(args.train_chunk_len, y.shape[0]):]
        loss = token_loss(logits_tail, y_tail).mean()

        for head_index, offset in enumerate((2,)):
            usable = future_logits[head_index, : -(offset - 1)]
            target = y[offset - 1:]
            usable = usable[-min(args.train_chunk_len, usable.shape[0]):]
            target = target[-min(args.train_chunk_len, target.shape[0]):]
            loss = loss + args.future_target_weight * token_loss(usable, target).mean()

        if auxiliary_logits is not None:
            if auxiliary_logits.ndim == logits.ndim + 1:
                aux = auxiliary_logits[:, -min(args.train_chunk_len, logits.shape[0]):]
                aux_targets = y_tail[None, :, None]
                aux_logp = jax.nn.log_softmax(aux.astype(jnp.float32), axis=-1)
                aux_losses = -jnp.take_along_axis(aux_logp, aux_targets, axis=-1)[..., 0]
                aux_loss = aux_losses.mean()
            else:
                aux = auxiliary_logits[-min(args.train_chunk_len, logits.shape[0]):]
                aux_loss = token_loss(aux, y_tail).mean()
            loss = loss + args.auxiliary_weight * aux_loss

            if auxiliary_future_logits is not None:
                aux_future_total = 0.0
                for head_index, offset in enumerate((2,)):
                    usable = auxiliary_future_logits[head_index, :, : -(offset - 1)]
                    target = y[offset - 1:]
                    usable = usable[:, -min(args.train_chunk_len, usable.shape[1]):]
                    target = target[-min(args.train_chunk_len, target.shape[0]):]
                    aux_future_logp = jax.nn.log_softmax(usable.astype(jnp.float32), axis=-1)
                    aux_future_losses = -jnp.take_along_axis(
                        aux_future_logp, target[None, :, None], axis=-1
                    )[..., 0]
                    aux_future_total = aux_future_total + aux_future_losses.mean()
                loss = loss + args.future_target_weight * aux_future_total

        return loss

    @jax.jit
    def update(params, state, recurrent_states, x, y):
        def objective(pp):
            outputs, new_states = factorized_deep_supervision_fwd_stateful(
                pp,
                x,
                cfg,
                tuple(CANONICAL["aux_layers"]),
                None,
                0.0,
                recurrent_states,
            )
            return stateful_loss_from_outputs(outputs, y), new_states

        (loss, new_states), grads = jax.value_and_grad(
            objective,
            has_aux=True,
        )(params)

        # The carry is returned as auxiliary output but is not differentiated
        # across optimizer updates. This is the actual 32-token BPTT boundary.
        new_states = jax.tree_util.tree_map(
            jax.lax.stop_gradient,
            new_states,
        )

        updates, state = tx.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        return params, state, new_states, loss


    # ------------------------------------------------------------------------
    # DATA
    # ------------------------------------------------------------------------

    raw = np.memmap(
        args.data_path,
        dtype=np.uint8,
        mode="r",
    )

    train, valid, test = load_split(raw)

    rng = np.random.default_rng(
        1000 + args.seed
    )

    rows = []
    start_step = 0
    elapsed_before = 0.0
    stream_path = outdir / "stream_state.json"
    if stream_path.exists():
        stream_state = json.loads(stream_path.read_text(encoding="utf-8"))
    else:
        stream_state = {}

    if args.resume and checkpoint_path.exists():
        print("=" * 80)
        print("RESUMING CHECKPOINT")
        print("=" * 80)

        with checkpoint_path.open("rb") as f:
            state = pickle.load(f)

        factorized_params = jax.device_put(
            state["params"],
            replicated,
        )
        opt_state = jax.device_put(
            state["opt_state"],
            replicated,
        )
        resume_recurrent_states = state.get("recurrent_states")

        rng.bit_generator.state = state[
            "rng_state"
        ]

        rows = state["rows"]
        start_step = state["step"]
        elapsed_before = state["elapsed_s"]

        print(
            f"RESUME step={start_step:,}"
        )

    # ------------------------------------------------------------------------
    # CONFIG OUTPUT
    # ------------------------------------------------------------------------

    config = {
        "experiment": "PB1-C final low-rank Q/K training",
        "model": MODEL_NAME,
        "candidate": args.candidate,
        "candidate_config": candidate,
        "canonical_budget": CANONICAL_PARAMS,
        "measured_parameters": measured_count,
        "dense_donor_parameters": dense_count,
        "qk_savings": factor_stats["qk_savings"],
        "q_rank": candidate["q_rank"],
        "k_rank": candidate["k_rank"],
        "batch": args.batch,
        "seq_len": args.input_seq_len,
        "loss_tail": args.loss_tail,
        "train_seq_len": args.train_chunk_len,
        "chars_per_step": chars_per_step,
        "target_chars": args.target_chars,
        "total_steps": total_steps,
        "stop_steps": stop_steps,
        "checkpoint_steps": checkpoint_steps,
        "eval_chunks": args.eval_chunks,
        "eval_batch": args.eval_batch,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "auxiliary_weight": args.auxiliary_weight,
        "future_target_weight": args.future_target_weight,
        "future_targets": [2],
        "optimizer": "adamw",
        "global_grad_clip": 1.0,
        "auxiliary_layers": list(
            CANONICAL["aux_layers"]
        ),
        "future_target_count": CANONICAL[
            "future_target_count"
        ],
        "seed": args.seed,
        "devices": [
            str(d)
            for d in jax.devices()
        ],
        "true_low_rank_matmuls": True,
        "recurrent_chunk": 32,
        "gradient_horizon": 32,
        "training_protocol": "32-token-truncated-BPTT-continuous-forward-state",
        "batch1_direct_forward": True,
        "train_chunk_len": args.train_chunk_len,
        "full_sequence_backward": False,
        "state_carry_across_updates": True,
    }

    atomic_json(
        config_path,
        config,
    )

    print("=" * 80)
    print("PB1-C CONFIG")
    print("=" * 80)
    print(
        json.dumps(
            config,
            indent=2,
        )
    )

    # ------------------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------------------

    started = time.perf_counter()

    for step in range(
        start_step + 1,
        stop_steps + 1,
    ):
        # One contiguous training stream is selected once and then advanced
        # by 32 tokens per optimizer update. The recurrent carry therefore
        # remains continuous in the forward pass while the gradient is
        # truncated at every optimizer-step boundary.
        if step == start_step + 1:
            stream_start = int(
                rng.integers(
                    0,
                    len(train) - max(args.train_chunk_len * total_steps + 1, 2),
                )
            )
            atomic_json(
                stream_path,
                {"stream_start": stream_start},
            )
        else:
            stream_start = stream_state["stream_start"]

        pos = stream_start + (step - start_step - 1) * args.train_chunk_len
        chunk = train[pos : pos + args.train_chunk_len + 1]
        x = jax.device_put(chunk[:-1][None, :].astype(np.int32), batch_sharding)
        y = jax.device_put(chunk[1:][None, :].astype(np.int32), batch_sharding)

        factorized_params, opt_state, recurrent_states, loss = update(
            factorized_params,
            opt_state,
            recurrent_states,
            x[0],
            y[0],
        )

        if (
            step == start_step + 1
            or step % 10 == 0
        ):
            block_tree(loss)

            elapsed = (
                elapsed_before
                + time.perf_counter()
                - started
            )

            print(
                f"PROGRESS step={step:,}/{total_steps:,} "
                f"loss={float(loss):.4f} "
                f"chars_s="
                f"{(step-start_step)*chars_per_step/max(elapsed-elapsed_before,1e-9):.0f}",
                flush=True,
            )

        if (
            step % checkpoint_steps == 0
            or step == stop_steps
        ):
            block_tree(
                (
                    factorized_params,
                    opt_state,
                )
            )

            elapsed = (
                elapsed_before
                + time.perf_counter()
                - started
            )

            val_bpc = evaluate(
                factorized_params,
                fwd_fn,
                valid,
                args.input_seq_len,
                args.eval_chunks,
                args.eval_batch,
                batch_sharding,
                args.loss_tail,
            )

            processed = (
                step * chars_per_step
            )

            reference = GPU_REFERENCES.get(
                processed
            )

            row = {
                "step": step,
                "processed_characters": processed,
                "loss": float(loss),
                "val_bpc": val_bpc,
                "gpu_reference_bpc": reference,
                "delta_to_gpu": (
                    None
                    if reference is None
                    else val_bpc - reference
                ),
                "elapsed_s": elapsed,
            }

            rows.append(row)

            atomic_json(
                progress_path,
                {
                    "config": config,
                    "rows": rows,
                },
            )

            atomic_pickle(
                checkpoint_path,
                {
                    "step": step,
                    "params": jax.device_get(
                        factorized_params
                    ),
                    "opt_state": jax.device_get(
                        opt_state
                    ),
                    "recurrent_states": jax.device_get(recurrent_states),
                    "rng_state": rng.bit_generator.state,
                    "rows": rows,
                    "elapsed_s": elapsed,
                },
            )

            print(
                "CHECKPOINT "
                + json.dumps(row),
                flush=True,
            )

            if not math.isfinite(val_bpc):
                raise RuntimeError(
                    "Non-finite validation BPC."
                )

    # ------------------------------------------------------------------------
    # FINAL TEST
    # ------------------------------------------------------------------------

    if stop_steps == total_steps:
        test_bpc = evaluate(
            factorized_params,
            fwd_fn,
            test,
            args.input_seq_len,
            args.eval_chunks,
            args.eval_batch,
            batch_sharding,
            args.loss_tail,
        )

        final = {
            "candidate": args.candidate,
            "processed_characters": (
                stop_steps * chars_per_step
            ),
            "validation_bpc": (
                rows[-1]["val_bpc"]
                if rows
                else None
            ),
            "test_bpc": test_bpc,
            "parameters": measured_count,
            "budget_error": (
                measured_count
                - CANONICAL_PARAMS
            ),
            "status": "COMPLETE",
        }

        atomic_json(
            outdir / "final_result.json",
            final,
        )

        print("=" * 80)
        print("PB1-C FINAL RESULT")
        print("=" * 80)
        print(
            f"Candidate                    : {args.candidate}"
        )
        print(
            f"Parameters                   : {measured_count:,}"
        )
        print(
            f"Budget error                 : "
            f"{measured_count-CANONICAL_PARAMS:+,}"
        )
        print(
            f"Validation BPC              : "
            f"{final['validation_bpc']:.6f}"
        )
        print(
            f"Test BPC                    : "
            f"{test_bpc:.6f}"
        )
        print(
            "Training status              : COMPLETE"
        )
    else:
        print("=" * 80)
        print("PB1-C SCREEN STOPPED")
        print("=" * 80)
        print(
            f"Stopped after {stop_steps:,} steps."
        )
        print(
            "Checkpoint retained for resume."
        )


if __name__ == "__main__":
    main()