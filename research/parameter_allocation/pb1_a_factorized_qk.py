from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import random


# ---------------------------------------------------------------------------
# Repository import
# ---------------------------------------------------------------------------

THIS_FILE = os.path.abspath(__file__)
RESEARCH_DIR = os.path.dirname(THIS_FILE)
REPO_ROOT = os.path.abspath(os.path.join(RESEARCH_DIR, "..", ".."))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from language.models import (  # noqa: E402
    ModelConfig,
    count_params,
    init_modus_x_memory_feedback_archive_lm,
    modus_x_memory_feedback_archive_lm_fwd,
    modus_x_memory_feedback_archive_lm_fwd_deep_supervision,
)


# ---------------------------------------------------------------------------
# Canonical experiment definition
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256
EMBED_DIM = 512
N_LAYERS = 12
N_HEADS = 8
SEQ_LEN = 512
ROUTER_HIDDEN = 32

AUX_LAYERS = (6,)
FUTURE_TARGET_COUNT = 1
DROPOUT = 0.0


@dataclass(frozen=True)
class Candidate:
    name: str
    q_rank: int
    k_rank: int
    hidden_dim: int
    mamba_state_dim: int
    ax_res: int


CANDIDATES = (
    Candidate(
        name="C1_q128_k128",
        q_rank=128,
        k_rank=128,
        hidden_dim=2016,
        mamba_state_dim=512,
        ax_res=576,
    ),
    Candidate(
        name="C2_q64_k64",
        q_rank=64,
        k_rank=64,
        hidden_dim=2272,
        mamba_state_dim=512,
        ax_res=608,
    ),
    Candidate(
        name="C3_q128_k64",
        q_rank=128,
        k_rank=64,
        hidden_dim=1760,
        mamba_state_dim=512,
        ax_res=608,
    ),
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def make_config(candidate: Candidate) -> ModelConfig:
    return ModelConfig(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        hidden_dim=candidate.hidden_dim,
        ax_res=candidate.ax_res,
        n_layers=N_LAYERS,
        n_heads_attn=N_HEADS,
        seq_len=SEQ_LEN,
        mamba_state_dim=candidate.mamba_state_dim,
        vector_router=True,
        router_hidden=ROUTER_HIDDEN,
    )


# ---------------------------------------------------------------------------
# Factorization utilities
# ---------------------------------------------------------------------------

def truncated_svd_factor(
    weight: jax.Array,
    rank: int,
) -> tuple[jax.Array, jax.Array]:
    """
    Factor W ~= A @ B using truncated SVD.

    W shape:
        [out_dim, in_dim]

    A shape:
        [out_dim, rank]

    B shape:
        [rank, in_dim]

    We split singular values symmetrically:
        A = U_r sqrt(S_r)
        B = sqrt(S_r) V_r^T
    """

    out_dim, in_dim = weight.shape

    if rank <= 0:
        raise ValueError(f"Rank must be positive, got {rank}")

    rank = min(rank, out_dim, in_dim)

    u, s, vh = jnp.linalg.svd(
        weight,
        full_matrices=False,
    )

    u_r = u[:, :rank]
    s_r = s[:rank]
    vh_r = vh[:rank, :]

    sqrt_s = jnp.sqrt(jnp.maximum(s_r, 0.0))

    a = u_r * sqrt_s[None, :]
    b = sqrt_s[:, None] * vh_r

    return a, b


def factorized_param_count(
    ax_res: int,
    embed_dim: int,
    q_rank: int,
    k_rank: int,
) -> int:
    """
    Parameter count for replacing dense Q/K with low-rank factors.
    """

    return (
        ax_res * q_rank
        + q_rank * embed_dim
        + ax_res * k_rank
        + k_rank * embed_dim
    )


def dense_qk_param_count(
    ax_res: int,
    embed_dim: int,
) -> int:
    return 2 * ax_res * embed_dim


# ---------------------------------------------------------------------------
# Factorized layer forward
# ---------------------------------------------------------------------------

def normalize(x: jax.Array) -> jax.Array:
    return x / (jnp.linalg.norm(x) + 1e-8)


def factorized_q_forward(
    layer: dict[str, jax.Array],
    e: jax.Array,
) -> jax.Array:
    """
    Factorized replacement for:

        normalize(layer["m_wq"] @ e)
    """

    return normalize(
        layer["m_wq_a"] @ (layer["m_wq_b"] @ e)
    )


def factorized_k_forward(
    layer: dict[str, jax.Array],
    e: jax.Array,
) -> jax.Array:
    """
    Factorized replacement for:

        normalize(layer["m_wk"] @ e)
    """

    return normalize(
        layer["m_wk_a"] @ (layer["m_wk_b"] @ e)
    )


# ---------------------------------------------------------------------------
# Convert dense layer -> factorized layer
# ---------------------------------------------------------------------------

def factorize_layer(
    layer: dict[str, jax.Array],
    q_rank: int,
    k_rank: int,
) -> dict[str, jax.Array]:
    """
    Convert one dense Modus_X MemoryFeedbackArchive layer into a
    Q/K-factorized layer.

    Every parameter except m_wq and m_wk is preserved exactly.

    The original dense matrices are removed.
    """

    q_a, q_b = truncated_svd_factor(
        layer["m_wq"],
        q_rank,
    )

    k_a, k_b = truncated_svd_factor(
        layer["m_wk"],
        k_rank,
    )

    factorized = dict(layer)

    factorized.pop("m_wq")
    factorized.pop("m_wk")

    factorized["m_wq_a"] = q_a
    factorized["m_wq_b"] = q_b

    factorized["m_wk_a"] = k_a
    factorized["m_wk_b"] = k_b

    return factorized


def stack_layers(
    layers: list[dict[str, jax.Array]],
) -> dict[str, jax.Array]:
    """
    Stack a list of layer dictionaries in exactly the same style as
    language.models initialization.
    """

    return jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs),
        *layers,
    )


# ---------------------------------------------------------------------------
# Build factorized model
# ---------------------------------------------------------------------------

def build_factorized_from_dense(
    key: jax.Array,
    cfg: ModelConfig,
    q_rank: int,
    k_rank: int,
) -> tuple[dict, dict]:
    """
    Create a dense MemoryFeedbackArchive model using the repository's
    canonical initializer, then replace only Q/K with factorized forms.

    Returns:
        dense_params
        factorized_params
    """

    dense_params = init_modus_x_memory_feedback_archive_lm(
        key,
        cfg,
    )

    dense_layers = jax.tree_util.tree_map(
        lambda x: x,
        dense_params["layers"],
    )

    factorized_layers = []

    for layer_index in range(cfg.n_layers):
        layer = jax.tree_util.tree_map(
            lambda x: x[layer_index],
            dense_layers,
        )

        factorized_layer = factorize_layer(
            layer,
            q_rank=q_rank,
            k_rank=k_rank,
        )

        factorized_layers.append(factorized_layer)

    factorized_params = {
        "embed": dense_params["embed"],
        "layers": stack_layers(factorized_layers),
        "head": dense_params["head"],
    }

    return dense_params, factorized_params


# ---------------------------------------------------------------------------
# Factorized MemoryFeedbackArchive forward
# ---------------------------------------------------------------------------

def layer_norm(
    x: jax.Array,
    g: jax.Array,
    b: jax.Array,
) -> jax.Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)

    return (
        g
        * (x - mean)
        / jnp.sqrt(var + 1e-5)
        + b
    )


def factorized_memory_feedback_layer_fwd(
    layer: dict[str, jax.Array],
    x_seq: jax.Array,
) -> jax.Array:

    r = layer["m_wq_a"].shape[0]

    def step(carry, e_raw):
        H_current, H_archive, s = carry

        e = layer_norm(
            e_raw,
            layer["pre_g"],
            layer["pre_b"],
        )

        # ---------------------------------------------------------------
        # Matrix memory stream
        # ---------------------------------------------------------------

        k = factorized_k_forward(
            layer,
            e,
        )

        q = factorized_q_forward(
            layer,
            e,
        )

        val = jnp.tanh(
            layer["m_wv"] @ e
        )

        eta = jax.nn.sigmoid(
            (
                layer["m_w_eta"] @ e
                + layer["m_b_eta"]
            )[0]
        )

        write = jax.nn.sigmoid(
            (
                layer["m_w_write"] @ e
                + layer["m_b_write"]
            )[0]
        )

        retain = jax.nn.sigmoid(
            (
                layer["m_w_ret"] @ e
                + layer["m_b_ret"]
            )[0]
        )

        old_current = H_current @ k

        H_current = (
            retain * H_current
            + (eta * write)
            * jnp.outer(
                val - old_current,
                k,
            )
        )

        archive_write = jax.nn.sigmoid(
            (
                layer["m_w_archive_write"] @ e
                + layer["m_b_archive_write"]
            )[0]
        )

        archive_retain = jax.nn.sigmoid(
            (
                layer["m_w_archive_ret"] @ e
                + layer["m_b_archive_ret"]
            )[0]
        )

        old_archive = H_archive @ k

        H_archive = (
            archive_retain * H_archive
            + (
                eta
                * write
                * archive_write
            )
            * jnp.outer(
                val - old_archive,
                k,
            )
        )

        read_gate = jax.nn.sigmoid(
            layer["m_w_read"] @ e
            + layer["m_b_read"]
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
            + (1.0 - archive_mix)
            * archive_context
        )

        proposal = (
            layer["m_proj_w"]
            @ jnp.concatenate(
                [e_raw, context]
            )
            + layer["m_proj_b"]
        )

        out_gate = jax.nn.sigmoid(
            layer["m_w_out"] @ e
            + layer["m_b_out"]
        )

        modus_out = out_gate * proposal

        # ---------------------------------------------------------------
        # Memory -> vector feedback
        # ---------------------------------------------------------------

        feedback_gate = jax.nn.sigmoid(
            (
                layer["s_w_memory_feedback"] @ e
                + layer["s_b_memory_feedback"]
            )[0]
        )

        compressed_memory = jnp.tanh(
            layer["s_memory_down"]
            @ context
        )

        memory_feedback = (
            layer["s_memory_up"]
            @ compressed_memory
        )

        e_vector = layer_norm(
            e_raw
            + feedback_gate * memory_feedback,
            layer["pre_g"],
            layer["pre_b"],
        )

        # ---------------------------------------------------------------
        # Mamba/vector stream
        # ---------------------------------------------------------------

        u = jnp.tanh(
            layer["s_wu"] @ e_vector
        )

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

        # ---------------------------------------------------------------
        # Router
        # ---------------------------------------------------------------

        r_hidden = jax.nn.gelu(
            layer["r_w"] @ e
            + layer["r_b"]
        )

        r_logits = (
            layer["r_proj"] @ r_hidden
            + layer["r_proj_b"]
        )

        router = jax.nn.sigmoid(
            r_logits
        )

        out = (
            router * modus_out
            + (1.0 - router)
            * mamba_out
        )

        return (
            H_current,
            H_archive,
            s,
        ), out

    initial_state = (
        jnp.zeros(
            (r, r),
            dtype=x_seq.dtype,
        ),
        jnp.zeros(
            (r, r),
            dtype=x_seq.dtype,
        ),
        jnp.zeros(
            layer["s_wu"].shape[0],
            dtype=x_seq.dtype,
        ),
    )

    _, out = jax.lax.scan(
        step,
        initial_state,
        x_seq,
    )

    return out


def factorized_lm_fwd(
    params: dict,
    x_ids: jax.Array,
    cfg: ModelConfig,
) -> jax.Array:

    x = params["embed"][x_ids]

    def scan_layer(x_in, layer):
        layer_out = factorized_memory_feedback_layer_fwd(
            layer,
            x_in,
        )

        return x_in + layer_out, None

    x, _ = jax.lax.scan(
        scan_layer,
        x,
        params["layers"],
    )

    h = jax.nn.gelu(
        x @ params["head"]["w1"].T
        + params["head"]["b1"]
    )

    return (
        h @ params["head"]["w2"].T
        + params["head"]["b2"]
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_factorized_shapes(
    params: dict,
    cfg: ModelConfig,
    q_rank: int,
    k_rank: int,
) -> None:

    layers = params["layers"]

    assert layers["m_wq_a"].shape == (
        cfg.n_layers,
        cfg.ax_res,
        q_rank,
    )

    assert layers["m_wq_b"].shape == (
        cfg.n_layers,
        q_rank,
        cfg.embed_dim,
    )

    assert layers["m_wk_a"].shape == (
        cfg.n_layers,
        cfg.ax_res,
        k_rank,
    )

    assert layers["m_wk_b"].shape == (
        cfg.n_layers,
        k_rank,
        cfg.embed_dim,
    )

    assert "m_wq" not in layers
    assert "m_wk" not in layers


def test_forward_runs(
    params: dict,
    cfg: ModelConfig,
    seed: int,
) -> float:

    key = random.PRNGKey(seed)

    x_ids = random.randint(
        key,
        (cfg.seq_len,),
        0,
        cfg.vocab_size,
    )

    logits = factorized_lm_fwd(
        params,
        x_ids,
        cfg,
    )

    expected_shape = (
        cfg.seq_len,
        cfg.vocab_size,
    )

    assert logits.shape == expected_shape, (
        f"Unexpected logits shape: "
        f"{logits.shape} != {expected_shape}"
    )

    logits.block_until_ready()

    return float(
        jnp.linalg.norm(logits)
    )


def measure_qk_compression(
    cfg: ModelConfig,
    q_rank: int,
    k_rank: int,
) -> dict[str, Any]:

    dense_count = dense_qk_param_count(
        cfg.ax_res,
        cfg.embed_dim,
    )

    factorized_count = factorized_param_count(
        cfg.ax_res,
        cfg.embed_dim,
        q_rank,
        k_rank,
    )

    savings = dense_count - factorized_count

    return {
        "dense_qk_params": dense_count,
        "factorized_qk_params": factorized_count,
        "qk_parameter_savings": savings,
        "qk_retained_fraction": (
            factorized_count / dense_count
        ),
    }


# ---------------------------------------------------------------------------
# Candidate audit
# ---------------------------------------------------------------------------

def run_candidate(
    candidate: Candidate,
    seed: int,
) -> dict[str, Any]:

    print()
    print("=" * 80)
    print(candidate.name)
    print("=" * 80)

    cfg = make_config(candidate)

    print("Configuration:")
    print(f"  hidden_dim       = {cfg.hidden_dim}")
    print(f"  mamba_state_dim  = {cfg.mamba_state_dim}")
    print(f"  ax_res           = {cfg.ax_res}")
    print(f"  Q rank           = {candidate.q_rank}")
    print(f"  K rank           = {candidate.k_rank}")

    key = random.PRNGKey(seed)

    dense_params, factorized_params = (
        build_factorized_from_dense(
            key,
            cfg,
            candidate.q_rank,
            candidate.k_rank,
        )
    )

    dense_count = count_params(
        dense_params
    )

    factorized_count = count_params(
        factorized_params
    )

    qk_stats = measure_qk_compression(
        cfg,
        candidate.q_rank,
        candidate.k_rank,
    )

    print()
    print("Parameter audit:")
    print(
        f"  Dense parameter count       : "
        f"{dense_count:,}"
    )

    print(
        f"  Factorized parameter count  : "
        f"{factorized_count:,}"
    )

    print(
        f"  Q/K parameter savings       : "
        f"{qk_stats['qk_parameter_savings']:,}"
    )

    print(
        f"  Q/K retained fraction       : "
        f"{qk_stats['qk_retained_fraction']:.6f}"
    )

    test_factorized_shapes(
        factorized_params,
        cfg,
        candidate.q_rank,
        candidate.k_rank,
    )

    print()
    print("Shape test: PASS")

    logits_norm = test_forward_runs(
        factorized_params,
        cfg,
        seed + 1000,
    )

    print(
        f"Forward test: PASS "
        f"(logits L2 = {logits_norm:.6f})"
    )

    return {
        "candidate": candidate.name,
        "hidden_dim": candidate.hidden_dim,
        "mamba_state_dim": candidate.mamba_state_dim,
        "ax_res": candidate.ax_res,
        "q_rank": candidate.q_rank,
        "k_rank": candidate.k_rank,
        "dense_parameter_count": dense_count,
        "factorized_parameter_count": factorized_count,
        "qk_dense_params": qk_stats["dense_qk_params"],
        "qk_factorized_params": qk_stats[
            "factorized_qk_params"
        ],
        "qk_parameter_savings": qk_stats[
            "qk_parameter_savings"
        ],
        "qk_retained_fraction": qk_stats[
            "qk_retained_fraction"
        ],
        "shape_test": "PASS",
        "forward_test": "PASS",
        "logits_norm": logits_norm,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "PB1-A: factorized Q/K architecture "
            "implementation and parameter audit."
        )
    )

    parser.add_argument(
        "--candidate",
        choices=[
            c.name
            for c in CANDIDATES
        ] + ["all"],
        default="all",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    selected = CANDIDATES

    if args.candidate != "all":
        selected = tuple(
            c
            for c in CANDIDATES
            if c.name == args.candidate
        )

    results = []

    print()
    print("=" * 80)
    print("PB1-A FACTORIZED Q/K")
    print("=" * 80)
    print()
    print(
        "This is an architecture implementation gate."
    )
    print(
        "It does NOT make the final model-selection decision."
    )
    print()

    for candidate in selected:
        result = run_candidate(
            candidate,
            args.seed,
        )

        results.append(result)

    print()
    print("=" * 80)
    print("PB1-A SUMMARY")
    print("=" * 80)

    for result in results:
        print()
        print(result["candidate"])
        print(
            f"  params: "
            f"{result['factorized_parameter_count']:,}"
        )
        print(
            f"  Q/K savings: "
            f"{result['qk_parameter_savings']:,}"
        )
        print(
            f"  Q/K retained: "
            f"{result['qk_retained_fraction']:.4f}"
        )
        print(
            f"  shape test: "
            f"{result['shape_test']}"
        )
        print(
            f"  forward test: "
            f"{result['forward_test']}"
        )

    print()
    print("=" * 80)
    print("PB1-A COMPLETE")
    print("=" * 80)
    print()
    print(
        "Next step: integrate the factorized architecture "
        "into the research training path and run matched "
        "enwik8 training."
    )


if __name__ == "__main__":
    main()