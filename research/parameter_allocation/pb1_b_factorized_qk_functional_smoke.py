"""
MODUS_X PB1-B — FACTORIZED Q/K FUNCTIONAL SMOKE TEST

Purpose
-------
PB1-A established the actual parameter counts for the Q/K-factorized
candidates. PB1-B now verifies that those candidates can use the SAME
official Modus_X DeepSupervision forward interface.

This script:
  1. builds each candidate with the repository's public make_model API;
  2. factorizes only layers.m_wq and layers.m_wk using truncated SVD;
  3. verifies the exact factorized parameter count;
  4. reconstructs Wq/Wk inside a research-only forward wrapper;
  5. calls the OFFICIAL repository DeepSupervision forward function;
  6. verifies output structure and shapes:
       final_logits
       auxiliary_logits
       future_logits
       auxiliary_future_logits
  7. checks all outputs are finite;
  8. reports dense-vs-factorized output differences.

IMPORTANT
---------
This is a FUNCTIONAL SMOKE TEST only.

It does NOT:
  - modify language/models.py;
  - modify checkpoints;
  - train anything;
  - claim the factorized implementation is compute-efficient;
  - replace the official forward implementation.

For this smoke test, Wq/Wk are reconstructed as A @ B immediately before
calling the official forward. This intentionally prioritizes correctness
of the parameterization and DeepSupervision interface over efficiency.

The subsequent training implementation can replace this reconstruction
wrapper with true low-rank matmuls.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, cast

import jax
import jax.numpy as jnp
import numpy as np


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LANGUAGE_DIR = PROJECT_ROOT / "language"
OUTPUT_DIR = PROJECT_ROOT / "research" / "parameter_allocation" / "outputs"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LANGUAGE_DIR))

from models import ModelConfig, count_params, make_model  # noqa: E402


# ============================================================================
# EXPERIMENT CONSTANTS
# ============================================================================

MODEL_NAME = "Modus_X_MemoryFeedbackArchive_DeepSupervision"

CANONICAL_PARAMS = 47_437_768

CANONICAL = {
    "vocab_size": 256,
    "embed_dim": 512,
    "hidden_dim": 1536,
    "mamba_state_dim": 512,
    "ax_res": 512,
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
    },
    "C2_q64_k64": {
        "hidden_dim": 2048,
        "mamba_state_dim": 640,
        "ax_res": 512,
        "q_rank": 64,
        "k_rank": 64,
    },
    "C3_q128_k64": {
        "hidden_dim": 1536,
        "mamba_state_dim": 640,
        "ax_res": 512,
        "q_rank": 128,
        "k_rank": 64,
    },
}

SMOKE_BATCH = 2
SMOKE_SEQ_LEN = 512
RNG_SEED = 20260911

# PB1-A verified counts.
EXPECTED_FACTORIZED_COUNTS = {
    "C1_q128_k128": 47_440_328,
    "C2_q64_k64": 47_441_864,
    "C3_q128_k64": 47_440_840,
}


# ============================================================================
# UTILITIES
# ============================================================================

def section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def count_tree(tree: Any) -> int:
    return int(count_params(tree))


def build_config(
    *,
    hidden_dim: int,
    mamba_state_dim: int,
    ax_res: int,
) -> ModelConfig:
    values: Dict[str, Any] = {
        "vocab_size": CANONICAL["vocab_size"],
        "embed_dim": CANONICAL["embed_dim"],
        "hidden_dim": hidden_dim,
        "ax_res": ax_res,
        "n_layers": CANONICAL["n_layers"],
        "n_heads_attn": CANONICAL["n_heads"],
        "seq_len": CANONICAL["seq_len"],
        "mamba_state_dim": mamba_state_dim,
        "vector_router": CANONICAL["vector_router"],
        "router_hidden": CANONICAL["router_hidden"],
    }

    if is_dataclass(ModelConfig):
        valid_fields = {f.name for f in fields(ModelConfig)}
    else:
        valid_fields = set(values)

    kwargs = {
        key: value
        for key, value in values.items()
        if key in valid_fields
    }

    return cast(Any, ModelConfig)(**kwargs)


def make_dense_model(
    *,
    hidden_dim: int,
    mamba_state_dim: int,
    ax_res: int,
    seed: int,
) -> Tuple[Any, Any, ModelConfig]:
    cfg = build_config(
        hidden_dim=hidden_dim,
        mamba_state_dim=mamba_state_dim,
        ax_res=ax_res,
    )

    params, fwd_fn = make_model(
        MODEL_NAME,
        jax.random.PRNGKey(seed),
        cfg,
        auxiliary_layers=CANONICAL["aux_layers"],
        future_target_count=CANONICAL["future_target_count"],
        dropout_rate=CANONICAL["dropout"],
    )

    return params, fwd_fn, cfg


# ============================================================================
# SVD FACTORIZATION
# ============================================================================

def factorize_weight(
    weight: Any,
    rank: int,
) -> Tuple[jax.Array, jax.Array]:
    """
    Factor each layer's W [ax_res, embed_dim] as:

        W ~= A @ B
        A [ax_res, rank]
        B [rank, embed_dim]

    using truncated SVD with balanced singular values.
    """
    weight = jnp.asarray(weight, dtype=jnp.float32)

    if weight.ndim != 3:
        raise ValueError(f"Expected rank-3 weight [layers,out,in], got {weight.shape}")

    n_layers, out_dim, in_dim = weight.shape

    if rank >= min(out_dim, in_dim):
        raise ValueError(
            f"Rank {rank} is not a strict low-rank factorization for "
            f"matrix shape {(out_dim, in_dim)}."
        )

    a_list = []
    b_list = []

    for layer_index in range(n_layers):
        matrix = weight[layer_index]
        u, s, vh = jnp.linalg.svd(matrix, full_matrices=False)

        s_r = s[:rank]
        root_s = jnp.sqrt(jnp.maximum(s_r, 0.0))

        a = u[:, :rank] * root_s[None, :]
        b = root_s[:, None] * vh[:rank, :]

        a_list.append(a)
        b_list.append(b)

    return jnp.stack(a_list), jnp.stack(b_list)


def make_factorized_params(
    dense_params: Dict[str, Any],
    *,
    q_rank: int,
    k_rank: int,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    layers = dense_params["layers"]

    if "m_wq" not in layers:
        raise KeyError("Missing layers.m_wq")
    if "m_wk" not in layers:
        raise KeyError("Missing layers.m_wk")

    wq = layers["m_wq"]
    wk = layers["m_wk"]

    q_a, q_b = factorize_weight(wq, q_rank)
    k_a, k_b = factorize_weight(wk, k_rank)

    new_layers = dict(layers)
    new_layers.pop("m_wq")
    new_layers.pop("m_wk")
    new_layers["m_wq_A"] = q_a
    new_layers["m_wq_B"] = q_b
    new_layers["m_wk_A"] = k_a
    new_layers["m_wk_B"] = k_b

    factorized_params = dict(dense_params)
    factorized_params["layers"] = new_layers

    q_dense = int(np.prod(wq.shape))
    k_dense = int(np.prod(wk.shape))

    q_factor = int(np.prod(q_a.shape) + np.prod(q_b.shape))
    k_factor = int(np.prod(k_a.shape) + np.prod(k_b.shape))

    stats = {
        "q_dense": q_dense,
        "q_factor": q_factor,
        "q_savings": q_dense - q_factor,
        "k_dense": k_dense,
        "k_factor": k_factor,
        "k_savings": k_dense - k_factor,
        "qk_savings": (q_dense - q_factor) + (k_dense - k_factor),
    }

    return factorized_params, stats


# ============================================================================
# RESEARCH-ONLY FORWARD WRAPPER
# ============================================================================

def reconstruct_dense_qk(
    factorized_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Reconstruct only Wq/Wk and return a dense parameter tree.

    This is deliberately used only for the PB1-B smoke test. The official
    repository forward function therefore remains completely untouched.
    """
    layers = factorized_params["layers"]

    q = jnp.einsum(
        "lor,lri->loi",
        layers["m_wq_A"],
        layers["m_wq_B"],
    )
    k = jnp.einsum(
        "lor,lri->loi",
        layers["m_wk_A"],
        layers["m_wk_B"],
    )

    new_layers = dict(layers)
    new_layers.pop("m_wq_A")
    new_layers.pop("m_wq_B")
    new_layers.pop("m_wk_A")
    new_layers.pop("m_wk_B")
    new_layers["m_wq"] = q
    new_layers["m_wk"] = k

    dense_view = dict(factorized_params)
    dense_view["layers"] = new_layers

    return dense_view


def factorized_forward(
    factorized_params: Dict[str, Any],
    x: jax.Array,
    official_fwd: Any,
) -> Any:
    dense_view = reconstruct_dense_qk(factorized_params)
    return official_fwd(dense_view, x)


# ============================================================================
# OUTPUT INSPECTION
# ============================================================================

def output_summary(outputs: Any) -> Dict[str, Any]:
    if not isinstance(outputs, tuple):
        raise RuntimeError(
            "DeepSupervision forward did not return a tuple. "
            f"Returned type: {type(outputs)}"
        )

    if len(outputs) != 4:
        raise RuntimeError(
            "Expected four DeepSupervision outputs "
            "(final, auxiliary, future, auxiliary_future), "
            f"got {len(outputs)}."
        )

    final_logits, auxiliary_logits, future_logits, auxiliary_future_logits = outputs

    arrays = {
        "final_logits": final_logits,
        "auxiliary_logits": auxiliary_logits,
        "future_logits": future_logits,
        "auxiliary_future_logits": auxiliary_future_logits,
    }

    summary: Dict[str, Any] = {}

    for name, value in arrays.items():
        arr = np.asarray(value)

        if not np.all(np.isfinite(arr)):
            raise RuntimeError(f"{name} contains non-finite values.")

        summary[name] = {
            "shape": [int(x) for x in arr.shape],
            "dtype": str(arr.dtype),
            "finite": True,
            "l2": float(np.linalg.norm(arr)),
        }

    return summary


def relative_l2(a: Any, b: Any) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    numerator = np.linalg.norm(aa - bb)
    denominator = max(np.linalg.norm(aa), 1e-12)
    return float(numerator / denominator)


def reconstruction_summary(
    dense_params: Dict[str, Any],
    factorized_params: Dict[str, Any],
) -> Dict[str, float]:
    dense_layers = dense_params["layers"]
    factor_layers = factorized_params["layers"]

    dense_q = np.asarray(dense_layers["m_wq"], dtype=np.float64)
    dense_k = np.asarray(dense_layers["m_wk"], dtype=np.float64)

    factor_q = np.asarray(
        jnp.einsum(
            "lor,lri->loi",
            factor_layers["m_wq_A"],
            factor_layers["m_wq_B"],
        ),
        dtype=np.float64,
    )
    factor_k = np.asarray(
        jnp.einsum(
            "lor,lri->loi",
            factor_layers["m_wk_A"],
            factor_layers["m_wk_B"],
        ),
        dtype=np.float64,
    )

    return {
        "q_relative_reconstruction_error": relative_l2(dense_q, factor_q),
        "k_relative_reconstruction_error": relative_l2(dense_k, factor_k),
    }


# ============================================================================
# SINGLE CANDIDATE
# ============================================================================

def run_candidate(
    name: str,
    candidate: Dict[str, int],
    x: jax.Array,
) -> Dict[str, Any]:
    print()
    print("-" * 80)
    print(name)
    print("-" * 80)

    params, official_fwd, cfg = make_dense_model(
        hidden_dim=candidate["hidden_dim"],
        mamba_state_dim=candidate["mamba_state_dim"],
        ax_res=candidate["ax_res"],
        seed=RNG_SEED,
    )

    dense_count = count_tree(params)

    factorized_params, factor_stats = make_factorized_params(
        params,
        q_rank=candidate["q_rank"],
        k_rank=candidate["k_rank"],
    )

    factorized_count = count_tree(factorized_params)

    expected_count = EXPECTED_FACTORIZED_COUNTS[name]

    print(f"hidden_dim                 : {candidate['hidden_dim']}")
    print(f"mamba_state_dim            : {candidate['mamba_state_dim']}")
    print(f"ax_res                     : {candidate['ax_res']}")
    print(f"Q rank                     : {candidate['q_rank']}")
    print(f"K rank                     : {candidate['k_rank']}")
    print(f"Dense parameter count      : {dense_count:,}")
    print(f"Factorized parameter count  : {factorized_count:,}")
    print(f"PB1-A expected count       : {expected_count:,}")
    print(f"Count difference           : {factorized_count - expected_count:+,}")

    if factorized_count != expected_count:
        raise RuntimeError(
            f"{name}: factorized parameter count mismatch. "
            f"Expected {expected_count:,}, found {factorized_count:,}."
        )

    recon = reconstruction_summary(params, factorized_params)

    print(
        "Q relative reconstruction  : "
        f"{recon['q_relative_reconstruction_error']:.8f}"
    )
    print(
        "K relative reconstruction  : "
        f"{recon['k_relative_reconstruction_error']:.8f}"
    )

    dense_outputs = official_fwd(params, x)
    factor_outputs = factorized_forward(
        factorized_params,
        x,
        official_fwd,
    )

    dense_summary = output_summary(dense_outputs)
    factor_summary = output_summary(factor_outputs)

    dense_final, dense_aux, dense_future, dense_aux_future = dense_outputs
    fact_final, fact_aux, fact_future, fact_aux_future = factor_outputs

    output_errors = {
        "final_logits_relative_l2": relative_l2(dense_final, fact_final),
        "auxiliary_logits_relative_l2": relative_l2(dense_aux, fact_aux),
        "future_logits_relative_l2": relative_l2(dense_future, fact_future),
        "auxiliary_future_logits_relative_l2": relative_l2(
            dense_aux_future,
            fact_aux_future,
        ),
    }

    print("DeepSupervision structure : PASS")
    print("Finite outputs             : PASS")
    print(f"Final logits relative L2  : {output_errors['final_logits_relative_l2']:.8f}")
    print(
        "Aux logits relative L2    : "
        f"{output_errors['auxiliary_logits_relative_l2']:.8f}"
    )
    print(
        "Future logits relative L2 : "
        f"{output_errors['future_logits_relative_l2']:.8f}"
    )
    print(
        "Aux-future relative L2    : "
        f"{output_errors['auxiliary_future_logits_relative_l2']:.8f}"
    )

    # Verify shapes match exactly between dense and factorized views.
    for dense_item, factor_item in zip(dense_outputs, factor_outputs):
        if tuple(dense_item.shape) != tuple(factor_item.shape):
            raise RuntimeError(
                f"{name}: dense/factorized output shape mismatch: "
                f"{dense_item.shape} vs {factor_item.shape}"
            )

    return {
        "candidate": name,
        "hidden_dim": candidate["hidden_dim"],
        "mamba_state_dim": candidate["mamba_state_dim"],
        "ax_res": candidate["ax_res"],
        "q_rank": candidate["q_rank"],
        "k_rank": candidate["k_rank"],
        "dense_parameter_count": dense_count,
        "factorized_parameter_count": factorized_count,
        "canonical_budget": CANONICAL_PARAMS,
        "budget_error": factorized_count - CANONICAL_PARAMS,
        "qk_savings": int(factor_stats["qk_savings"]),
        "q_relative_reconstruction_error": recon[
            "q_relative_reconstruction_error"
        ],
        "k_relative_reconstruction_error": recon[
            "k_relative_reconstruction_error"
        ],
        **output_errors,
        "dense_output_summary": dense_summary,
        "factorized_output_summary": factor_summary,
        "status": "PASS",
    }


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    section("MODUS_X PB1-B — FACTORIZED Q/K FUNCTIONAL SMOKE TEST")

    print(f"Project root                 : {PROJECT_ROOT}")
    print(f"Language dir                 : {LANGUAGE_DIR}")
    print(f"Output dir                   : {OUTPUT_DIR}")
    print(f"Model                        : {MODEL_NAME}")
    print(f"Canonical parameter budget  : {CANONICAL_PARAMS:,}")
    print(f"Smoke batch                 : {SMOKE_BATCH}")
    print(f"Smoke sequence length       : {SMOKE_SEQ_LEN}")
    print(f"Auxiliary layers            : {CANONICAL['aux_layers']}")
    print(f"Future target count         : {CANONICAL['future_target_count']}")
    print("Official model modified     : NO")

    if SMOKE_SEQ_LEN != CANONICAL["seq_len"]:
        raise RuntimeError("Smoke sequence length must match canonical seq_len.")

    section("CANONICAL PUBLIC API CHECK")

    canonical_params, canonical_fwd, canonical_cfg = make_dense_model(
        hidden_dim=CANONICAL["hidden_dim"],
        mamba_state_dim=CANONICAL["mamba_state_dim"],
        ax_res=CANONICAL["ax_res"],
        seed=RNG_SEED,
    )

    canonical_count = count_tree(canonical_params)

    print(f"Expected canonical count   : {CANONICAL_PARAMS:,}")
    print(f"Measured canonical count   : {canonical_count:,}")
    print(f"Difference                 : {canonical_count - CANONICAL_PARAMS:+,}")

    if canonical_count != CANONICAL_PARAMS:
        raise RuntimeError(
            f"Canonical parameter mismatch: expected {CANONICAL_PARAMS:,}, "
            f"found {canonical_count:,}."
        )

    print("Canonical count             : PASS")
    print("Public make_model API       : PASS")

    rng = np.random.default_rng(RNG_SEED)
    x_np = rng.integers(
        0,
        CANONICAL["vocab_size"],
        size=(SMOKE_SEQ_LEN,),
        dtype=np.int32,
    )
    x = jnp.asarray(x_np)

    # Verify canonical DeepSupervision output interface first.
    canonical_outputs = canonical_fwd(canonical_params, x)
    canonical_summary = output_summary(canonical_outputs)

    print("Canonical DeepSupervision   : PASS")
    for name, info in canonical_summary.items():
        print(f"  {name:28s}: {info['shape']}")

    section("PB1-B CANDIDATE TESTS")

    results = []

    for candidate_name, candidate in CANDIDATES.items():
        results.append(
            run_candidate(
                candidate_name,
                candidate,
                x,
            )
        )

    section("PB1-B FINAL RESULT")

    for result in results:
        print(
            f"{result['candidate']:18s} "
            f"params={result['factorized_parameter_count']:,} "
            f"budget_error={result['budget_error']:+,} "
            f"status={result['status']}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUT_DIR / "PB1_B_FACTORIZED_QK_FUNCTIONAL_SMOKE.json"
    csv_path = OUTPUT_DIR / "PB1_B_FACTORIZED_QK_FUNCTIONAL_SMOKE.csv"
    md_path = OUTPUT_DIR / "PB1_B_FACTORIZED_QK_FUNCTIONAL_SMOKE.md"

    payload = {
        "experiment": "PB1-B factorized Q/K functional smoke test",
        "model": MODEL_NAME,
        "canonical_parameters": CANONICAL_PARAMS,
        "canonical_config": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in CANONICAL.items()
        },
        "smoke_batch": SMOKE_BATCH,
        "smoke_sequence_length": SMOKE_SEQ_LEN,
        "seed": RNG_SEED,
        "candidates": results,
        "overall_status": "PASS",
    }

    json_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    fieldnames = [
        "candidate",
        "hidden_dim",
        "mamba_state_dim",
        "ax_res",
        "q_rank",
        "k_rank",
        "dense_parameter_count",
        "factorized_parameter_count",
        "budget_error",
        "qk_savings",
        "q_relative_reconstruction_error",
        "k_relative_reconstruction_error",
        "final_logits_relative_l2",
        "auxiliary_logits_relative_l2",
        "future_logits_relative_l2",
        "auxiliary_future_logits_relative_l2",
        "status",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {field: result.get(field) for field in fieldnames}
            )

    lines = [
        "# PB1-B Factorized Q/K Functional Smoke Test",
        "",
        f"- Model: `{MODEL_NAME}`",
        f"- Canonical parameters: **{CANONICAL_PARAMS:,}**",
        f"- Sequence length: `{SMOKE_SEQ_LEN}`",
        f"- Seed: `{RNG_SEED}`",
        "- Intervention: factorize only `m_wq` and `m_wk`.",
        "- Official `make_model` and DeepSupervision forward remain untouched.",
        "",
        "## Results",
        "",
        "| Candidate | Hidden | Mamba | ax_res | Q | K | Factorized params | Budget error | Q recon err | K recon err | Final logits rel L2 | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for result in results:
        lines.append(
            "| "
            f"{result['candidate']} | "
            f"{result['hidden_dim']} | "
            f"{result['mamba_state_dim']} | "
            f"{result['ax_res']} | "
            f"{result['q_rank']} | "
            f"{result['k_rank']} | "
            f"{result['factorized_parameter_count']:,} | "
            f"{result['budget_error']:+,} | "
            f"{result['q_relative_reconstruction_error']:.8f} | "
            f"{result['k_relative_reconstruction_error']:.8f} | "
            f"{result['final_logits_relative_l2']:.8f} | "
            f"{result['status']} |"
        )

    lines.extend(
        [
            "",
            "## Gate",
            "",
            "PB1-B PASS requires:",
            "",
            "1. canonical parameter count matches exactly;",
            "2. official DeepSupervision returns four outputs;",
            "3. every candidate matches its PB1-A factorized parameter count;",
            "4. dense and factorized output shapes match;",
            "5. all factorized outputs are finite.",
            "",
            "The reconstruction wrapper is a smoke-test mechanism, not the final",
            "efficient training implementation. The next training implementation",
            "must compute Wq/Wk through the A/B factors directly.",
        ]
    )

    md_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print(f"Wrote: {json_path}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")

    section("PB1-B COMPLETE")
    print("Canonical count validation : PASS")
    print("DeepSupervision interface  : PASS")
    print("C1/C2/C3 functional smoke : PASS")
    print("Training                   : NOT RUN")
    print("Owner model modified       : NO")


if __name__ == "__main__":
    main()
