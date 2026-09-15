"""
MODUS_X PB1-A — ACTUAL FACTORIZED Q/K BUDGET CORRECTION

Purpose
-------
PB0-C remains the completed dense-model allocation study.

PB1-A exposed an implementation-aware correction:
the actual Q/K factorization savings depend on ax_res because
m_wq and m_wk have shape [ax_res, embed_dim].

This script therefore:
  1. validates the canonical 47,437,768-parameter model;
  2. searches a larger vector-capacity grid for each PB0-B-approved
     Q/K rank allocation;
  3. computes a fast exact accounting estimate;
  4. instantiates the actual dense repository model for the closest
     candidates;
  5. replaces only m_wq and m_wk by explicit low-rank A/B factors
     for parameter counting;
  6. verifies the resulting factorized parameter count;
  7. writes JSON/CSV/Markdown reports.

IMPORTANT
---------
This is an AUDIT / BUDGET-CORRECTION script only.

It does NOT:
  - edit language/models.py;
  - train anything;
  - change m_wv, m_w_read, m_w_out, m_proj_w, feedback, router, etc.;
  - use vector_router=True;
  - claim functional equivalence.

The factorized training implementation comes after this budget audit.

Canonical model:
    Modus_X_MemoryFeedbackArchive_DeepSupervision

Canonical:
    47,437,768 parameters
    embed_dim=512
    hidden_dim=1536
    ax_res=512
    mamba_state_dim=512
    n_layers=12
    router_hidden=32
    vector_router=False
    auxiliary_layers=(6,)
    future_target_count=1
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast


# ============================================================================
# PROJECT PATHS
# ============================================================================

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LANGUAGE_DIR = PROJECT_ROOT / "language"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "research"
    / "parameter_allocation"
    / "outputs"
)

# Make the repository's language package importable.
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LANGUAGE_DIR))

from models import ModelConfig, count_params, make_model  # noqa: E402




# ============================================================================
# CANONICAL MODEL / BUDGET
# ============================================================================

CANONICAL_PARAMS = 47_437_768

CANONICAL = {
    "vocab_size": 256,
    "embed_dim": 512,
    "hidden_dim": 1536,
    "mamba_state_dim": 512,
    "ax_res": 512,
    "n_layers": 12,
    "router_hidden": 32,
    "seq_len": 512,
    "n_heads": 8,
    "dropout": 0.0,
    "vector_router": False,
    "aux_layers": (6,),
    "future_target_count": 1,
}

MODEL_NAME = "Modus_X_MemoryFeedbackArchive_DeepSupervision"

# PB0-B-supported interventions only.
MATRIX_CANDIDATES = {
    "C1_q128_k128": {"m_wq_rank": 128, "m_wk_rank": 128},
    "C2_q64_k64": {"m_wq_rank": 64, "m_wk_rank": 64},
    "C3_q128_k64": {"m_wq_rank": 128, "m_wk_rank": 64},
}

# Search grid.
#
# The previous PB0-C sweep stopped at hidden_dim=2304.  That was sufficient
# for the accounting-only search but not for the actual factorized budget
# correction, because the factorization can recover more parameters than
# the dense proxy allocated for a particular ax_res.
#
# Keep the grid aligned to the repository's tested 32-wide increments.
HIDDEN_DIM_VALUES = list(range(1536, 4097, 32))
MAMBA_STATE_VALUES = list(range(512, 1025, 32))
AX_RES_VALUES = list(range(512, 1025, 32))

# Only the closest accounting candidates are instantiated with the real
# repository model.  This keeps the audit fast while retaining exact
# verification for the final shortlist.
EXACT_VERIFY_COUNT = 40


# ============================================================================
# UTILITIES
# ============================================================================

def section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def format_params(value: int) -> str:
    return f"{value:,}"


def count_tree(params: Any) -> int:
    return int(count_params(params))


def build_model_config(
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

    # ``is_dataclass`` narrows the class to ``DataclassInstance`` in some
    # type checkers, even though it is still the callable configuration type.
    return cast(Any, ModelConfig)(**kwargs)


def make_dense_model(
    *,
    hidden_dim: int,
    mamba_state_dim: int,
    ax_res: int,
) -> Tuple[Any, Any, ModelConfig]:
    """
    Instantiate the exact repository model used by the canonical trainer.

    Returns:
        params, forward_fn, cfg
    """
    import jax

    cfg = build_model_config(
        hidden_dim=hidden_dim,
        mamba_state_dim=mamba_state_dim,
        ax_res=ax_res,
    )

    params, forward_fn = make_model(
        MODEL_NAME,
        jax.random.PRNGKey(0),
        cfg,
        auxiliary_layers=CANONICAL["aux_layers"],
        future_target_count=CANONICAL["future_target_count"],
        dropout_rate=CANONICAL["dropout"],
    )

    return params, forward_fn, cfg


# ============================================================================
# CANONICAL VALIDATION
# ============================================================================
def validate_canonical() -> int:
    section("VALIDATING CANONICAL MODEL")

    params, _, _ = make_dense_model(
        hidden_dim=CANONICAL["hidden_dim"],
        mamba_state_dim=CANONICAL["mamba_state_dim"],
        ax_res=CANONICAL["ax_res"],
    )

    measured = count_tree(params)

    print(f"Model                     : {MODEL_NAME}")
    print(f"Expected parameter count : {CANONICAL_PARAMS:,}")
    print(f"Measured parameter count  : {measured:,}")
    print(f"Difference                : {measured - CANONICAL_PARAMS:+,}")

    if measured != CANONICAL_PARAMS:
        raise RuntimeError(
            "CANONICAL PARAMETER COUNT MISMATCH.\n"
            f"Expected {CANONICAL_PARAMS:,}, found {measured:,}."
        )

    print("Canonical validation     : PASS")

    return measured


# ============================================================================
# FAST ACCOUNTING
# ============================================================================

def factorized_projection_params(
    input_dim: int,
    output_dim: int,
    rank: int,
) -> int:
    return input_dim * rank + rank * output_dim


def qk_savings(
    *,
    ax_res: int,
    embed_dim: int,
    n_layers: int,
    q_rank: int,
    k_rank: int,
) -> int:
    """
    Dense:
        Wq = [ax_res, embed_dim]
        Wk = [ax_res, embed_dim]

    Factorized:
        Wq = A_q @ B_q
        A_q = [ax_res, q_rank]
        B_q = [q_rank, embed_dim]

        Wk = A_k @ B_k
        A_k = [ax_res, k_rank]
        B_k = [k_rank, embed_dim]
    """
    dense_q = ax_res * embed_dim
    fact_q = factorized_projection_params(ax_res, embed_dim, q_rank)

    dense_k = ax_res * embed_dim
    fact_k = factorized_projection_params(ax_res, embed_dim, k_rank)

    return n_layers * ((dense_q - fact_q) + (dense_k - fact_k))


# These one-dimensional deltas were measured in PB0-C from actual repository
# model instantiation.  They are used only to rank the large search space.
HIDDEN_DELTA_PER_32 = 49_216
MAMBA_DELTA_PER_32 = 983_808
AX_RES_DELTA_PER_32 = 1_193_472


def fast_dense_delta(
    *,
    hidden_dim: int,
    mamba_state_dim: int,
    ax_res: int,
) -> int:
    return (
        ((hidden_dim - 1536) // 32) * HIDDEN_DELTA_PER_32
        + ((mamba_state_dim - 512) // 32) * MAMBA_DELTA_PER_32
        + ((ax_res - 512) // 32) * AX_RES_DELTA_PER_32
    )


def fast_factorized_total(
    *,
    q_rank: int,
    k_rank: int,
    hidden_dim: int,
    mamba_state_dim: int,
    ax_res: int,
) -> int:
    dense_total = CANONICAL_PARAMS + fast_dense_delta(
        hidden_dim=hidden_dim,
        mamba_state_dim=mamba_state_dim,
        ax_res=ax_res,
    )

    return dense_total - qk_savings(
        ax_res=ax_res,
        embed_dim=CANONICAL["embed_dim"],
        n_layers=CANONICAL["n_layers"],
        q_rank=q_rank,
        k_rank=k_rank,
    )


# ============================================================================
# ACTUAL FACTORIZATION FOR COUNTING
# ============================================================================

def factorized_count(
    params: Dict[str, Any],
    *,
    q_rank: int,
    k_rank: int,
) -> Tuple[int, Dict[str, Any]]:
    """
    Construct the factorized parameter tree for exact parameter counting.

    The dense tensors are replaced only for:
        layers.m_wq
        layers.m_wk

    Each is represented as:
        A: [layers, ax_res, rank]
        B: [layers, rank, embed_dim]

    No other parameter is changed.

    Values are initialized from the dense tensor by SVD so the generated
    factor tree is also structurally valid for later PB1 training work.
    """
    import jax.numpy as jnp

    layers = params["layers"]

    if "m_wq" not in layers or "m_wk" not in layers:
        raise KeyError(
            "Expected layers.m_wq and layers.m_wk in the repository model."
        )

    wq = layers["m_wq"]
    wk = layers["m_wk"]

    if len(wq.shape) != 3:
        raise ValueError(f"Unexpected m_wq shape: {wq.shape}")
    if len(wk.shape) != 3:
        raise ValueError(f"Unexpected m_wk shape: {wk.shape}")

    n_layers_q, ax_res_q, embed_dim_q = wq.shape
    n_layers_k, ax_res_k, embed_dim_k = wk.shape

    if n_layers_q != n_layers_k:
        raise ValueError("m_wq/m_wk layer counts differ.")

    if ax_res_q != ax_res_k:
        raise ValueError("m_wq/m_wk ax_res dimensions differ.")

    if embed_dim_q != embed_dim_k:
        raise ValueError("m_wq/m_wk embed dimensions differ.")

    if q_rank >= min(ax_res_q, embed_dim_q):
        raise ValueError(
            f"q_rank={q_rank} is not a proper low-rank factorization "
            f"for shape {wq.shape[1:]}."
        )

    if k_rank >= min(ax_res_k, embed_dim_k):
        raise ValueError(
            f"k_rank={k_rank} is not a proper low-rank factorization "
            f"for shape {wk.shape[1:]}."
        )

    def svd_factors(weight: Any, rank: int) -> Tuple[Any, Any]:
        """
        For W [out_dim, in_dim]:
            W ~= A @ B
            A [out_dim, rank]
            B [rank, in_dim]
        """
        factor_list_a = []
        factor_list_b = []

        for layer_index in range(weight.shape[0]):
            matrix = jnp.asarray(weight[layer_index], dtype=jnp.float32)
            u, s, vh = jnp.linalg.svd(matrix, full_matrices=False)

            s_r = s[:rank]
            root_s = jnp.sqrt(jnp.maximum(s_r, 0.0))

            a = u[:, :rank] * root_s[None, :]
            b = root_s[:, None] * vh[:rank, :]

            factor_list_a.append(a)
            factor_list_b.append(b)

        return jnp.stack(factor_list_a), jnp.stack(factor_list_b)

    q_a, q_b = svd_factors(wq, q_rank)
    k_a, k_b = svd_factors(wk, k_rank)

    # Rebuild the tree without the dense Q/K leaves.
    new_layers = dict(layers)
    new_layers.pop("m_wq")
    new_layers.pop("m_wk")

    new_layers["m_wq_A"] = q_a
    new_layers["m_wq_B"] = q_b
    new_layers["m_wk_A"] = k_a
    new_layers["m_wk_B"] = k_b

    new_params = dict(params)
    new_params["layers"] = new_layers

    total = count_tree(new_params)

    return total, new_params


# ============================================================================
# SEARCH
# ============================================================================

def search_fast_candidates() -> List[Dict[str, Any]]:
    section("FAST FACTORIZED BUDGET SEARCH")

    all_rows: List[Dict[str, Any]] = []

    total_grid = (
        len(HIDDEN_DIM_VALUES)
        * len(MAMBA_STATE_VALUES)
        * len(AX_RES_VALUES)
    )

    print(f"Search grid per candidate       : {total_grid:,}")
    print(f"Hidden dimensions               : {HIDDEN_DIM_VALUES[0]}..{HIDDEN_DIM_VALUES[-1]}")
    print(f"Mamba state dimensions         : {MAMBA_STATE_VALUES[0]}..{MAMBA_STATE_VALUES[-1]}")
    print(f"ax_res dimensions               : {AX_RES_VALUES[0]}..{AX_RES_VALUES[-1]}")

    for candidate_name, ranks in MATRIX_CANDIDATES.items():
        rows: List[Dict[str, Any]] = []

        for hidden_dim in HIDDEN_DIM_VALUES:
            for mamba_state_dim in MAMBA_STATE_VALUES:
                for ax_res in AX_RES_VALUES:
                    estimated_total = fast_factorized_total(
                        q_rank=ranks["m_wq_rank"],
                        k_rank=ranks["m_wk_rank"],
                        hidden_dim=hidden_dim,
                        mamba_state_dim=mamba_state_dim,
                        ax_res=ax_res,
                    )

                    error = estimated_total - CANONICAL_PARAMS

                    rows.append(
                        {
                            "candidate": candidate_name,
                            "m_wq_rank": ranks["m_wq_rank"],
                            "m_wk_rank": ranks["m_wk_rank"],
                            "hidden_dim": hidden_dim,
                            "mamba_state_dim": mamba_state_dim,
                            "ax_res": ax_res,
                            "estimated_total_params": estimated_total,
                            "estimated_budget_error": error,
                            "estimated_abs_budget_error": abs(error),
                        }
                    )

        rows.sort(key=lambda row: row["estimated_abs_budget_error"])

        all_rows.extend(rows)

        print()
        print(candidate_name)
        print("-" * len(candidate_name))

        for rank, row in enumerate(rows[:10], start=1):
            print(
                f"{rank:2d}. "
                f"h={row['hidden_dim']:4d} "
                f"mamba={row['mamba_state_dim']:4d} "
                f"ax={row['ax_res']:4d} "
                f"estimated={row['estimated_total_params']:,} "
                f"error={row['estimated_budget_error']:+,}"
            )

    return all_rows


# ============================================================================
# EXACT VERIFICATION
# ============================================================================

def verify_exact(
    candidate_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    section("EXACT FACTORIZED COUNT VERIFICATION")

    verified: List[Dict[str, Any]] = []

    for index, row in enumerate(candidate_rows[:EXACT_VERIFY_COUNT], start=1):
        print(
            f"[{index:02d}/{min(EXACT_VERIFY_COUNT, len(candidate_rows)):02d}] "
            f"{row['candidate']} "
            f"h={row['hidden_dim']} "
            f"mamba={row['mamba_state_dim']} "
            f"ax={row['ax_res']} "
            f"q={row['m_wq_rank']} "
            f"k={row['m_wk_rank']}"
        )

        params, _, cfg = make_dense_model(
            hidden_dim=row["hidden_dim"],
            mamba_state_dim=row["mamba_state_dim"],
            ax_res=row["ax_res"],
        )

        dense_total = count_tree(params)

        factor_total, factor_params = factorized_count(
            params,
            q_rank=row["m_wq_rank"],
            k_rank=row["m_wk_rank"],
        )

        actual_error = factor_total - CANONICAL_PARAMS

        q_dense = cfg.n_layers * cfg.ax_res * cfg.embed_dim
        k_dense = cfg.n_layers * cfg.ax_res * cfg.embed_dim

        q_factor = cfg.n_layers * (
            cfg.ax_res * row["m_wq_rank"]
            + row["m_wq_rank"] * cfg.embed_dim
        )
        k_factor = cfg.n_layers * (
            cfg.ax_res * row["m_wk_rank"]
            + row["m_wk_rank"] * cfg.embed_dim
        )

        actual_savings = (q_dense - q_factor) + (k_dense - k_factor)

        measured_fast_delta = (
            dense_total
            - CANONICAL_PARAMS
            - actual_savings
        )

        result = dict(row)
        result.update(
            {
                "dense_total_params": dense_total,
                "factorized_total_params": factor_total,
                "actual_budget_error": actual_error,
                "actual_abs_budget_error": abs(actual_error),
                "actual_qk_savings": actual_savings,
                "fast_estimate_error": (
                    factor_total - row["estimated_total_params"]
                ),
                "dense_count_delta_from_canonical": (
                    dense_total - CANONICAL_PARAMS
                ),
                "dense_vs_pb0c_delta": measured_fast_delta,
                "shape_q": [
                    int(cfg.n_layers),
                    int(cfg.ax_res),
                    int(cfg.embed_dim),
                ],
                "shape_k": [
                    int(cfg.n_layers),
                    int(cfg.ax_res),
                    int(cfg.embed_dim),
                ],
            }
        )

        # Force a traversal so the factorized tree is definitely a valid
        # JAX parameter pytree.
        _ = count_tree(factor_params)

        print(
            f"    dense={dense_total:,} "
            f"factorized={factor_total:,} "
            f"error={actual_error:+,} "
            f"QK_savings={actual_savings:,}"
        )

        verified.append(result)

    verified.sort(key=lambda row: row["actual_abs_budget_error"])
    return verified


# ============================================================================
# OUTPUT
# ============================================================================

def write_outputs(
    *,
    fast_rows: List[Dict[str, Any]],
    exact_rows: List[Dict[str, Any]],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUT_DIR / "PB1_A_CORRECTED_BUDGET.json"
    csv_path = OUTPUT_DIR / "PB1_A_CORRECTED_BUDGET.csv"
    md_path = OUTPUT_DIR / "PB1_A_CORRECTED_BUDGET.md"

    payload = {
        "experiment": "PB1-A actual factorized Q/K budget correction",
        "model": MODEL_NAME,
        "canonical_parameters": CANONICAL_PARAMS,
        "canonical_config": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in CANONICAL.items()
        },
        "matrix_candidates": MATRIX_CANDIDATES,
        "search_space": {
            "hidden_dim": [
                HIDDEN_DIM_VALUES[0],
                HIDDEN_DIM_VALUES[-1],
                32,
            ],
            "mamba_state_dim": [
                MAMBA_STATE_VALUES[0],
                MAMBA_STATE_VALUES[-1],
                32,
            ],
            "ax_res": [
                AX_RES_VALUES[0],
                AX_RES_VALUES[-1],
                32,
            ],
        },
        "exact_verify_count": EXACT_VERIFY_COUNT,
        "exact_rows": exact_rows,
        "top_exact": exact_rows[:15],
    }

    json_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    fieldnames = [
        "candidate",
        "m_wq_rank",
        "m_wk_rank",
        "hidden_dim",
        "mamba_state_dim",
        "ax_res",
        "estimated_total_params",
        "estimated_budget_error",
        "dense_total_params",
        "factorized_total_params",
        "actual_budget_error",
        "actual_qk_savings",
        "fast_estimate_error",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in exact_rows:
            writer.writerow(
                {
                    key: row.get(key)
                    for key in fieldnames
                }
            )

    lines = [
        "# PB1-A Actual Factorized Q/K Budget Correction",
        "",
        f"- Model: `{MODEL_NAME}`",
        f"- Canonical parameters: **{CANONICAL_PARAMS:,}**",
        "- Intervention: factorize only `m_wq` and `m_wk`.",
        "- `vector_router=False`.",
        "- Auxiliary layer: `(6,)`.",
        "- Future target count: `1`.",
        "",
        "## Exact verified candidates",
        "",
        "| Rank | Candidate | Hidden | Mamba | ax_res | Q rank | K rank | Dense params | Factorized params | Budget error | Q/K savings |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for rank, row in enumerate(exact_rows[:15], start=1):
        lines.append(
            "| "
            f"{rank} | "
            f"{row['candidate']} | "
            f"{row['hidden_dim']} | "
            f"{row['mamba_state_dim']} | "
            f"{row['ax_res']} | "
            f"{row['m_wq_rank']} | "
            f"{row['m_wk_rank']} | "
            f"{row['dense_total_params']:,} | "
            f"{row['factorized_total_params']:,} | "
            f"{row['actual_budget_error']:+,} | "
            f"{row['actual_qk_savings']:,} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "PB0-C is retained as the completed dense allocation study.",
            "PB1-A is correcting only the implementation-aware budget count.",
            "No training candidate should be promoted until its factorized count",
            "has been verified here.",
            "",
            "The best candidate is the row with the smallest absolute",
            "actual budget error. Prefer a slightly-under-budget candidate",
            "over exceeding the fixed budget unless a separate explicit",
            "budget policy is adopted.",
            "",
            "This script does not establish that a factorized candidate is",
            "functionally equivalent to the dense model. Functional validation",
            "belongs to the subsequent PB1-A/PB1-B architecture smoke test and",
            "matched training.",
            "",
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


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    section("MODUS_X PB1-A — ACTUAL FACTORIZED Q/K BUDGET CORRECTION")

    print(f"Project root                 : {PROJECT_ROOT}")
    print(f"Language dir                 : {LANGUAGE_DIR}")
    print(f"Output dir                   : {OUTPUT_DIR}")
    print(f"Canonical budget             : {CANONICAL_PARAMS:,}")
    print(f"Model                        : {MODEL_NAME}")
    print("Q/K factorization            : m_wq + m_wk only")
    print("vector_router                : False")
    print("Auxiliary layers             : (6,)")
    print("Future target count          : 1")

    validate_canonical()

    fast_rows = search_fast_candidates()

    # Keep the globally closest accounting candidates.  This is sufficient
    # because the exact verifier below recomputes the count from actual model
    # tensors rather than trusting the fast estimator.
    fast_rows.sort(key=lambda row: row["estimated_abs_budget_error"])

    exact_rows = verify_exact(fast_rows)

    if not exact_rows:
        raise RuntimeError("No exact candidates were verified.")

    section("BEST ACTUAL FACTORIZED CANDIDATES")

    for rank, row in enumerate(exact_rows[:15], start=1):
        print(
            f"{rank:2d}. "
            f"{row['candidate']} "
            f"h={row['hidden_dim']} "
            f"mamba={row['mamba_state_dim']} "
            f"ax={row['ax_res']} "
            f"q={row['m_wq_rank']} "
            f"k={row['m_wk_rank']} "
            f"params={row['factorized_total_params']:,} "
            f"error={row['actual_budget_error']:+,}"
        )

    best = exact_rows[0]

    print()
    print("BEST VERIFIED CANDIDATE")
    print(f"  Candidate       : {best['candidate']}")
    print(f"  hidden_dim      : {best['hidden_dim']}")
    print(f"  mamba_state_dim : {best['mamba_state_dim']}")
    print(f"  ax_res           : {best['ax_res']}")
    print(f"  Q rank           : {best['m_wq_rank']}")
    print(f"  K rank           : {best['m_wk_rank']}")
    print(f"  Dense params     : {best['dense_total_params']:,}")
    print(f"  Factorized params: {best['factorized_total_params']:,}")
    print(f"  Budget error     : {best['actual_budget_error']:+,}")
    print(f"  Q/K savings      : {best['actual_qk_savings']:,}")

    write_outputs(
        fast_rows=fast_rows,
        exact_rows=exact_rows,
    )

    section("PB1-A CORRECTED BUDGET AUDIT COMPLETE")
    print("Canonical count validation : PASS")
    print("Actual factorized counts   : VERIFIED")
    print("Training                   : NOT RUN")
    print("Owner model modified       : NO")


if __name__ == "__main__":
    main()
