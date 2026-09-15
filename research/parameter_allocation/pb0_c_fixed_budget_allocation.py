"""
MODUS_X PB0-C — OPTIMIZED FIXED-BUDGET PARAMETER ALLOCATION SEARCH

Purpose
-------
Find fixed ~47.44M parameter allocations after PB0-B identified
functionally compressible matrix-memory projections.

PB0-C is ACCOUNTING ONLY.
No factorized architecture is implemented here.
No training is performed here.

Canonical budget:
    47,437,768 parameters

PB0-B-supported matrix candidates:

    C1: m_wq=128, m_wk=128
    C2: m_wq=64,  m_wk=64
    C3: m_wq=128, m_wk=64

Vector-side dimensions explored:

    hidden_dim
    mamba_state_dim
    ax_res

IMPORTANT
---------
The previous implementation constructed every Cartesian combination:

    25 * 17 * 17 * 3 = 21,675 model constructions.

That is unnecessarily expensive.

This implementation first measures one-dimensional parameter-count
effects and performs the large search mathematically. Only a small
number of final candidates are constructed for exact verification.

The final candidate architectures must still be trained in PB1-A/B.
"""

from __future__ import annotations

import csv
import inspect
import json
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


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

sys.path.insert(0, str(LANGUAGE_DIR))

from models import ModelConfig, count_params, make_model  # noqa: E402


# ============================================================================
# CANONICAL MODEL
# ============================================================================

CANONICAL_PARAMS = 47_437_768

CANONICAL = {
    "vocab_size": 256,
    "embed_dim": 512,
    "hidden_dim": 1536,
    "ax_res": 512,
    "n_layers": 12,
    "mamba_state_dim": 512,
    "router_hidden": 32,
    "seq_len": 512,
    "n_heads": 8,
    "dropout": 0.0,
    "vector_router": False,
    "aux_layers": (6,),
    "future_target_count": 1,
}


# ============================================================================
# PB0-B MATRIX CANDIDATES
# ============================================================================

MATRIX_CANDIDATES = {
    "C1_q128_k128": {
        "m_wq_rank": 128,
        "m_wk_rank": 128,
    },
    "C2_q64_k64": {
        "m_wq_rank": 64,
        "m_wk_rank": 64,
    },
    "C3_q128_k64": {
        "m_wq_rank": 128,
        "m_wk_rank": 64,
    },
}


# ============================================================================
# VECTOR SEARCH SPACE
# ============================================================================

# Keep dimensions aligned with the original search grid.

HIDDEN_DIM_VALUES = list(range(1536, 2305, 32))

MAMBA_STATE_VALUES = list(range(512, 1025, 32))

AX_RES_VALUES = list(range(512, 1025, 32))


# ============================================================================
# FINAL VERIFICATION SIZE
# ============================================================================

# Number of mathematically promising candidates to verify with actual model
# construction per matrix candidate.

FINAL_VERIFY_COUNT = 30


# ============================================================================
# OUTPUT HELPERS
# ============================================================================

def section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def format_params(value: int) -> str:
    return f"{value:,}"


def pct(value: int, total: int = CANONICAL_PARAMS) -> float:
    return 100.0 * value / total


# ============================================================================
# CONFIGURATION
# ============================================================================

def candidate_config_dict(
    *,
    hidden_dim: int,
    mamba_state_dim: int,
    ax_res: int,
) -> Dict[str, Any]:
    cfg = dict(CANONICAL)

    cfg["hidden_dim"] = hidden_dim
    cfg["mamba_state_dim"] = mamba_state_dim
    cfg["ax_res"] = ax_res

    return cfg


def build_model_config(values: Dict[str, Any]) -> ModelConfig:
    """
    Construct ModelConfig using only fields that exist in the installed
    repository version.
    """

    if is_dataclass(ModelConfig):
        valid_fields = {
            f.name
            for f in fields(ModelConfig)
        }
    else:
        try:
            valid_fields = set(
                inspect.signature(ModelConfig).parameters
            )
        except Exception:
            valid_fields = set(values)

    kwargs: Dict[str, Any] = {}

    for key, value in values.items():
        if key in valid_fields:
            kwargs[key] = value

    return ModelConfig(**kwargs)


# ============================================================================
# EXACT MODEL CONSTRUCTION / COUNT
# ============================================================================

def make_counted_model(
    *,
    hidden_dim: int,
    mamba_state_dim: int,
    ax_res: int,
) -> Tuple[Any, int]:
    """
    Construct the canonical dense Modus_X model and return:

        (parameter_tree, parameter_count)

    Exact repository API:

        make_model(
            name,
            key,
            cfg,
            auxiliary_layers=None,
            future_target_count=0,
            dropout_rate=0.0,
        )
    """

    import jax

    values = candidate_config_dict(
        hidden_dim=hidden_dim,
        mamba_state_dim=mamba_state_dim,
        ax_res=ax_res,
    )

    cfg = build_model_config(values)

    key = jax.random.PRNGKey(0)

    model_name = (
        "Modus_X_MemoryFeedbackArchive_DeepSupervision"
    )

    params, _ = make_model(
        model_name,
        key,
        cfg,
        auxiliary_layers=CANONICAL["aux_layers"],
        future_target_count=CANONICAL["future_target_count"],
        dropout_rate=CANONICAL["dropout"],
    )

    total = int(count_params(params))

    return params, total


# ============================================================================
# CANONICAL VALIDATION
# ============================================================================

def validate_canonical_model() -> int:

    section("VALIDATING CANONICAL MODEL")

    model, total = make_counted_model(
        hidden_dim=CANONICAL["hidden_dim"],
        mamba_state_dim=CANONICAL["mamba_state_dim"],
        ax_res=CANONICAL["ax_res"],
    )

    del model

    print(
        f"Expected parameter count : "
        f"{CANONICAL_PARAMS:,}"
    )

    print(
        f"Measured parameter count : "
        f"{total:,}"
    )

    print(
        f"Difference               : "
        f"{total - CANONICAL_PARAMS:+,}"
    )

    if total != CANONICAL_PARAMS:
        raise RuntimeError(
            "\nCANONICAL PARAMETER COUNT MISMATCH.\n"
            f"Expected : {CANONICAL_PARAMS:,}\n"
            f"Measured : {total:,}\n"
            f"Delta    : {total - CANONICAL_PARAMS:+,}\n\n"
            "Do NOT continue with PB0-C."
        )

    print("Canonical validation      : PASS")

    return total


# ============================================================================
# MATRIX DONOR ACCOUNTING
# ============================================================================

def factorized_projection_params(
    input_dim: int,
    output_dim: int,
    rank: int,
) -> int:
    """
    Dense:
        input_dim * output_dim

    Rank-r factorization:
        input_dim * rank + rank * output_dim
    """

    return (
        input_dim * rank
        + rank * output_dim
    )


def projection_savings(
    *,
    dim: int,
    rank: int,
) -> int:

    dense = dim * dim

    factorized = factorized_projection_params(
        dim,
        dim,
        rank,
    )

    return dense - factorized


def calculate_matrix_savings(
    m_wq_rank: int,
    m_wk_rank: int,
    n_layers: int,
) -> Dict[str, int]:

    q_per_layer = projection_savings(
        dim=512,
        rank=m_wq_rank,
    )

    k_per_layer = projection_savings(
        dim=512,
        rank=m_wk_rank,
    )

    q_total = q_per_layer * n_layers
    k_total = k_per_layer * n_layers

    total = q_total + k_total

    return {
        "m_wq_rank": m_wq_rank,
        "m_wk_rank": m_wk_rank,
        "m_wq_savings_per_layer": q_per_layer,
        "m_wk_savings_per_layer": k_per_layer,
        "m_wq_total_savings": q_total,
        "m_wk_total_savings": k_total,
        "total_matrix_savings": total,
    }


# ============================================================================
# ONE-DIMENSIONAL PARAMETER SWEEPS
# ============================================================================

def measure_hidden_sweep() -> Dict[int, int]:

    section("MEASURING HIDDEN-DIMENSION PARAMETER SWEEP")

    result: Dict[int, int] = {}

    for index, hidden_dim in enumerate(
        HIDDEN_DIM_VALUES,
        start=1,
    ):
        print(
            f"[hidden {index:02d}/{len(HIDDEN_DIM_VALUES):02d}] "
            f"hidden_dim={hidden_dim}"
        )

        model, total = make_counted_model(
            hidden_dim=hidden_dim,
            mamba_state_dim=CANONICAL["mamba_state_dim"],
            ax_res=CANONICAL["ax_res"],
        )

        del model

        delta = total - CANONICAL_PARAMS

        result[hidden_dim] = delta

        print(
            f"    total={total:,} "
            f"delta={delta:+,}"
        )

    return result


def measure_mamba_state_sweep() -> Dict[int, int]:

    section("MEASURING MAMBA-STATE PARAMETER SWEEP")

    result: Dict[int, int] = {}

    for index, state_dim in enumerate(
        MAMBA_STATE_VALUES,
        start=1,
    ):
        print(
            f"[mamba {index:02d}/{len(MAMBA_STATE_VALUES):02d}] "
            f"mamba_state_dim={state_dim}"
        )

        model, total = make_counted_model(
            hidden_dim=CANONICAL["hidden_dim"],
            mamba_state_dim=state_dim,
            ax_res=CANONICAL["ax_res"],
        )

        del model

        delta = total - CANONICAL_PARAMS

        result[state_dim] = delta

        print(
            f"    total={total:,} "
            f"delta={delta:+,}"
        )

    return result


def measure_ax_res_sweep() -> Dict[int, int]:

    section("MEASURING AX-RES PARAMETER SWEEP")

    result: Dict[int, int] = {}

    for index, ax_res in enumerate(
        AX_RES_VALUES,
        start=1,
    ):
        print(
            f"[ax_res {index:02d}/{len(AX_RES_VALUES):02d}] "
            f"ax_res={ax_res}"
        )

        model, total = make_counted_model(
            hidden_dim=CANONICAL["hidden_dim"],
            mamba_state_dim=CANONICAL["mamba_state_dim"],
            ax_res=ax_res,
        )

        del model

        delta = total - CANONICAL_PARAMS

        result[ax_res] = delta

        print(
            f"    total={total:,} "
            f"delta={delta:+,}"
        )

    return result


# ============================================================================
# COMBINED ESTIMATOR
# ============================================================================

def estimate_vector_delta(
    *,
    hidden_dim: int,
    mamba_state_dim: int,
    ax_res: int,
    hidden_delta: Dict[int, int],
    mamba_delta: Dict[int, int],
    ax_delta: Dict[int, int],
) -> int:
    """
    First-order additive estimate of the vector-side parameter delta.

    The canonical point is zero for each component.

    IMPORTANT:
    This is an efficient SEARCH ESTIMATOR, not the final truth.

    Final candidates are always checked by constructing the actual model.
    """

    return (
        hidden_delta[hidden_dim]
        + mamba_delta[mamba_state_dim]
        + ax_delta[ax_res]
    )


# ============================================================================
# SEARCH
# ============================================================================

def generate_search_candidates(
    candidate_name: str,
    matrix_info: Dict[str, int],
    hidden_delta: Dict[int, int],
    mamba_delta: Dict[int, int],
    ax_delta: Dict[int, int],
) -> List[Dict[str, Any]]:

    saved = matrix_info["total_matrix_savings"]

    section(
        f"SEARCHING {candidate_name.upper()}"
    )

    print(
        f"Recovered matrix budget : "
        f"{saved:,}"
    )

    results: List[Dict[str, Any]] = []

    combinations = (
        len(HIDDEN_DIM_VALUES)
        * len(MAMBA_STATE_VALUES)
        * len(AX_RES_VALUES)
    )

    print(
        f"Mathematical combinations : "
        f"{combinations:,}"
    )

    print(
        "No model construction is performed during "
        "this Cartesian search."
    )

    checked = 0

    for hidden_dim in HIDDEN_DIM_VALUES:

        for mamba_state_dim in MAMBA_STATE_VALUES:

            for ax_res in AX_RES_VALUES:

                checked += 1

                vector_delta = estimate_vector_delta(
                    hidden_dim=hidden_dim,
                    mamba_state_dim=mamba_state_dim,
                    ax_res=ax_res,
                    hidden_delta=hidden_delta,
                    mamba_delta=mamba_delta,
                    ax_delta=ax_delta,
                )

                if vector_delta < 0:
                    continue

                if vector_delta > saved:
                    continue

                candidate_total = (
                    CANONICAL_PARAMS
                    - saved
                    + vector_delta
                )

                budget_error = (
                    candidate_total
                    - CANONICAL_PARAMS
                )

                result = {
                    "candidate": candidate_name,
                    "m_wq_rank": matrix_info["m_wq_rank"],
                    "m_wk_rank": matrix_info["m_wk_rank"],
                    "matrix_saved_params": saved,
                    "hidden_dim": hidden_dim,
                    "mamba_state_dim": mamba_state_dim,
                    "ax_res": ax_res,
                    "estimated_vector_side_delta": vector_delta,
                    "estimated_candidate_total_params": (
                        candidate_total
                    ),
                    "estimated_budget_error": budget_error,
                    "estimated_exact_budget": (
                        budget_error == 0
                    ),
                    "estimated_unused_recovered_params": (
                        saved - vector_delta
                    ),
                }

                results.append(result)

    print(
        f"Combinations mathematically checked : "
        f"{checked:,}"
    )

    print(
        f"Feasible mathematical candidates     : "
        f"{len(results):,}"
    )

    return results


# ============================================================================
# RANKING
# ============================================================================

def rank_estimated_results(
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    return sorted(
        results,
        key=lambda x: (
            abs(x["estimated_budget_error"]),
            -x["estimated_vector_side_delta"],
            -x["mamba_state_dim"],
            -x["hidden_dim"],
            -x["ax_res"],
        ),
    )


# ============================================================================
# EXACT VERIFICATION
# ============================================================================

def verify_candidates(
    candidate_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    section("EXACT MODEL VERIFICATION")

    ranked = rank_estimated_results(
        candidate_results
    )

    # Select a small verification pool.
    verify_pool = ranked[
        :FINAL_VERIFY_COUNT
    ]

    print(
        f"Verifying {len(verify_pool)} "
        f"candidate architectures."
    )

    verified: List[Dict[str, Any]] = []

    for index, candidate in enumerate(
        verify_pool,
        start=1,
    ):

        hidden_dim = candidate["hidden_dim"]
        mamba_state_dim = candidate["mamba_state_dim"]
        ax_res = candidate["ax_res"]

        print()
        print(
            f"[verify {index:02d}/{len(verify_pool):02d}] "
            f"hidden={hidden_dim} "
            f"mamba_state={mamba_state_dim} "
            f"ax_res={ax_res}"
        )

        model, actual_total = make_counted_model(
            hidden_dim=hidden_dim,
            mamba_state_dim=mamba_state_dim,
            ax_res=ax_res,
        )

        del model

        actual_vector_delta = (
            actual_total
            - CANONICAL_PARAMS
        )

        saved = candidate[
            "matrix_saved_params"
        ]

        actual_candidate_total = (
            CANONICAL_PARAMS
            - saved
            + actual_vector_delta
        )

        actual_budget_error = (
            actual_candidate_total
            - CANONICAL_PARAMS
        )

        estimator_error = (
            actual_vector_delta
            - candidate[
                "estimated_vector_side_delta"
            ]
        )

        verified_result = dict(candidate)

        verified_result.update(
            {
                "actual_vector_side_delta": (
                    actual_vector_delta
                ),
                "actual_vector_model_total": (
                    actual_total
                ),
                "actual_candidate_total_params": (
                    actual_candidate_total
                ),
                "actual_budget_error": (
                    actual_budget_error
                ),
                "actual_exact_budget": (
                    actual_budget_error == 0
                ),
                "estimator_error_params": (
                    estimator_error
                ),
                "actual_unused_recovered_params": (
                    saved - actual_vector_delta
                ),
            }
        )

        verified.append(
            verified_result
        )

        print(
            f"    estimated vector delta : "
            f"{candidate['estimated_vector_side_delta']:+,}"
        )

        print(
            f"    actual vector delta    : "
            f"{actual_vector_delta:+,}"
        )

        print(
            f"    estimator error        : "
            f"{estimator_error:+,}"
        )

        print(
            f"    actual candidate total : "
            f"{actual_candidate_total:,}"
        )

        print(
            f"    actual budget error    : "
            f"{actual_budget_error:+,}"
        )

        print(
            f"    EXACT                  : "
            f"{'YES' if actual_budget_error == 0 else 'NO'}"
        )

    return verified


# ============================================================================
# VERIFIED RANKING
# ============================================================================

def rank_verified_results(
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    return sorted(
        results,
        key=lambda x: (
            abs(x["actual_budget_error"]),
            -x["actual_vector_side_delta"],
            -x["mamba_state_dim"],
            -x["hidden_dim"],
            -x["ax_res"],
        ),
    )


# ============================================================================
# JSON REPORT
# ============================================================================

def write_json(
    matrix_accounting: Dict[str, Dict[str, int]],
    sweeps: Dict[str, Dict[int, int]],
    estimated_results: Dict[str, List[Dict[str, Any]]],
    verified_results: Dict[str, List[Dict[str, Any]]],
) -> Path:

    path = (
        OUTPUT_DIR
        / "PB0_C_FIXED_BUDGET_ALLOCATION.json"
    )

    payload = {
        "experiment": "PB0-C",
        "status": "complete",
        "canonical_parameter_count": (
            CANONICAL_PARAMS
        ),
        "canonical_config": CANONICAL,
        "matrix_accounting": matrix_accounting,
        "one_dimensional_sweeps": sweeps,
        "estimated_results": estimated_results,
        "verified_results": verified_results,
        "method": {
            "cartesian_search": (
                "mathematical_only"
            ),
            "exact_model_construction": (
                "one_dimensional_sweeps_and_final_verification"
            ),
            "final_training": (
                "deferred_to_PB1"
            ),
        },
    }

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return path


# ============================================================================
# CSV REPORT
# ============================================================================

def write_csv(
    verified_results: Dict[str, List[Dict[str, Any]]],
) -> Path:

    path = (
        OUTPUT_DIR
        / "PB0_C_FIXED_BUDGET_ALLOCATION.csv"
    )

    rows: List[Dict[str, Any]] = []

    for candidate_name, results in (
        verified_results.items()
    ):

        ranked = rank_verified_results(
            results
        )

        for rank, result in enumerate(
            ranked,
            start=1,
        ):

            row = dict(result)
            row["verified_rank"] = rank
            rows.append(row)

    if not rows:
        path.write_text(
            "",
            encoding="utf-8",
        )
        return path

    fieldnames = list(rows[0].keys())

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    return path


# ============================================================================
# MARKDOWN REPORT
# ============================================================================

def write_markdown(
    matrix_accounting: Dict[str, Dict[str, int]],
    verified_results: Dict[str, List[Dict[str, Any]]],
) -> Path:

    path = (
        OUTPUT_DIR
        / "PB0_C_FIXED_BUDGET_ALLOCATION.md"
    )

    lines: List[str] = []

    lines.append(
        "# PB0-C — Fixed-Budget Parameter Allocation"
    )

    lines.append("")

    lines.append(
        f"Canonical parameter budget: "
        f"**{CANONICAL_PARAMS:,}**"
    )

    lines.append("")

    lines.append(
        "PB0-C is an accounting/search stage. "
        "No training is performed."
    )

    lines.append("")

    lines.append(
        "## Method"
    )

    lines.append("")

    lines.append(
        "The full Cartesian allocation space is searched "
        "mathematically using measured one-dimensional "
        "parameter-count deltas."
    )

    lines.append("")

    lines.append(
        "Only the final promising candidates are constructed "
        "with the actual Modus_X model for exact verification."
    )

    lines.append("")

    lines.append(
        "## Matrix donor accounting"
    )

    lines.append("")

    lines.append(
        "| Candidate | m_wq rank | m_wk rank | "
        "Recovered parameters | Model share |"
    )

    lines.append(
        "|---|---:|---:|---:|---:|"
    )

    for name, info in (
        matrix_accounting.items()
    ):

        saved = info[
            "total_matrix_savings"
        ]

        lines.append(
            f"| {name} | "
            f"{info['m_wq_rank']} | "
            f"{info['m_wk_rank']} | "
            f"{saved:,} | "
            f"{pct(saved):.3f}% |"
        )

    lines.append("")

    lines.append(
        "## Verified candidates"
    )

    lines.append("")

    for name, results in (
        verified_results.items()
    ):

        lines.append(
            f"### {name}"
        )

        lines.append("")

        ranked = rank_verified_results(
            results
        )

        lines.append(
            "| Rank | hidden_dim | mamba_state_dim | "
            "ax_res | Actual vector Δ | "
            "Actual total | Budget error | Exact | "
            "Estimator error |"
        )

        lines.append(
            "|---:|---:|---:|---:|---:|---:|---:|:---:|---:|"
        )

        for rank, result in enumerate(
            ranked[:15],
            start=1,
        ):

            lines.append(
                f"| {rank} | "
                f"{result['hidden_dim']} | "
                f"{result['mamba_state_dim']} | "
                f"{result['ax_res']} | "
                f"{result['actual_vector_side_delta']:+,} | "
                f"{result['actual_candidate_total_params']:,} | "
                f"{result['actual_budget_error']:+,} | "
                f"{'YES' if result['actual_exact_budget'] else 'NO'} | "
                f"{result['estimator_error_params']:+,} |"
            )

        lines.append("")

    lines.append(
        "## Interpretation"
    )

    lines.append("")

    lines.append(
        "PB0-B provides the justification for treating "
        "m_wq and m_wk as candidate donor projections."
    )

    lines.append("")

    lines.append(
        "PB0-C does not claim that increased vector/Mamba "
        "capacity improves BPC. It only identifies "
        "parameter-valid allocations for training."
    )

    lines.append("")

    lines.append(
        "An exact 47,437,768-parameter allocation is "
        "preferred for the fixed-budget comparison."
    )

    lines.append("")

    lines.append(
        "The final architecture must be selected using "
        "matched training and validation/test BPC."
    )

    lines.append("")

    lines.append(
        "## Next experiment"
    )

    lines.append("")

    lines.append(
        "PB1-A: implement the factorized m_wq/m_wk "
        "architecture and train the best verified "
        "fixed-budget candidates against the canonical "
        "47M baseline."
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return path


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    section(
        "MODUS_X PB0-C OPTIMIZED FIXED-BUDGET SEARCH"
    )

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"Language dir : {LANGUAGE_DIR}"
    )

    print(
        f"Output dir   : {OUTPUT_DIR}"
    )

    print(
        f"Canonical    : {CANONICAL_PARAMS:,}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------------
    # 1. Canonical validation.
    # ------------------------------------------------------------------------

    validate_canonical_model()

    # ------------------------------------------------------------------------
    # 2. Matrix donor accounting.
    # ------------------------------------------------------------------------

    section(
        "MATRIX DONOR ACCOUNTING"
    )

    matrix_accounting: Dict[
        str,
        Dict[str, int]
    ] = {}

    for name, ranks in (
        MATRIX_CANDIDATES.items()
    ):

        info = calculate_matrix_savings(
            m_wq_rank=ranks["m_wq_rank"],
            m_wk_rank=ranks["m_wk_rank"],
            n_layers=CANONICAL["n_layers"],
        )

        matrix_accounting[name] = info

        print()
        print(name)
        print("-" * 60)

        print(
            f"m_wq rank       : "
            f"{info['m_wq_rank']}"
        )

        print(
            f"m_wk rank       : "
            f"{info['m_wk_rank']}"
        )

        print(
            f"m_wq saved      : "
            f"{info['m_wq_total_savings']:,}"
        )

        print(
            f"m_wk saved      : "
            f"{info['m_wk_total_savings']:,}"
        )

        print(
            f"TOTAL RECOVERED : "
            f"{info['total_matrix_savings']:,}"
        )

    # ------------------------------------------------------------------------
    # 3. One-dimensional exact sweeps.
    # ------------------------------------------------------------------------

    section(
        "ONE-DIMENSIONAL EXACT PARAMETER SWEEPS"
    )

    hidden_delta = (
        measure_hidden_sweep()
    )

    mamba_delta = (
        measure_mamba_state_sweep()
    )

    ax_delta = (
        measure_ax_res_sweep()
    )

    sweeps = {
        "hidden_dim_delta": hidden_delta,
        "mamba_state_dim_delta": mamba_delta,
        "ax_res_delta": ax_delta,
    }

    # ------------------------------------------------------------------------
    # 4. Mathematical Cartesian search.
    # ------------------------------------------------------------------------

    estimated_results: Dict[
        str,
        List[Dict[str, Any]]
    ] = {}

    for name, info in (
        matrix_accounting.items()
    ):

        results = generate_search_candidates(
            candidate_name=name,
            matrix_info=info,
            hidden_delta=hidden_delta,
            mamba_delta=mamba_delta,
            ax_delta=ax_delta,
        )

        estimated_results[name] = results

    # ------------------------------------------------------------------------
    # 5. Exact verification.
    # ------------------------------------------------------------------------

    verified_results: Dict[
        str,
        List[Dict[str, Any]]
    ] = {}

    for name, results in (
        estimated_results.items()
    ):

        verified = verify_candidates(
            results
        )

        verified_results[name] = verified

    # ------------------------------------------------------------------------
    # 6. Reports.
    # ------------------------------------------------------------------------

    section(
        "WRITING PB0-C REPORTS"
    )

    json_path = write_json(
        matrix_accounting=matrix_accounting,
        sweeps=sweeps,
        estimated_results=estimated_results,
        verified_results=verified_results,
    )

    csv_path = write_csv(
        verified_results
    )

    md_path = write_markdown(
        matrix_accounting=matrix_accounting,
        verified_results=verified_results,
    )

    print()
    print(
        f"JSON : {json_path}"
    )

    print(
        f"CSV  : {csv_path}"
    )

    print(
        f"MD   : {md_path}"
    )

    # ------------------------------------------------------------------------
    # 7. Final recommendations.
    # ------------------------------------------------------------------------

    section(
        "PB0-C VERIFIED RESULTS"
    )

    for name, results in (
        verified_results.items()
    ):

        print()
        print(name)
        print("-" * 75)

        ranked = rank_verified_results(
            results
        )

        exact = [
            r
            for r in ranked
            if r["actual_exact_budget"]
        ]

        if exact:

            print(
                f"Exact-budget candidates found: "
                f"{len(exact)}"
            )

            for index, result in enumerate(
                exact[:5],
                start=1,
            ):

                print(
                    f"{index}. "
                    f"hidden={result['hidden_dim']} | "
                    f"mamba_state={result['mamba_state_dim']} | "
                    f"ax_res={result['ax_res']} | "
                    f"total={result['actual_candidate_total_params']:,}"
                )

        else:

            print(
                "No exact-budget candidate in the "
                "verified pool."
            )

            print(
                "Closest verified candidates:"
            )

            for index, result in enumerate(
                ranked[:5],
                start=1,
            ):

                print(
                    f"{index}. "
                    f"hidden={result['hidden_dim']} | "
                    f"mamba_state={result['mamba_state_dim']} | "
                    f"ax_res={result['ax_res']} | "
                    f"budget_error={result['actual_budget_error']:+,}"
                )

    section(
        "PB0-C COMPLETE"
    )

    print(
        "Next gate: use the verified fixed-budget "
        "architectures for PB1-A training."
    )


if __name__ == "__main__":
    main()