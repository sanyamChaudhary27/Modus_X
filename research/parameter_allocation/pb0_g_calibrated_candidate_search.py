from __future__ import annotations
import csv
import itertools
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "research" / "parameter_allocation" / "outputs"

PB0_F_PATH = (
    OUTPUT_DIR / "PB0_F_PARAMETER_FORMULA_CALIBRATION.json"
)

B0_CENSUS_PATH = OUTPUT_DIR / "b0_parameter_census.json"

OUTPUT_JSON_PATH = (
    OUTPUT_DIR / "PB0_G_CALIBRATED_CANDIDATE_SEARCH.json"
)

OUTPUT_MARKDOWN_PATH = (
    OUTPUT_DIR / "PB0_G_CALIBRATED_CANDIDATE_SEARCH.md"
)

OUTPUT_CSV_PATH = (
    OUTPUT_DIR / "PB0_G_CALIBRATED_CANDIDATES.csv"
)


# =============================================================================
# SEARCH CONSTANTS
# =============================================================================

CANONICAL_R = 512
CANONICAL_N = 512
CANONICAL_H = 32

DEFAULT_TARGET_PARAMETER_COUNT = 47_437_768

MAX_RESULTS_TO_STORE = 500

MAX_ABSOLUTE_PARAMETER_ERROR = 50_000

R_MIN = 16
R_MAX = 512
R_STEP = 8

N_MIN = 16
N_MAX = 768
N_STEP = 8

H_MIN = 4
H_MAX = 128
H_STEP = 1


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass(frozen=True)
class CalibratedFormula:
    intercept: int
    r_coefficient: int
    n_coefficient: int
    h_coefficient: int
    feedback_r_coefficient: int
    feedback_constant_coefficient: int
    feedback_cap: int


@dataclass(frozen=True)
class Candidate:
    rank_r: int
    vector_dimension_n: int
    router_hidden_h: int
    feedback_rank_f: int
    parameter_count: int
    parameter_error: int
    matrix_parameter_change: int
    vector_parameter_change: int
    router_parameter_change: int
    feedback_parameter_change: int
    rank_ratio: float
    vector_ratio: float
    router_ratio: float
    allocation_distance: float
    score: float
    exact_match: bool

@dataclass(frozen=True)
class CalibratedFormula:
    """
    Exact parameter formula calibrated against the actual canonical
    Modus_X initializer during PB0-F.
    """

    intercept: int
    r_coefficient: int
    n_coefficient: int
    h_coefficient: int
    feedback_r_coefficient: int
    feedback_constant_coefficient: int
    feedback_cap: int

    def feedback_rank(
        self,
        matrix_rank: int,
        vector_dim: int,
    ) -> int:
        """
        Resolve the feedback bottleneck rank according to the canonical
        architecture rule.
        """

        if matrix_rank <= 0:
            raise ValueError(
                f"matrix_rank must be positive, got {matrix_rank}"
            )

        if vector_dim <= 0:
            raise ValueError(
                f"vector_dim must be positive, got {vector_dim}"
            )

        return min(
            self.feedback_cap,
            matrix_rank,
            vector_dim,
        )

    def parameter_count(
        self,
        matrix_rank: int,
        vector_dim: int,
        router_hidden: int,
    ) -> int:
        """
        Compute the exact calibrated parameter count.
        """

        if matrix_rank <= 0:
            raise ValueError(
                f"matrix_rank must be positive, got {matrix_rank}"
            )

        if vector_dim <= 0:
            raise ValueError(
                f"vector_dim must be positive, got {vector_dim}"
            )

        if router_hidden <= 0:
            raise ValueError(
                f"router_hidden must be positive, got {router_hidden}"
            )

        feedback_rank = self.feedback_rank(
            matrix_rank=matrix_rank,
            vector_dim=vector_dim,
        )

        return (
            self.intercept
            + self.r_coefficient * matrix_rank
            + self.n_coefficient * vector_dim
            + self.h_coefficient * router_hidden
            + self.feedback_r_coefficient
            * feedback_rank
            * matrix_rank
            + self.feedback_constant_coefficient
            * feedback_rank
        )

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_banner(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input file does not exist:\n{path}"
        )

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Failed to parse JSON file:\n{path}\n\n{exc}"
        ) from exc

    if not isinstance(data, dict):
        raise TypeError(
            f"Expected JSON object at root of:\n{path}\n"
            f"Received: {type(data).__name__}"
        )

    return data


def find_integer_value(
    data: Any,
    candidate_keys: tuple[str, ...],
) -> int | None:
    if isinstance(data, dict):
        for key in candidate_keys:
            value = data.get(key)

            if isinstance(value, int):
                return value

        for value in data.values():
            result = find_integer_value(value, candidate_keys)

            if result is not None:
                return result

    elif isinstance(data, list):
        for item in data:
            result = find_integer_value(item, candidate_keys)

            if result is not None:
                return result

    return None


# =============================================================================
# INPUT LOADING
# =============================================================================

def load_target_parameter_count() -> int:
    data = load_json(B0_CENSUS_PATH)

    preferred_keys = (
        "actual_parameter_count",
        "expected_parameter_count",
        "total_parameters",
        "parameter_count",
    )

    value = find_integer_value(data, preferred_keys)

    if value is None:
        print(
            "WARNING: Could not resolve canonical parameter count "
            "from B0 census."
        )
        print(
            f"Using fallback target: "
            f"{DEFAULT_TARGET_PARAMETER_COUNT:,}"
        )

        return DEFAULT_TARGET_PARAMETER_COUNT

    if value <= 0:
        raise ValueError(
            "Resolved canonical parameter count is not positive:\n"
            f"{value}"
        )

    return value

def load_calibrated_formula():
    """
    Load the exact calibrated parameter formula produced by PB0-F.

    PB0-F stores formula coefficients in JSON as a dictionary. PB0-G
    converts those coefficients into the CalibratedFormula dataclass so
    the rest of the search code can use typed attribute access.
    """

    if not PB0_F_PATH.exists():
        raise FileNotFoundError(
            f"PB0-F calibration output was not found:\n{PB0_F_PATH}"
        )

    with PB0_F_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise TypeError(
            "PB0-F calibration JSON must contain a top-level object."
        )

    coefficients = data.get("formula_coefficients")

    if not isinstance(coefficients, dict):
        raise KeyError(
            "Could not find a valid 'formula_coefficients' object in:\n"
            f"{PB0_F_PATH}"
        )

    required_fields = (
        "intercept",
        "matrix_rank",
        "vector_dim",
        "router_hidden",
        "feedback_rank_times_matrix_rank",
        "feedback_rank",
    )

    missing_fields = [
        field_name
        for field_name in required_fields
        if field_name not in coefficients
    ]

    if missing_fields:
        raise KeyError(
            "PB0-F formula_coefficients is missing required fields:\n"
            + "\n".join(
                f"  - {field_name}"
                for field_name in missing_fields
            )
        )

    canonical_configuration = data.get("canonical_configuration")

    if not isinstance(canonical_configuration, dict):
        raise KeyError(
            "PB0-F calibration output does not contain a valid "
            "'canonical_configuration' object."
        )

    feedback_rule = data.get("feedback_rank_rule")

    if feedback_rule != "min(32, ax_res, mamba_state_dim)":
        raise ValueError(
            "PB0-G currently expects the canonical feedback rank rule:\n"
            "min(32, ax_res, mamba_state_dim)\n\n"
            f"Found:\n{feedback_rule!r}"
        )

    return CalibratedFormula(
        intercept=int(coefficients["intercept"]),
        r_coefficient=int(coefficients["matrix_rank"]),
        n_coefficient=int(coefficients["vector_dim"]),
        h_coefficient=int(coefficients["router_hidden"]),
        feedback_r_coefficient=int(
            coefficients["feedback_rank_times_matrix_rank"]
        ),
        feedback_constant_coefficient=int(
            coefficients["feedback_rank"]
        ),
        feedback_cap=32,
    )

# =============================================================================
# PARAMETER FORMULA
# =============================================================================

def resolve_feedback_rank(
    rank_r: int,
    vector_dimension_n: int,
    feedback_cap: int,
) -> int:
    if rank_r <= 0:
        raise ValueError(
            f"rank_r must be positive. Received: {rank_r}"
        )

    if vector_dimension_n <= 0:
        raise ValueError(
            "vector_dimension_n must be positive. "
            f"Received: {vector_dimension_n}"
        )

    if feedback_cap <= 0:
        raise ValueError(
            f"feedback_cap must be positive. "
            f"Received: {feedback_cap}"
        )

    return min(
        feedback_cap,
        rank_r,
        vector_dimension_n,
    )


def calculate_parameter_count(
    formula: CalibratedFormula,
    rank_r: int,
    vector_dimension_n: int,
    router_hidden_h: int,
) -> tuple[int, int]:
    if rank_r <= 0:
        raise ValueError(
            f"rank_r must be positive. Received: {rank_r}"
        )

    if vector_dimension_n <= 0:
        raise ValueError(
            "vector_dimension_n must be positive. "
            f"Received: {vector_dimension_n}"
        )

    if router_hidden_h <= 0:
        raise ValueError(
            "router_hidden_h must be positive. "
            f"Received: {router_hidden_h}"
        )

    feedback_rank = resolve_feedback_rank(
        rank_r=rank_r,
        vector_dimension_n=vector_dimension_n,
        feedback_cap=formula.feedback_cap,
    )

    parameter_count = (
        formula.intercept
        + formula.r_coefficient * rank_r
        + formula.n_coefficient * vector_dimension_n
        + formula.h_coefficient * router_hidden_h
        + formula.feedback_r_coefficient
        * feedback_rank
        * rank_r
        + formula.feedback_constant_coefficient
        * feedback_rank
    )

    return parameter_count, feedback_rank


# =============================================================================
# BASELINE CONTRIBUTION ESTIMATES
# =============================================================================

def calculate_variable_contributions(
    formula: CalibratedFormula,
    rank_r: int,
    vector_dimension_n: int,
    router_hidden_h: int,
) -> tuple[int, int, int, int]:
    feedback_rank = resolve_feedback_rank(
        rank_r=rank_r,
        vector_dimension_n=vector_dimension_n,
        feedback_cap=formula.feedback_cap,
    )

    matrix_contribution = (
        formula.r_coefficient * rank_r
    )

    vector_contribution = (
        formula.n_coefficient * vector_dimension_n
    )

    router_contribution = (
        formula.h_coefficient * router_hidden_h
    )

    feedback_contribution = (
        formula.feedback_r_coefficient
        * feedback_rank
        * rank_r
        + formula.feedback_constant_coefficient
        * feedback_rank
    )

    return (
        matrix_contribution,
        vector_contribution,
        router_contribution,
        feedback_contribution,
    )


# =============================================================================
# CANDIDATE SCORING
# =============================================================================

def calculate_candidate_score(
    parameter_error: int,
    rank_r: int,
    vector_dimension_n: int,
    router_hidden_h: int,
) -> tuple[float, float]:
    rank_ratio = rank_r / CANONICAL_R
    vector_ratio = vector_dimension_n / CANONICAL_N
    router_ratio = router_hidden_h / CANONICAL_H

    allocation_distance = (
        abs(rank_ratio - 1.0)
        + abs(vector_ratio - 1.0)
        + 0.50 * abs(router_ratio - 1.0)
    )

    score = (
        parameter_error
        + allocation_distance * 1_000_000.0
    )

    return score, allocation_distance


# =============================================================================
# SEARCH
# =============================================================================

def generate_candidates(
    formula: CalibratedFormula,
    target_parameter_count: int,
) -> list[Candidate]:
    candidates: list[Candidate] = []

    baseline_matrix, baseline_vector, baseline_router, baseline_feedback = (
        calculate_variable_contributions(
            formula=formula,
            rank_r=CANONICAL_R,
            vector_dimension_n=CANONICAL_N,
            router_hidden_h=CANONICAL_H,
        )
    )

    rank_values = range(
        R_MIN,
        R_MAX + 1,
        R_STEP,
    )

    vector_values = range(
        N_MIN,
        N_MAX + 1,
        N_STEP,
    )

    router_values = range(
        H_MIN,
        H_MAX + 1,
        H_STEP,
    )

    total_search_space = (
        len(list(rank_values))
        * len(list(vector_values))
        * len(list(router_values))
    )

    print(
        f"Search space size: {total_search_space:,} candidates"
    )

    checked = 0

    for rank_r, vector_dimension_n, router_hidden_h in itertools.product(
        rank_values,
        vector_values,
        router_values,
    ):
        checked += 1

        parameter_count, feedback_rank = (
            calculate_parameter_count(
                formula=formula,
                rank_r=rank_r,
                vector_dimension_n=vector_dimension_n,
                router_hidden_h=router_hidden_h,
            )
        )

        parameter_error = (
            parameter_count - target_parameter_count
        )

        absolute_parameter_error = abs(
            parameter_error
        )

        if (
            absolute_parameter_error
            > MAX_ABSOLUTE_PARAMETER_ERROR
        ):
            continue

        (
            matrix_contribution,
            vector_contribution,
            router_contribution,
            feedback_contribution,
        ) = calculate_variable_contributions(
            formula=formula,
            rank_r=rank_r,
            vector_dimension_n=vector_dimension_n,
            router_hidden_h=router_hidden_h,
        )

        (
            score,
            allocation_distance,
        ) = calculate_candidate_score(
            parameter_error=absolute_parameter_error,
            rank_r=rank_r,
            vector_dimension_n=vector_dimension_n,
            router_hidden_h=router_hidden_h,
        )

        candidates.append(
            Candidate(
                rank_r=rank_r,
                vector_dimension_n=vector_dimension_n,
                router_hidden_h=router_hidden_h,
                feedback_rank_f=feedback_rank,
                parameter_count=parameter_count,
                parameter_error=parameter_error,
                matrix_parameter_change=(
                    matrix_contribution
                    - baseline_matrix
                ),
                vector_parameter_change=(
                    vector_contribution
                    - baseline_vector
                ),
                router_parameter_change=(
                    router_contribution
                    - baseline_router
                ),
                feedback_parameter_change=(
                    feedback_contribution
                    - baseline_feedback
                ),
                rank_ratio=rank_r / CANONICAL_R,
                vector_ratio=(
                    vector_dimension_n
                    / CANONICAL_N
                ),
                router_ratio=(
                    router_hidden_h
                    / CANONICAL_H
                ),
                allocation_distance=allocation_distance,
                score=score,
                exact_match=(
                    parameter_count
                    == target_parameter_count
                ),
            )
        )

    print(
        f"Candidates within ±{MAX_ABSOLUTE_PARAMETER_ERROR:,}: "
        f"{len(candidates):,}"
    )

    candidates.sort(
        key=lambda candidate: (
            0 if candidate.exact_match else 1,
            abs(candidate.parameter_error),
            candidate.score,
            candidate.allocation_distance,
            -candidate.rank_r,
            -candidate.vector_dimension_n,
        )
    )

    if len(candidates) > MAX_RESULTS_TO_STORE:
        candidates = candidates[
            :MAX_RESULTS_TO_STORE
        ]

    return candidates


# =============================================================================
# REPORTING
# =============================================================================

def write_csv(
    candidates: list[Candidate],
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "rank_r",
        "vector_dimension_n",
        "router_hidden_h",
        "feedback_rank_f",
        "parameter_count",
        "parameter_error",
        "absolute_parameter_error",
        "exact_match",
        "matrix_parameter_change",
        "vector_parameter_change",
        "router_parameter_change",
        "feedback_parameter_change",
        "rank_ratio",
        "vector_ratio",
        "router_ratio",
        "allocation_distance",
        "score",
    ]

    with OUTPUT_CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for candidate in candidates:
            row = asdict(candidate)

            row[
                "absolute_parameter_error"
            ] = abs(
                candidate.parameter_error
            )

            writer.writerow(row)


def write_json_output(
    formula: CalibratedFormula,
    target_parameter_count: int,
    candidates: list[Candidate],
) -> None:
    exact_candidates = [
        candidate
        for candidate in candidates
        if candidate.exact_match
    ]

    best_candidate = (
        asdict(candidates[0])
        if candidates
        else None
    )

    output = {
        "report": (
            "PB0-G Calibrated Candidate Search"
        ),
        "generated_at_utc": (
            datetime.now(timezone.utc)
            .isoformat()
        ),
        "project_root": str(PROJECT_ROOT),
        "target_parameter_count": (
            target_parameter_count
        ),
        "canonical_dimensions": {
            "rank_r": CANONICAL_R,
            "vector_dimension_n": (
                CANONICAL_N
            ),
            "router_hidden_h": (
                CANONICAL_H
            ),
        },
        "search_ranges": {
            "rank_r": {
                "minimum": R_MIN,
                "maximum": R_MAX,
                "step": R_STEP,
            },
            "vector_dimension_n": {
                "minimum": N_MIN,
                "maximum": N_MAX,
                "step": N_STEP,
            },
            "router_hidden_h": {
                "minimum": H_MIN,
                "maximum": H_MAX,
                "step": H_STEP,
            },
            "maximum_absolute_parameter_error": (
                MAX_ABSOLUTE_PARAMETER_ERROR
            ),
        },
        "calibrated_formula": asdict(
            formula
        ),
        "stored_candidate_count": (
            len(candidates)
        ),
        "exact_candidate_count": (
            len(exact_candidates)
        ),
        "best_candidate": (
            best_candidate
        ),
        "candidates": [
            asdict(candidate)
            for candidate in candidates
        ],
    }

    with OUTPUT_JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )


def write_markdown_output(
    formula: CalibratedFormula,
    target_parameter_count: int,
    candidates: list[Candidate],
) -> None:
    exact_candidates = [
        candidate
        for candidate in candidates
        if candidate.exact_match
    ]

    lines: list[str] = []

    lines.append(
        "# PB0-G Calibrated Candidate Search"
    )
    lines.append("")

    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "PB0-G searches the actual calibrated "
        "MemoryFeedbackArchive parameter space for "
        "integer dimension configurations near the "
        "frozen canonical parameter budget."
    )
    lines.append("")

    lines.append("## Target")
    lines.append("")
    lines.append(
        f"- Frozen canonical parameter budget: "
        f"**{target_parameter_count:,}**"
    )
    lines.append("")

    lines.append(
        "## Calibrated Formula"
    )
    lines.append("")
    lines.append(
        "```text"
    )
    lines.append(
        f"P = {formula.intercept} "
        f"+ ({formula.r_coefficient} * r) "
        f"+ ({formula.n_coefficient} * n) "
        f"+ ({formula.h_coefficient} * h) "
        f"+ ({formula.feedback_r_coefficient} * f * r) "
        f"+ ({formula.feedback_constant_coefficient} * f)"
    )
    lines.append("")
    lines.append(
        f"f = min({formula.feedback_cap}, r, n)"
    )
    lines.append(
        "```"
    )
    lines.append("")

    lines.append(
        "## Search Result"
    )
    lines.append("")

    lines.append(
        f"- Stored candidates: **{len(candidates):,}**"
    )
    lines.append(
        f"- Exact parameter matches: "
        f"**{len(exact_candidates):,}**"
    )
    lines.append("")

    if candidates:
        best = candidates[0]

        lines.append(
            "## Best Candidate"
        )
        lines.append("")
        lines.append(
            f"- Matrix rank `r`: **{best.rank_r}**"
        )
        lines.append(
            f"- Vector dimension `n`: "
            f"**{best.vector_dimension_n}**"
        )
        lines.append(
            f"- Router width `h`: "
            f"**{best.router_hidden_h}**"
        )
        lines.append(
            f"- Feedback rank `f`: "
            f"**{best.feedback_rank_f}**"
        )
        lines.append(
            f"- Parameter count: "
            f"**{best.parameter_count:,}**"
        )
        lines.append(
            f"- Parameter error: "
            f"**{best.parameter_error:+,}**"
        )
        lines.append(
            f"- Exact match: "
            f"**{best.exact_match}**"
        )
        lines.append("")

    lines.append(
        "## Top Candidates"
    )
    lines.append("")

    lines.append(
        "| Rank r | Vector n | Router h | "
        "Feedback f | Parameters | Error | Exact |"
    )
    lines.append(
        "|---:|---:|---:|---:|---:|---:|---:|"
    )

    for candidate in candidates[:25]:
        lines.append(
            "| "
            f"{candidate.rank_r} | "
            f"{candidate.vector_dimension_n} | "
            f"{candidate.router_hidden_h} | "
            f"{candidate.feedback_rank_f} | "
            f"{candidate.parameter_count:,} | "
            f"{candidate.parameter_error:+,} | "
            f"{candidate.exact_match} |"
        )

    lines.append("")

    lines.append(
        "## Next Gate"
    )
    lines.append("")
    lines.append(
        "Do not train any candidate yet. "
        "The selected candidate must be instantiated "
        "through the actual Modus_X initializer and "
        "verified with an exact parameter-tree recount."
    )
    lines.append("")

    with OUTPUT_MARKDOWN_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "\n".join(lines)
        )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print_banner(
        "MODUS_X PB0-G CALIBRATED CANDIDATE SEARCH"
    )

    print(
        f"Project root : {PROJECT_ROOT}"
    )
    print(
        f"PB0-F input  : {PB0_F_PATH}"
    )
    print(
        f"B0 input     : {B0_CENSUS_PATH}"
    )
    print(
        f"Output dir   : {OUTPUT_DIR}"
    )

    print_banner(
        "LOADING CALIBRATED PARAMETER FORMULA"
    )

    formula = load_calibrated_formula()

    print(
        f"Intercept                    : "
        f"{formula.intercept:,}"
    )
    print(
        f"r coefficient                : "
        f"{formula.r_coefficient:,}"
    )
    print(
        f"n coefficient                : "
        f"{formula.n_coefficient:,}"
    )
    print(
        f"h coefficient                : "
        f"{formula.h_coefficient:,}"
    )
    print(
        f"feedback r coefficient       : "
        f"{formula.feedback_r_coefficient:,}"
    )
    print(
        f"feedback constant coefficient: "
        f"{formula.feedback_constant_coefficient:,}"
    )
    print(
        f"feedback cap                 : "
        f"{formula.feedback_cap}"
    )

    print_banner(
        "LOADING FROZEN CANONICAL PARAMETER BUDGET"
    )

    target_parameter_count = (
        load_target_parameter_count()
    )

    print(
        f"Target parameter count: "
        f"{target_parameter_count:,}"
    )

    print_banner(
        "SEARCHING CALIBRATED ARCHITECTURE SPACE"
    )

    candidates = generate_candidates(
        formula=formula,
        target_parameter_count=(
            target_parameter_count
        ),
    )

    if not candidates:
        raise RuntimeError(
            "PB0-G found no candidates within the "
            "configured parameter-error window."
        )

    print_banner(
        "BEST CALIBRATED CANDIDATES"
    )

    for index, candidate in enumerate(
        candidates[:20],
        start=1,
    ):
        print(
            f"{index:03d} "
            f"r={candidate.rank_r:3d} "
            f"n={candidate.vector_dimension_n:3d} "
            f"h={candidate.router_hidden_h:3d} "
            f"f={candidate.feedback_rank_f:2d} "
            f"params={candidate.parameter_count:,} "
            f"error={candidate.parameter_error:+,} "
            f"exact={candidate.exact_match}"
        )

    print_banner(
        "WRITING PB0-G OUTPUTS"
    )

    write_csv(
        candidates=candidates
    )

    write_json_output(
        formula=formula,
        target_parameter_count=(
            target_parameter_count
        ),
        candidates=candidates,
    )

    write_markdown_output(
        formula=formula,
        target_parameter_count=(
            target_parameter_count
        ),
        candidates=candidates,
    )

    exact_candidates = [
        candidate
        for candidate in candidates
        if candidate.exact_match
    ]

    print_banner(
        "PB0-G CALIBRATED CANDIDATE SEARCH COMPLETE"
    )

    print(
        f"Target parameter count : "
        f"{target_parameter_count:,}"
    )
    print(
        f"Stored candidates      : "
        f"{len(candidates):,}"
    )
    print(
        f"Exact matches          : "
        f"{len(exact_candidates):,}"
    )

    best = candidates[0]

    print()
    print(
        "Best candidate:"
    )
    print(
        f"  r = {best.rank_r}"
    )
    print(
        f"  n = {best.vector_dimension_n}"
    )
    print(
        f"  h = {best.router_hidden_h}"
    )
    print(
        f"  f = {best.feedback_rank_f}"
    )
    print(
        f"  parameters = "
        f"{best.parameter_count:,}"
    )
    print(
        f"  error      = "
        f"{best.parameter_error:+,}"
    )
    print(
        f"  exact      = "
        f"{best.exact_match}"
    )

    print()
    print(
        "Output files:"
    )
    print(
        f"  {OUTPUT_JSON_PATH}"
    )
    print(
        f"  {OUTPUT_MARKDOWN_PATH}"
    )
    print(
        f"  {OUTPUT_CSV_PATH}"
    )

    print()
    print(
        "NEXT GATE:"
    )
    print(
        "PB0-H must instantiate the selected "
        "candidate through the real Modus_X "
        "initializer and perform an exact "
        "parameter-tree recount."
    )
    print(
        "Do not train the candidate yet."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 80)
        print(
            "PB0-G CALIBRATED CANDIDATE SEARCH FAILED"
        )
        print("=" * 80)
        print()
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print()

        raise