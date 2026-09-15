from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

SCRIPT_PATH = Path(__file__).resolve()

# Expected location:
#
# E:\Modus_X\research\parameter_allocation\b4_matrix_budget_analysis.py
#
# SCRIPT_PATH.parent        -> parameter_allocation
# SCRIPT_PATH.parent.parent -> research
# SCRIPT_PATH.parent.parent.parent -> Modus_X

ANALYSIS_DIR = SCRIPT_PATH.parent
RESEARCH_DIR = ANALYSIS_DIR.parent
PROJECT_ROOT = RESEARCH_DIR.parent

OUTPUT_DIR = ANALYSIS_DIR / "outputs"

B0_JSON_PATH = OUTPUT_DIR / "b0_parameter_census.json"
B0_CSV_PATH = OUTPUT_DIR / "b0_parameter_leaves.csv"

OUTPUT_JSON_PATH = OUTPUT_DIR / "B4_MATRIX_BUDGET_ANALYSIS.json"
OUTPUT_MARKDOWN_PATH = OUTPUT_DIR / "B4_MATRIX_BUDGET_ANALYSIS.md"
OUTPUT_CSV_PATH = OUTPUT_DIR / "B4_MATRIX_BUDGET_CANDIDATES.csv"


# =============================================================================
# CANONICAL MATRIX TARGETS
# =============================================================================

MATRIX_PROJECTION_PATHS: tuple[str, ...] = (
    "layers.m_wk",
    "layers.m_wq",
    "layers.m_wv",
    "layers.m_w_read",
    "layers.m_w_out",
)

MEMORY_INPUT_PROJECTION_PATH = "layers.m_proj_w"


# =============================================================================
# EXPERIMENT SEARCH SPACE
# =============================================================================

# Low-rank decomposition candidates for the five repeated 512 x 512 matrices.
#
# Canonical:
#
#     W ∈ R^(512 x 512)
#
# Parameters per matrix per layer:
#
#     512 * 512 = 262,144
#
# Rank-r factorization:
#
#     A ∈ R^(512 x r)
#     B ∈ R^(r x 512)
#
# Parameters:
#
#     512*r + r*512 = 1024*r

FACTORIZATION_RANKS: tuple[int, ...] = (
    64,
    96,
    128,
    160,
    192,
    256,
    320,
    384,
)


# Canonical m_proj_w:
#
#     [12, 512, 1024]
#
# Candidate accounting scenarios:
#
#     [12, 512, candidate_width]

MEMORY_INPUT_WIDTHS: tuple[int, ...] = (
    256,
    384,
    512,
    640,
    768,
    896,
)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class ParameterLeaf:
    path: str
    shape: tuple[int, ...]
    elements: int
    component: str


@dataclass(frozen=True)
class BudgetCandidate:
    experiment_family: str
    candidate_name: str
    modification_target: str
    strategy: str

    canonical_parameters: int
    candidate_parameters: int
    parameters_saved: int

    savings_percent_of_target: float

    projected_model_parameters: int
    projected_model_reduction: int
    projected_model_reduction_percent: float

    notes: str


# =============================================================================
# ERROR HANDLING
# =============================================================================

def fail(message: str) -> None:
    print()
    print("=" * 80)
    print("B4 ANALYSIS FAILED")
    print("=" * 80)
    print()
    print(message)
    print()
    sys.exit(1)


# =============================================================================
# BASIC HELPERS
# =============================================================================

def format_number(value: int) -> str:
    return f"{value:,}"


def percentage(
    numerator: int,
    denominator: int,
) -> float:
    if denominator <= 0:
        return 0.0

    return 100.0 * numerator / denominator


# =============================================================================
# JSON LOADING
# =============================================================================

def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        fail(
            f"Required input JSON file does not exist:\n{path}\n\n"
            "Run b0_parameter_census.py before running B4."
        )

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

    except OSError as exc:
        fail(
            f"Could not read JSON file:\n{path}\n\n"
            f"Error: {exc}"
        )

    except json.JSONDecodeError as exc:
        fail(
            f"Could not parse JSON file:\n{path}\n\n"
            f"Error: {exc}"
        )

    if not isinstance(data, dict):
        fail(
            f"Expected a JSON object at the root of:\n{path}\n\n"
            f"Found type: {type(data).__name__}"
        )

    return data


# =============================================================================
# CSV LOADING
# =============================================================================

def load_csv_rows(
    path: Path,
) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        fail(
            f"Required input CSV file does not exist:\n{path}\n\n"
            "Run b0_parameter_census.py before running B4."
        )

    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:

            reader = csv.DictReader(file)

            if reader.fieldnames is None:
                fail(
                    f"CSV file has no header row:\n{path}"
                )

            fieldnames = list(reader.fieldnames)
            rows = list(reader)

    except OSError as exc:
        fail(
            f"Could not read CSV file:\n{path}\n\n"
            f"Error: {exc}"
        )

    if not rows:
        fail(
            f"CSV contains no parameter rows:\n{path}\n\n"
            "The B0 census appears incomplete."
        )

    return rows, fieldnames


# =============================================================================
# COLUMN DETECTION
# =============================================================================

def identify_column(
    fieldnames: list[str],
    candidates: tuple[str, ...],
    description: str,
) -> str:

    normalized_to_original: dict[str, str] = {}

    for fieldname in fieldnames:

        cleaned = fieldname.strip()

        if cleaned:
            normalized_to_original[
                cleaned.lower()
            ] = fieldname

    for candidate in candidates:

        if candidate in normalized_to_original:
            return normalized_to_original[candidate]

    raise RuntimeError(
        f"Could not identify {description}.\n"
        f"Available columns: {fieldnames}\n"
        f"Expected one of: {', '.join(candidates)}"
    )


def identify_path_column(
    fieldnames: list[str],
) -> str:

    return identify_column(
        fieldnames=fieldnames,
        candidates=(
            "path",
            "parameter_path",
            "name",
            "parameter_name",
        ),
        description="parameter path column",
    )


def identify_shape_column(
    fieldnames: list[str],
) -> str:

    return identify_column(
        fieldnames=fieldnames,
        candidates=(
            "shape",
            "parameter_shape",
            "tensor_shape",
        ),
        description="parameter shape column",
    )


def identify_parameter_column(
    fieldnames: list[str],
) -> str:

    return identify_column(
        fieldnames=fieldnames,
        candidates=(
            "elements",
            "parameter_count",
            "parameters",
            "num_parameters",
            "num_params",
            "params",
            "count",
            "size",
        ),
        description="parameter-count column",
    )


def identify_component_column(
    fieldnames: list[str],
) -> str:

    return identify_column(
        fieldnames=fieldnames,
        candidates=(
            "component",
            "parameter_component",
            "group",
            "category",
        ),
        description="component column",
    )


# =============================================================================
# SHAPE PARSING
# =============================================================================

def parse_shape(
    value: str,
) -> tuple[int, ...]:

    cleaned = value.strip()

    if not cleaned:
        raise ValueError(
            "Shape value is empty."
        )

    if (
        cleaned.startswith("[")
        and cleaned.endswith("]")
    ):
        cleaned = cleaned[1:-1]

    if not cleaned:
        return ()

    dimensions: list[int] = []

    for item in cleaned.split(","):

        dimension_text = item.strip()

        if not dimension_text:
            raise ValueError(
                f"Invalid shape value: '{value}'"
            )

        try:
            dimension = int(dimension_text)

        except ValueError as exc:
            raise ValueError(
                f"Could not parse dimension "
                f"'{dimension_text}' in '{value}'"
            ) from exc

        if dimension < 0:
            raise ValueError(
                f"Negative dimension {dimension} "
                f"in shape '{value}'"
            )

        dimensions.append(dimension)

    return tuple(dimensions)


# =============================================================================
# PARAMETER LEAF EXTRACTION
# =============================================================================

def parse_parameter_leaves(
    csv_path: Path,
) -> tuple[
    list[ParameterLeaf],
    dict[str, str],
]:

    rows, fieldnames = load_csv_rows(
        csv_path
    )

    try:
        path_column = identify_path_column(
            fieldnames
        )

        shape_column = identify_shape_column(
            fieldnames
        )

        parameter_column = identify_parameter_column(
            fieldnames
        )

        component_column = identify_component_column(
            fieldnames
        )

    except RuntimeError as exc:
        fail(str(exc))

    leaves: list[ParameterLeaf] = []

    for row_number, row in enumerate(
        rows,
        start=2,
    ):

        raw_path = row.get(
            path_column
        )

        raw_shape = row.get(
            shape_column
        )

        raw_elements = row.get(
            parameter_column
        )

        raw_component = row.get(
            component_column
        )

        if (
            raw_path is None
            or not raw_path.strip()
        ):
            fail(
                f"Missing parameter path at CSV row "
                f"{row_number}."
            )

        if raw_shape is None:
            fail(
                f"Missing parameter shape at CSV row "
                f"{row_number}."
            )

        if raw_elements is None:
            fail(
                f"Missing parameter count at CSV row "
                f"{row_number}."
            )

        if (
            raw_component is None
            or not raw_component.strip()
        ):
            fail(
                f"Missing component at CSV row "
                f"{row_number}."
            )

        try:
            shape = parse_shape(
                raw_shape
            )

        except ValueError as exc:
            fail(
                f"Invalid shape at CSV row "
                f"{row_number}: {exc}"
            )

        try:
            elements = int(
                raw_elements
            )

        except ValueError:
            fail(
                f"Could not parse parameter count "
                f"at CSV row {row_number}.\n"
                f"Column: {parameter_column}\n"
                f"Value: '{raw_elements}'"
            )

        if elements < 0:
            fail(
                f"Negative parameter count at "
                f"CSV row {row_number}: "
                f"{elements}"
            )

        leaves.append(
            ParameterLeaf(
                path=raw_path.strip(),
                shape=shape,
                elements=elements,
                component=raw_component.strip(),
            )
        )

    detected_columns = {
        "path": path_column,
        "shape": shape_column,
        "parameters": parameter_column,
        "component": component_column,
    }

    return (
        leaves,
        detected_columns,
    )


# =============================================================================
# PARAMETER TOTAL EXTRACTION
# =============================================================================

def find_integer_value(
    value: Any,
) -> int | None:

    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value

    if (
        isinstance(value, float)
        and value.is_integer()
    ):
        return int(value)

    if isinstance(value, str):

        cleaned = (
            value
            .replace(",", "")
            .strip()
        )

        if cleaned.isdigit():
            return int(cleaned)

    return None


def extract_reported_total(
    data: dict[str, Any],
) -> int | None:

    candidate_keys = (
        "total_parameters",
        "parameter_count",
        "total_parameter_count",
        "canonical_parameter_count",
        "params",
        "total_params",
        "model_parameters",
        "num_parameters",
        "num_params",
    )

    # Check root first.

    for key in candidate_keys:

        value = data.get(key)

        parsed = find_integer_value(
            value
        )

        if parsed is not None:
            return parsed

    # Recursively inspect nested dictionaries.

    def search(
        value: Any,
    ) -> int | None:

        if isinstance(
            value,
            dict,
        ):

            for key in candidate_keys:

                if key in value:

                    parsed = find_integer_value(
                        value[key]
                    )

                    if parsed is not None:
                        return parsed

            for nested_value in value.values():

                result = search(
                    nested_value
                )

                if result is not None:
                    return result

        elif isinstance(
            value,
            list,
        ):

            for item in value:

                result = search(
                    item
                )

                if result is not None:
                    return result

        return None

    return search(
        data
    )


def calculate_total_from_leaves(
    leaves: list[ParameterLeaf],
) -> int:

    return sum(
        leaf.elements
        for leaf in leaves
    )


# =============================================================================
# LEAF INDEXING
# =============================================================================

def build_leaf_index(
    leaves: list[ParameterLeaf],
) -> dict[str, ParameterLeaf]:

    index: dict[
        str,
        ParameterLeaf,
    ] = {}

    for leaf in leaves:

        if leaf.path in index:

            fail(
                "Duplicate parameter path found "
                "in B0 census:\n"
                f"{leaf.path}"
            )

        index[
            leaf.path
        ] = leaf

    return index


def require_leaf(
    index: dict[
        str,
        ParameterLeaf,
    ],
    path: str,
) -> ParameterLeaf:

    leaf = index.get(
        path
    )

    if leaf is None:

        available = sorted(
            index.keys()
        )

        fail(
            f"Required canonical parameter "
            f"was not found:\n{path}\n\n"
            f"Available paths include:\n"
            + "\n".join(
                available[:30]
            )
        )

    return leaf


# =============================================================================
# CANONICAL SHAPE VALIDATION
# =============================================================================

def validate_matrix_leaf(
    leaf: ParameterLeaf,
    expected_shape_tail: tuple[
        int,
        int,
    ],
) -> None:

    if (
        leaf.component
        != "matrix_memory"
    ):
        fail(
            f"Expected '{leaf.path}' to be "
            f"classified as matrix_memory.\n"
            f"Actual component: "
            f"{leaf.component}"
        )

    if len(
        leaf.shape
    ) != 3:
        fail(
            f"Expected 3D tensor for "
            f"'{leaf.path}'.\n"
            f"Actual shape: {leaf.shape}"
        )

    actual_tail = (
        leaf.shape[1],
        leaf.shape[2],
    )

    if (
        actual_tail
        != expected_shape_tail
    ):
        fail(
            f"Unexpected shape for "
            f"'{leaf.path}'.\n"
            f"Expected final dimensions: "
            f"{expected_shape_tail}\n"
            f"Actual shape: "
            f"{leaf.shape}"
        )

    implied_elements = (
        leaf.shape[0]
        * leaf.shape[1]
        * leaf.shape[2]
    )

    if (
        leaf.elements
        != implied_elements
    ):
        fail(
            f"Parameter count does not match "
            f"shape for '{leaf.path}'.\n"
            f"Shape implies: "
            f"{implied_elements:,}\n"
            f"Census reports: "
            f"{leaf.elements:,}"
        )


# =============================================================================
# CANDIDATE CREATION
# =============================================================================

def create_candidate(
    *,
    experiment_family: str,
    candidate_name: str,
    modification_target: str,
    strategy: str,
    canonical_parameters: int,
    candidate_parameters: int,
    total_model_parameters: int,
    notes: str,
) -> BudgetCandidate:

    if canonical_parameters <= 0:
        raise ValueError(
            "Canonical parameter count must "
            "be positive."
        )

    if candidate_parameters < 0:
        raise ValueError(
            "Candidate parameter count cannot "
            "be negative."
        )

    if (
        candidate_parameters
        > canonical_parameters
    ):
        raise ValueError(
            "Candidate uses more parameters "
            "than its canonical target.\n"
            f"Canonical: "
            f"{canonical_parameters:,}\n"
            f"Candidate: "
            f"{candidate_parameters:,}\n"
            f"Candidate name: "
            f"{candidate_name}"
        )

    parameters_saved = (
        canonical_parameters
        - candidate_parameters
    )

    projected_model_parameters = (
        total_model_parameters
        - parameters_saved
    )

    if (
        projected_model_parameters
        <= 0
    ):
        raise ValueError(
            "Projected model parameter count "
            "must remain positive."
        )

    return BudgetCandidate(
        experiment_family=experiment_family,
        candidate_name=candidate_name,
        modification_target=modification_target,
        strategy=strategy,
        canonical_parameters=canonical_parameters,
        candidate_parameters=candidate_parameters,
        parameters_saved=parameters_saved,
        savings_percent_of_target=percentage(
            parameters_saved,
            canonical_parameters,
        ),
        projected_model_parameters=(
            projected_model_parameters
        ),
        projected_model_reduction=(
            parameters_saved
        ),
        projected_model_reduction_percent=percentage(
            parameters_saved,
            total_model_parameters,
        ),
        notes=notes,
    )


# =============================================================================
# FACTORIZATION ANALYSIS
# =============================================================================

def analyze_factorization_candidates(
    leaves: list[ParameterLeaf],
    total_model_parameters: int,
) -> list[BudgetCandidate]:

    if not leaves:
        fail(
            "No projection leaves supplied "
            "for factorization analysis."
        )

    reference_shape = (
        leaves[0].shape
    )

    if len(
        reference_shape
    ) != 3:
        fail(
            "Factorization reference leaf "
            "does not have a 3D shape."
        )

    layer_count = (
        reference_shape[0]
    )

    input_width = (
        reference_shape[1]
    )

    output_width = (
        reference_shape[2]
    )

    for leaf in leaves:

        if (
            leaf.shape
            != reference_shape
        ):
            fail(
                "Projection group contains "
                "inconsistent shapes.\n"
                f"Expected: "
                f"{reference_shape}\n"
                f"Found: "
                f"{leaf.shape}\n"
                f"Path: "
                f"{leaf.path}"
            )

    canonical_parameters = sum(
        leaf.elements
        for leaf in leaves
    )

    matrix_count = len(
        leaves
    )

    candidates: list[
        BudgetCandidate
    ] = []

    for rank in FACTORIZATION_RANKS:

        if rank <= 0:
            fail(
                f"Invalid factorization "
                f"rank: {rank}"
            )

        # A rank equal to or above the
        # smaller dimension is not a
        # reduction candidate.

        if (
            rank
            >= min(
                input_width,
                output_width,
            )
        ):
            continue

        parameters_per_matrix_per_layer = (
            input_width * rank
            + rank * output_width
        )

        candidate_parameters = (
            layer_count
            * matrix_count
            * parameters_per_matrix_per_layer
        )

        # Explicitly verify that this
        # search point is actually a
        # reduction before creating it.

        if (
            candidate_parameters
            >= canonical_parameters
        ):
            continue

        candidate = create_candidate(
            experiment_family="PB1-A",
            candidate_name=(
                "PB1-A-factorized-"
                f"{input_width}x"
                f"{output_width}-"
                f"rank-{rank}"
            ),
            modification_target=(
                "layers.m_wk, "
                "layers.m_wq, "
                "layers.m_wv, "
                "layers.m_w_read, "
                "layers.m_w_out"
            ),
            strategy=(
                "Replace each dense "
                f"{input_width}x"
                f"{output_width} projection "
                "with a two-factor "
                f"low-rank decomposition "
                f"of rank {rank}"
            ),
            canonical_parameters=(
                canonical_parameters
            ),
            candidate_parameters=(
                candidate_parameters
            ),
            total_model_parameters=(
                total_model_parameters
            ),
            notes=(
                "Accounting-only scenario. "
                "Each dense matrix is "
                f"represented as "
                f"{input_width}x{rank} "
                "followed by "
                f"{rank}x{output_width}. "
                "No functional equivalence "
                "or training result is implied."
            ),
        )

        candidates.append(
            candidate
        )

    return candidates


# =============================================================================
# MEMORY INPUT WIDTH ANALYSIS
# =============================================================================

def analyze_memory_input_width_candidates(
    leaf: ParameterLeaf,
    total_model_parameters: int,
) -> list[BudgetCandidate]:

    if len(
        leaf.shape
    ) != 3:
        fail(
            f"Expected 3D shape for "
            f"'{leaf.path}'.\n"
            f"Actual shape: "
            f"{leaf.shape}"
        )

    layer_count = (
        leaf.shape[0]
    )

    input_width = (
        leaf.shape[1]
    )

    canonical_output_width = (
        leaf.shape[2]
    )

    canonical_parameters = (
        leaf.elements
    )

    candidates: list[
        BudgetCandidate
    ] = []

    for candidate_output_width in (
        MEMORY_INPUT_WIDTHS
    ):

        if (
            candidate_output_width
            <= 0
        ):
            fail(
                "Memory input width must "
                "be positive.\n"
                f"Invalid width: "
                f"{candidate_output_width}"
            )

        # B4 is reduction analysis.
        # Widths greater than or equal
        # to canonical do not belong
        # in this candidate set.

        if (
            candidate_output_width
            >= canonical_output_width
        ):
            continue

        candidate_parameters = (
            layer_count
            * input_width
            * candidate_output_width
        )

        if (
            candidate_parameters
            >= canonical_parameters
        ):
            continue

        candidate = create_candidate(
            experiment_family="PB1-A",
            candidate_name=(
                "PB1-A-memory-input-width-"
                f"{candidate_output_width}"
            ),
            modification_target=(
                leaf.path
            ),
            strategy=(
                "Reduce memory input "
                "projection output width "
                f"from "
                f"{canonical_output_width} "
                f"to "
                f"{candidate_output_width}"
            ),
            canonical_parameters=(
                canonical_parameters
            ),
            candidate_parameters=(
                candidate_parameters
            ),
            total_model_parameters=(
                total_model_parameters
            ),
            notes=(
                "Accounting-only scenario. "
                f"Canonical shape "
                f"{list(leaf.shape)} becomes "
                f"[{layer_count}, "
                f"{input_width}, "
                f"{candidate_output_width}]. "
                "Implementation may require "
                "coordinated downstream "
                "architecture changes."
            ),
        )

        candidates.append(
            candidate
        )

    return candidates


# =============================================================================
# COMBINED CANDIDATES
# =============================================================================

def analyze_combined_candidates(
    factorization_candidates: list[
        BudgetCandidate
    ],
    width_candidates: list[
        BudgetCandidate
    ],
    total_model_parameters: int,
) -> list[BudgetCandidate]:

    candidates: list[
        BudgetCandidate
    ] = []

    for factorization_candidate in (
        factorization_candidates
    ):

        for width_candidate in (
            width_candidates
        ):

            # These two canonical targets are
            # disjoint parameter groups.
            #
            # Group 1:
            # five repeated 512x512 projections
            #
            # Group 2:
            # m_proj_w
            #
            # Therefore their parameter savings
            # can be added directly.

            canonical_parameters = (
                factorization_candidate
                .canonical_parameters
                + width_candidate
                .canonical_parameters
            )

            candidate_parameters = (
                factorization_candidate
                .candidate_parameters
                + width_candidate
                .candidate_parameters
            )

            if (
                candidate_parameters
                >= canonical_parameters
            ):
                continue

            candidate = create_candidate(
                experiment_family="PB1-A",
                candidate_name=(
                    f"{factorization_candidate.candidate_name}"
                    "__"
                    f"{width_candidate.candidate_name}"
                ),
                modification_target=(
                    f"{factorization_candidate.modification_target}; "
                    f"{width_candidate.modification_target}"
                ),
                strategy=(
                    f"{factorization_candidate.strategy}; "
                    f"{width_candidate.strategy}"
                ),
                canonical_parameters=(
                    canonical_parameters
                ),
                candidate_parameters=(
                    candidate_parameters
                ),
                total_model_parameters=(
                    total_model_parameters
                ),
                notes=(
                    "Combined accounting "
                    "scenario covering two "
                    "disjoint parameter groups. "
                    "No implementation or "
                    "training result is implied."
                ),
            )

            candidates.append(
                candidate
            )

    return candidates


# =============================================================================
# PB1-B REALLOCATION ANALYSIS
# =============================================================================

def build_reallocation_options(
    candidates: list[
        BudgetCandidate
    ],
    canonical_model_parameters: int,
) -> list[dict[str, Any]]:

    options: list[
        dict[str, Any]
    ] = []

    for candidate in candidates:

        saved_parameters = (
            candidate.parameters_saved
        )

        restored_budget = (
            candidate.projected_model_parameters
            + saved_parameters
        )

        exact_restoration_difference = (
            canonical_model_parameters
            - restored_budget
        )

        options.append(
            {
                "source_candidate": (
                    candidate.candidate_name
                ),
                "parameters_available_for_reallocation": (
                    saved_parameters
                ),
                "canonical_model_parameters": (
                    canonical_model_parameters
                ),
                "projected_parameters_after_reduction": (
                    candidate.projected_model_parameters
                ),
                "budget_after_full_reallocation": (
                    restored_budget
                ),
                "difference_from_canonical_budget": (
                    exact_restoration_difference
                ),
                "interpretation": (
                    "The saved parameters define "
                    "the first-order PB1-B budget "
                    "available for reallocation "
                    "while preserving the canonical "
                    "total parameter count."
                ),
            }
        )

    return options


# =============================================================================
# COMPONENT TOTALS
# =============================================================================

def calculate_component_totals(
    leaves: list[
        ParameterLeaf
    ],
) -> dict[str, int]:

    totals: dict[
        str,
        int,
    ] = defaultdict(int)

    for leaf in leaves:

        totals[
            leaf.component
        ] += leaf.elements

    return dict(
        sorted(
            totals.items(),
            key=lambda item: (
                item[1]
            ),
            reverse=True,
        )
    )


# =============================================================================
# OUTPUT: JSON
# =============================================================================

def write_json_output(
    *,
    total_model_parameters: int,
    detected_columns: dict[
        str,
        str,
    ],
    projection_leaves: list[
        ParameterLeaf
    ],
    memory_input_leaf: ParameterLeaf,
    candidates: list[
        BudgetCandidate
    ],
    reallocation_options: list[
        dict[str, Any]
    ],
    component_totals: dict[
        str,
        int,
    ],
) -> None:

    payload: dict[
        str,
        Any,
    ] = {
        "analysis": (
            "B4 Matrix Budget Analysis"
        ),
        "project_root": (
            str(PROJECT_ROOT)
        ),
        "total_model_parameters": (
            total_model_parameters
        ),
        "detected_columns": (
            detected_columns
        ),
        "canonical_targets": {
            "projection_group": [
                {
                    "path": (
                        leaf.path
                    ),
                    "shape": (
                        list(leaf.shape)
                    ),
                    "parameters": (
                        leaf.elements
                    ),
                }
                for leaf
                in projection_leaves
            ],
            "memory_input_projection": {
                "path": (
                    memory_input_leaf.path
                ),
                "shape": (
                    list(
                        memory_input_leaf.shape
                    )
                ),
                "parameters": (
                    memory_input_leaf.elements
                ),
            },
        },
        "component_totals": (
            component_totals
        ),
        "candidate_count": (
            len(candidates)
        ),
        "candidates": [
            asdict(candidate)
            for candidate
            in candidates
        ],
        "pb1_b_reallocation_options": (
            reallocation_options
        ),
        "interpretation": {
            "purpose": (
                "Quantify parameter-budget "
                "consequences before modifying "
                "the canonical architecture."
            ),
            "not_an_architecture_change": (
                True
            ),
            "not_a_training_result": (
                True
            ),
            "not_a_claim_of_functional_equivalence": (
                True
            ),
        },
    }

    try:

        with OUTPUT_JSON_PATH.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                payload,
                file,
                indent=2,
            )

            file.write(
                "\n"
            )

    except OSError as exc:

        fail(
            f"Could not write JSON output:\n"
            f"{OUTPUT_JSON_PATH}\n\n"
            f"Error: {exc}"
        )


# =============================================================================
# OUTPUT: CSV
# =============================================================================

def write_csv_output(
    candidates: list[
        BudgetCandidate
    ],
) -> None:

    fieldnames = [
        "experiment_family",
        "candidate_name",
        "modification_target",
        "strategy",
        "canonical_parameters",
        "candidate_parameters",
        "parameters_saved",
        "savings_percent_of_target",
        "projected_model_parameters",
        "projected_model_reduction",
        "projected_model_reduction_percent",
        "notes",
    ]

    try:

        with OUTPUT_CSV_PATH.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for candidate in candidates:

                writer.writerow(
                    asdict(candidate)
                )

    except OSError as exc:

        fail(
            f"Could not write CSV output:\n"
            f"{OUTPUT_CSV_PATH}\n\n"
            f"Error: {exc}"
        )


# =============================================================================
# OUTPUT: MARKDOWN
# =============================================================================

def write_markdown_output(
    *,
    total_model_parameters: int,
    projection_leaves: list[
        ParameterLeaf
    ],
    memory_input_leaf: ParameterLeaf,
    candidates: list[
        BudgetCandidate
    ],
    component_totals: dict[
        str,
        int,
    ],
) -> None:

    lines: list[
        str
    ] = []

    lines.append(
        "# B4 Matrix Budget Analysis"
    )

    lines.append("")

    lines.append(
        "## Scope"
    )

    lines.append("")

    lines.append(
        "This analysis converts the measured "
        "B0/B3 parameter census into explicit "
        "parameter-budget scenarios."
    )

    lines.append("")

    lines.append(
        "No architecture, checkpoint, training "
        "configuration, or canonical model file "
        "is modified."
    )

    lines.append("")

    lines.append(
        "## Canonical Model Budget"
    )

    lines.append("")

    lines.append(
        f"- Total canonical model parameters: "
        f"**{format_number(total_model_parameters)}**"
    )

    lines.append("")

    lines.append(
        "## Canonical Matrix Targets"
    )

    lines.append("")

    lines.append(
        "### Repeated 512x512 Projection Group"
    )

    lines.append("")

    lines.append(
        "| Parameter path | Shape | Parameters |"
    )

    lines.append(
        "|---|---|---:|"
    )

    for leaf in projection_leaves:

        lines.append(
            f"| `{leaf.path}` | "
            f"`{list(leaf.shape)}` | "
            f"{format_number(leaf.elements)} |"
        )

    projection_total = sum(
        leaf.elements
        for leaf
        in projection_leaves
    )

    lines.append("")

    lines.append(
        f"- Projection group total: "
        f"**{format_number(projection_total)}**"
    )

    lines.append("")

    lines.append(
        "### Memory Input Projection"
    )

    lines.append("")

    lines.append(
        f"- Path: "
        f"`{memory_input_leaf.path}`"
    )

    lines.append(
        f"- Shape: "
        f"`{list(memory_input_leaf.shape)}`"
    )

    lines.append(
        f"- Parameters: "
        f"**{format_number(memory_input_leaf.elements)}**"
    )

    lines.append("")

    lines.append(
        "## Component Budget Context"
    )

    lines.append("")

    lines.append(
        "| Component | Parameters | Share of model |"
    )

    lines.append(
        "|---|---:|---:|"
    )

    for (
        component,
        parameters,
    ) in component_totals.items():

        lines.append(
            f"| {component} | "
            f"{format_number(parameters)} | "
            f"{percentage(parameters, total_model_parameters):.4f}% |"
        )

    lines.append("")

    lines.append(
        "## PB1-A Candidate Budgets"
    )

    lines.append("")

    lines.append(
        "| Candidate | Saved Parameters | "
        "Target Reduction | "
        "Projected Model Parameters | "
        "Model Reduction |"
    )

    lines.append(
        "|---|---:|---:|---:|---:|"
    )

    sorted_candidates = sorted(
        candidates,
        key=lambda candidate: (
            candidate.parameters_saved,
            candidate.candidate_name,
        ),
        reverse=True,
    )

    for candidate in (
        sorted_candidates
    ):

        lines.append(
            f"| `{candidate.candidate_name}` | "
            f"{format_number(candidate.parameters_saved)} | "
            f"{candidate.savings_percent_of_target:.2f}% | "
            f"{format_number(candidate.projected_model_parameters)} | "
            f"{candidate.projected_model_reduction_percent:.2f}% |"
        )

    lines.append("")

    lines.append(
        "## PB1-B Reallocation Interpretation"
    )

    lines.append("")

    lines.append(
        "For every PB1-A reduction candidate, "
        "the saved parameter count is the "
        "maximum first-order budget available "
        "for a fixed-budget PB1-B reallocation."
    )

    lines.append("")

    lines.append(
        "The later PB1-B experiment should compare:"
    )

    lines.append("")

    lines.append(
        "1. Canonical Modus_X at the measured "
        "canonical parameter budget."
    )

    lines.append(
        "2. The PB1-A reduced matrix-memory "
        "candidate."
    )

    lines.append(
        "3. A PB1-B candidate that reallocates "
        "approximately the saved parameters "
        "elsewhere."
    )

    lines.append("")

    lines.append(
        "## Interpretation"
    )

    lines.append("")

    lines.append(
        "B4 does not select a winning "
        "architecture. It defines the "
        "quantitative candidate search space "
        "before architecture modifications "
        "and training experiments."
    )

    lines.append("")

    try:

        with OUTPUT_MARKDOWN_PATH.open(
            "w",
            encoding="utf-8",
        ) as file:

            file.write(
                "\n".join(lines)
            )

            file.write(
                "\n"
            )

    except OSError as exc:

        fail(
            f"Could not write Markdown output:\n"
            f"{OUTPUT_MARKDOWN_PATH}\n\n"
            f"Error: {exc}"
        )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 80)
    print(
        "MODUS_X B4 MATRIX MEMORY BUDGET ANALYSIS"
    )
    print("=" * 80)
    print()

    print(
        f"Project root : "
        f"{PROJECT_ROOT}"
    )

    print(
        f"Input JSON   : "
        f"{B0_JSON_PATH}"
    )

    print(
        f"Input CSV    : "
        f"{B0_CSV_PATH}"
    )

    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    b0_data = load_json(
        B0_JSON_PATH
    )

    (
        leaves,
        detected_columns,
    ) = parse_parameter_leaves(
        B0_CSV_PATH
    )

    print(
        f"Detected path column      : "
        f"{detected_columns['path']}"
    )

    print(
        f"Detected shape column     : "
        f"{detected_columns['shape']}"
    )

    print(
        f"Detected parameter column : "
        f"{detected_columns['parameters']}"
    )

    print(
        f"Detected component column : "
        f"{detected_columns['component']}"
    )

    print()

    total_from_json = (
        extract_reported_total(
            b0_data
        )
    )

    total_from_leaves = (
        calculate_total_from_leaves(
            leaves
        )
    )

    if (
        total_from_json
        is not None
    ):

        if (
            total_from_json
            != total_from_leaves
        ):

            fail(
                "B0 JSON parameter total does not "
                "match the sum of B0 CSV leaves.\n\n"
                f"JSON total: "
                f"{total_from_json:,}\n"
                f"CSV total : "
                f"{total_from_leaves:,}\n\n"
                "Do not perform budget analysis "
                "on inconsistent census inputs."
            )

        total_model_parameters = (
            total_from_json
        )

    else:

        total_model_parameters = (
            total_from_leaves
        )

    leaf_index = (
        build_leaf_index(
            leaves
        )
    )

    projection_leaves: list[
        ParameterLeaf
    ] = []

    for path in (
        MATRIX_PROJECTION_PATHS
    ):

        leaf = require_leaf(
            leaf_index,
            path,
        )

        validate_matrix_leaf(
            leaf,
            expected_shape_tail=(
                512,
                512,
            ),
        )

        projection_leaves.append(
            leaf
        )

    memory_input_leaf = (
        require_leaf(
            leaf_index,
            MEMORY_INPUT_PROJECTION_PATH,
        )
    )

    validate_matrix_leaf(
        memory_input_leaf,
        expected_shape_tail=(
            512,
            1024,
        ),
    )

    component_totals = (
        calculate_component_totals(
            leaves
        )
    )

    factorization_candidates = (
        analyze_factorization_candidates(
            projection_leaves,
            total_model_parameters,
        )
    )

    width_candidates = (
        analyze_memory_input_width_candidates(
            memory_input_leaf,
            total_model_parameters,
        )
    )

    combined_candidates = (
        analyze_combined_candidates(
            factorization_candidates,
            width_candidates,
            total_model_parameters,
        )
    )

    all_candidates = (
        factorization_candidates
        + width_candidates
        + combined_candidates
    )

    if not all_candidates:

        fail(
            "B4 generated zero valid reduction "
            "candidates.\n\n"
            "Inspect the configured ranks, widths, "
            "and canonical tensor shapes."
        )

    reallocation_options = (
        build_reallocation_options(
            all_candidates,
            total_model_parameters,
        )
    )

    write_json_output(
        total_model_parameters=(
            total_model_parameters
        ),
        detected_columns=(
            detected_columns
        ),
        projection_leaves=(
            projection_leaves
        ),
        memory_input_leaf=(
            memory_input_leaf
        ),
        candidates=(
            all_candidates
        ),
        reallocation_options=(
            reallocation_options
        ),
        component_totals=(
            component_totals
        ),
    )

    write_csv_output(
        all_candidates
    )

    write_markdown_output(
        total_model_parameters=(
            total_model_parameters
        ),
        projection_leaves=(
            projection_leaves
        ),
        memory_input_leaf=(
            memory_input_leaf
        ),
        candidates=(
            all_candidates
        ),
        component_totals=(
            component_totals
        ),
    )

    print()
    print("=" * 80)
    print(
        "B4 ANALYSIS COMPLETE"
    )
    print("=" * 80)
    print()

    print(
        f"Measured total model parameters : "
        f"{total_model_parameters:,}"
    )

    print(
        f"Repeated 512x512 projections   : "
        f"{len(projection_leaves)}"
    )

    print(
        f"Factorization cases             : "
        f"{len(factorization_candidates)}"
    )

    print(
        f"Memory input width cases        : "
        f"{len(width_candidates)}"
    )

    print(
        f"Combined budget cases           : "
        f"{len(combined_candidates)}"
    )

    print(
        f"Total PB1-A candidates          : "
        f"{len(all_candidates)}"
    )

    print()

    print(
        "Largest parameter-saving candidates:"
    )

    largest_savings = sorted(
        all_candidates,
        key=lambda candidate: (
            candidate.parameters_saved,
            candidate.candidate_name,
        ),
        reverse=True,
    )[:10]

    for candidate in (
        largest_savings
    ):

        print()

        print(
            f"  {candidate.candidate_name}"
        )

        print(
            f"    saved parameters : "
            f"{candidate.parameters_saved:,}"
        )

        print(
            f"    projected model  : "
            f"{candidate.projected_model_parameters:,}"
        )

        print(
            f"    model reduction  : "
            f"{candidate.projected_model_reduction_percent:.4f}%"
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
        "NEXT GATE: inspect "
        "B4_MATRIX_BUDGET_ANALYSIS.md and select "
        "a small controlled PB1-A candidate set "
        "before modifying language/models.py."
    )


if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print(
            "B4 analysis interrupted by user."
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print("=" * 80)
        print(
            "UNHANDLED B4 ERROR"
        )
        print("=" * 80)
        print()

        print(
            f"{type(exc).__name__}: {exc}"
        )

        print()

        raise