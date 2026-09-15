from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Mapping, NoReturn, Sequence


# =============================================================================
# PROJECT PATHS
# =============================================================================


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

RESEARCH_DIR: Final[Path] = (
    PROJECT_ROOT
    / "research"
    / "parameter_allocation"
)

OUTPUT_DIR: Final[Path] = (
    RESEARCH_DIR
    / "outputs"
)

PB0_C_AUDIT_JSON: Final[Path] = (
    OUTPUT_DIR
    / "PB0_PARAMETER_BUDGET_AUDIT.json"
)

OUTPUT_JSON: Final[Path] = (
    OUTPUT_DIR
    / "PB0_FIXED_BUDGET_ALLOCATION.json"
)

OUTPUT_MARKDOWN: Final[Path] = (
    OUTPUT_DIR
    / "PB0_FIXED_BUDGET_ALLOCATION.md"
)

OUTPUT_CSV: Final[Path] = (
    OUTPUT_DIR
    / "PB0_FIXED_BUDGET_CANDIDATES.csv"
)

OUTPUT_COMPONENT_CSV: Final[Path] = (
    OUTPUT_DIR
    / "PB0_FIXED_BUDGET_COMPONENTS.csv"
)


# =============================================================================
# FIXED EXPERIMENT CONSTANTS
# =============================================================================


EXPECTED_BASELINE_PARAMETER_COUNT: Final[int] = 47_437_768

FROZEN_VOCABULARY_SIZE: Final[int] = 256
FROZEN_LAYER_COUNT: Final[int] = 12
FROZEN_EMBEDDING_WIDTH: Final[int] = 512
FROZEN_HIDDEN_WIDTH: Final[int] = 1_536
FROZEN_MATRIX_VECTOR_STATE_WIDTH: Final[int] = 512

TRAIN_START_BYTE: Final[int] = 0
TRAIN_END_BYTE: Final[int] = 90_000_000

VALIDATION_START_BYTE: Final[int] = 90_000_000
VALIDATION_END_BYTE: Final[int] = 95_000_000

TEST_START_BYTE: Final[int] = 95_000_000
TEST_END_BYTE: Final[int] = 100_000_000


# =============================================================================
# COMPONENT DEFINITIONS
# =============================================================================


COMPONENT_MATRIX_MEMORY: Final[str] = "matrix_memory"
COMPONENT_VECTOR_MAMBA: Final[str] = "vector_mamba"
COMPONENT_ROUTER: Final[str] = "router"
COMPONENT_FEEDBACK_BRIDGE: Final[str] = "feedback_bridge"
COMPONENT_EMBEDDINGS: Final[str] = "embeddings"
COMPONENT_OUTPUT_HEAD: Final[str] = "output_head"
COMPONENT_NORMALIZATION: Final[str] = "normalization"
COMPONENT_AUXILIARY: Final[str] = "auxiliary"
COMPONENT_OTHER: Final[str] = "other"


COMPONENT_ORDER: Final[tuple[str, ...]] = (
    COMPONENT_MATRIX_MEMORY,
    COMPONENT_VECTOR_MAMBA,
    COMPONENT_ROUTER,
    COMPONENT_FEEDBACK_BRIDGE,
    COMPONENT_EMBEDDINGS,
    COMPONENT_OUTPUT_HEAD,
    COMPONENT_NORMALIZATION,
    COMPONENT_AUXILIARY,
    COMPONENT_OTHER,
)


COMPONENT_DISPLAY_NAMES: Final[dict[str, str]] = {
    COMPONENT_MATRIX_MEMORY: "Matrix Memory",
    COMPONENT_VECTOR_MAMBA: "Vector / Mamba Pathway",
    COMPONENT_ROUTER: "Router",
    COMPONENT_FEEDBACK_BRIDGE: "Feedback Bridge",
    COMPONENT_EMBEDDINGS: "Embeddings",
    COMPONENT_OUTPUT_HEAD: "Output Head",
    COMPONENT_NORMALIZATION: "Normalization",
    COMPONENT_AUXILIARY: "Auxiliary",
    COMPONENT_OTHER: "Other",
}


# Components that PB0-D is allowed to deliberately reallocate at the
# theoretical budget-matrix stage.
#
# Embeddings, output head, normalization, auxiliary heads, and "other"
# infrastructure remain frozen in the candidate hypotheses. Their real
# architectural implementation is preserved unless a later experiment
# explicitly opens them as an independent research variable.
REALLOCATION_COMPONENTS: Final[tuple[str, ...]] = (
    COMPONENT_MATRIX_MEMORY,
    COMPONENT_VECTOR_MAMBA,
    COMPONENT_ROUTER,
    COMPONENT_FEEDBACK_BRIDGE,
)


FROZEN_COMPONENTS: Final[tuple[str, ...]] = (
    COMPONENT_EMBEDDINGS,
    COMPONENT_OUTPUT_HEAD,
    COMPONENT_NORMALIZATION,
    COMPONENT_AUXILIARY,
    COMPONENT_OTHER,
)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass(frozen=True)
class ComponentAllocation:
    component: str
    display_name: str
    parameter_count: int
    parameter_fraction: float
    parameter_percent: float


@dataclass(frozen=True)
class CandidateAllocation:
    candidate_id: str
    candidate_name: str
    hypothesis: str
    status: str
    selection_order: int
    selected_for_next_gate: bool
    matrix_memory_parameters: int
    vector_mamba_parameters: int
    router_parameters: int
    feedback_bridge_parameters: int
    embeddings_parameters: int
    output_head_parameters: int
    normalization_parameters: int
    auxiliary_parameters: int
    other_parameters: int
    total_parameters: int
    baseline_total_parameters: int
    total_difference_from_baseline: int
    max_absolute_component_shift: int
    max_absolute_component_shift_percent: float
    allocation_distance_l1_parameters: int
    allocation_distance_l1_percent: float
    matrix_memory_change: int
    vector_mamba_change: int
    router_change: int
    feedback_bridge_change: int
    matrix_memory_change_percent: float
    vector_mamba_change_percent: float
    router_change_percent: float
    feedback_bridge_change_percent: float


@dataclass(frozen=True)
class CandidateComponentRow:
    candidate_id: str
    candidate_name: str
    component: str
    display_name: str
    parameter_count: int
    baseline_parameter_count: int
    parameter_difference: int
    parameter_difference_percent_of_baseline: float
    parameter_fraction_of_total: float
    parameter_percent_of_total: float


@dataclass(frozen=True)
class PB0DResult:
    generated_at_utc: str
    project_root: str
    baseline_audit_path: str
    baseline_parameter_count: int
    reallocatable_budget: int
    frozen_parameter_count: int
    architecture_constraints: Mapping[str, int]
    dataset_protocol: Mapping[str, Any]
    baseline_components: tuple[ComponentAllocation, ...]
    candidates: tuple[CandidateAllocation, ...]
    selected_candidate_id: str


# =============================================================================
# CONSOLE HELPERS
# =============================================================================


def banner(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def fail(message: str) -> NoReturn:
    raise RuntimeError(message)


def format_int(value: int) -> str:
    return f"{value:,}"


def format_percent(value: float) -> str:
    return f"{value:.3f}%"


def format_signed_int(value: int) -> str:
    return f"{value:+,}"


def format_signed_percent(value: float) -> str:
    return f"{value:+.3f}%"


# =============================================================================
# FILE VALIDATION
# =============================================================================


def ensure_output_directory() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_json_file(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        fail(
            "Required PB0-C audit output does not exist: "
            f"{path}"
        )

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as json_file:
            payload = json.load(
                json_file
            )
    except json.JSONDecodeError as error:
        fail(
            "PB0-C audit JSON could not be decoded: "
            f"{path}. Error: {error}"
        )

    if not isinstance(
        payload,
        dict,
    ):
        fail(
            "PB0-C audit JSON root must be an object: "
            f"{path}"
        )

    return payload


def require_mapping(
    value: Any,
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(
        value,
        Mapping,
    ):
        fail(
            f"Expected '{name}' to be a JSON object."
        )

    return value


def require_sequence(
    value: Any,
    name: str,
) -> Sequence[Any]:
    if isinstance(
        value,
        (str, bytes),
    ):
        fail(
            f"Expected '{name}' to be a JSON array."
        )

    if not isinstance(
        value,
        Sequence,
    ):
        fail(
            f"Expected '{name}' to be a JSON array."
        )

    return value


def require_int(
    value: Any,
    name: str,
) -> int:
    if isinstance(
        value,
        bool,
    ):
        fail(
            f"Expected '{name}' to be an integer, "
            "not a boolean."
        )

    if isinstance(
        value,
        int,
    ):
        return value

    if isinstance(
        value,
        float,
    ):
        if not math.isfinite(
            value
        ):
            fail(
                f"Expected '{name}' to be finite."
            )

        if value.is_integer():
            return int(
                value
            )

    fail(
        f"Expected '{name}' to be an integer."
    )


def require_str(
    value: Any,
    name: str,
) -> str:
    if not isinstance(
        value,
        str,
    ):
        fail(
            f"Expected '{name}' to be a string."
        )

    stripped = value.strip()

    if not stripped:
        fail(
            f"Expected '{name}' to be non-empty."
        )

    return stripped


# =============================================================================
# PB0-C BASELINE EXTRACTION
# =============================================================================


def find_seed_audits(
    payload: Mapping[str, Any],
) -> Sequence[Any]:
    for key in (
        "seed_audits",
        "audits",
        "seed_results",
    ):
        if key in payload:
            return require_sequence(
                payload[key],
                key,
            )

    fail(
        "Could not locate seed audits in PB0-C audit JSON. "
        "Expected one of: seed_audits, audits, seed_results."
    )


def extract_component_summaries(
    seed_audit: Mapping[str, Any],
) -> Sequence[Any]:
    for key in (
        "component_summaries",
        "components",
        "component_allocation",
    ):
        if key in seed_audit:
            return require_sequence(
                seed_audit[key],
                key,
            )

    fail(
        "Could not locate component summaries in a PB0-C seed audit."
    )


def extract_baseline_parameter_count(
    payload: Mapping[str, Any],
) -> int:
    for key in (
        "baseline_parameter_count",
        "expected_parameter_count",
        "parameter_count",
    ):
        if key in payload:
            return require_int(
                payload[key],
                key,
            )

    seed_audits = find_seed_audits(
        payload
    )

    if not seed_audits:
        fail(
            "PB0-C audit contains no seed audits."
        )

    first_audit = require_mapping(
        seed_audits[0],
        "seed_audits[0]",
    )

    if "parameter_count" not in first_audit:
        fail(
            "PB0-C seed audit does not expose parameter_count."
        )

    return require_int(
        first_audit["parameter_count"],
        "seed_audits[0].parameter_count",
    )


def extract_baseline_components(
    payload: Mapping[str, Any],
    baseline_parameter_count: int,
) -> dict[str, ComponentAllocation]:
    seed_audits = find_seed_audits(
        payload
    )

    if not seed_audits:
        fail(
            "PB0-C audit contains no seed audits."
        )

    first_seed = require_mapping(
        seed_audits[0],
        "seed_audits[0]",
    )

    component_summaries = (
        extract_component_summaries(
            first_seed
        )
    )

    components: dict[
        str,
        ComponentAllocation,
    ] = {}

    for index, raw_summary in enumerate(
        component_summaries
    ):
        summary = require_mapping(
            raw_summary,
            f"component_summaries[{index}]",
        )

        component = require_str(
            summary.get(
                "component"
            ),
            f"component_summaries[{index}].component",
        )

        display_name = (
            summary.get(
                "display_name"
            )
        )

        if display_name is None:
            display_name = (
                COMPONENT_DISPLAY_NAMES.get(
                    component,
                    component,
                )
            )

        display_name = require_str(
            display_name,
            f"component_summaries[{index}].display_name",
        )

        parameter_count = require_int(
            summary.get(
                "parameter_count"
            ),
            f"component_summaries[{index}].parameter_count",
        )

        if parameter_count < 0:
            fail(
                f"Component '{component}' has "
                "a negative parameter count."
            )

        if component in components:
            fail(
                f"Duplicate component '{component}' "
                "in PB0-C audit."
            )

        fraction = (
            parameter_count
            / baseline_parameter_count
        )

        components[component] = (
            ComponentAllocation(
                component=component,
                display_name=display_name,
                parameter_count=parameter_count,
                parameter_fraction=fraction,
                parameter_percent=(
                    fraction
                    * 100.0
                ),
            )
        )

    missing_components = [
        component
        for component in COMPONENT_ORDER
        if component not in components
    ]

    if missing_components:
        fail(
            "PB0-C audit is missing required components: "
            + ", ".join(
                missing_components
            )
        )

    component_total = sum(
        components[
            component
        ].parameter_count
        for component in COMPONENT_ORDER
    )

    if (
        component_total
        != baseline_parameter_count
    ):
        fail(
            "PB0-C component total does not equal "
            "the baseline parameter count. "
            f"Components={component_total:,}, "
            f"Baseline={baseline_parameter_count:,}"
        )

    return components


def validate_pb0_c_baseline(
    payload: Mapping[str, Any],
    baseline_parameter_count: int,
    components: Mapping[
        str,
        ComponentAllocation,
    ],
) -> None:
    if (
        baseline_parameter_count
        != EXPECTED_BASELINE_PARAMETER_COUNT
    ):
        fail(
            "PB0-C baseline parameter count does not "
            "match the frozen research baseline. "
            f"Observed={baseline_parameter_count:,}, "
            f"Expected={EXPECTED_BASELINE_PARAMETER_COUNT:,}"
        )

    seed_audits = find_seed_audits(
        payload
    )

    if len(
        seed_audits
    ) < 2:
        fail(
            "PB0-D requires at least two PB0-C seed audits "
            "to confirm topology consistency."
        )

    first_seed = require_mapping(
        seed_audits[0],
        "seed_audits[0]",
    )

    second_seed = require_mapping(
        seed_audits[1],
        "seed_audits[1]",
    )

    first_count = require_int(
        first_seed.get(
            "parameter_count"
        ),
        "seed_audits[0].parameter_count",
    )

    second_count = require_int(
        second_seed.get(
            "parameter_count"
        ),
        "seed_audits[1].parameter_count",
    )

    if (
        first_count
        != second_count
    ):
        fail(
            "PB0-C seed parameter counts differ. "
            f"Seed1={first_count:,}, "
            f"Seed2={second_count:,}"
        )

    if (
        first_count
        != baseline_parameter_count
    ):
        fail(
            "PB0-C seed parameter count does not match "
            "the extracted baseline parameter count."
        )

    expected_baseline = {
        COMPONENT_MATRIX_MEMORY: 25_233_468,
        COMPONENT_VECTOR_MAMBA: 18_898_944,
        COMPONENT_ROUTER: 399_744,
        COMPONENT_FEEDBACK_BRIDGE: 6_156,
        COMPONENT_EMBEDDINGS: 131_072,
        COMPONENT_OUTPUT_HEAD: 1_181_440,
        COMPONENT_NORMALIZATION: 12_288,
        COMPONENT_AUXILIARY: 1_181_440,
        COMPONENT_OTHER: 393_216,
    }

    for component, expected_count in (
        expected_baseline.items()
    ):
        observed_count = (
            components[
                component
            ].parameter_count
        )

        if (
            observed_count
            != expected_count
        ):
            fail(
                "PB0-C component allocation differs from "
                "the frozen baseline. "
                f"Component={component}, "
                f"Observed={observed_count:,}, "
                f"Expected={expected_count:,}"
            )


# =============================================================================
# FIXED-BUDGET REALLOCATION HELPERS
# =============================================================================


def frozen_parameter_count(
    baseline_components: Mapping[
        str,
        ComponentAllocation,
    ],
) -> int:
    return sum(
        baseline_components[
            component
        ].parameter_count
        for component in FROZEN_COMPONENTS
    )


def reallocatable_parameter_budget(
    baseline_components: Mapping[
        str,
        ComponentAllocation,
    ],
) -> int:
    return sum(
        baseline_components[
            component
        ].parameter_count
        for component in REALLOCATION_COMPONENTS
    )


def proportional_remainder_distribution(
    *,
    budget: int,
    weights: Mapping[str, float],
) -> dict[str, int]:
    if budget < 0:
        fail(
            "Budget must not be negative."
        )

    if not weights:
        fail(
            "Weights must not be empty."
        )

    weight_sum = sum(
        weights.values()
    )

    if (
        not math.isfinite(
            weight_sum
        )
        or weight_sum <= 0.0
    ):
        fail(
            "Weights must have a positive finite sum."
        )

    raw_values: dict[
        str,
        float,
    ] = {}

    allocations: dict[
        str,
        int,
    ] = {}

    assigned = 0

    for component in (
        REALLOCATION_COMPONENTS
    ):
        if component not in weights:
            fail(
                f"Missing allocation weight for "
                f"'{component}'."
            )

        weight = weights[
            component
        ]

        if (
            not math.isfinite(
                weight
            )
            or weight < 0.0
        ):
            fail(
                f"Invalid allocation weight for "
                f"'{component}': {weight}"
            )

        raw_value = (
            budget
            * weight
            / weight_sum
        )

        floor_value = math.floor(
            raw_value
        )

        raw_values[
            component
        ] = raw_value

        allocations[
            component
        ] = floor_value

        assigned += floor_value

    remaining = (
        budget
        - assigned
    )

    ranking = sorted(
        REALLOCATION_COMPONENTS,
        key=lambda component: (
            raw_values[
                component
            ]
            - allocations[
                component
            ],
            -REALLOCATION_COMPONENTS.index(
                component
            ),
        ),
        reverse=True,
    )

    for component in ranking[
        :remaining
    ]:
        allocations[
            component
        ] += 1

    final_total = sum(
        allocations.values()
    )

    if (
        final_total
        != budget
    ):
        fail(
            "Largest-remainder allocation failed to "
            "preserve the exact budget. "
            f"Allocated={final_total:,}, "
            f"Budget={budget:,}"
        )

    return allocations


def validate_reallocation_weights(
    weights: Mapping[
        str,
        float,
    ],
) -> None:
    if set(
        weights.keys()
    ) != set(
        REALLOCATION_COMPONENTS
    ):
        missing = (
            set(
                REALLOCATION_COMPONENTS
            )
            - set(
                weights.keys()
            )
        )

        unexpected = (
            set(
                weights.keys()
            )
            - set(
                REALLOCATION_COMPONENTS
            )
        )

        fail(
            "Candidate weights must contain exactly "
            "the reallocatable components. "
            f"Missing={sorted(missing)}, "
            f"Unexpected={sorted(unexpected)}"
        )

    total = 0.0

    for component in (
        REALLOCATION_COMPONENTS
    ):
        value = weights[
            component
        ]

        if (
            not isinstance(
                value,
                (int, float),
            )
            or isinstance(
                value,
                bool,
            )
        ):
            fail(
                f"Candidate weight for '{component}' "
                "must be numeric."
            )

        numeric_value = float(
            value
        )

        if (
            not math.isfinite(
                numeric_value
            )
            or numeric_value < 0.0
        ):
            fail(
                f"Candidate weight for '{component}' "
                "must be finite and non-negative."
            )

        total += numeric_value

    if total <= 0.0:
        fail(
            "Candidate allocation weights must sum "
            "to a positive value."
        )


# =============================================================================
# CANDIDATE DEFINITIONS
# =============================================================================


def candidate_weight_matrix() -> list[
    tuple[
        str,
        str,
        str,
        int,
        bool,
        dict[str, float],
    ]
]:
    return [
        (
            "BASELINE",
            "Baseline Allocation",
            (
                "Frozen PB0-C allocation retained as the "
                "control reference. This candidate is not "
                "a new architecture and exists to preserve "
                "the exact audited allocation as the "
                "comparison origin."
            ),
            0,
            False,
            {
                COMPONENT_MATRIX_MEMORY: 1.0,
                COMPONENT_VECTOR_MAMBA: 1.0,
                COMPONENT_ROUTER: 1.0,
                COMPONENT_FEEDBACK_BRIDGE: 1.0,
            },
        ),
        (
            "A",
            "Matrix-to-Vector Rebalance",
            (
                "First controlled hypothesis: modestly "
                "reduce the dominant matrix-memory allocation "
                "and reallocate capacity primarily into the "
                "vector/Mamba pathway, with a smaller targeted "
                "increase to the feedback bridge and router. "
                "The objective is to test whether the current "
                "53.193% matrix allocation is over-provisioned "
                "relative to vector computation and matrix-to-"
                "vector information transfer."
            ),
            1,
            True,
            {
                COMPONENT_MATRIX_MEMORY: 0.92,
                COMPONENT_VECTOR_MAMBA: 1.08,
                COMPONENT_ROUTER: 1.15,
                COMPONENT_FEEDBACK_BRIDGE: 1.50,
            },
        ),
        (
            "B",
            "Matrix Capacity Reduction",
            (
                "Stronger matrix-capacity reduction hypothesis. "
                "Tests whether matrix parameters exhibit "
                "substantial diminishing returns when capacity "
                "is shifted toward vector computation while "
                "keeping router and feedback growth modest."
            ),
            2,
            False,
            {
                COMPONENT_MATRIX_MEMORY: 0.82,
                COMPONENT_VECTOR_MAMBA: 1.18,
                COMPONENT_ROUTER: 1.10,
                COMPONENT_FEEDBACK_BRIDGE: 1.25,
            },
        ),
        (
            "C",
            "Feedback Expansion",
            (
                "Feedback-bottleneck hypothesis. Keeps the "
                "major matrix/vector balance relatively close "
                "to baseline while directing more of the "
                "reallocatable budget toward the matrix-to-vector "
                "feedback bridge."
            ),
            3,
            False,
            {
                COMPONENT_MATRIX_MEMORY: 0.95,
                COMPONENT_VECTOR_MAMBA: 1.03,
                COMPONENT_ROUTER: 1.10,
                COMPONENT_FEEDBACK_BRIDGE: 4.00,
            },
        ),
        (
            "D",
            "Router Capacity Expansion",
            (
                "Coordination hypothesis. Tests whether routing "
                "capacity is under-allocated relative to the "
                "two dominant memory pathways."
            ),
            4,
            False,
            {
                COMPONENT_MATRIX_MEMORY: 0.95,
                COMPONENT_VECTOR_MAMBA: 1.02,
                COMPONENT_ROUTER: 3.00,
                COMPONENT_FEEDBACK_BRIDGE: 1.20,
            },
        ),
    ]


# =============================================================================
# CANDIDATE CONSTRUCTION
# =============================================================================


def build_candidate(
    *,
    candidate_id: str,
    candidate_name: str,
    hypothesis: str,
    selection_order: int,
    selected_for_next_gate: bool,
    weights: Mapping[
        str,
        float,
    ],
    baseline_components: Mapping[
        str,
        ComponentAllocation,
    ],
    baseline_parameter_count: int,
) -> CandidateAllocation:
    validate_reallocation_weights(
        weights
    )

    baseline_reallocatable_budget = (
        reallocatable_parameter_budget(
            baseline_components
        )
    )

    baseline_weights = {
        component: float(
            baseline_components[
                component
            ].parameter_count
        )
        for component in REALLOCATION_COMPONENTS
    }

    is_baseline_candidate = (
        candidate_id
        == "BASELINE"
    )

    if is_baseline_candidate:
        reallocations = {
            component: baseline_components[
                component
            ].parameter_count
            for component in REALLOCATION_COMPONENTS
        }
    else:
        effective_weights = {
            component: (
                baseline_weights[
                    component
                ]
                * weights[
                    component
                ]
            )
            for component in REALLOCATION_COMPONENTS
        }

        reallocations = (
            proportional_remainder_distribution(
                budget=(
                    baseline_reallocatable_budget
                ),
                weights=(
                    effective_weights
                ),
            )
        )

    component_counts: dict[
        str,
        int,
    ] = {}

    for component in COMPONENT_ORDER:
        if component in (
            REALLOCATION_COMPONENTS
        ):
            component_counts[
                component
            ] = reallocations[
                component
            ]
        else:
            component_counts[
                component
            ] = baseline_components[
                component
            ].parameter_count

    total_parameters = sum(
        component_counts.values()
    )

    if (
        total_parameters
        != baseline_parameter_count
    ):
        fail(
            f"Candidate '{candidate_id}' violates the "
            "fixed parameter budget. "
            f"Candidate={total_parameters:,}, "
            f"Baseline={baseline_parameter_count:,}"
        )

    changes = {
        component: (
            component_counts[
                component
            ]
            - baseline_components[
                component
            ].parameter_count
        )
        for component in REALLOCATION_COMPONENTS
    }

    change_percents: dict[
        str,
        float,
    ] = {}

    for component in (
        REALLOCATION_COMPONENTS
    ):
        baseline_count = (
            baseline_components[
                component
            ].parameter_count
        )

        if baseline_count <= 0:
            fail(
                f"Baseline component '{component}' "
                "has a non-positive parameter count."
            )

        change_percents[
            component
        ] = (
            changes[
                component
            ]
            / baseline_count
            * 100.0
        )

    all_component_shifts = [
        abs(
            component_counts[
                component
            ]
            - baseline_components[
                component
            ].parameter_count
        )
        for component in COMPONENT_ORDER
    ]

    max_absolute_component_shift = max(
        all_component_shifts
    )

    max_absolute_component_shift_percent = (
        max_absolute_component_shift
        / baseline_parameter_count
        * 100.0
    )

    allocation_distance_l1_parameters = sum(
        all_component_shifts
    )

    allocation_distance_l1_percent = (
        allocation_distance_l1_parameters
        / baseline_parameter_count
        * 100.0
    )

    if candidate_id == "BASELINE":
        status = "control"
    elif selected_for_next_gate:
        status = "selected_hypothesis"
    else:
        status = "unselected_candidate"

    return CandidateAllocation(
        candidate_id=candidate_id,
        candidate_name=candidate_name,
        hypothesis=hypothesis,
        status=status,
        selection_order=selection_order,
        selected_for_next_gate=(
            selected_for_next_gate
        ),
        matrix_memory_parameters=(
            component_counts[
                COMPONENT_MATRIX_MEMORY
            ]
        ),
        vector_mamba_parameters=(
            component_counts[
                COMPONENT_VECTOR_MAMBA
            ]
        ),
        router_parameters=(
            component_counts[
                COMPONENT_ROUTER
            ]
        ),
        feedback_bridge_parameters=(
            component_counts[
                COMPONENT_FEEDBACK_BRIDGE
            ]
        ),
        embeddings_parameters=(
            component_counts[
                COMPONENT_EMBEDDINGS
            ]
        ),
        output_head_parameters=(
            component_counts[
                COMPONENT_OUTPUT_HEAD
            ]
        ),
        normalization_parameters=(
            component_counts[
                COMPONENT_NORMALIZATION
            ]
        ),
        auxiliary_parameters=(
            component_counts[
                COMPONENT_AUXILIARY
            ]
        ),
        other_parameters=(
            component_counts[
                COMPONENT_OTHER
            ]
        ),
        total_parameters=(
            total_parameters
        ),
        baseline_total_parameters=(
            baseline_parameter_count
        ),
        total_difference_from_baseline=(
            total_parameters
            - baseline_parameter_count
        ),
        max_absolute_component_shift=(
            max_absolute_component_shift
        ),
        max_absolute_component_shift_percent=(
            max_absolute_component_shift_percent
        ),
        allocation_distance_l1_parameters=(
            allocation_distance_l1_parameters
        ),
        allocation_distance_l1_percent=(
            allocation_distance_l1_percent
        ),
        matrix_memory_change=(
            changes[
                COMPONENT_MATRIX_MEMORY
            ]
        ),
        vector_mamba_change=(
            changes[
                COMPONENT_VECTOR_MAMBA
            ]
        ),
        router_change=(
            changes[
                COMPONENT_ROUTER
            ]
        ),
        feedback_bridge_change=(
            changes[
                COMPONENT_FEEDBACK_BRIDGE
            ]
        ),
        matrix_memory_change_percent=(
            change_percents[
                COMPONENT_MATRIX_MEMORY
            ]
        ),
        vector_mamba_change_percent=(
            change_percents[
                COMPONENT_VECTOR_MAMBA
            ]
        ),
        router_change_percent=(
            change_percents[
                COMPONENT_ROUTER
            ]
        ),
        feedback_bridge_change_percent=(
            change_percents[
                COMPONENT_FEEDBACK_BRIDGE
            ]
        ),
    )


def build_candidates(
    baseline_components: Mapping[
        str,
        ComponentAllocation,
    ],
    baseline_parameter_count: int,
) -> list[
    CandidateAllocation
]:
    candidates: list[
        CandidateAllocation
    ] = []

    selected_count = 0

    for (
        candidate_id,
        candidate_name,
        hypothesis,
        selection_order,
        selected_for_next_gate,
        weights,
    ) in candidate_weight_matrix():
        candidate = build_candidate(
            candidate_id=candidate_id,
            candidate_name=candidate_name,
            hypothesis=hypothesis,
            selection_order=selection_order,
            selected_for_next_gate=(
                selected_for_next_gate
            ),
            weights=weights,
            baseline_components=(
                baseline_components
            ),
            baseline_parameter_count=(
                baseline_parameter_count
            ),
        )

        if candidate.selected_for_next_gate:
            selected_count += 1

        candidates.append(
            candidate
        )

    if selected_count != 1:
        fail(
            "PB0-D must select exactly one candidate "
            "for the next gate. "
            f"Selected count={selected_count}"
        )

    candidate_ids = [
        candidate.candidate_id
        for candidate in candidates
    ]

    if len(
        candidate_ids
    ) != len(
        set(
            candidate_ids
        )
    ):
        fail(
            "Candidate identifiers must be unique."
        )

    if "A" not in candidate_ids:
        fail(
            "Candidate A is missing from the fixed-budget "
            "candidate matrix."
        )

    selected_candidate = next(
        (
            candidate
            for candidate in candidates
            if candidate.selected_for_next_gate
        ),
        None,
    )

    if selected_candidate is None:
        fail(
            "No selected PB0-D candidate exists."
        )

    if (
        selected_candidate.candidate_id
        != "A"
    ):
        fail(
            "Candidate A must be the first selected "
            "controlled allocation hypothesis."
        )

    return candidates


# =============================================================================
# CANDIDATE COMPONENT ROWS
# =============================================================================


def candidate_component_rows(
    candidates: Sequence[
        CandidateAllocation
    ],
    baseline_components: Mapping[
        str,
        ComponentAllocation,
    ],
) -> list[
    CandidateComponentRow
]:
    rows: list[
        CandidateComponentRow
    ] = []

    candidate_field_by_component = {
        COMPONENT_MATRIX_MEMORY:
            "matrix_memory_parameters",
        COMPONENT_VECTOR_MAMBA:
            "vector_mamba_parameters",
        COMPONENT_ROUTER:
            "router_parameters",
        COMPONENT_FEEDBACK_BRIDGE:
            "feedback_bridge_parameters",
        COMPONENT_EMBEDDINGS:
            "embeddings_parameters",
        COMPONENT_OUTPUT_HEAD:
            "output_head_parameters",
        COMPONENT_NORMALIZATION:
            "normalization_parameters",
        COMPONENT_AUXILIARY:
            "auxiliary_parameters",
        COMPONENT_OTHER:
            "other_parameters",
    }

    for candidate in candidates:
        for component in COMPONENT_ORDER:
            field_name = (
                candidate_field_by_component[
                    component
                ]
            )

            parameter_count = getattr(
                candidate,
                field_name,
            )

            baseline_count = (
                baseline_components[
                    component
                ].parameter_count
            )

            difference = (
                parameter_count
                - baseline_count
            )

            if baseline_count <= 0:
                difference_percent = 0.0
            else:
                difference_percent = (
                    difference
                    / baseline_count
                    * 100.0
                )

            fraction = (
                parameter_count
                / candidate.total_parameters
            )

            rows.append(
                CandidateComponentRow(
                    candidate_id=(
                        candidate.candidate_id
                    ),
                    candidate_name=(
                        candidate.candidate_name
                    ),
                    component=component,
                    display_name=(
                        COMPONENT_DISPLAY_NAMES[
                            component
                        ]
                    ),
                    parameter_count=(
                        parameter_count
                    ),
                    baseline_parameter_count=(
                        baseline_count
                    ),
                    parameter_difference=(
                        difference
                    ),
                    parameter_difference_percent_of_baseline=(
                        difference_percent
                    ),
                    parameter_fraction_of_total=(
                        fraction
                    ),
                    parameter_percent_of_total=(
                        fraction
                        * 100.0
                    ),
                )
            )

    return rows


# =============================================================================
# OUTPUT WRITERS
# =============================================================================


def write_json_output(
    result: PB0DResult,
) -> None:
    payload = asdict(
        result
    )

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            payload,
            json_file,
            indent=2,
            sort_keys=False,
        )

        json_file.write(
            "\n"
        )


def write_candidate_csv(
    candidates: Sequence[
        CandidateAllocation
    ],
) -> None:
    if not candidates:
        fail(
            "No candidates available for CSV output."
        )

    rows = [
        asdict(
            candidate
        )
        for candidate in candidates
    ]

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(
                rows[0].keys()
            ),
            extrasaction="raise",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                row
            )


def write_component_csv(
    rows: Sequence[
        CandidateComponentRow
    ],
) -> None:
    if not rows:
        fail(
            "No candidate component rows available."
        )

    serialized_rows = [
        asdict(
            row
        )
        for row in rows
    ]

    with OUTPUT_COMPONENT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(
                serialized_rows[0].keys()
            ),
            extrasaction="raise",
        )

        writer.writeheader()

        for row in serialized_rows:
            writer.writerow(
                row
            )


def write_markdown_output(
    *,
    result: PB0DResult,
    component_rows: Sequence[
        CandidateComponentRow
    ],
) -> None:
    baseline_component_map = {
        component.component:
            component
        for component in (
            result.baseline_components
        )
    }

    selected_candidate = next(
        (
            candidate
            for candidate in result.candidates
            if candidate.selected_for_next_gate
        ),
        None,
    )

    if selected_candidate is None:
        fail(
            "Cannot write PB0-D markdown without "
            "a selected candidate."
        )

    lines: list[
        str
    ] = []

    lines.append(
        "# PB0-D Fixed-Budget Parameter Allocation Matrix"
    )

    lines.append(
        ""
    )

    lines.append(
        "## Objective"
    )

    lines.append(
        ""
    )

    lines.append(
        "Given the exact fixed parameter budget established "
        "by PB0-C, construct controlled parameter-allocation "
        "hypotheses without changing the total parameter count."
    )

    lines.append(
        ""
    )

    lines.append(
        "PB0-D does not train any candidate, does not evaluate "
        "validation BPC, and does not read the test split."
    )

    lines.append(
        ""
    )

    lines.append(
        "## Frozen Budget"
    )

    lines.append(
        ""
    )

    lines.append(
        f"- Exact baseline: "
        f"`{result.baseline_parameter_count:,}` parameters"
    )

    lines.append(
        f"- Reallocatable budget: "
        f"`{result.reallocatable_budget:,}` parameters"
    )

    lines.append(
        f"- Frozen non-reallocation components: "
        f"`{result.frozen_parameter_count:,}` parameters"
    )

    lines.append(
        ""
    )

    lines.append(
        "## Frozen Architecture Constraints"
    )

    lines.append(
        ""
    )

    lines.append(
        "| Field | Value |"
    )

    lines.append(
        "|---|---:|"
    )

    for key, value in (
        result.architecture_constraints.items()
    ):
        lines.append(
            f"| {key} | {value:,} |"
        )

    lines.append(
        ""
    )

    lines.append(
        "## Dataset Guardrails"
    )

    lines.append(
        ""
    )

    lines.append(
        f"- Training range: "
        f"`{TRAIN_START_BYTE:,}:{TRAIN_END_BYTE:,}`"
    )

    lines.append(
        f"- Validation range: "
        f"`{VALIDATION_START_BYTE:,}:{VALIDATION_END_BYTE:,}`"
    )

    lines.append(
        f"- Test range: "
        f"`{TEST_START_BYTE:,}:{TEST_END_BYTE:,}`"
    )

    lines.append(
        "- Test status: **unread until a candidate winner is frozen**"
    )

    lines.append(
        ""
    )

    lines.append(
        "## PB0-C Baseline Allocation"
    )

    lines.append(
        ""
    )

    lines.append(
        "| Component | Parameters | Percent of Total |"
    )

    lines.append(
        "|---|---:|---:|"
    )

    for component in (
        result.baseline_components
    ):
        lines.append(
            "| "
            f"{component.display_name} | "
            f"{component.parameter_count:,} | "
            f"{component.parameter_percent:.3f}% |"
        )

    lines.append(
        ""
    )

    lines.append(
        "## Candidate Matrix"
    )

    lines.append(
        ""
    )

    lines.append(
        "| Candidate | Status | Total | Matrix Δ | Vector Δ | Router Δ | Feedback Δ |"
    )

    lines.append(
        "|---|---|---:|---:|---:|---:|---:|"
    )

    for candidate in (
        result.candidates
    ):
        lines.append(
            "| "
            f"{candidate.candidate_id} "
            f"({candidate.candidate_name}) | "
            f"{candidate.status} | "
            f"{candidate.total_parameters:,} | "
            f"{candidate.matrix_memory_change:+,} | "
            f"{candidate.vector_mamba_change:+,} | "
            f"{candidate.router_change:+,} | "
            f"{candidate.feedback_bridge_change:+,} |"
        )

    lines.append(
        ""
    )

    lines.append(
        "## Candidate A"
    )

    lines.append(
        ""
    )

    lines.append(
        "**Candidate A is the selected first controlled "
        "allocation hypothesis. It is not trained by PB0-D.**"
    )

    lines.append(
        ""
    )

    lines.append(
        selected_candidate.hypothesis
    )

    lines.append(
        ""
    )

    lines.append(
        "| Component | Baseline | Candidate A | Difference |"
    )

    lines.append(
        "|---|---:|---:|---:|"
    )

    candidate_a_rows = [
        row
        for row in component_rows
        if row.candidate_id == "A"
    ]

    for row in candidate_a_rows:
        lines.append(
            "| "
            f"{row.display_name} | "
            f"{row.baseline_parameter_count:,} | "
            f"{row.parameter_count:,} | "
            f"{row.parameter_difference:+,} |"
        )

    lines.append(
        ""
    )

    lines.append(
        "## Next Gate"
    )

    lines.append(
        ""
    )

    lines.append(
        "1. Freeze Candidate A allocation."
    )

    lines.append(
        "2. Map the allocation to actual `model.py` dimensions."
    )

    lines.append(
        "3. Instantiate the modified architecture."
    )

    lines.append(
        "4. Perform an exact parameter recount."
    )

    lines.append(
        "5. Reject the mapping if it cannot preserve the fixed budget."
    )

    lines.append(
        "6. Only after the recount passes, train Candidate A."
    )

    lines.append(
        "7. Evaluate validation BPC only."
    )

    lines.append(
        "8. Keep the test split unread until the candidate winner is frozen."
    )

    lines.append(
        ""
    )

    lines.append(
        "## Explicit Limitation"
    )

    lines.append(
        ""
    )

    lines.append(
        "The allocations in PB0-D are parameter-budget targets. "
        "They are not yet evidence that every target is realizable "
        "by changing integer model dimensions. Architectural mapping "
        "and exact recount are mandatory before training."
    )

    lines.append(
        ""
    )

    with OUTPUT_MARKDOWN.open(
        "w",
        encoding="utf-8",
    ) as markdown_file:
        markdown_file.write(
            "\n".join(
                lines
            )
        )

        markdown_file.write(
            "\n"
        )


# =============================================================================
# CONSOLE REPORTING
# =============================================================================


def print_baseline(
    baseline_components: Mapping[
        str,
        ComponentAllocation,
    ],
    baseline_parameter_count: int,
) -> None:
    banner(
        "PB0-D BASELINE PARAMETER ALLOCATION"
    )

    print(
        "Exact baseline parameter count : "
        f"{format_int(baseline_parameter_count)}"
    )

    print()

    for component in (
        COMPONENT_ORDER
    ):
        allocation = (
            baseline_components[
                component
            ]
        )

        print(
            f"{allocation.display_name:28s} "
            f"{format_int(allocation.parameter_count):>14s} "
            f"{format_percent(allocation.parameter_percent):>10s}"
        )


def print_candidate_matrix(
    candidates: Sequence[
        CandidateAllocation
    ],
) -> None:
    banner(
        "PB0-D FIXED-BUDGET CANDIDATE MATRIX"
    )

    for candidate in candidates:
        print(
            f"{candidate.candidate_id:8s} "
            f"{candidate.candidate_name}"
        )

        print(
            f"  Status       : "
            f"{candidate.status}"
        )

        print(
            f"  Total        : "
            f"{format_int(candidate.total_parameters)}"
        )

        print(
            f"  Matrix       : "
            f"{format_signed_int(candidate.matrix_memory_change)} "
            f"({format_signed_percent(candidate.matrix_memory_change_percent)})"
        )

        print(
            f"  Vector       : "
            f"{format_signed_int(candidate.vector_mamba_change)} "
            f"({format_signed_percent(candidate.vector_mamba_change_percent)})"
        )

        print(
            f"  Router       : "
            f"{format_signed_int(candidate.router_change)} "
            f"({format_signed_percent(candidate.router_change_percent)})"
        )

        print(
            f"  Feedback     : "
            f"{format_signed_int(candidate.feedback_bridge_change)} "
            f"({format_signed_percent(candidate.feedback_bridge_change_percent)})"
        )

        print(
            f"  L1 shift     : "
            f"{format_int(candidate.allocation_distance_l1_parameters)}"
        )

        print()


# =============================================================================
# MAIN
# =============================================================================


def main() -> int:
    banner(
        "MODUS_X PB0-D FIXED-BUDGET PARAMETER ALLOCATION"
    )

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"PB0-C audit  : {PB0_C_AUDIT_JSON}"
    )

    print(
        f"Output dir   : {OUTPUT_DIR}"
    )

    print()

    print(
        "Loading PB0-C exact parameter budget audit..."
    )

    ensure_output_directory()

    payload = load_json_file(
        PB0_C_AUDIT_JSON
    )

    baseline_parameter_count = (
        extract_baseline_parameter_count(
            payload
        )
    )

    baseline_components = (
        extract_baseline_components(
            payload,
            baseline_parameter_count,
        )
    )

    validate_pb0_c_baseline(
        payload,
        baseline_parameter_count,
        baseline_components,
    )

    print_baseline(
        baseline_components,
        baseline_parameter_count,
    )

    reallocatable_budget = (
        reallocatable_parameter_budget(
            baseline_components
        )
    )

    frozen_budget = (
        frozen_parameter_count(
            baseline_components
        )
    )

    if (
        reallocatable_budget
        + frozen_budget
        != baseline_parameter_count
    ):
        fail(
            "PB0-D budget partition does not reconstruct "
            "the baseline parameter count."
        )

    candidates = build_candidates(
        baseline_components,
        baseline_parameter_count,
    )

    print_candidate_matrix(
        candidates
    )

    selected_candidate = next(
        (
            candidate
            for candidate in candidates
            if candidate.selected_for_next_gate
        ),
        None,
    )

    if selected_candidate is None:
        fail(
            "No candidate was selected for the next gate."
        )

    component_rows = (
        candidate_component_rows(
            candidates,
            baseline_components,
        )
    )

    result = PB0DResult(
        generated_at_utc=(
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        project_root=str(
            PROJECT_ROOT
        ),
        baseline_audit_path=str(
            PB0_C_AUDIT_JSON
        ),
        baseline_parameter_count=(
            baseline_parameter_count
        ),
        reallocatable_budget=(
            reallocatable_budget
        ),
        frozen_parameter_count=(
            frozen_budget
        ),
        architecture_constraints={
            "vocabulary_size":
                FROZEN_VOCABULARY_SIZE,
            "layer_count":
                FROZEN_LAYER_COUNT,
            "embedding_width":
                FROZEN_EMBEDDING_WIDTH,
            "hidden_width":
                FROZEN_HIDDEN_WIDTH,
            "matrix_vector_state_width":
                FROZEN_MATRIX_VECTOR_STATE_WIDTH,
        },
        dataset_protocol={
            "train_start_byte":
                TRAIN_START_BYTE,
            "train_end_byte":
                TRAIN_END_BYTE,
            "validation_start_byte":
                VALIDATION_START_BYTE,
            "validation_end_byte":
                VALIDATION_END_BYTE,
            "test_start_byte":
                TEST_START_BYTE,
            "test_end_byte":
                TEST_END_BYTE,
            "test_split_policy":
                "unread_until_candidate_winner_is_frozen",
            "pb0_d_reads_dataset":
                False,
        },
        baseline_components=tuple(
            baseline_components[
                component
            ]
            for component in COMPONENT_ORDER
        ),
        candidates=tuple(
            candidates
        ),
        selected_candidate_id=(
            selected_candidate.candidate_id
        ),
    )

    banner(
        "WRITING PB0-D OUTPUTS"
    )

    write_json_output(
        result
    )

    write_candidate_csv(
        candidates
    )

    write_component_csv(
        component_rows
    )

    write_markdown_output(
        result=result,
        component_rows=(
            component_rows
        ),
    )

    banner(
        "PB0-D FIXED-BUDGET ALLOCATION COMPLETE"
    )

    print(
        "Baseline parameter count : "
        f"{format_int(baseline_parameter_count)}"
    )

    print(
        "Reallocatable budget     : "
        f"{format_int(reallocatable_budget)}"
    )

    print(
        "Frozen budget            : "
        f"{format_int(frozen_budget)}"
    )

    print()

    print(
        "Selected candidate       : "
        f"{selected_candidate.candidate_id} "
        f"({selected_candidate.candidate_name})"
    )

    print(
        "Selected candidate total : "
        f"{format_int(selected_candidate.total_parameters)}"
    )

    print()

    print(
        "PB0-D does not train a model."
    )

    print(
        "PB0-D does not read validation data."
    )

    print(
        "PB0-D does not read test data."
    )

    print()

    print(
        "Output files:"
    )

    print(
        f"  {OUTPUT_JSON}"
    )

    print(
        f"  {OUTPUT_MARKDOWN}"
    )

    print(
        f"  {OUTPUT_CSV}"
    )

    print(
        f"  {OUTPUT_COMPONENT_CSV}"
    )

    print()

    print(
        "NEXT GATE: freeze Candidate A, map its allocation "
        "to actual model.py dimensions, then perform an "
        "exact parameter recount before any training."
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )
    except KeyboardInterrupt:
        print(
            "\nPB0-D interrupted by user.",
            file=sys.stderr,
        )
        raise SystemExit(
            130
        )
    except Exception as error:
        banner(
            "PB0-D FIXED-BUDGET ALLOCATION FAILED"
        )

        print(
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )

        raise