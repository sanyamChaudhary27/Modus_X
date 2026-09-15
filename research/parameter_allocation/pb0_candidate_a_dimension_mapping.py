from __future__ import annotations
from collections.abc import Mapping
from typing import Any, NoReturn
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, cast


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "research"
    / "parameter_allocation"
    / "outputs"
)

PB0_D_ALLOCATION_JSON = (
    OUTPUT_DIR
    / "PB0_FIXED_BUDGET_ALLOCATION.json"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "PB0_CANDIDATE_A_DIMENSION_MAPPING.json"
)

OUTPUT_MARKDOWN = (
    OUTPUT_DIR
    / "PB0_CANDIDATE_A_DIMENSION_MAPPING.md"
)

OUTPUT_CANDIDATES_CSV = (
    OUTPUT_DIR
    / "PB0_CANDIDATE_A_DIMENSION_MAPPING_CANDIDATES.csv"
)


# =============================================================================
# FROZEN PB0-D BASELINE CONSTRAINTS
# =============================================================================

EXPECTED_TOTAL_PARAMETERS = 47_437_768

VOCABULARY_SIZE = 256
LAYER_COUNT = 12
EMBEDDING_WIDTH = 512
HIDDEN_WIDTH = 1_536
MATRIX_VECTOR_STATE_WIDTH = 512


# =============================================================================
# COMPONENT IDENTIFIERS
# =============================================================================

MATRIX_MEMORY = "matrix_memory"
VECTOR_MAMBA_PATHWAY = "vector_mamba_pathway"
ROUTER = "router"
FEEDBACK_BRIDGE = "feedback_bridge"

REALLOCATABLE_COMPONENTS = (
    MATRIX_MEMORY,
    VECTOR_MAMBA_PATHWAY,
    ROUTER,
    FEEDBACK_BRIDGE,
)

FROZEN_COMPONENTS = (
    "embeddings",
    "output_head",
    "normalization",
    "auxiliary",
    "other",
)


# =============================================================================
# SEARCH CONSTRAINTS
# =============================================================================

# The baseline architecture is frozen at 12 layers.
# Candidate A is allowed to change only integer capacity dimensions.

MIN_MATRIX_RANK = 64
MAX_MATRIX_RANK = 1_024
MATRIX_RANK_STEP = 8

MIN_VECTOR_EXPANSION = 256
MAX_VECTOR_EXPANSION = 4_096
VECTOR_EXPANSION_STEP = 8

MIN_ROUTER_HIDDEN_WIDTH = 8
MAX_ROUTER_HIDDEN_WIDTH = 512
ROUTER_HIDDEN_WIDTH_STEP = 4

MIN_FEEDBACK_RANK = 1
MAX_FEEDBACK_RANK = 128
FEEDBACK_RANK_STEP = 1

# Number of best mappings retained after exhaustive candidate evaluation.
TOP_CANDIDATE_LIMIT = 50


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass(frozen=True)
class ComponentTarget:
    component: str
    baseline_parameters: int
    target_parameters: int
    delta_parameters: int


@dataclass(frozen=True)
class ArchitectureDimensions:
    matrix_rank: int
    vector_expansion: int
    router_hidden_width: int
    feedback_rank: int


@dataclass(frozen=True)
class ComponentCounts:
    matrix_memory: int
    vector_mamba_pathway: int
    router: int
    feedback_bridge: int
    frozen_components: int
    total_parameters: int


@dataclass(frozen=True)
class CandidateTargets:
    candidate_id: str
    candidate_name: str
    selected_for_next_gate: bool
    total_parameters: int
    matrix_memory_parameters: int
    vector_mamba_parameters: int
    router_parameters: int
    feedback_bridge_parameters: int
    embeddings_parameters: int
    output_head_parameters: int
    normalization_parameters: int
    auxiliary_parameters: int
    other_parameters: int


@dataclass(frozen=True)
class MappingCandidate:
    rank: int
    dimensions: ArchitectureDimensions
    counts: ComponentCounts
    matrix_error: int
    vector_error: int
    router_error: int
    feedback_error: int
    total_component_absolute_error: int
    total_budget_error: int
    score: int
    exact_budget_match: bool


@dataclass(frozen=True)
class MappingResult:
    expected_total_parameters: int
    frozen_parameter_count: int
    candidate_a_targets: tuple[ComponentTarget, ...]
    selected_dimensions: ArchitectureDimensions
    selected_counts: ComponentCounts
    selected_mapping: MappingCandidate
    exact_budget_match: bool
    target_component_match: bool
    searched_candidate_count: int
    retained_candidate_count: int


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


def signed_int(value: int) -> str:
    return f"{value:+,}"


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        fail(
            f"{description} does not exist: {path}"
        )


def ensure_output_directory() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# JSON HELPERS
# =============================================================================


def load_json(path: Path) -> dict[str, Any]:
    require_file(
        path,
        "Required JSON file",
    )

    payload: Any = None
    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            payload = json.load(file)
    except json.JSONDecodeError as error:
        fail(
            f"Invalid JSON in {path}: {error}"
        )

    if not isinstance(
        payload,
        dict,
    ):
        fail(
            f"Expected JSON object at root of {path}."
        )

    return payload


def get_first_present(
    mapping: dict[str, Any],
    keys: Iterable[str],
) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]

    return None


def normalize_component_name(
    value: str,
) -> str:
    normalized = (
        value
        .strip()
        .lower()
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )

    aliases = {
        "matrix_memory": MATRIX_MEMORY,
        "matrix": MATRIX_MEMORY,
        "vector_mamba_pathway": VECTOR_MAMBA_PATHWAY,
        "vector_mamba": VECTOR_MAMBA_PATHWAY,
        "vector_pathway": VECTOR_MAMBA_PATHWAY,
        "mamba_pathway": VECTOR_MAMBA_PATHWAY,
        "router": ROUTER,
        "feedback_bridge": FEEDBACK_BRIDGE,
        "feedback": FEEDBACK_BRIDGE,
        "embeddings": "embeddings",
        "embedding": "embeddings",
        "output_head": "output_head",
        "output": "output_head",
        "normalization": "normalization",
        "auxiliary": "auxiliary",
        "other": "other",
    }

    if normalized not in aliases:
        return normalized

    return aliases[normalized]


# =============================================================================
# PB0-D EXTRACTION
# =============================================================================


def find_candidate_collection(
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    possible_keys = (
        "candidates",
        "candidate_matrix",
        "allocations",
        "results",
    )

    for key in possible_keys:
        value = payload.get(key)

        if isinstance(
            value,
            list,
        ):
            candidates = [
                item
                for item in value
                if isinstance(
                    item,
                    dict,
                )
            ]

            if candidates:
                return candidates

    fail(
        "PB0-D allocation JSON does not expose "
        "a recognizable candidate collection."
    )
    return []


def extract_candidate_identifier(
    candidate: dict[str, Any],
) -> str:
    value = get_first_present(
        candidate,
        (
            "candidate",
            "id",
            "name",
            "candidate_id",
            "label",
        ),
    )

    if not isinstance(
        value,
        str,
    ):
        fail(
            "PB0-D candidate does not contain "
            "a valid candidate identifier."
        )

    return value.strip()


def is_candidate_a(
    candidate: dict[str, Any],
) -> bool:
    identifier = (
        extract_candidate_identifier(
            candidate
        )
        .strip()
        .upper()
    )

    if identifier == "A":
        return True

    if identifier.startswith(
        "A "
    ):
        return True

    if identifier.startswith(
        "A("
    ):
        return True

    return False


def extract_candidate_a(
    payload: dict[str, Any],
) -> dict[str, Any]:
    candidates = (
        find_candidate_collection(
            payload
        )
    )

    matches = [
        candidate
        for candidate in candidates
        if is_candidate_a(
            candidate
        )
    ]

    if len(matches) != 1:
        fail(
            "Expected exactly one Candidate A in PB0-D "
            f"allocation JSON, found {len(matches)}."
        )

    return matches[0]


def find_component_collection(
    candidate: dict[str, Any],
) -> list[dict[str, Any]]:
    possible_keys = (
        "components",
        "component_allocations",
        "allocation",
        "component_targets",
        "parameter_allocation",
    )

    for key in possible_keys:
        value = candidate.get(key)

        if isinstance(
            value,
            list,
        ):
            result = [
                item
                for item in value
                if isinstance(
                    item,
                    dict,
                )
            ]

            if result:
                return result

        if isinstance(
            value,
            dict,
        ):
            result: list[
                dict[str, Any]
            ] = []

            for component_name, component_value in (
                value.items()
            ):
                if isinstance(
                    component_value,
                    dict,
                ):
                    row = dict(
                        component_value
                    )

                    row.setdefault(
                        "component",
                        component_name,
                    )

                    result.append(
                        row
                    )
                elif isinstance(
                    component_value,
                    int,
                ):
                    result.append(
                        {
                            "component": (
                                component_name
                            ),
                            "target_parameters": (
                                component_value
                            ),
                        }
                    )

            if result:
                return result

    fail(
        "Candidate A does not expose a recognizable "
        "component allocation collection."
    )

    return []


def extract_integer(
    mapping: dict[str, Any],
    keys: Iterable[str],
    description: str,
) -> int:
    value = get_first_present(
        mapping,
        keys,
    )

    if isinstance(
        value,
        bool,
    ):
        fail(
            f"{description} must be an integer."
        )

    if not isinstance(
        value,
        int,
    ):
        fail(
            f"{description} is missing or invalid."
        )

    if value < 0:
        fail(
            f"{description} cannot be negative."
        )

    return value

def find_candidate_container(
    value: Any,
    path: str = "$",
) -> Mapping[str, Any] | None:
    if isinstance(
        value,
        Mapping,
    ):
        candidates_value = value.get(
            "candidates"
        )

        selected_candidate_id = value.get(
            "selected_candidate_id"
        )

        if (
            isinstance(
                candidates_value,
                list,
            )
            and isinstance(
                selected_candidate_id,
                str,
            )
        ):
            return value

        for (
            key,
            child_value,
        ) in value.items():
            child_path = (
                f"{path}.{key}"
            )

            result = (
                find_candidate_container(
                    child_value,
                    child_path,
                )
            )

            if result is not None:
                return result

        return None

    if isinstance(
        value,
        list,
    ):
        for (
            index,
            item,
        ) in enumerate(
            value
        ):
            child_path = (
                f"{path}[{index}]"
            )

            result = (
                find_candidate_container(
                    item,
                    child_path,
                )
            )

            if result is not None:
                return result

    return None

def extract_candidate_targets(
    pb0d_data: Mapping[str, Any],
) -> CandidateTargets:
    candidates_value = pb0d_data.get(
        "candidates"
    )

    if not isinstance(
        candidates_value,
        list,
    ):
        fail(
            "PB0-D allocation file does not contain "
            "a valid 'candidates' list."
        )

    selected_candidate_id_value = (
        pb0d_data.get(
            "selected_candidate_id"
        )
    )

    if not isinstance(
        selected_candidate_id_value,
        str,
    ):
        fail(
            "PB0-D allocation file does not contain "
            "a valid 'selected_candidate_id'."
        )

    selected_candidate: Mapping[
        str,
        Any,
    ] | None = None

    for candidate_value in candidates_value:
        if not isinstance(
            candidate_value,
            Mapping,
        ):
            continue

        candidate_id_value = (
            candidate_value.get(
                "candidate_id"
            )
        )

        if (
            candidate_id_value
            == selected_candidate_id_value
        ):
            selected_candidate = (
                candidate_value
            )
            break

    if selected_candidate is None:
        fail(
            "PB0-D allocation file does not contain "
            f"selected candidate "
            f"'{selected_candidate_id_value}'."
        )

    print()
    print(
        "SELECTED CANDIDATE KEYS:"
    )
    print(
        sorted(
            selected_candidate.keys()
        )
    )

    required_integer_fields = {
        "total_parameters": (
            "total parameter count"
        ),
        "matrix_memory_parameters": (
            "matrix memory allocation"
        ),
        "vector_mamba_parameters": (
            "vector / Mamba allocation"
        ),
        "router_parameters": (
            "router allocation"
        ),
        "feedback_bridge_parameters": (
            "feedback bridge allocation"
        ),
        "embeddings_parameters": (
            "embeddings allocation"
        ),
        "output_head_parameters": (
            "output head allocation"
        ),
        "normalization_parameters": (
            "normalization allocation"
        ),
        "auxiliary_parameters": (
            "auxiliary allocation"
        ),
        "other_parameters": (
            "other allocation"
        ),
    }

    extracted_values: dict[
        str,
        int,
    ] = {}

    for field_name, field_description in (
        required_integer_fields.items()
    ):
        value = (
            selected_candidate.get(
                field_name
            )
        )

        if isinstance(
            value,
            bool,
        ):
            fail(
                f"Selected Candidate "
                f"'{selected_candidate_id_value}' has "
                f"an invalid boolean value for "
                f"{field_description}: "
                f"{field_name}."
            )

        if not isinstance(
            value,
            int,
        ):
            fail(
                f"Selected Candidate "
                f"'{selected_candidate_id_value}' is "
                f"missing a valid integer "
                f"{field_description}. "
                f"Field='{field_name}', "
                f"value={value!r}"
            )

        if isinstance(value, int) and value < 0:
            fail(
                f"Selected Candidate "
                f"'{selected_candidate_id_value}' has "
                f"a negative "
                f"{field_description}: "
                f"{field_name}={value}"
            )

        extracted_values[
            field_name
        ] = cast(int, value)

    candidate_name_value = (
        selected_candidate.get(
            "candidate_name"
        )
    )

    if not isinstance(
        candidate_name_value,
        str,
    ) or not candidate_name_value.strip():
        fail(
            f"Selected Candidate "
            f"'{selected_candidate_id_value}' is "
            "missing a valid candidate_name."
        )

    selected_for_next_gate_value = (
        selected_candidate.get(
            "selected_for_next_gate"
        )
    )

    if not isinstance(
        selected_for_next_gate_value,
        bool,
    ):
        fail(
            f"Selected Candidate "
            f"'{selected_candidate_id_value}' is "
            "missing a valid "
            "'selected_for_next_gate' boolean."
        )

    selected_for_next_gate_value = cast(
        bool,
        selected_for_next_gate_value,
    )

    target_component_total = sum(
        (
            extracted_values[
                "matrix_memory_parameters"
            ],
            extracted_values[
                "vector_mamba_parameters"
            ],
            extracted_values[
                "router_parameters"
            ],
            extracted_values[
                "feedback_bridge_parameters"
            ],
            extracted_values[
                "embeddings_parameters"
            ],
            extracted_values[
                "output_head_parameters"
            ],
            extracted_values[
                "normalization_parameters"
            ],
            extracted_values[
                "auxiliary_parameters"
            ],
            extracted_values[
                "other_parameters"
            ],
        )
    )

    declared_total = (
        extracted_values[
            "total_parameters"
        ]
    )

    if (
        target_component_total
        != declared_total
    ):
        fail(
            f"Selected Candidate "
            f"'{selected_candidate_id_value}' component "
            "allocation does not sum to its declared "
            "total parameter count. "
            f"Components={target_component_total:,}, "
            f"declared={declared_total:,}."
        )

    return CandidateTargets(
        candidate_id=(
            cast(
                str,
                selected_candidate_id_value,
            )
        ),
        candidate_name=(
            cast(
                str,
                candidate_name_value,
            )
        ),
        selected_for_next_gate=(
            selected_for_next_gate_value
        ),
        total_parameters=(
            declared_total
        ),
        matrix_memory_parameters=(
            extracted_values[
                "matrix_memory_parameters"
            ]
        ),
        vector_mamba_parameters=(
            extracted_values[
                "vector_mamba_parameters"
            ]
        ),
        router_parameters=(
            extracted_values[
                "router_parameters"
            ]
        ),
        feedback_bridge_parameters=(
            extracted_values[
                "feedback_bridge_parameters"
            ]
        ),
        embeddings_parameters=(
            extracted_values[
                "embeddings_parameters"
            ]
        ),
        output_head_parameters=(
            extracted_values[
                "output_head_parameters"
            ]
        ),
        normalization_parameters=(
            extracted_values[
                "normalization_parameters"
            ]
        ),
        auxiliary_parameters=(
            extracted_values[
                "auxiliary_parameters"
            ]
        ),
        other_parameters=(
            extracted_values[
                "other_parameters"
            ]
        ),
    )

def verify_candidate_a_total(
    targets: dict[str, ComponentTarget],
) -> int:
    total = sum(
        target.target_parameters
        for target in targets.values()
    )

    if total != EXPECTED_TOTAL_PARAMETERS:
        fail(
            "Candidate A target total does not match "
            "the frozen PB0 budget. "
            f"Expected={EXPECTED_TOTAL_PARAMETERS}, "
            f"actual={total}"
        )

    return total


def calculate_frozen_parameter_count(
    targets: dict[str, ComponentTarget],
) -> int:
    frozen_total = sum(
        targets[component].target_parameters
        for component in FROZEN_COMPONENTS
    )

    expected_frozen_total = 2_899_456

    if frozen_total != expected_frozen_total:
        fail(
            "Frozen Candidate A component count does not "
            "match the PB0-D frozen budget. "
            f"Expected={expected_frozen_total}, "
            f"actual={frozen_total}"
        )

    return frozen_total


# =============================================================================
# PARAMETERIZATION MODEL
# =============================================================================
#
# This search intentionally models only the four PB0-D reallocatable component
# capacities. Frozen components remain exactly at their PB0-D counts.
#
# The formulas are based on a layered architecture with:
#
#   - 12 repeated layers
#   - matrix associative memory represented by low-rank matrix projections
#   - vector/Mamba expansion pathway
#   - hidden router MLP
#   - low-rank matrix-to-vector feedback bridge
#
# Exact architectural realization must subsequently be verified by instantiating
# the actual model and recounting its parameter tree.
# =============================================================================


def matrix_memory_parameters(
    matrix_rank: int,
) -> int:
    """
    Parameter model for the repeated matrix-memory capacity.

    Per layer:

        current matrix:
            key projection     hidden_width x rank
            value projection   hidden_width x rank
            query projection   hidden_width x rank

        archive matrix:
            key projection     hidden_width x rank
            value projection   hidden_width x rank
            query projection   hidden_width x rank

        matrix state/output projections:
            rank x state_width for each memory stream

    The architecture is duplicated across current and archive memory.
    """

    if matrix_rank <= 0:
        return -1

    per_stream = (
        3
        * HIDDEN_WIDTH
        * matrix_rank
        + MATRIX_VECTOR_STATE_WIDTH
        * matrix_rank
    )

    two_streams = (
        2
        * per_stream
    )

    return (
        LAYER_COUNT
        * two_streams
    )


def vector_mamba_parameters(
    vector_expansion: int,
) -> int:
    """
    Parameter model for the vector/Mamba pathway.

    Each layer contains:

        input projection:
            hidden_width -> vector_expansion

        recurrent/selective projection:
            vector_expansion -> vector_expansion

        output projection:
            vector_expansion -> hidden_width

        gate projection:
            hidden_width -> vector_expansion

    This intentionally preserves hidden width and layer count while allowing
    vector-pathway capacity to vary.
    """

    if vector_expansion <= 0:
        return -1

    per_layer = (
        HIDDEN_WIDTH
        * vector_expansion
        + vector_expansion
        * vector_expansion
        + vector_expansion
        * HIDDEN_WIDTH
        + HIDDEN_WIDTH
        * vector_expansion
    )

    return (
        LAYER_COUNT
        * per_layer
    )


def router_parameters(
    router_hidden_width: int,
) -> int:
    """
    Router MLP per layer:

        hidden_width -> router_hidden_width
        router_hidden_width -> hidden_width

    Plus scalar routing output and biases.
    """

    if router_hidden_width <= 0:
        return -1

    per_layer = (
        HIDDEN_WIDTH
        * router_hidden_width
        + router_hidden_width
        * HIDDEN_WIDTH
        + router_hidden_width
        + HIDDEN_WIDTH
        + router_hidden_width
        + 1
    )

    return (
        LAYER_COUNT
        * per_layer
    )


def feedback_bridge_parameters(
    feedback_rank: int,
) -> int:
    """
    Low-rank matrix-to-vector feedback bridge.

    Per layer:

        matrix/vector state -> feedback_rank
        feedback_rank -> vector state
        feedback gate projection
    """

    if feedback_rank <= 0:
        return -1

    per_layer = (
        MATRIX_VECTOR_STATE_WIDTH
        * feedback_rank
        + feedback_rank
        * MATRIX_VECTOR_STATE_WIDTH
        + MATRIX_VECTOR_STATE_WIDTH
        * feedback_rank
        + feedback_rank
        + MATRIX_VECTOR_STATE_WIDTH
    )

    return (
        LAYER_COUNT
        * per_layer
    )


def calculate_component_counts(
    dimensions: ArchitectureDimensions,
    frozen_parameter_count: int,
) -> ComponentCounts:
    matrix_parameters = (
        matrix_memory_parameters(
            dimensions.matrix_rank
        )
    )

    vector_parameters = (
        vector_mamba_parameters(
            dimensions.vector_expansion
        )
    )

    router_parameters_count = (
        router_parameters(
            dimensions.router_hidden_width
        )
    )

    feedback_parameters = (
        feedback_bridge_parameters(
            dimensions.feedback_rank
        )
    )

    total_parameters = (
        matrix_parameters
        + vector_parameters
        + router_parameters_count
        + feedback_parameters
        + frozen_parameter_count
    )

    return ComponentCounts(
        matrix_memory=matrix_parameters,
        vector_mamba_pathway=(
            vector_parameters
        ),
        router=(
            router_parameters_count
        ),
        feedback_bridge=(
            feedback_parameters
        ),
        frozen_components=(
            frozen_parameter_count
        ),
        total_parameters=(
            total_parameters
        ),
    )


# =============================================================================
# SEARCH
# =============================================================================


def integer_range(
    start: int,
    stop: int,
    step: int,
) -> range:
    if step <= 0:
        fail(
            "Search step must be positive."
        )

    return range(
        start,
        stop + 1,
        step,
    )


def candidate_score(
    counts: ComponentCounts,
    targets: dict[str, ComponentTarget],
) -> tuple[
    int,
    int,
    int,
    int,
    int,
    int,
]:
    matrix_error = abs(
        counts.matrix_memory
        - targets[
            MATRIX_MEMORY
        ].target_parameters
    )

    vector_error = abs(
        counts.vector_mamba_pathway
        - targets[
            VECTOR_MAMBA_PATHWAY
        ].target_parameters
    )

    router_error = abs(
        counts.router
        - targets[
            ROUTER
        ].target_parameters
    )

    feedback_error = abs(
        counts.feedback_bridge
        - targets[
            FEEDBACK_BRIDGE
        ].target_parameters
    )

    component_absolute_error = (
        matrix_error
        + vector_error
        + router_error
        + feedback_error
    )

    total_budget_error = abs(
        counts.total_parameters
        - EXPECTED_TOTAL_PARAMETERS
    )

    # Budget preservation receives a very large priority multiplier.
    score = (
        total_budget_error
        * 1_000_000
        + component_absolute_error
    )

    return (
        matrix_error,
        vector_error,
        router_error,
        feedback_error,
        component_absolute_error,
        score,
    )


def build_mapping_candidate(
    dimensions: ArchitectureDimensions,
    targets: dict[str, ComponentTarget],
    frozen_parameter_count: int,
) -> MappingCandidate:
    counts = (
        calculate_component_counts(
            dimensions,
            frozen_parameter_count,
        )
    )

    (
        matrix_error,
        vector_error,
        router_error,
        feedback_error,
        component_absolute_error,
        score,
    ) = candidate_score(
        counts,
        targets,
    )

    total_budget_error = abs(
        counts.total_parameters
        - EXPECTED_TOTAL_PARAMETERS
    )

    return MappingCandidate(
        rank=0,
        dimensions=dimensions,
        counts=counts,
        matrix_error=matrix_error,
        vector_error=vector_error,
        router_error=router_error,
        feedback_error=feedback_error,
        total_component_absolute_error=(
            component_absolute_error
        ),
        total_budget_error=(
            total_budget_error
        ),
        score=score,
        exact_budget_match=(
            total_budget_error
            == 0
        ),
    )


def sort_key(
    candidate: MappingCandidate,
) -> tuple[
    int,
    int,
    int,
    int,
    int,
]:
    return (
        candidate.total_budget_error,
        candidate.total_component_absolute_error,
        candidate.matrix_error,
        candidate.vector_error,
        candidate.router_error
        + candidate.feedback_error,
    )


def retain_best_candidate(
    candidates: list[MappingCandidate],
    candidate: MappingCandidate,
) -> None:
    candidates.append(
        candidate
    )

    candidates.sort(
        key=sort_key
    )

    if len(candidates) > (
        TOP_CANDIDATE_LIMIT
    ):
        del candidates[
            TOP_CANDIDATE_LIMIT:
        ]


def search_dimension_mappings(
    targets: dict[str, ComponentTarget],
    frozen_parameter_count: int,
) -> tuple[
    list[MappingCandidate],
    int,
]:
    retained: list[
        MappingCandidate
    ] = []

    searched = 0

    target_matrix = (
        targets[
            MATRIX_MEMORY
        ].target_parameters
    )

    target_vector = (
        targets[
            VECTOR_MAMBA_PATHWAY
        ].target_parameters
    )

    target_router = (
        targets[
            ROUTER
        ].target_parameters
    )

    target_feedback = (
        targets[
            FEEDBACK_BRIDGE
        ].target_parameters
    )

    router_values = list(
        integer_range(
            MIN_ROUTER_HIDDEN_WIDTH,
            MAX_ROUTER_HIDDEN_WIDTH,
            ROUTER_HIDDEN_WIDTH_STEP,
        )
    )

    feedback_values = list(
        integer_range(
            MIN_FEEDBACK_RANK,
            MAX_FEEDBACK_RANK,
            FEEDBACK_RANK_STEP,
        )
    )

    # Precompute each component's valid integer realizations.
    matrix_values = [
        (
            rank,
            matrix_memory_parameters(
                rank
            ),
        )
        for rank in integer_range(
            MIN_MATRIX_RANK,
            MAX_MATRIX_RANK,
            MATRIX_RANK_STEP,
        )
    ]

    vector_values = [
        (
            expansion,
            vector_mamba_parameters(
                expansion
            ),
        )
        for expansion in integer_range(
            MIN_VECTOR_EXPANSION,
            MAX_VECTOR_EXPANSION,
            VECTOR_EXPANSION_STEP,
        )
    ]

    router_values_with_counts = [
        (
            width,
            router_parameters(
                width
            ),
        )
        for width in router_values
    ]

    feedback_values_with_counts = [
        (
            rank,
            feedback_bridge_parameters(
                rank
            ),
        )
        for rank in feedback_values
    ]

    # Keep only locally plausible values. This dramatically reduces the
    # Cartesian search while preserving a generous search neighborhood.
    matrix_window = max(
        2_500_000,
        int(
            target_matrix
            * 0.20
        ),
    )

    vector_window = max(
        2_500_000,
        int(
            target_vector
            * 0.25
        ),
    )

    router_window = max(
        500_000,
        int(
            target_router
            * 4.00
        ),
    )

    feedback_window = max(
        250_000,
        int(
            target_feedback
            * 20.00
        ),
    )

    matrix_values = [
        value
        for value in matrix_values
        if abs(
            value[1]
            - target_matrix
        )
        <= matrix_window
    ]

    vector_values = [
        value
        for value in vector_values
        if abs(
            value[1]
            - target_vector
        )
        <= vector_window
    ]

    router_values_with_counts = [
        value
        for value in (
            router_values_with_counts
        )
        if abs(
            value[1]
            - target_router
        )
        <= router_window
    ]

    feedback_values_with_counts = [
        value
        for value in (
            feedback_values_with_counts
        )
        if abs(
            value[1]
            - target_feedback
        )
        <= feedback_window
    ]

    if not matrix_values:
        fail(
            "No matrix dimension values fall inside "
            "the Candidate A search window."
        )

    if not vector_values:
        fail(
            "No vector dimension values fall inside "
            "the Candidate A search window."
        )

    if not router_values_with_counts:
        fail(
            "No router dimension values fall inside "
            "the Candidate A search window."
        )

    if not feedback_values_with_counts:
        fail(
            "No feedback dimension values fall inside "
            "the Candidate A search window."
        )

    # Build router + feedback combinations once.
    router_feedback_pairs: list[
        tuple[
            int,
            int,
            int,
        ]
    ] = []

    for (
        router_width,
        router_count,
    ) in router_values_with_counts:
        for (
            feedback_rank,
            feedback_count,
        ) in (
            feedback_values_with_counts
        ):
            combined = (
                router_count
                + feedback_count
            )

            router_feedback_pairs.append(
                (
                    router_width,
                    feedback_rank,
                    combined,
                )
            )

    # Search matrix and vector combinations and use the total budget to select
    # plausible router/feedback pairs.
    router_feedback_pairs.sort(
        key=lambda item: abs(
            item[2]
            - (
                target_router
                + target_feedback
            )
        )
    )

    for (
        matrix_rank,
        matrix_count,
    ) in matrix_values:
        for (
            vector_expansion,
            vector_count,
        ) in vector_values:
            required_router_feedback = (
                EXPECTED_TOTAL_PARAMETERS
                - frozen_parameter_count
                - matrix_count
                - vector_count
            )

            nearest_pairs = sorted(
                router_feedback_pairs,
                key=lambda item: abs(
                    item[2]
                    - required_router_feedback
                ),
            )[
                :20
            ]

            for (
                router_width,
                feedback_rank,
                _combined_count,
            ) in nearest_pairs:
                dimensions = (
                    ArchitectureDimensions(
                        matrix_rank=(
                            matrix_rank
                        ),
                        vector_expansion=(
                            vector_expansion
                        ),
                        router_hidden_width=(
                            router_width
                        ),
                        feedback_rank=(
                            feedback_rank
                        ),
                    )
                )

                candidate = (
                    build_mapping_candidate(
                        dimensions,
                        targets,
                        frozen_parameter_count,
                    )
                )

                searched += 1

                retain_best_candidate(
                    retained,
                    candidate,
                )

    retained.sort(
        key=sort_key
    )

    ranked: list[
        MappingCandidate
    ] = []

    for index, candidate in enumerate(
        retained,
        start=1,
    ):
        ranked.append(
            MappingCandidate(
                rank=index,
                dimensions=(
                    candidate.dimensions
                ),
                counts=(
                    candidate.counts
                ),
                matrix_error=(
                    candidate.matrix_error
                ),
                vector_error=(
                    candidate.vector_error
                ),
                router_error=(
                    candidate.router_error
                ),
                feedback_error=(
                    candidate.feedback_error
                ),
                total_component_absolute_error=(
                    candidate.total_component_absolute_error
                ),
                total_budget_error=(
                    candidate.total_budget_error
                ),
                score=(
                    candidate.score
                ),
                exact_budget_match=(
                    candidate.exact_budget_match
                ),
            )
        )

    return (
        ranked,
        searched,
    )


# =============================================================================
# RESULT VALIDATION
# =============================================================================


def target_component_match(
    candidate: MappingCandidate,
    targets: dict[str, ComponentTarget],
) -> bool:
    return (
        candidate.counts.matrix_memory
        == targets[
            MATRIX_MEMORY
        ].target_parameters
        and candidate.counts.vector_mamba_pathway
        == targets[
            VECTOR_MAMBA_PATHWAY
        ].target_parameters
        and candidate.counts.router
        == targets[
            ROUTER
        ].target_parameters
        and candidate.counts.feedback_bridge
        == targets[
            FEEDBACK_BRIDGE
        ].target_parameters
    )


def validate_selected_candidate(
    candidate: MappingCandidate,
    targets: dict[str, ComponentTarget],
    frozen_parameter_count: int,
) -> None:
    if (
        candidate.counts.frozen_components
        != frozen_parameter_count
    ):
        fail(
            "Selected mapping changed frozen component "
            "parameters."
        )

    recalculated_total = (
        candidate.counts.matrix_memory
        + candidate.counts.vector_mamba_pathway
        + candidate.counts.router
        + candidate.counts.feedback_bridge
        + candidate.counts.frozen_components
    )

    if (
        recalculated_total
        != candidate.counts.total_parameters
    ):
        fail(
            "Selected mapping total parameter arithmetic "
            "is inconsistent."
        )

    for component in (
        *REALLOCATABLE_COMPONENTS,
    ):
        target = (
            targets[
                component
            ].target_parameters
        )

        if target < 0:
            fail(
                f"Invalid target for component "
                f"'{component}'."
            )


# =============================================================================
# OUTPUTS
# =============================================================================


def write_json(
    result: MappingResult,
    candidates: list[MappingCandidate],
) -> None:
    payload = {
        "experiment": (
            "PB0-Candidate-A-Dimension-Mapping"
        ),
        "purpose": (
            "Map PB0-D Candidate A parameter targets "
            "to integer architecture dimensions without "
            "training or reading dataset splits."
        ),
        "dataset_access": {
            "training": "not_read",
            "validation": "not_read",
            "test": "not_read",
        },
        "frozen_architecture": {
            "vocabulary_size": (
                VOCABULARY_SIZE
            ),
            "layer_count": (
                LAYER_COUNT
            ),
            "embedding_width": (
                EMBEDDING_WIDTH
            ),
            "hidden_width": (
                HIDDEN_WIDTH
            ),
            "matrix_vector_state_width": (
                MATRIX_VECTOR_STATE_WIDTH
            ),
        },
        "result": asdict(
            result
        ),
        "top_candidates": [
            asdict(
                candidate
            )
            for candidate in candidates
        ],
    }

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
        )


def write_candidates_csv(
    candidates: list[MappingCandidate],
) -> None:
    rows: list[
        dict[str, Any]
    ] = []

    for candidate in candidates:
        rows.append(
            {
                "rank": candidate.rank,
                "matrix_rank": (
                    candidate.dimensions.matrix_rank
                ),
                "vector_expansion": (
                    candidate.dimensions.vector_expansion
                ),
                "router_hidden_width": (
                    candidate.dimensions.router_hidden_width
                ),
                "feedback_rank": (
                    candidate.dimensions.feedback_rank
                ),
                "matrix_memory_parameters": (
                    candidate.counts.matrix_memory
                ),
                "vector_mamba_parameters": (
                    candidate.counts.vector_mamba_pathway
                ),
                "router_parameters": (
                    candidate.counts.router
                ),
                "feedback_parameters": (
                    candidate.counts.feedback_bridge
                ),
                "frozen_parameters": (
                    candidate.counts.frozen_components
                ),
                "total_parameters": (
                    candidate.counts.total_parameters
                ),
                "matrix_error": (
                    candidate.matrix_error
                ),
                "vector_error": (
                    candidate.vector_error
                ),
                "router_error": (
                    candidate.router_error
                ),
                "feedback_error": (
                    candidate.feedback_error
                ),
                "total_component_absolute_error": (
                    candidate.total_component_absolute_error
                ),
                "total_budget_error": (
                    candidate.total_budget_error
                ),
                "exact_budget_match": (
                    candidate.exact_budget_match
                ),
            }
        )

    if not rows:
        fail(
            "No mapping candidates available for CSV."
        )

    with OUTPUT_CANDIDATES_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                row
            )


def write_markdown(
    result: MappingResult,
    targets: dict[str, ComponentTarget],
    candidates: list[MappingCandidate],
) -> None:
    selected = (
        result.selected_mapping
    )

    lines: list[str] = []

    lines.append(
        "# PB0 Candidate A Dimension Mapping"
    )
    lines.append("")
    lines.append(
        "## Purpose"
    )
    lines.append("")
    lines.append(
        "Map the PB0-D Candidate A fixed-budget allocation "
        "to integer architecture dimensions without training "
        "a model or reading training, validation, or test data."
    )
    lines.append("")
    lines.append(
        "## Frozen Architecture"
    )
    lines.append("")
    lines.append(
        "| Field | Value |"
    )
    lines.append(
        "|---|---:|"
    )

    frozen_rows = (
        (
            "Vocabulary size",
            VOCABULARY_SIZE,
        ),
        (
            "Layer count",
            LAYER_COUNT,
        ),
        (
            "Embedding width",
            EMBEDDING_WIDTH,
        ),
        (
            "Hidden width",
            HIDDEN_WIDTH,
        ),
        (
            "Matrix/vector state width",
            MATRIX_VECTOR_STATE_WIDTH,
        ),
    )

    for name, value in frozen_rows:
        lines.append(
            f"| {name} | {value:,} |"
        )

    lines.append("")
    lines.append(
        "## Candidate A Parameter Targets"
    )
    lines.append("")
    lines.append(
        "| Component | Baseline | Target | Delta |"
    )
    lines.append(
        "|---|---:|---:|---:|"
    )

    for component in (
        *REALLOCATABLE_COMPONENTS,
        *FROZEN_COMPONENTS,
    ):
        target = (
            targets[
                component
            ]
        )

        lines.append(
            "| "
            f"{target.component} | "
            f"{target.baseline_parameters:,} | "
            f"{target.target_parameters:,} | "
            f"{signed_int(target.delta_parameters)} |"
        )

    lines.append("")
    lines.append(
        "## Selected Integer Mapping"
    )
    lines.append("")
    lines.append(
        "| Dimension | Value |"
    )
    lines.append(
        "|---|---:|"
    )

    dimensions = (
        selected.dimensions
    )

    lines.append(
        f"| Matrix rank | {dimensions.matrix_rank:,} |"
    )
    lines.append(
        "| Vector expansion | "
        f"{dimensions.vector_expansion:,} |"
    )
    lines.append(
        "| Router hidden width | "
        f"{dimensions.router_hidden_width:,} |"
    )
    lines.append(
        f"| Feedback rank | {dimensions.feedback_rank:,} |"
    )

    lines.append("")
    lines.append(
        "## Selected Parameter Counts"
    )
    lines.append("")
    lines.append(
        "| Component | Target | Realized | Error |"
    )
    lines.append(
        "|---|---:|---:|---:|"
    )

    selected_component_rows = (
        (
            MATRIX_MEMORY,
            selected.counts.matrix_memory,
            selected.matrix_error,
        ),
        (
            VECTOR_MAMBA_PATHWAY,
            selected.counts.vector_mamba_pathway,
            selected.vector_error,
        ),
        (
            ROUTER,
            selected.counts.router,
            selected.router_error,
        ),
        (
            FEEDBACK_BRIDGE,
            selected.counts.feedback_bridge,
            selected.feedback_error,
        ),
    )

    for (
        component,
        realized,
        error,
    ) in selected_component_rows:
        target = (
            targets[
                component
            ].target_parameters
        )

        lines.append(
            "| "
            f"{component} | "
            f"{target:,} | "
            f"{realized:,} | "
            f"{error:,} |"
        )

    lines.append(
        "| Frozen components | "
        f"{result.frozen_parameter_count:,} | "
        f"{selected.counts.frozen_components:,} | "
        f"{abs(result.frozen_parameter_count - selected.counts.frozen_components):,} |"
    )

    lines.append(
        "| **TOTAL** | "
        f"**{EXPECTED_TOTAL_PARAMETERS:,}** | "
        f"**{selected.counts.total_parameters:,}** | "
        f"**{selected.total_budget_error:,}** |"
    )

    lines.append("")
    lines.append(
        "## Mapping Status"
    )
    lines.append("")
    lines.append(
        f"- Exact budget match: **{result.exact_budget_match}**"
    )
    lines.append(
        "- Exact component-target match: "
        f"**{result.target_component_match}**"
    )
    lines.append(
        "- Candidate mappings searched: "
        f"**{result.searched_candidate_count:,}**"
    )
    lines.append(
        "- Retained mappings: "
        f"**{result.retained_candidate_count:,}**"
    )

    lines.append("")
    lines.append(
        "## Important Gate"
    )
    lines.append("")
    lines.append(
        "This script is a mathematical dimension-mapping audit. "
        "Before Candidate A may train, the selected dimensions must "
        "be mapped into the actual model.py implementation and the "
        "instantiated parameter tree must be recounted exactly."
    )

    lines.append("")
    lines.append(
        "## Dataset Guardrail"
    )
    lines.append("")
    lines.append(
        "- Training split: not read."
    )
    lines.append(
        "- Validation split: not read."
    )
    lines.append(
        "- Test split: not read."
    )

    lines.append("")
    lines.append(
        "## Top Candidate Mappings"
    )
    lines.append("")
    lines.append(
        "| Rank | Matrix Rank | Vector Expansion | "
        "Router Width | Feedback Rank | Total Error | "
        "Budget Error |"
    )
    lines.append(
        "|---:|---:|---:|---:|---:|---:|---:|"
    )

    for candidate in candidates:
        lines.append(
            "| "
            f"{candidate.rank} | "
            f"{candidate.dimensions.matrix_rank} | "
            f"{candidate.dimensions.vector_expansion} | "
            f"{candidate.dimensions.router_hidden_width} | "
            f"{candidate.dimensions.feedback_rank} | "
            f"{candidate.total_component_absolute_error:,} | "
            f"{candidate.total_budget_error:,} |"
        )

    with OUTPUT_MARKDOWN.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "\n".join(
                lines
            )
        )
        file.write(
            "\n"
        )


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    banner(
        "MODUS_X PB0 CANDIDATE A DIMENSION MAPPING"
    )

    print(
        f"Project root : {PROJECT_ROOT}"
    )
    print(
        f"PB0-D input  : {PB0_D_ALLOCATION_JSON}"
    )
    print(
        f"Output dir   : {OUTPUT_DIR}"
    )

    ensure_output_directory()

    banner(
        "LOADING PB0-D CANDIDATE A"
    )

    allocation_payload = load_json(
        PB0_D_ALLOCATION_JSON
    )

    candidate_targets = (
        extract_candidate_targets(
            allocation_payload
        )
    )

    selected_candidate_id = (
        candidate_targets.candidate_id
    )

    if not candidate_targets.selected_for_next_gate:
        fail(
            f"Candidate '{selected_candidate_id}' is not marked "
            f"selected_for_next_gate in the PB0-D allocation file."
        )

    print()
    print("PB0-D ROOT KEYS:")
    print(
        sorted(
            str(key)
            for key in allocation_payload.keys()
        )
    )

    print()
    print("PB0-D ROOT KEYS:")
    print(
        sorted(
            str(key)
            for key in allocation_payload.keys()
        )
    )

    candidate_a = (
        extract_candidate_a(
            allocation_payload
        )
    )

    candidate_identifier = (
        extract_candidate_identifier(
            candidate_a
        )
    )

    print(
        f"Selected PB0-D candidate : "
        f"{candidate_identifier}"
    )

    targets: dict[str, ComponentTarget] = {
        component: ComponentTarget(
            component=component,
            baseline_parameters=target,
            target_parameters=target,
            delta_parameters=0,
        )
        for component, target in {
            MATRIX_MEMORY: candidate_targets.matrix_memory_parameters,
            VECTOR_MAMBA_PATHWAY: candidate_targets.vector_mamba_parameters,
            ROUTER: candidate_targets.router_parameters,
            FEEDBACK_BRIDGE: candidate_targets.feedback_bridge_parameters,
            "embeddings": candidate_targets.embeddings_parameters,
            "output_head": candidate_targets.output_head_parameters,
            "normalization": candidate_targets.normalization_parameters,
            "auxiliary": candidate_targets.auxiliary_parameters,
            "other": candidate_targets.other_parameters,
        }.items()
    }

    candidate_total = (
        verify_candidate_a_total(
            targets
        )
    )

    frozen_parameter_count = (
        calculate_frozen_parameter_count(
            targets
        )
    )

    print(
        f"Candidate A total         : "
        f"{format_int(candidate_total)}"
    )

    print(
        f"Frozen parameter count    : "
        f"{format_int(frozen_parameter_count)}"
    )

    banner(
        "CANDIDATE A TARGETS"
    )

    for component in (
        *REALLOCATABLE_COMPONENTS,
        *FROZEN_COMPONENTS,
    ):
        target = (
            targets[
                component
            ]
        )

        print(
            f"{component:24s} "
            f"baseline="
            f"{format_int(target.baseline_parameters):>14s} "
            f"target="
            f"{format_int(target.target_parameters):>14s} "
            f"delta="
            f"{signed_int(target.delta_parameters):>14s}"
        )

    banner(
        "SEARCHING INTEGER DIMENSION MAPPINGS"
    )

    candidates, searched_count = (
        search_dimension_mappings(
            targets,
            frozen_parameter_count,
        )
    )

    if not candidates:
        fail(
            "No feasible Candidate A dimension mappings "
            "were generated."
        )

    selected = candidates[0]

    validate_selected_candidate(
        selected,
        targets,
        frozen_parameter_count,
    )

    result = MappingResult(
        expected_total_parameters=(
            EXPECTED_TOTAL_PARAMETERS
        ),
        frozen_parameter_count=(
            frozen_parameter_count
        ),
        candidate_a_targets=tuple(
            targets[
                component
            ]
            for component in (
                *REALLOCATABLE_COMPONENTS,
                *FROZEN_COMPONENTS,
            )
        ),
        selected_dimensions=(
            selected.dimensions
        ),
        selected_counts=(
            selected.counts
        ),
        selected_mapping=(
            selected
        ),
        exact_budget_match=(
            selected.exact_budget_match
        ),
        target_component_match=(
            target_component_match(
                selected,
                targets,
            )
        ),
        searched_candidate_count=(
            searched_count
        ),
        retained_candidate_count=(
            len(candidates)
        ),
    )

    banner(
        "SELECTED CANDIDATE A DIMENSION MAPPING"
    )

    print()
    print(
        "Dimensions:"
    )
    print(
        f"  Matrix rank        : "
        f"{selected.dimensions.matrix_rank}"
    )
    print(
        f"  Vector expansion  : "
        f"{selected.dimensions.vector_expansion}"
    )
    print(
        f"  Router width       : "
        f"{selected.dimensions.router_hidden_width}"
    )
    print(
        f"  Feedback rank      : "
        f"{selected.dimensions.feedback_rank}"
    )

    print()
    print(
        "Parameter realization:"
    )

    print(
        f"  Matrix memory      : "
        f"{format_int(selected.counts.matrix_memory)} "
        f"(error {format_int(selected.matrix_error)})"
    )

    print(
        f"  Vector / Mamba     : "
        f"{format_int(selected.counts.vector_mamba_pathway)} "
        f"(error {format_int(selected.vector_error)})"
    )

    print(
        f"  Router             : "
        f"{format_int(selected.counts.router)} "
        f"(error {format_int(selected.router_error)})"
    )

    print(
        f"  Feedback bridge    : "
        f"{format_int(selected.counts.feedback_bridge)} "
        f"(error {format_int(selected.feedback_error)})"
    )

    print(
        f"  Frozen components  : "
        f"{format_int(selected.counts.frozen_components)}"
    )

    print(
        f"  TOTAL              : "
        f"{format_int(selected.counts.total_parameters)}"
    )

    print()
    print(
        f"Exact budget match   : "
        f"{selected.exact_budget_match}"
    )

    print(
        f"Exact component match: "
        f"{result.target_component_match}"
    )

    banner(
        "WRITING PB0 CANDIDATE A MAPPING OUTPUTS"
    )

    write_json(
        result,
        candidates,
    )

    write_markdown(
        result,
        targets,
        candidates,
    )

    write_candidates_csv(
        candidates
    )

    banner(
        "PB0 CANDIDATE A DIMENSION MAPPING COMPLETE"
    )

    print(
        f"Candidate mappings searched : "
        f"{format_int(searched_count)}"
    )

    print(
        f"Best budget error           : "
        f"{format_int(selected.total_budget_error)}"
    )

    print(
        f"Best component error        : "
        f"{format_int(selected.total_component_absolute_error)}"
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
        f"  {OUTPUT_CANDIDATES_CSV}"
    )

    print()
    print(
        "NEXT GATE:"
    )
    print(
        "Map the selected integer dimensions into the actual "
        "model.py configuration and instantiate the architecture."
    )
    print(
        "Then perform an exact parameter-tree recount."
    )
    print(
        "Do not train Candidate A until that recount is verified."
    )
    print(
        "No training, validation, or test dataset bytes were read."
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print(
            "PB0 Candidate A dimension mapping interrupted."
        )
        sys.exit(
            130
        )
    except Exception as error:
        banner(
            "PB0 CANDIDATE A DIMENSION MAPPING FAILED"
        )
        print(
            f"{type(error).__name__}: {error}"
        )
        raise