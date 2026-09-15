from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

SCRIPT_PATH = Path(__file__).resolve()

PARAMETER_ALLOCATION_DIR = SCRIPT_PATH.parent
RESEARCH_DIR = PARAMETER_ALLOCATION_DIR.parent
PROJECT_ROOT = RESEARCH_DIR.parent

OUTPUT_DIR = PARAMETER_ALLOCATION_DIR / "outputs"

PB0_D_INPUT_PATH = (
    OUTPUT_DIR
    / "PB0_FIXED_BUDGET_ALLOCATION.json"
)

PB0_CANDIDATE_A_MAPPING_PATH = (
    OUTPUT_DIR
    / "PB0_CANDIDATE_A_DIMENSION_MAPPING.json"
)

OUTPUT_JSON_PATH = (
    OUTPUT_DIR
    / "PB0_E_ARCHITECTURE_MAPPING.json"
)

OUTPUT_MARKDOWN_PATH = (
    OUTPUT_DIR
    / "PB0_E_ARCHITECTURE_MAPPING.md"
)

OUTPUT_CSV_PATH = (
    OUTPUT_DIR
    / "PB0_E_ARCHITECTURE_PARAMETER_PLAN.csv"
)


# =============================================================================
# CANONICAL ARCHITECTURE CONSTANTS
# =============================================================================

CANONICAL_EMBED_DIM = 512
CANONICAL_N_LAYERS = 12
CANONICAL_VOCAB_SIZE = 256

CANONICAL_FEEDBACK_GATE_OUTPUTS = 1

CANONICAL_MATRIX_ARCHIVE_GATE_OUTPUTS = 1

CANONICAL_VECTOR_ROUTER = True


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class ArchitectureDimensions:
    """
    Concrete dimensions for Candidate A.

    d:
        Main hidden / embedding width.

    r:
        Matrix-memory rank and matrix state width.

    n:
        Vector recurrent expansion / state width.

    router_hidden:
        Router hidden dimension.

    feedback_rank:
        Bottleneck width used by the matrix-to-vector feedback bridge.

    n_layers:
        Number of stacked recurrent layers.

    vocab_size:
        Byte vocabulary size.
    """

    d: int
    r: int
    n: int
    router_hidden: int
    feedback_rank: int
    n_layers: int
    vocab_size: int


@dataclass(frozen=True)
class ParameterPlanEntry:
    """
    One concrete parameter tensor family in the Candidate A architecture.
    """

    component: str
    parameter_path: str
    per_layer_parameters: int
    total_parameters: int
    formula: str
    shape_description: str


# =============================================================================
# ERROR HANDLING
# =============================================================================

def fail(message: str, exit_code: int = 1) -> None:
    print()
    print("=" * 80)
    print("PB0-E ARCHITECTURE MAPPING FAILED")
    print("=" * 80)
    print()
    print(message)
    print()

    raise SystemExit(exit_code)


def require_positive_integer(
    value: Any,
    field_name: str,
) -> int:
    """
    Validate that a value is a strictly positive integer.
    """

    if isinstance(value, bool):
        raise RuntimeError(
            f"Field '{field_name}' must be a positive integer, "
            "but received a boolean."
        )

    if isinstance(value, int):
        parsed_value = value

    elif isinstance(value, float):
        if not value.is_integer():
            raise RuntimeError(
                f"Field '{field_name}' must be an integer, "
                f"but received {value!r}."
            )

        parsed_value = int(value)

    elif isinstance(value, str):
        cleaned = value.replace(",", "").strip()

        if not cleaned.isdigit():
            raise RuntimeError(
                f"Field '{field_name}' must contain an integer, "
                f"but received {value!r}."
            )

        parsed_value = int(cleaned)

    else:
        raise RuntimeError(
            f"Field '{field_name}' must be an integer-compatible value, "
            f"but received type {type(value).__name__}."
        )

    if parsed_value <= 0:
        raise RuntimeError(
            f"Field '{field_name}' must be greater than zero, "
            f"but received {parsed_value}."
        )

    return parsed_value


# =============================================================================
# JSON LOADING
# =============================================================================

def load_json(
    path: Path,
    description: str,
) -> dict[str, Any]:
    """
    Load and validate a JSON object.
    """

    if not path.exists():
        fail(
            f"Required {description} file does not exist:\n"
            f"{path}"
        )

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

    except json.JSONDecodeError as exc:
        fail(
            f"Could not parse {description} JSON:\n"
            f"{path}\n\n"
            f"JSON error: {exc}"
        )

    except OSError as exc:
        fail(
            f"Could not read {description} file:\n"
            f"{path}\n\n"
            f"Operating system error: {exc}"
        )

    if not isinstance(data, dict):
        fail(
            f"{description} must contain a JSON object at the root.\n"
            f"Found: {type(data).__name__}"
        )

    return data


# =============================================================================
# PB0-D CANDIDATE RESOLUTION
# =============================================================================

def extract_selected_candidate_id(
    pb0_d_data: dict[str, Any],
) -> str:
    """
    Extract the selected candidate identifier from PB0-D.
    """

    value = pb0_d_data.get(
        "selected_candidate_id"
    )

    if not isinstance(value, str):
        raise RuntimeError(
            "PB0-D does not contain a valid "
            "'selected_candidate_id' string."
        )

    candidate_id = value.strip()

    if not candidate_id:
        raise RuntimeError(
            "PB0-D 'selected_candidate_id' is empty."
        )

    return candidate_id


def extract_candidate(
    pb0_d_data: dict[str, Any],
    candidate_id: str,
) -> dict[str, Any]:
    """
    Locate the selected candidate in PB0-D.
    """

    candidates = pb0_d_data.get(
        "candidates"
    )

    if not isinstance(candidates, list):
        raise RuntimeError(
            "PB0-D field 'candidates' must be a list."
        )

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue

        current_id = candidate.get(
            "candidate_id"
        )

        if current_id == candidate_id:
            return candidate

    raise RuntimeError(
        f"Could not find PB0-D candidate "
        f"'{candidate_id}'."
    )


# =============================================================================
# DIMENSION EXTRACTION
# =============================================================================

def extract_dimension_mapping(
    mapping_data: dict[str, Any],
) -> ArchitectureDimensions:
    """
    Extract the selected integer architecture dimensions.

    The function accepts the currently observed PB0 Candidate A mapping
    structure and searches recursively for the dimension values required
    to instantiate the candidate.
    """

    dimension_sources: list[dict[str, Any]] = []

    def visit(
        value: Any,
    ) -> None:
        if isinstance(value, dict):
            dimension_sources.append(value)

            for child_value in value.values():
                visit(child_value)

        elif isinstance(value, list):
            for child_value in value:
                visit(child_value)

    visit(mapping_data)

    matrix_rank: Any = None
    vector_expansion: Any = None
    router_width: Any = None
    feedback_rank: Any = None

    matrix_keys = (
        "matrix_rank",
        "matrix_memory_rank",
        "selected_matrix_rank",
    )

    vector_keys = (
        "vector_expansion",
        "vector_width",
        "vector_state_width",
        "selected_vector_expansion",
    )

    router_keys = (
        "router_width",
        "router_hidden",
        "router_hidden_width",
        "selected_router_width",
    )

    feedback_keys = (
        "feedback_rank",
        "feedback_width",
        "feedback_bottleneck",
        "selected_feedback_rank",
    )

    for source in dimension_sources:
        if matrix_rank is None:
            for key in matrix_keys:
                if key in source:
                    matrix_rank = source[key]
                    break

        if vector_expansion is None:
            for key in vector_keys:
                if key in source:
                    vector_expansion = source[key]
                    break

        if router_width is None:
            for key in router_keys:
                if key in source:
                    router_width = source[key]
                    break

        if feedback_rank is None:
            for key in feedback_keys:
                if key in source:
                    feedback_rank = source[key]
                    break

    if matrix_rank is None:
        raise RuntimeError(
            "Could not locate 'matrix_rank' in the "
            "PB0 Candidate A dimension mapping JSON."
        )

    if vector_expansion is None:
        raise RuntimeError(
            "Could not locate 'vector_expansion' in the "
            "PB0 Candidate A dimension mapping JSON."
        )

    if router_width is None:
        raise RuntimeError(
            "Could not locate 'router_width' in the "
            "PB0 Candidate A dimension mapping JSON."
        )

    if feedback_rank is None:
        raise RuntimeError(
            "Could not locate 'feedback_rank' in the "
            "PB0 Candidate A dimension mapping JSON."
        )

    r = require_positive_integer(
        matrix_rank,
        "matrix_rank",
    )

    n = require_positive_integer(
        vector_expansion,
        "vector_expansion",
    )

    router_hidden = require_positive_integer(
        router_width,
        "router_width",
    )

    feedback = require_positive_integer(
        feedback_rank,
        "feedback_rank",
    )

    return ArchitectureDimensions(
        d=CANONICAL_EMBED_DIM,
        r=r,
        n=n,
        router_hidden=router_hidden,
        feedback_rank=feedback,
        n_layers=CANONICAL_N_LAYERS,
        vocab_size=CANONICAL_VOCAB_SIZE,
    )


# =============================================================================
# PARAMETER ACCOUNTING
# =============================================================================

def build_parameter_plan(
    dimensions: ArchitectureDimensions,
) -> list[ParameterPlanEntry]:
    """
    Build an explicit parameter accounting plan matching the canonical
    parameterization used by the current Modus_X MemoryFeedbackArchive
    architecture.

    This does not instantiate JAX arrays and does not modify models.py.
    It only translates concrete dimensions into exact tensor formulas.
    """

    d = dimensions.d
    r = dimensions.r
    n = dimensions.n
    h = dimensions.router_hidden
    f = dimensions.feedback_rank
    layers = dimensions.n_layers
    vocab = dimensions.vocab_size

    plan: list[ParameterPlanEntry] = []

    def add_layer_parameter(
        component: str,
        parameter_path: str,
        per_layer_parameters: int,
        formula: str,
        shape_description: str,
    ) -> None:
        plan.append(
            ParameterPlanEntry(
                component=component,
                parameter_path=parameter_path,
                per_layer_parameters=per_layer_parameters,
                total_parameters=(
                    per_layer_parameters * layers
                ),
                formula=formula,
                shape_description=shape_description,
            )
        )

    # -------------------------------------------------------------------------
    # MATRIX MEMORY
    # -------------------------------------------------------------------------

    add_layer_parameter(
        component="matrix_memory",
        parameter_path="layers.m_wk",
        per_layer_parameters=r * d,
        formula="r * d",
        shape_description=f"[{r}, {d}]",
    )

    add_layer_parameter(
        component="matrix_memory",
        parameter_path="layers.m_wq",
        per_layer_parameters=r * d,
        formula="r * d",
        shape_description=f"[{r}, {d}]",
    )

    add_layer_parameter(
        component="matrix_memory",
        parameter_path="layers.m_wv",
        per_layer_parameters=r * d,
        formula="r * d",
        shape_description=f"[{r}, {d}]",
    )

    add_layer_parameter(
        component="matrix_memory",
        parameter_path="layers.m_w_read",
        per_layer_parameters=r * d,
        formula="r * d",
        shape_description=f"[{r}, {d}]",
    )

    add_layer_parameter(
        component="matrix_memory",
        parameter_path="layers.m_w_out",
        per_layer_parameters=d * d,
        formula="d * d",
        shape_description=f"[{d}, {d}]",
    )

    add_layer_parameter(
        component="matrix_memory",
        parameter_path="layers.m_proj_w",
        per_layer_parameters=d * (d + r),
        formula="d * (d + r)",
        shape_description=f"[{d}, {d + r}]",
    )

    add_layer_parameter(
        component="matrix_memory",
        parameter_path="layers.m_b_read",
        per_layer_parameters=r,
        formula="r",
        shape_description=f"[{r}]",
    )

    add_layer_parameter(
        component="matrix_memory",
        parameter_path="layers.m_b_out",
        per_layer_parameters=d,
        formula="d",
        shape_description=f"[{d}]",
    )

    add_layer_parameter(
        component="matrix_memory",
        parameter_path="layers.m_proj_b",
        per_layer_parameters=d,
        formula="d",
        shape_description=f"[{d}]",
    )

    add_layer_parameter(
        component="matrix_memory",
        parameter_path="layers.m_ln_g",
        per_layer_parameters=r,
        formula="r",
        shape_description=f"[{r}]",
    )

    add_layer_parameter(
        component="matrix_memory",
        parameter_path="layers.m_ln_b",
        per_layer_parameters=r,
        formula="r",
        shape_description=f"[{r}]",
    )

    add_layer_parameter(
        component="matrix_memory",
        parameter_path="layers.m_w_eta",
        per_layer_parameters=d,
        formula="1 * d",
        shape_description=f"[1, {d}]",
    )

    add_layer_parameter(
        component="matrix_memory",
        parameter_path="layers.m_b_eta",
        per_layer_parameters=1,
        formula="1",
        shape_description="[1]",
    )

    add_layer_parameter(
        component="matrix_memory",
        parameter_path="layers.m_w_write",
        per_layer_parameters=d,
        formula="1 * d",
        shape_description=f"[1, {d}]",
    )

    add_layer_parameter(
        component="matrix_memory",
        parameter_path="layers.m_b_write",
        per_layer_parameters=1,
        formula="1",
        shape_description="[1]",
    )

    add_layer_parameter(
        component="matrix_memory",
        parameter_path="layers.m_w_ret",
        per_layer_parameters=d,
        formula="1 * d",
        shape_description=f"[1, {d}]",
    )

    add_layer_parameter(
        component="matrix_memory",
        parameter_path="layers.m_b_ret",
        per_layer_parameters=1,
        formula="1",
        shape_description="[1]",
    )

    add_layer_parameter(
        component="archive_memory_control",
        parameter_path="layers.m_w_archive_write",
        per_layer_parameters=d,
        formula="1 * d",
        shape_description=f"[1, {d}]",
    )

    add_layer_parameter(
        component="archive_memory_control",
        parameter_path="layers.m_b_archive_write",
        per_layer_parameters=1,
        formula="1",
        shape_description="[1]",
    )

    add_layer_parameter(
        component="archive_memory_control",
        parameter_path="layers.m_w_archive_ret",
        per_layer_parameters=d,
        formula="1 * d",
        shape_description=f"[1, {d}]",
    )

    add_layer_parameter(
        component="archive_memory_control",
        parameter_path="layers.m_b_archive_ret",
        per_layer_parameters=1,
        formula="1",
        shape_description="[1]",
    )

    add_layer_parameter(
        component="archive_memory_control",
        parameter_path="layers.m_w_archive_mix",
        per_layer_parameters=r * d,
        formula="r * d",
        shape_description=f"[{r}, {d}]",
    )

    add_layer_parameter(
        component="archive_memory_control",
        parameter_path="layers.m_b_archive_mix",
        per_layer_parameters=r,
        formula="r",
        shape_description=f"[{r}]",
    )

    # -------------------------------------------------------------------------
    # VECTOR RECURRENCE
    # -------------------------------------------------------------------------

    add_layer_parameter(
        component="vector_mamba_pathway",
        parameter_path="layers.s_wu",
        per_layer_parameters=n * d,
        formula="n * d",
        shape_description=f"[{n}, {d}]",
    )

    add_layer_parameter(
        component="vector_mamba_pathway",
        parameter_path="layers.s_w_delta",
        per_layer_parameters=n * d,
        formula="n * d",
        shape_description=f"[{n}, {d}]",
    )

    add_layer_parameter(
        component="vector_mamba_pathway",
        parameter_path="layers.s_b_delta",
        per_layer_parameters=n,
        formula="n",
        shape_description=f"[{n}]",
    )

    add_layer_parameter(
        component="vector_mamba_pathway",
        parameter_path="layers.s_w_ret",
        per_layer_parameters=n * d,
        formula="n * d",
        shape_description=f"[{n}, {d}]",
    )

    add_layer_parameter(
        component="vector_mamba_pathway",
        parameter_path="layers.s_b_ret",
        per_layer_parameters=n,
        formula="n",
        shape_description=f"[{n}]",
    )

    add_layer_parameter(
        component="vector_mamba_pathway",
        parameter_path="layers.s_w_c",
        per_layer_parameters=n * d,
        formula="n * d",
        shape_description=f"[{n}, {d}]",
    )

    add_layer_parameter(
        component="vector_mamba_pathway",
        parameter_path="layers.s_w_gate",
        per_layer_parameters=d * d,
        formula="d * d",
        shape_description=f"[{d}, {d}]",
    )

    add_layer_parameter(
        component="vector_mamba_pathway",
        parameter_path="layers.s_b_gate",
        per_layer_parameters=d,
        formula="d",
        shape_description=f"[{d}]",
    )

    add_layer_parameter(
        component="vector_mamba_pathway",
        parameter_path="layers.s_proj_w",
        per_layer_parameters=d * n,
        formula="d * n",
        shape_description=f"[{d}, {n}]",
    )

    add_layer_parameter(
        component="vector_mamba_pathway",
        parameter_path="layers.s_proj_b",
        per_layer_parameters=d,
        formula="d",
        shape_description=f"[{d}]",
    )

    # -------------------------------------------------------------------------
    # MATRIX -> VECTOR FEEDBACK
    # -------------------------------------------------------------------------

    add_layer_parameter(
        component="feedback_bridge",
        parameter_path="layers.s_w_memory_feedback",
        per_layer_parameters=d,
        formula="1 * d",
        shape_description=f"[1, {d}]",
    )

    add_layer_parameter(
        component="feedback_bridge",
        parameter_path="layers.s_b_memory_feedback",
        per_layer_parameters=1,
        formula="1",
        shape_description="[1]",
    )

    add_layer_parameter(
        component="feedback_bridge",
        parameter_path="layers.s_memory_down",
        per_layer_parameters=f * r,
        formula="feedback_rank * r",
        shape_description=f"[{f}, {r}]",
    )

    add_layer_parameter(
        component="feedback_bridge",
        parameter_path="layers.s_memory_up",
        per_layer_parameters=d * f,
        formula="d * feedback_rank",
        shape_description=f"[{d}, {f}]",
    )

    # -------------------------------------------------------------------------
    # ROUTER
    # -------------------------------------------------------------------------

    add_layer_parameter(
        component="router",
        parameter_path="layers.r_w",
        per_layer_parameters=h * d,
        formula="router_hidden * d",
        shape_description=f"[{h}, {d}]",
    )

    add_layer_parameter(
        component="router",
        parameter_path="layers.r_b",
        per_layer_parameters=h,
        formula="router_hidden",
        shape_description=f"[{h}]",
    )

    router_outputs = (
        d
        if CANONICAL_VECTOR_ROUTER
        else 1
    )

    add_layer_parameter(
        component="router",
        parameter_path="layers.r_proj",
        per_layer_parameters=router_outputs * h,
        formula="router_outputs * router_hidden",
        shape_description=f"[{router_outputs}, {h}]",
    )

    add_layer_parameter(
        component="router",
        parameter_path="layers.r_proj_b",
        per_layer_parameters=router_outputs,
        formula="router_outputs",
        shape_description=f"[{router_outputs}]",
    )

    # -------------------------------------------------------------------------
    # PRE-NORMALIZATION
    # -------------------------------------------------------------------------

    add_layer_parameter(
        component="normalization",
        parameter_path="layers.pre_g",
        per_layer_parameters=d,
        formula="d",
        shape_description=f"[{d}]",
    )

    add_layer_parameter(
        component="normalization",
        parameter_path="layers.pre_b",
        per_layer_parameters=d,
        formula="d",
        shape_description=f"[{d}]",
    )

    # -------------------------------------------------------------------------
    # EMBEDDING
    # -------------------------------------------------------------------------

    plan.append(
        ParameterPlanEntry(
            component="embeddings",
            parameter_path="embed",
            per_layer_parameters=(
                vocab * d
            ),
            total_parameters=(
                vocab * d
            ),
            formula="vocab_size * d",
            shape_description=f"[{vocab}, {d}]",
        )
    )

    # -------------------------------------------------------------------------
    # OUTPUT HEAD
    #
    # The canonical census should remain the source of truth for exact
    # head accounting. PB0-E freezes the canonical head dimensions.
    #
    # For byte LM architecture, the observed canonical large leaves are:
    # head.w1 [1536, 512]
    # head.w2 [256, 1536]
    # -------------------------------------------------------------------------

    head_hidden = 3 * d

    plan.append(
        ParameterPlanEntry(
            component="output_head",
            parameter_path="head.w1",
            per_layer_parameters=head_hidden * d,
            total_parameters=head_hidden * d,
            formula="(3 * d) * d",
            shape_description=f"[{head_hidden}, {d}]",
        )
    )

    plan.append(
        ParameterPlanEntry(
            component="output_head",
            parameter_path="head.w2",
            per_layer_parameters=vocab * head_hidden,
            total_parameters=vocab * head_hidden,
            formula="vocab_size * (3 * d)",
            shape_description=f"[{vocab}, {head_hidden}]",
        )
    )

    # -------------------------------------------------------------------------
    # FUTURE PREDICTION HEAD
    #
    # The canonical census reports:
    # future_heads.w1 [1, 1536, 512]
    # future_heads.w2 [1, 256, 1536]
    # -------------------------------------------------------------------------

    plan.append(
        ParameterPlanEntry(
            component="auxiliary",
            parameter_path="future_heads.w1",
            per_layer_parameters=head_hidden * d,
            total_parameters=head_hidden * d,
            formula="1 * (3 * d) * d",
            shape_description=f"[1, {head_hidden}, {d}]",
        )
    )

    plan.append(
        ParameterPlanEntry(
            component="auxiliary",
            parameter_path="future_heads.w2",
            per_layer_parameters=vocab * head_hidden,
            total_parameters=vocab * head_hidden,
            formula="1 * vocab_size * (3 * d)",
            shape_description=f"[1, {vocab}, {head_hidden}]",
        )
    )

    return plan


# =============================================================================
# SUMMARY BUILDING
# =============================================================================

def build_component_summary(
    plan: list[ParameterPlanEntry],
) -> list[dict[str, Any]]:
    """
    Aggregate parameter-plan entries by component.
    """

    totals: dict[str, int] = {}

    for entry in plan:
        totals.setdefault(
            entry.component,
            0,
        )

        totals[
            entry.component
        ] += entry.total_parameters

    total_parameters = sum(
        totals.values()
    )

    summary: list[dict[str, Any]] = []

    for component, parameters in sorted(
        totals.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    ):
        percent = (
            100.0
            * parameters
            / total_parameters
            if total_parameters > 0
            else 0.0
        )

        summary.append(
            {
                "component": component,
                "parameters": parameters,
                "percent_of_model": percent,
            }
        )

    return summary


def find_mapping_reported_total(
    mapping_data: dict[str, Any],
) -> int | None:
    """
    Recursively search for a reported selected-candidate total.
    """

    candidate_keys = (
        "total_parameters",
        "candidate_total_parameters",
        "realized_total_parameters",
        "TOTAL",
    )

    def visit(
        value: Any,
    ) -> int | None:
        if isinstance(value, dict):
            for key in candidate_keys:
                if key in value:
                    try:
                        return require_positive_integer(
                            value[key],
                            key,
                        )
                    except RuntimeError:
                        pass

            for child_value in value.values():
                found = visit(
                    child_value
                )

                if found is not None:
                    return found

        elif isinstance(value, list):
            for child_value in value:
                found = visit(
                    child_value
                )

                if found is not None:
                    return found

        return None

    return visit(
        mapping_data
    )


# =============================================================================
# OUTPUT WRITING
# =============================================================================

def write_json_output(
    output: dict[str, Any],
) -> None:
    try:
        with OUTPUT_JSON_PATH.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                output,
                file,
                indent=2,
                sort_keys=False,
            )

    except OSError as exc:
        fail(
            f"Could not write JSON output:\n"
            f"{OUTPUT_JSON_PATH}\n\n"
            f"Error: {exc}"
        )


def write_csv_output(
    plan: list[ParameterPlanEntry],
) -> None:
    try:
        with OUTPUT_CSV_PATH.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "component",
                    "parameter_path",
                    "per_layer_parameters",
                    "total_parameters",
                    "formula",
                    "shape_description",
                ],
            )

            writer.writeheader()

            for entry in plan:
                writer.writerow(
                    asdict(entry)
                )

    except OSError as exc:
        fail(
            f"Could not write CSV output:\n"
            f"{OUTPUT_CSV_PATH}\n\n"
            f"Error: {exc}"
        )


def write_markdown_output(
    dimensions: ArchitectureDimensions,
    candidate_id: str,
    target_budget: int,
    mapping_reported_total: int | None,
    computed_total: int,
    component_summary: list[dict[str, Any]],
    plan: list[ParameterPlanEntry],
) -> None:
    lines: list[str] = []

    lines.append(
        "# PB0-E Candidate Architecture Mapping"
    )

    lines.append("")
    lines.append("## Scope")
    lines.append("")

    lines.append(
        "PB0-E translates the selected PB0 Candidate A integer dimensions "
        "into a concrete parameterization plan for the existing Modus_X "
        "architecture."
    )

    lines.append("")

    lines.append(
        "No canonical model source file is modified by this analysis."
    )

    lines.append("")
    lines.append("## Selected Candidate")
    lines.append("")

    lines.append(
        f"- Candidate ID: **{candidate_id}**"
    )

    lines.append(
        f"- Frozen target budget: **{target_budget:,}**"
    )

    if mapping_reported_total is not None:
        lines.append(
            f"- PB0 mapping reported total: "
            f"**{mapping_reported_total:,}**"
        )

    lines.append(
        f"- PB0-E computed parameter-plan total: "
        f"**{computed_total:,}**"
    )

    lines.append(
        f"- Computed difference from frozen target: "
        f"**{computed_total - target_budget:+,}**"
    )

    lines.append("")
    lines.append("## Concrete Dimensions")
    lines.append("")

    lines.append(
        "| Dimension | Value |"
    )

    lines.append(
        "|---|---:|"
    )

    lines.append(
        f"| Main width `d` | {dimensions.d} |"
    )

    lines.append(
        f"| Matrix rank `r` | {dimensions.r} |"
    )

    lines.append(
        f"| Vector expansion `n` | {dimensions.n} |"
    )

    lines.append(
        f"| Router hidden width | {dimensions.router_hidden} |"
    )

    lines.append(
        f"| Feedback bottleneck rank | "
        f"{dimensions.feedback_rank} |"
    )

    lines.append(
        f"| Layers | {dimensions.n_layers} |"
    )

    lines.append(
        f"| Vocabulary size | {dimensions.vocab_size} |"
    )

    lines.append("")
    lines.append("## Component Parameter Plan")
    lines.append("")

    lines.append(
        "| Component | Parameters | Share |"
    )

    lines.append(
        "|---|---:|---:|"
    )

    for row in component_summary:
        lines.append(
            f"| {row['component']} | "
            f"{row['parameters']:,} | "
            f"{row['percent_of_model']:.4f}% |"
        )

    lines.append("")
    lines.append("## Tensor Mapping")
    lines.append("")

    lines.append(
        "| Component | Path | Shape | Formula | Per Layer | Total |"
    )

    lines.append(
        "|---|---|---|---|---:|---:|"
    )

    for entry in plan:
        lines.append(
            f"| {entry.component} | "
            f"`{entry.parameter_path}` | "
            f"`{entry.shape_description}` | "
            f"`{entry.formula}` | "
            f"{entry.per_layer_parameters:,} | "
            f"{entry.total_parameters:,} |"
        )

    lines.append("")
    lines.append("## Architectural Mapping Decision")
    lines.append("")

    lines.append(
        "The selected PB0 dimensions are mapped onto the existing "
        "MemoryFeedbackArchive architecture as follows:"
    )

    lines.append("")

    lines.append(
        "- `r` changes matrix-memory projection dimensions and recurrent "
        "matrix state dimensions."
    )

    lines.append(
        "- `n` changes the vector recurrent state and associated "
        "input/output projections."
    )

    lines.append(
        "- `router_hidden` changes only the router hidden representation."
    )

    lines.append(
        "- `feedback_rank` changes the matrix-to-vector bottleneck."
    )

    lines.append(
        "- `d = 512`, layer count, vocabulary, embedding, and output "
        "interfaces remain frozen."
    )

    lines.append("")
    lines.append("## Important Gate")
    lines.append("")

    lines.append(
        "This parameter plan is an analytical mapping, not yet proof that "
        "the canonical initializer and forward implementation instantiate "
        "exactly this tree."
    )

    lines.append("")

    lines.append(
        "The next gate is to create an isolated experimental architecture "
        "configuration, instantiate it through the actual JAX model "
        "initializer, and perform an exact parameter-tree recount."
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
    print("MODUS_X PB0-E CANDIDATE ARCHITECTURE MAPPING")
    print("=" * 80)
    print()

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"PB0-D input  : {PB0_D_INPUT_PATH}"
    )

    print(
        f"Dimension map: {PB0_CANDIDATE_A_MAPPING_PATH}"
    )

    print(
        f"Output dir   : {OUTPUT_DIR}"
    )

    print()

    if not OUTPUT_DIR.exists():
        fail(
            f"Output directory does not exist:\n"
            f"{OUTPUT_DIR}"
        )

    # -------------------------------------------------------------------------
    # LOAD INPUTS
    # -------------------------------------------------------------------------

    print("=" * 80)
    print("LOADING PB0 INPUTS")
    print("=" * 80)
    print()

    pb0_d_data = load_json(
        PB0_D_INPUT_PATH,
        "PB0-D fixed-budget allocation",
    )

    mapping_data = load_json(
        PB0_CANDIDATE_A_MAPPING_PATH,
        "PB0 Candidate A dimension mapping",
    )

    candidate_id = (
        extract_selected_candidate_id(
            pb0_d_data
        )
    )

    selected_candidate = (
        extract_candidate(
            pb0_d_data,
            candidate_id,
        )
    )

    baseline_parameter_count = (
        require_positive_integer(
            pb0_d_data.get(
                "baseline_parameter_count"
            ),
            "baseline_parameter_count",
        )
    )

    frozen_parameter_count = (
        require_positive_integer(
            pb0_d_data.get(
                "frozen_parameter_count"
            ),
            "frozen_parameter_count",
        )
    )

    candidate_total = (
        require_positive_integer(
            selected_candidate.get(
                "total_parameters"
            ),
            "candidate.total_parameters",
        )
    )

    print(
        f"Selected candidate ID        : {candidate_id}"
    )

    print(
        f"Baseline parameter count     : "
        f"{baseline_parameter_count:,}"
    )

    print(
        f"Frozen parameter count       : "
        f"{frozen_parameter_count:,}"
    )

    print(
        f"PB0-D candidate total        : "
        f"{candidate_total:,}"
    )

    print()

    # -------------------------------------------------------------------------
    # DIMENSION EXTRACTION
    # -------------------------------------------------------------------------

    print("=" * 80)
    print("RESOLVING CONCRETE ARCHITECTURE DIMENSIONS")
    print("=" * 80)
    print()

    dimensions = (
        extract_dimension_mapping(
            mapping_data
        )
    )

    print(
        f"Main width d                 : "
        f"{dimensions.d}"
    )

    print(
        f"Matrix rank r                : "
        f"{dimensions.r}"
    )

    print(
        f"Vector expansion n           : "
        f"{dimensions.n}"
    )

    print(
        f"Router hidden width          : "
        f"{dimensions.router_hidden}"
    )

    print(
        f"Feedback bottleneck rank     : "
        f"{dimensions.feedback_rank}"
    )

    print(
        f"Layer count                  : "
        f"{dimensions.n_layers}"
    )

    print(
        f"Vocabulary size              : "
        f"{dimensions.vocab_size}"
    )

    print()

    # -------------------------------------------------------------------------
    # PARAMETER PLAN
    # -------------------------------------------------------------------------

    print("=" * 80)
    print("BUILDING CONCRETE PARAMETER PLAN")
    print("=" * 80)
    print()

    parameter_plan = (
        build_parameter_plan(
            dimensions
        )
    )

    computed_total = sum(
        entry.total_parameters
        for entry in parameter_plan
    )

    component_summary = (
        build_component_summary(
            parameter_plan
        )
    )

    mapping_reported_total = (
        find_mapping_reported_total(
            mapping_data
        )
    )

    difference_from_candidate = (
        computed_total
        - candidate_total
    )

    difference_from_baseline = (
        computed_total
        - baseline_parameter_count
    )

    print(
        f"PB0-E computed total         : "
        f"{computed_total:,}"
    )

    print(
        f"Difference from PB0-D target : "
        f"{difference_from_candidate:+,}"
    )

    print(
        f"Difference from baseline     : "
        f"{difference_from_baseline:+,}"
    )

    print()

    print("Component totals:")
    print()

    for row in component_summary:
        print(
            f"  {row['component']:<28} "
            f"{row['parameters']:>12,}  "
            f"{row['percent_of_model']:>8.4f}%"
        )

    print()

    # -------------------------------------------------------------------------
    # STRUCTURED OUTPUT
    # -------------------------------------------------------------------------

    output: dict[str, Any] = {
        "analysis_name": (
            "Modus_X PB0-E Candidate Architecture Mapping"
        ),
        "generated_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "project_root": (
            str(PROJECT_ROOT)
        ),
        "inputs": {
            "pb0_d_fixed_budget_allocation": (
                str(PB0_D_INPUT_PATH)
            ),
            "pb0_candidate_a_dimension_mapping": (
                str(
                    PB0_CANDIDATE_A_MAPPING_PATH
                )
            ),
        },
        "selected_candidate": {
            "candidate_id": candidate_id,
            "pb0_d_candidate_total": (
                candidate_total
            ),
            "baseline_parameter_count": (
                baseline_parameter_count
            ),
            "frozen_parameter_count": (
                frozen_parameter_count
            ),
        },
        "architecture_dimensions": (
            asdict(
                dimensions
            )
        ),
        "parameter_accounting": {
            "pb0_mapping_reported_total": (
                mapping_reported_total
            ),
            "pb0_e_computed_total": (
                computed_total
            ),
            "difference_from_pb0_d_candidate": (
                difference_from_candidate
            ),
            "difference_from_baseline": (
                difference_from_baseline
            ),
            "exact_match_to_pb0_d_candidate": (
                computed_total
                == candidate_total
            ),
            "exact_match_to_baseline": (
                computed_total
                == baseline_parameter_count
            ),
        },
        "component_summary": (
            component_summary
        ),
        "parameter_plan": [
            asdict(entry)
            for entry in parameter_plan
        ],
        "next_gate": (
            "Create an isolated experimental architecture configuration, "
            "instantiate the actual JAX parameter tree, and perform an "
            "exact parameter recount before training or checkpoint loading."
        ),
    }

    # -------------------------------------------------------------------------
    # WRITE OUTPUTS
    # -------------------------------------------------------------------------

    print("=" * 80)
    print("WRITING PB0-E OUTPUTS")
    print("=" * 80)
    print()

    write_json_output(
        output
    )

    write_csv_output(
        parameter_plan
    )

    write_markdown_output(
        dimensions=dimensions,
        candidate_id=candidate_id,
        target_budget=candidate_total,
        mapping_reported_total=(
            mapping_reported_total
        ),
        computed_total=computed_total,
        component_summary=component_summary,
        plan=parameter_plan,
    )

    # -------------------------------------------------------------------------
    # COMPLETE
    # -------------------------------------------------------------------------

    print("=" * 80)
    print("PB0-E ARCHITECTURE MAPPING COMPLETE")
    print("=" * 80)
    print()

    print(
        f"PB0-E computed parameter total : "
        f"{computed_total:,}"
    )

    print(
        f"PB0-D target parameter total   : "
        f"{candidate_total:,}"
    )

    print(
        f"Difference                      : "
        f"{difference_from_candidate:+,}"
    )

    print()

    print("Output files:")
    print()

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

    print("NEXT GATE:")
    print(
        "Instantiate these dimensions through the actual Modus_X "
        "initializer and perform an exact parameter-tree recount."
    )

    print(
        "Do not train Candidate A yet."
    )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print()
        print(
            "PB0-E interrupted by user."
        )
        sys.exit(130)

    except SystemExit:
        raise

    except Exception as exc:
        print()
        print("=" * 80)
        print("UNHANDLED PB0-E ERROR")
        print("=" * 80)
        print()
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print()

        raise