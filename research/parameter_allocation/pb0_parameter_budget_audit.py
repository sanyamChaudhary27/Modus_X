from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


# =============================================================================
# PROJECT CONFIGURATION
# =============================================================================


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
OUTPUT_DIR = SCRIPT_PATH.parent / "outputs"

CHECKPOINT_FILENAME = "checkpoint.pkl"

DEFAULT_SEED_DIRECTORIES: dict[str, Path] = {
    "seed1": Path(r"E:\seed1"),
    "seed2": Path(r"E:\seed2"),
}

EXPECTED_BASELINE_PARAMETER_COUNT = 47_437_768

OUTPUT_JSON = (
    OUTPUT_DIR
    / "PB0_PARAMETER_BUDGET_AUDIT.json"
)

OUTPUT_MARKDOWN = (
    OUTPUT_DIR
    / "PB0_PARAMETER_BUDGET_AUDIT.md"
)

OUTPUT_COMPONENTS_CSV = (
    OUTPUT_DIR
    / "PB0_PARAMETER_BUDGET_COMPONENTS.csv"
)

OUTPUT_PARAMETERS_CSV = (
    OUTPUT_DIR
    / "PB0_PARAMETER_BUDGET_PARAMETERS.csv"
)

OUTPUT_BY_LAYER_CSV = (
    OUTPUT_DIR
    / "PB0_PARAMETER_BUDGET_BY_LAYER.csv"
)

OUTPUT_SEED_COMPARISON_CSV = (
    OUTPUT_DIR
    / "PB0_PARAMETER_BUDGET_SEED_COMPARISON.csv"
)


# =============================================================================
# COMPONENT DEFINITIONS
# =============================================================================


COMPONENT_ORDER: tuple[str, ...] = (
    "matrix_memory",
    "vector_mamba",
    "router",
    "feedback_bridge",
    "embeddings",
    "output_head",
    "normalization",
    "auxiliary",
    "other",
)


COMPONENT_DISPLAY_NAMES: dict[str, str] = {
    "matrix_memory": "Matrix Memory",
    "vector_mamba": "Vector / Mamba Pathway",
    "router": "Router",
    "feedback_bridge": "Feedback Bridge",
    "embeddings": "Embeddings",
    "output_head": "Output Head",
    "normalization": "Normalization",
    "auxiliary": "Auxiliary",
    "other": "Other",
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass(frozen=True)
class ParameterLeaf:
    path: str
    shape: tuple[int, ...]
    dtype: str
    elements: int
    bytes: int


@dataclass(frozen=True)
class ParameterRecord:
    seed: str
    component: str
    parameter_path: str
    top_level_group: str
    layer: int | None
    shape: tuple[int, ...]
    dtype: str
    parameters: int
    bytes: int
    percent_of_seed_total: float


@dataclass(frozen=True)
class ComponentSummary:
    seed: str
    component: str
    display_name: str
    parameter_count: int
    byte_count: int
    parameter_fraction: float
    parameter_percent: float
    parameter_tensors: int
    layer_parameter_count: int
    global_parameter_count: int


@dataclass(frozen=True)
class LayerComponentSummary:
    seed: str
    layer: int
    component: str
    display_name: str
    parameter_count: int
    parameter_fraction_of_seed_total: float
    parameter_percent_of_seed_total: float
    parameter_fraction_of_layer_total: float
    parameter_percent_of_layer_total: float


@dataclass(frozen=True)
class SeedAudit:
    seed: str
    checkpoint_path: str
    checkpoint_size_bytes: int
    checkpoint_step: int | str
    parameter_count: int
    parameter_tensor_count: int
    component_summaries: tuple[ComponentSummary, ...]
    parameter_records: tuple[ParameterRecord, ...]
    layer_summaries: tuple[LayerComponentSummary, ...]


@dataclass(frozen=True)
class SeedComparison:
    component: str
    display_name: str
    seed1_parameters: int
    seed2_parameters: int
    absolute_difference: int
    relative_difference_percent: float | None
    topology_consistent: bool


@dataclass(frozen=True)
class AuditResult:
    seed_audits: tuple[SeedAudit, ...]
    seed_comparisons: tuple[SeedComparison, ...]
    topology_identical: bool
    baseline_parameter_count_verified: bool


# =============================================================================
# CONSOLE HELPERS
# =============================================================================


def banner(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def fail(message: str) -> None:
    raise RuntimeError(message)


def format_int(value: int) -> str:
    return f"{value:,}"


def format_float(
    value: float,
    digits: int = 6,
) -> str:
    if math.isnan(value):
        return "NaN"

    if math.isinf(value):
        return "Inf"

    return f"{value:.{digits}f}"


def format_percent(
    value: float,
    digits: int = 4,
) -> str:
    return f"{value * 100.0:.{digits}f}%"


# =============================================================================
# ARGUMENT PARSING
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "PB0-C exact parameter budget audit for Modus_X. "
            "Audits frozen Seed1 and Seed2 checkpoints and decomposes "
            "the approximately 47M parameter budget into architectural "
            "components."
        )
    )

    parser.add_argument(
        "--seed1-dir",
        type=Path,
        default=DEFAULT_SEED_DIRECTORIES["seed1"],
        help=(
            "Directory containing Seed1 checkpoint.pkl. "
            f"Default: {DEFAULT_SEED_DIRECTORIES['seed1']}"
        ),
    )

    parser.add_argument(
        "--seed2-dir",
        type=Path,
        default=DEFAULT_SEED_DIRECTORIES["seed2"],
        help=(
            "Directory containing Seed2 checkpoint.pkl. "
            f"Default: {DEFAULT_SEED_DIRECTORIES['seed2']}"
        ),
    )

    parser.add_argument(
        "--allow-unexpected-count",
        action="store_true",
        help=(
            "Do not fail if the checkpoint parameter count differs from "
            f"{EXPECTED_BASELINE_PARAMETER_COUNT:,}. "
            "The audit will still report the actual count."
        ),
    )

    return parser.parse_args()


# =============================================================================
# CHECKPOINT LOADING
# =============================================================================


def validate_seed_directory(
    seed: str,
    directory: Path,
) -> Path:
    if not directory.exists():
        fail(
            f"Seed directory does not exist for {seed}:\n"
            f"{directory}"
        )

    if not directory.is_dir():
        fail(
            f"Seed path is not a directory for {seed}:\n"
            f"{directory}"
        )

    checkpoint_path = (
        directory
        / CHECKPOINT_FILENAME
    )

    if not checkpoint_path.exists():
        fail(
            f"Checkpoint does not exist for {seed}:\n"
            f"{checkpoint_path}"
        )

    if not checkpoint_path.is_file():
        fail(
            f"Checkpoint path is not a regular file for {seed}:\n"
            f"{checkpoint_path}"
        )

    return checkpoint_path


def load_checkpoint(
    checkpoint_path: Path,
) -> Mapping[str, Any]:
    try:
        with checkpoint_path.open(
            "rb"
        ) as checkpoint_file:
            checkpoint = pickle.load(
                checkpoint_file
            )
    except Exception as exc:
        fail(
            f"Unable to load checkpoint:\n"
            f"{checkpoint_path}\n\n"
            f"{type(exc).__name__}: {exc}"
        )

    if not isinstance(
        checkpoint,
        Mapping,
    ):
        fail(
            f"Checkpoint root is not a mapping:\n"
            f"{checkpoint_path}\n"
            f"Actual type: {type(checkpoint).__name__}"
        )

    return checkpoint


def extract_checkpoint_params(
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
) -> Mapping[str, Any]:
    if "params" not in checkpoint:
        fail(
            f"Checkpoint is missing top-level 'params':\n"
            f"{checkpoint_path}\n\n"
            "Available keys:\n"
            + "\n".join(
                f"  - {key}"
                for key in sorted(
                    str(key)
                    for key in checkpoint.keys()
                )
            )
        )

    params = checkpoint["params"]

    if not isinstance(
        params,
        Mapping,
    ):
        fail(
            f"Checkpoint params are not a mapping:\n"
            f"{checkpoint_path}\n"
            f"Actual type: {type(params).__name__}"
        )

    return params


# =============================================================================
# PARAMETER TREE COLLECTION
# =============================================================================


def recursively_collect_parameter_leaves(
    tree: Any,
) -> list[ParameterLeaf]:
    leaves: list[ParameterLeaf] = []

    def walk(
        value: Any,
        prefix: str,
    ) -> None:
        if isinstance(
            value,
            Mapping,
        ):
            for key, child in value.items():
                key_text = str(key)

                child_path = (
                    key_text
                    if prefix == ""
                    else (
                        f"{prefix}.{key_text}"
                    )
                )

                walk(
                    child,
                    child_path,
                )

            return

        try:
            array = np.asarray(value)
        except Exception:
            return

        if array.ndim == 0:
            return

        if not np.issubdtype(
            array.dtype,
            np.number,
        ):
            return

        leaves.append(
            ParameterLeaf(
                path=prefix,
                shape=tuple(
                    int(dimension)
                    for dimension
                    in array.shape
                ),
                dtype=str(array.dtype),
                elements=int(array.size),
                bytes=int(array.nbytes),
            )
        )

    walk(
        tree,
        "",
    )

    return leaves


def extract_layer_from_path(
    parameter_path: str,
) -> int | None:
    parts = parameter_path.split(
        "."
    )

    for index, part in enumerate(parts):
        if (
            part == "layers"
            and index + 1 < len(parts)
        ):
            possible_layer = (
                parts[index + 1]
            )

            try:
                return int(
                    possible_layer
                )
            except ValueError:
                return None

    return None


def determine_top_level_group(
    parameter_path: str,
) -> str:
    if not parameter_path:
        return "root"

    return parameter_path.split(
        ".",
        maxsplit=1,
    )[0]


# =============================================================================
# PARAMETER CLASSIFICATION
# =============================================================================


def classify_parameter_path(
    parameter_path: str,
) -> str:
    """
    Classify a checkpoint parameter using the actual Modus_X naming scheme.

    Classification is intentionally based on parameter family semantics rather
    than broad substring matching such as simply matching 'out', because
    parameters like m_w_out belong to matrix memory rather than the LM head.
    """

    path = parameter_path.lower()
    leaf_name = path.split(".")[-1]

    # -------------------------------------------------------------------------
    # EMBEDDINGS
    # -------------------------------------------------------------------------

    if (
        path == "embed"
        or path.startswith("embed.")
        or leaf_name == "embed"
        or leaf_name.startswith("embed_")
        or path == "pos"
        or path.startswith("pos.")
        or leaf_name == "pos"
        or "position_embed" in path
        or "pos_embed" in path
    ):
        return "embeddings"

    # -------------------------------------------------------------------------
    # OUTPUT HEAD
    # -------------------------------------------------------------------------

    if (
        path == "head"
        or path.startswith("head.")
        or path.startswith("lm_head.")
        or path.startswith("output_head.")
        or leaf_name.startswith("lm_head")
        or leaf_name.startswith("output_head")
        or leaf_name == "head_w"
        or leaf_name == "head_b"
    ):
        return "output_head"

    # -------------------------------------------------------------------------
    # FEEDBACK BRIDGE
    # -------------------------------------------------------------------------

    if (
        "feedback" in path
        or "fb_" in leaf_name
        or leaf_name.startswith("fb")
        or "bridge" in path
        or "memory_feedback" in path
    ):
        return "feedback_bridge"

    # -------------------------------------------------------------------------
    # AUXILIARY / DEEP SUPERVISION
    # -------------------------------------------------------------------------

    if (
        "aux" in path
        or "auxiliary" in path
        or "deep_supervision" in path
        or "future_head" in path
        or "future_proj" in path
    ):
        return "auxiliary"

    # -------------------------------------------------------------------------
    # MATRIX MEMORY
    #
    # Actual Modus_X matrix stream:
    #
    # m_wk
    # m_wq
    # m_wv
    # m_w_eta
    # m_b_eta
    # m_w_write
    # m_b_write
    # m_w_ret
    # m_b_ret
    # m_w_read
    # m_b_read
    # m_w_out
    # m_b_out
    # m_proj_w
    # m_proj_b
    # m_ln_g
    # m_ln_b
    #
    # Current/archive extension:
    #
    # m_w_archive_write
    # m_b_archive_write
    # m_w_archive_ret
    # m_b_archive_ret
    # m_w_archive_mix
    # m_b_archive_mix
    # -------------------------------------------------------------------------

    if (
        leaf_name.startswith("m_w")
        or leaf_name.startswith("m_b")
        or leaf_name.startswith("m_proj")
        or leaf_name.startswith("m_ln")
        or leaf_name.startswith("matrix_")
        or "archive" in leaf_name
    ):
        return "matrix_memory"

    # -------------------------------------------------------------------------
    # VECTOR / MAMBA PATHWAY
    #
    # Actual Modus_X vector stream:
    #
    # s_wu
    # s_w_delta
    # s_b_delta
    # s_w_ret
    # s_b_ret
    # s_w_c
    # s_w_gate
    # s_b_gate
    # s_proj_w
    # s_proj_b
    # -------------------------------------------------------------------------

    if (
        leaf_name.startswith("s_w")
        or leaf_name.startswith("s_b")
        or leaf_name.startswith("s_proj")
        or "mamba" in path
        or "vector_state" in path
    ):
        return "vector_mamba"

    # -------------------------------------------------------------------------
    # ROUTER
    #
    # Actual Modus_X router:
    #
    # r_w
    # r_b
    # r_proj
    # r_proj_b
    # -------------------------------------------------------------------------

    if (
        leaf_name.startswith("r_w")
        or leaf_name.startswith("r_b")
        or leaf_name.startswith("r_proj")
        or "router" in path
    ):
        return "router"

    # -------------------------------------------------------------------------
    # NORMALIZATION
    # -------------------------------------------------------------------------

    if (
        leaf_name.endswith("_g")
        or leaf_name.endswith("_b")
        or "norm" in path
        or "layer_norm" in path
        or leaf_name.startswith("ln_")
    ):
        return "normalization"

    return "other"


# =============================================================================
# INVENTORY BUILDING
# =============================================================================


def build_parameter_records(
    *,
    seed: str,
    leaves: Sequence[ParameterLeaf],
    total_parameter_count: int,
) -> list[ParameterRecord]:
    if total_parameter_count <= 0:
        fail(
            "Total parameter count must be positive."
        )

    records: list[
        ParameterRecord
    ] = []

    for leaf in leaves:
        component = (
            classify_parameter_path(
                leaf.path
            )
        )

        layer = extract_layer_from_path(
            leaf.path
        )

        top_level_group = (
            determine_top_level_group(
                leaf.path
            )
        )

        records.append(
            ParameterRecord(
                seed=seed,
                component=component,
                parameter_path=leaf.path,
                top_level_group=top_level_group,
                layer=layer,
                shape=leaf.shape,
                dtype=leaf.dtype,
                parameters=leaf.elements,
                bytes=leaf.bytes,
                percent_of_seed_total=(
                    leaf.elements
                    / total_parameter_count
                    * 100.0
                ),
            )
        )

    return records


def validate_inventory_total(
    records: Sequence[ParameterRecord],
    expected_total: int,
    seed: str,
) -> None:
    observed_total = sum(
        record.parameters
        for record in records
    )

    if observed_total != expected_total:
        fail(
            f"Parameter inventory total mismatch for {seed}.\n\n"
            f"Expected: {format_int(expected_total)}\n"
            f"Observed: {format_int(observed_total)}"
        )


def write_seed_comparison_csv(
    comparisons: Sequence[SeedComparison],
) -> None:
    rows: list[
        dict[str, Any]
    ] = []

    for comparison in comparisons:
        rows.append(
            {
                "component": (
                    comparison.component
                ),
                "display_name": (
                    comparison.display_name
                ),
                "seed1_parameters": (
                    comparison.seed1_parameters
                ),
                "seed2_parameters": (
                    comparison.seed2_parameters
                ),
                "absolute_difference": (
                    comparison.absolute_difference
                ),
                "relative_difference_percent": (
                    comparison.relative_difference_percent
                ),
                "topology_consistent": (
                    comparison.topology_consistent
                ),
            }
        )

    if not rows:
        fail(
            "No seed-comparison rows are available."
        )

    with OUTPUT_SEED_COMPARISON_CSV.open(
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

# =============================================================================
# COMPONENT AGGREGATION
# =============================================================================


def summarize_components(
    *,
    seed: str,
    records: Sequence[ParameterRecord],
    total_parameter_count: int,
) -> list[ComponentSummary]:
    component_parameters: dict[
        str,
        int,
    ] = defaultdict(int)

    component_bytes: dict[
        str,
        int,
    ] = defaultdict(int)

    component_tensor_counts: dict[
        str,
        int,
    ] = defaultdict(int)

    component_layer_parameters: dict[
        str,
        int,
    ] = defaultdict(int)

    component_global_parameters: dict[
        str,
        int,
    ] = defaultdict(int)

    for record in records:
        component = record.component

        component_parameters[
            component
        ] += record.parameters

        component_bytes[
            component
        ] += record.bytes

        component_tensor_counts[
            component
        ] += 1

        if record.layer is None:
            component_global_parameters[
                component
            ] += record.parameters
        else:
            component_layer_parameters[
                component
            ] += record.parameters

    summaries: list[
        ComponentSummary
    ] = []

    for component in COMPONENT_ORDER:
        parameter_count = (
            component_parameters.get(
                component,
                0,
            )
        )

        parameter_fraction = (
            parameter_count
            / total_parameter_count
        )

        summaries.append(
            ComponentSummary(
                seed=seed,
                component=component,
                display_name=(
                    COMPONENT_DISPLAY_NAMES[
                        component
                    ]
                ),
                parameter_count=parameter_count,
                byte_count=(
                    component_bytes.get(
                        component,
                        0,
                    )
                ),
                parameter_fraction=(
                    parameter_fraction
                ),
                parameter_percent=(
                    parameter_fraction
                    * 100.0
                ),
                parameter_tensors=(
                    component_tensor_counts.get(
                        component,
                        0,
                    )
                ),
                layer_parameter_count=(
                    component_layer_parameters.get(
                        component,
                        0,
                    )
                ),
                global_parameter_count=(
                    component_global_parameters.get(
                        component,
                        0,
                    )
                ),
            )
        )

    summarized_total = sum(
        summary.parameter_count
        for summary in summaries
    )

    if (
        summarized_total
        != total_parameter_count
    ):
        fail(
            f"Component aggregation mismatch for {seed}.\n\n"
            f"Expected: {format_int(total_parameter_count)}\n"
            f"Observed: {format_int(summarized_total)}"
        )

    return summaries


# =============================================================================
# LAYER AGGREGATION
# =============================================================================

def summarize_layers(
    *,
    seed: str,
    records: Sequence[ParameterRecord],
    total_parameter_count: int,
) -> list[LayerComponentSummary]:
    if total_parameter_count <= 0:
        fail(
            f"Cannot summarize layers for seed '{seed}' because "
            "the total parameter count is non-positive."
        )

    layer_component_counts: dict[
        tuple[int, str],
        int,
    ] = defaultdict(int)

    layer_totals: dict[
        int,
        int,
    ] = defaultdict(int)

    discovered_layers: set[
        int
    ] = set()

    records_with_layers = 0

    for record in records:
        if record.layer is None:
            continue

        records_with_layers += 1

        discovered_layers.add(
            record.layer
        )

        layer_component_counts[
            (
                record.layer,
                record.component,
            )
        ] += record.parameters

        layer_totals[
            record.layer
        ] += record.parameters

    if records_with_layers == 0:
        print()
        print(
            "WARNING: No numeric layer identifiers were discovered "
            "in checkpoint parameter paths."
        )
        print(
            "WARNING: Per-layer parameter allocation cannot be "
            "reconstructed from the checkpoint naming topology."
        )
        print(
            "WARNING: Skipping by-layer summaries while preserving "
            "the full component-level parameter audit."
        )

        return []

    summaries: list[
        LayerComponentSummary
    ] = []

    for layer in sorted(
        discovered_layers
    ):
        layer_total = layer_totals[
            layer
        ]

        if layer_total <= 0:
            fail(
                f"Layer {layer} in seed '{seed}' has a non-positive "
                "parameter total."
            )

        for component in COMPONENT_ORDER:
            parameter_count = (
                layer_component_counts.get(
                    (
                        layer,
                        component,
                    ),
                    0,
                )
            )

            seed_fraction = (
                parameter_count
                / total_parameter_count
            )

            layer_fraction = (
                parameter_count
                / layer_total
            )

            summaries.append(
                LayerComponentSummary(
                    seed=seed,
                    layer=layer,
                    component=component,
                    display_name=(
                        COMPONENT_DISPLAY_NAMES[
                            component
                        ]
                    ),
                    parameter_count=(
                        parameter_count
                    ),
                    parameter_fraction_of_seed_total=(
                        seed_fraction
                    ),
                    parameter_percent_of_seed_total=(
                        seed_fraction
                        * 100.0
                    ),
                    parameter_fraction_of_layer_total=(
                        layer_fraction
                    ),
                    parameter_percent_of_layer_total=(
                        layer_fraction
                        * 100.0
                    ),
                )
            )

    return summaries


# =============================================================================
# SEED AUDIT
# =============================================================================


def audit_seed(
    *,
    seed: str,
    checkpoint_path: Path,
) -> SeedAudit:
    banner(
        f"AUDITING {seed.upper()}"
    )

    print(
        f"Checkpoint : {checkpoint_path}"
    )

    checkpoint_size_bytes = (
        checkpoint_path.stat().st_size
    )

    print(
        "Size       : "
        f"{checkpoint_size_bytes / (1024 * 1024):.2f} MiB"
    )

    checkpoint = load_checkpoint(
        checkpoint_path
    )

    params = extract_checkpoint_params(
        checkpoint,
        checkpoint_path,
    )

    checkpoint_step = checkpoint.get(
        "step",
        "unknown",
    )

    if isinstance(
        checkpoint_step,
        np.generic,
    ):
        checkpoint_step = (
            checkpoint_step.item()
        )

    leaves = (
        recursively_collect_parameter_leaves(
            params
        )
    )

    if not leaves:
        fail(
            f"No numeric parameter tensors found in {seed}."
        )

    parameter_count = sum(
        leaf.elements
        for leaf in leaves
    )

    records = (
        build_parameter_records(
            seed=seed,
            leaves=leaves,
            total_parameter_count=(
                parameter_count
            ),
        )
    )

    validate_inventory_total(
        records,
        parameter_count,
        seed,
    )

    component_summaries = (
        summarize_components(
            seed=seed,
            records=records,
            total_parameter_count=(
                parameter_count
            ),
        )
    )

    layer_summaries = (
        summarize_layers(
            seed=seed,
            records=records,
            total_parameter_count=(
                parameter_count
            ),
        )
    )

    print()
    print(
        "Checkpoint step : "
        f"{checkpoint_step}"
    )
    print(
        "Parameter tensors: "
        f"{format_int(len(leaves))}"
    )
    print(
        "Parameter count  : "
        f"{format_int(parameter_count)}"
    )

    print()
    print("Component allocation:")

    for summary in component_summaries:
        print(
            f"  {summary.display_name:24s} "
            f"{format_int(summary.parameter_count):>14s} "
            f"{summary.parameter_percent:8.3f}%"
        )

    return SeedAudit(
        seed=seed,
        checkpoint_path=str(
            checkpoint_path
        ),
        checkpoint_size_bytes=(
            checkpoint_size_bytes
        ),
        checkpoint_step=(
            checkpoint_step
        ),
        parameter_count=(
            parameter_count
        ),
        parameter_tensor_count=(
            len(leaves)
        ),
        component_summaries=tuple(
            component_summaries
        ),
        parameter_records=tuple(
            records
        ),
        layer_summaries=tuple(
            layer_summaries
        ),
    )


# =============================================================================
# TOPOLOGY COMPARISON
# =============================================================================


def record_signature(
    record: ParameterRecord,
) -> tuple[
    str,
    str,
    int | None,
    tuple[int, ...],
]:
    return (
        record.component,
        record.parameter_path,
        record.layer,
        record.shape,
    )


def compare_parameter_topology(
    seed1: SeedAudit,
    seed2: SeedAudit,
) -> bool:
    seed1_signatures = {
        record_signature(record)
        for record in seed1.parameter_records
    }

    seed2_signatures = {
        record_signature(record)
        for record in seed2.parameter_records
    }

    if (
        seed1_signatures
        == seed2_signatures
    ):
        return True

    only_seed1 = (
        seed1_signatures
        - seed2_signatures
    )

    only_seed2 = (
        seed2_signatures
        - seed1_signatures
    )

    message_lines = [
        "Seed parameter topology differs.",
    ]

    if only_seed1:
        message_lines.append(
            ""
        )
        message_lines.append(
            "Only present in Seed1:"
        )

        for item in sorted(
            only_seed1
        )[:20]:
            message_lines.append(
                f"  - {item}"
            )

    if only_seed2:
        message_lines.append(
            ""
        )
        message_lines.append(
            "Only present in Seed2:"
        )

        for item in sorted(
            only_seed2
        )[:20]:
            message_lines.append(
                f"  - {item}"
            )

    fail(
        "\n".join(
            message_lines
        )
    )

    return False


def component_summary_map(
    audit: SeedAudit,
) -> dict[
    str,
    ComponentSummary,
]:
    return {
        summary.component: summary
        for summary
        in audit.component_summaries
    }


def build_seed_comparisons(
    seed1: SeedAudit,
    seed2: SeedAudit,
) -> list[SeedComparison]:
    seed1_map = (
        component_summary_map(
            seed1
        )
    )

    seed2_map = (
        component_summary_map(
            seed2
        )
    )

    comparisons: list[
        SeedComparison
    ] = []

    for component in COMPONENT_ORDER:
        seed1_summary = (
            seed1_map[component]
        )

        seed2_summary = (
            seed2_map[component]
        )

        difference = abs(
            seed1_summary.parameter_count
            - seed2_summary.parameter_count
        )

        baseline = (
            abs(
                seed1_summary.parameter_count
            )
            + abs(
                seed2_summary.parameter_count
            )
        ) / 2.0

        relative_difference_percent: (
            float
            | None
        )

        if baseline <= 0.0:
            relative_difference_percent = (
                None
            )
        else:
            relative_difference_percent = (
                difference
                / baseline
                * 100.0
            )

        comparisons.append(
            SeedComparison(
                component=component,
                display_name=(
                    COMPONENT_DISPLAY_NAMES[
                        component
                    ]
                ),
                seed1_parameters=(
                    seed1_summary.parameter_count
                ),
                seed2_parameters=(
                    seed2_summary.parameter_count
                ),
                absolute_difference=(
                    difference
                ),
                relative_difference_percent=(
                    relative_difference_percent
                ),
                topology_consistent=(
                    difference == 0
                ),
            )
        )

    return comparisons


# =============================================================================
# CSV OUTPUT
# =============================================================================


def write_components_csv(
    audits: Sequence[SeedAudit],
) -> None:
    rows: list[
        dict[str, Any]
    ] = []

    for audit in audits:
        for summary in (
            audit.component_summaries
        ):
            rows.append(
                {
                    "seed": summary.seed,
                    "component": summary.component,
                    "display_name": (
                        summary.display_name
                    ),
                    "parameter_count": (
                        summary.parameter_count
                    ),
                    "parameter_fraction": (
                        summary.parameter_fraction
                    ),
                    "parameter_percent": (
                        summary.parameter_percent
                    ),
                    "parameter_tensors": (
                        summary.parameter_tensors
                    ),
                    "layer_parameter_count": (
                        summary.layer_parameter_count
                    ),
                    "global_parameter_count": (
                        summary.global_parameter_count
                    ),
                    "byte_count": (
                        summary.byte_count
                    ),
                }
            )

    if not rows:
        fail(
            "No component rows available."
        )

    with OUTPUT_COMPONENTS_CSV.open(
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


def write_parameters_csv(
    audits: Sequence[SeedAudit],
) -> None:
    rows: list[
        dict[str, Any]
    ] = []

    for audit in audits:
        for record in (
            audit.parameter_records
        ):
            rows.append(
                {
                    "seed": record.seed,
                    "component": (
                        record.component
                    ),
                    "parameter_path": (
                        record.parameter_path
                    ),
                    "top_level_group": (
                        record.top_level_group
                    ),
                    "layer": record.layer,
                    "shape": "x".join(
                        str(dimension)
                        for dimension
                        in record.shape
                    ),
                    "dtype": record.dtype,
                    "parameters": (
                        record.parameters
                    ),
                    "bytes": record.bytes,
                    "percent_of_seed_total": (
                        record.percent_of_seed_total
                    ),
                }
            )

    if not rows:
        fail(
            "No parameter inventory rows available."
        )

    with OUTPUT_PARAMETERS_CSV.open(
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


def write_by_layer_csv(
    audits: Sequence[SeedAudit],
) -> None:
    rows: list[
        dict[str, Any]
    ] = []

    for audit in audits:
        for summary in audit.layer_summaries:
            rows.append(
                {
                    "seed": summary.seed,
                    "layer": summary.layer,
                    "component": (
                        summary.component
                    ),
                    "display_name": (
                        summary.display_name
                    ),
                    "parameter_count": (
                        summary.parameter_count
                    ),
                    "parameter_fraction_of_seed_total": (
                        summary.parameter_fraction_of_seed_total
                    ),
                    "parameter_percent_of_seed_total": (
                        summary.parameter_percent_of_seed_total
                    ),
                    "parameter_fraction_of_layer_total": (
                        summary.parameter_fraction_of_layer_total
                    ),
                    "parameter_percent_of_layer_total": (
                        summary.parameter_percent_of_layer_total
                    ),
                }
            )

    if not rows:
        print()
        print(
            "WARNING: No layer-level parameter rows were produced."
        )
        print(
            "WARNING: Skipping by-layer CSV because checkpoint "
            "parameter names do not expose reconstructable layer IDs."
        )

        if OUTPUT_BY_LAYER_CSV.exists():
            OUTPUT_BY_LAYER_CSV.unlink()

        return

    with OUTPUT_BY_LAYER_CSV.open(
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

# =============================================================================
# JSON OUTPUT
# =============================================================================


def make_json_safe(
    value: Any,
) -> Any:
    if isinstance(
        value,
        Path,
    ):
        return str(value)

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if isinstance(
        value,
        tuple,
    ):
        return [
            make_json_safe(
                item
            )
            for item in value
        ]

    if isinstance(
        value,
        list,
    ):
        return [
            make_json_safe(
                item
            )
            for item in value
        ]

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): make_json_safe(
                item
            )
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        float,
    ):
        if (
            math.isnan(value)
            or math.isinf(value)
        ):
            return None

    return value


def write_json_report(
    result: AuditResult,
) -> None:
    payload = {
        "analysis": {
            "name": (
                "PB0-C Parameter Budget Audit"
            ),
            "purpose": (
                "Exact decomposition of the frozen Modus_X "
                "parameter budget before fixed-budget "
                "allocation experiments."
            ),
            "expected_baseline_parameter_count": (
                EXPECTED_BASELINE_PARAMETER_COUNT
            ),
            "component_order": list(
                COMPONENT_ORDER
            ),
        },
        "project_root": str(
            PROJECT_ROOT
        ),
        "topology_identical": (
            result.topology_identical
        ),
        "baseline_parameter_count_verified": (
            result.baseline_parameter_count_verified
        ),
        "seed_audits": [
            asdict(audit)
            for audit
            in result.seed_audits
        ],
        "seed_comparisons": [
            asdict(comparison)
            for comparison
            in result.seed_comparisons
        ],
    }

    OUTPUT_JSON.write_text(
        json.dumps(
            make_json_safe(
                payload
            ),
            indent=2,
        ),
        encoding="utf-8",
    )


# =============================================================================
# MARKDOWN HELPERS
# =============================================================================


def markdown_table(
    headers: Sequence[str],
    rows: Sequence[
        Sequence[str]
    ],
) -> str:
    header_line = (
        "| "
        + " | ".join(
            headers
        )
        + " |"
    )

    separator_line = (
        "|"
        + "|".join(
            "---"
            for _
            in headers
        )
        + "|"
    )

    body_lines = [
        "| "
        + " | ".join(
            row
        )
        + " |"
        for row
        in rows
    ]

    return "\n".join(
        [
            header_line,
            separator_line,
            *body_lines,
        ]
    )


def write_markdown_report(
    result: AuditResult,
) -> None:
    lines: list[
        str
    ] = []

    lines.append(
        "# PB0-C Parameter Budget Audit"
    )
    lines.append("")

    lines.append(
        "## Research Question"
    )
    lines.append("")
    lines.append(
        "Given a fixed approximately 47M parameter budget, "
        "what allocation between matrix memory, vector/Mamba "
        "pathway, router, feedback bridge, embeddings, and "
        "output head gives the lowest validation/test BPC?"
    )
    lines.append("")

    lines.append(
        "This audit establishes the baseline allocation before "
        "any parameter reallocation experiment."
    )
    lines.append("")

    lines.append(
        "## Baseline Verification"
    )
    lines.append("")

    baseline_verified = (
        "PASS"
        if result.baseline_parameter_count_verified
        else "FAIL"
    )

    topology_verified = (
        "PASS"
        if result.topology_identical
        else "FAIL"
    )

    lines.append(
        markdown_table(
            [
                "Check",
                "Result",
            ],
            [
                [
                    "Expected baseline parameter count",
                    format_int(
                        EXPECTED_BASELINE_PARAMETER_COUNT
                    ),
                ],
                [
                    "Baseline count verification",
                    baseline_verified,
                ],
                [
                    "Seed topology verification",
                    topology_verified,
                ],
            ],
        )
    )

    lines.append("")

    lines.append(
        "## Checkpoints"
    )
    lines.append("")

    checkpoint_rows: list[
        list[str]
    ] = []

    for audit in (
        result.seed_audits
    ):
        checkpoint_rows.append(
            [
                audit.seed,
                audit.checkpoint_path,
                format_int(
                    audit.parameter_count
                ),
                format_int(
                    audit.parameter_tensor_count
                ),
                str(
                    audit.checkpoint_step
                ),
            ]
        )

    lines.append(
        markdown_table(
            [
                "Seed",
                "Checkpoint",
                "Parameters",
                "Parameter tensors",
                "Step",
            ],
            checkpoint_rows,
        )
    )

    lines.append("")

    lines.append(
        "## Baseline Parameter Allocation"
    )
    lines.append("")

    component_rows: list[
        list[str]
    ] = []

    for audit in (
        result.seed_audits
    ):
        for summary in (
            audit.component_summaries
        ):
            component_rows.append(
                [
                    audit.seed,
                    summary.display_name,
                    format_int(
                        summary.parameter_count
                    ),
                    format_percent(
                        summary.parameter_fraction
                    ),
                    format_int(
                        summary.parameter_tensors
                    ),
                    format_int(
                        summary.layer_parameter_count
                    ),
                    format_int(
                        summary.global_parameter_count
                    ),
                ]
            )

    lines.append(
        markdown_table(
            [
                "Seed",
                "Component",
                "Parameters",
                "% of total",
                "Tensors",
                "Layer params",
                "Global params",
            ],
            component_rows,
        )
    )

    lines.append("")

    lines.append(
        "## Cross-Seed Component Consistency"
    )
    lines.append("")

    comparison_rows: list[
        list[str]
    ] = []

    for comparison in (
        result.seed_comparisons
    ):
        relative_difference = (
            "N/A"
            if comparison.relative_difference_percent
            is None
            else (
                f"{comparison.relative_difference_percent:.6f}%"
            )
        )

        comparison_rows.append(
            [
                comparison.display_name,
                format_int(
                    comparison.seed1_parameters
                ),
                format_int(
                    comparison.seed2_parameters
                ),
                format_int(
                    comparison.absolute_difference
                ),
                relative_difference,
                (
                    "PASS"
                    if comparison.topology_consistent
                    else "FAIL"
                ),
            ]
        )

    lines.append(
        markdown_table(
            [
                "Component",
                "Seed1",
                "Seed2",
                "Difference",
                "Relative difference",
                "Consistent",
            ],
            comparison_rows,
        )
    )

    lines.append("")

    lines.append(
        "## Allocation Vector"
    )
    lines.append("")

    lines.append(
        "The baseline allocation vector is represented as:"
    )
    lines.append("")
    lines.append(
        "`[matrix_memory, vector_mamba, router, "
        "feedback_bridge, embeddings, output_head, "
        "normalization, auxiliary, other]`"
    )
    lines.append("")

    for audit in (
        result.seed_audits
    ):
        component_map = {
            summary.component: summary
            for summary
            in audit.component_summaries
        }

        vector = [
            component_map[
                component
            ].parameter_count
            for component
            in COMPONENT_ORDER
        ]

        lines.append(
            f"### {audit.seed.upper()}"
        )
        lines.append("")
        lines.append(
            "`"
            + ", ".join(
                format_int(value)
                for value
                in vector
            )
            + "`"
        )
        lines.append("")

    lines.append(
        "## Interpretation Boundary"
    )
    lines.append("")

    lines.append(
        "PB0-C measures where parameters are allocated. "
        "It does not determine which component should receive "
        "more or fewer parameters."
    )
    lines.append("")

    lines.append(
        "That causal question requires the subsequent fixed-budget "
        "allocation matrix and empirical validation/test BPC "
        "experiments."
    )
    lines.append("")

    lines.append(
        "## Next Experimental Gate"
    )
    lines.append("")

    lines.append(
        "PB0-D will use this exact baseline allocation to construct "
        "architecturally valid candidates under the fixed parameter "
        f"budget of approximately {EXPECTED_BASELINE_PARAMETER_COUNT:,} "
        "parameters."
    )
    lines.append("")

    OUTPUT_MARKDOWN.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    args = parse_args()

    banner(
        "MODUS_X PB0-C PARAMETER BUDGET AUDIT"
    )

    print(
        f"Project root : {PROJECT_ROOT}"
    )
    print(
        f"Output dir   : {OUTPUT_DIR}"
    )
    print(
        "Expected parameter count : "
        f"{format_int(EXPECTED_BASELINE_PARAMETER_COUNT)}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    

    seed_directories: dict[
        str,
        Path,
    ] = {
        "seed1": args.seed1_dir,
        "seed2": args.seed2_dir,
    }

    checkpoint_paths: dict[
        str,
        Path,
    ] = {}

    for seed, directory in (
        seed_directories.items()
    ):
        checkpoint_paths[seed] = (
            validate_seed_directory(
                seed,
                directory,
            )
        )

    seed_audits: list[
        SeedAudit
    ] = []

    for seed, checkpoint_path in (
        checkpoint_paths.items()
    ):
        audit = audit_seed(
            seed=seed,
            checkpoint_path=(
                checkpoint_path
            ),
        )

        seed_audits.append(
            audit
        )

    if len(seed_audits) != 2:
        fail(
            "PB0-C requires exactly two seed audits."
        )

    seed1_audit = next(
        (
            audit
            for audit
            in seed_audits
            if audit.seed == "seed1"
        ),
        None,
    )

    seed2_audit = next(
        (
            audit
            for audit
            in seed_audits
            if audit.seed == "seed2"
        ),
        None,
    )

    if (
        seed1_audit is None
        or seed2_audit is None
    ):
        fail(
            "Unable to locate both Seed1 and Seed2 audit results."
        )

    banner(
        "VERIFYING PARAMETER TOPOLOGY"
    )

    topology_identical = (
        compare_parameter_topology(
            seed1_audit,
            seed2_audit,
        )
    )

    print(
        "Seed parameter topology: PASS"
    )

    seed1_count = (
        seed1_audit.parameter_count
    )

    seed2_count = (
        seed2_audit.parameter_count
    )

    baseline_parameter_count_verified = (
        seed1_count
        == EXPECTED_BASELINE_PARAMETER_COUNT
        and seed2_count
        == EXPECTED_BASELINE_PARAMETER_COUNT
    )

    if (
        not baseline_parameter_count_verified
        and not args.allow_unexpected_count
    ):
        fail(
            "Checkpoint parameter count does not match the "
            "expected PB0 baseline.\n\n"
            f"Expected: {format_int(EXPECTED_BASELINE_PARAMETER_COUNT)}\n"
            f"Seed1  : {format_int(seed1_count)}\n"
            f"Seed2  : {format_int(seed2_count)}\n\n"
            "Use --allow-unexpected-count only if this checkpoint "
            "is intentionally a different baseline."
        )

    banner(
        "BUILDING CROSS-SEED COMPARISON"
    )

    comparisons = (
        build_seed_comparisons(
            seed1_audit,
            seed2_audit,
        )
    )

    for comparison in comparisons:
        print(
            f"{comparison.display_name:24s} | "
            f"Seed1 {format_int(comparison.seed1_parameters):>14s} | "
            f"Seed2 {format_int(comparison.seed2_parameters):>14s} | "
            f"Difference {format_int(comparison.absolute_difference):>10s}"
        )

    result = AuditResult(
        seed_audits=tuple(
            seed_audits
        ),
        seed_comparisons=tuple(
            comparisons
        ),
        topology_identical=(
            topology_identical
        ),
        baseline_parameter_count_verified=(
            baseline_parameter_count_verified
        ),
    )

    banner(
        "WRITING OUTPUTS"
    )

    write_components_csv(
        seed_audits
    )

    write_parameters_csv(
        seed_audits
    )

    write_by_layer_csv(
        seed_audits
    )

    write_seed_comparison_csv(
        comparisons
    )

    write_json_report(
        result
    )

    write_markdown_report(
        result
    )

    banner(
        "PB0-C PARAMETER BUDGET AUDIT COMPLETE"
    )

    print()
    print(
        "Seed1 parameters : "
        f"{format_int(seed1_count)}"
    )
    print(
        "Seed2 parameters : "
        f"{format_int(seed2_count)}"
    )
    print(
        "Expected baseline: "
        f"{format_int(EXPECTED_BASELINE_PARAMETER_COUNT)}"
    )
    print()
    print(
        "Topology identical : "
        f"{topology_identical}"
    )
    print(
        "Baseline verified  : "
        f"{baseline_parameter_count_verified}"
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
        f"  {OUTPUT_COMPONENTS_CSV}"
    )
    print(
        f"  {OUTPUT_PARAMETERS_CSV}"
    )
    print(
        f"  {OUTPUT_BY_LAYER_CSV}"
    )
    print(
        f"  {OUTPUT_SEED_COMPARISON_CSV}"
    )

    print()
    print(
        "NEXT GATE: use the exact baseline component allocation "
        "to construct PB0-D fixed-budget parameter allocation "
        "candidates."
    )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        banner(
            "PB0-C AUDIT INTERRUPTED"
        )

        print(
            "The parameter budget audit was interrupted by the user."
        )

        sys.exit(
            130
        )

    except Exception as exc:
        banner(
            "PB0-C AUDIT FAILED"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )
        print()

        traceback.print_exc()

        sys.exit(
            1
        )