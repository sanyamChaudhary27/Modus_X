from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp


# ============================================================================
# Repository paths
# ============================================================================

SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[2]
LANGUAGE_DIR = REPOSITORY_ROOT / "language"
MODELS_PATH = LANGUAGE_DIR / "models.py"
OUTPUT_DIR = SCRIPT_PATH.parent / "outputs"

if not MODELS_PATH.exists():
    raise FileNotFoundError(
        f"Canonical model file was not found.\n"
        f"Expected: {MODELS_PATH}\n"
        f"Repository root resolved as: {REPOSITORY_ROOT}"
    )

if str(LANGUAGE_DIR) not in sys.path:
    sys.path.insert(0, str(LANGUAGE_DIR))


try:
    from models import ModelConfig
    from models import count_params
    from models import make_model
except ImportError as exc:
    raise ImportError(
        "Failed to import the canonical language/models.py module. "
        "Run this script from the Modus_X repository without moving "
        "language/models.py."
    ) from exc


# ============================================================================
# Frozen B0 baseline configuration
# ============================================================================

EXPECTED_PARAMETER_COUNT = 47_437_768

BASELINE_CONFIG = {
    "vocab_size": 256,
    "embed_dim": 512,
    "hidden_dim": 1536,
    "ax_res": 512,
    "n_layers": 12,
    "n_heads_attn": 8,
    "seq_len": 512,
    "mamba_state_dim": 512,
    "vector_router": False,
    "router_hidden": 32,
}

BASELINE_MODEL_NAME = "Modus_X_MemoryFeedbackArchive_DeepSupervision"

BASELINE_MODEL_KEY_CANDIDATES = (
    "memory_feedback_archive",
    "modus_x_memory_feedback_archive",
    "Modus_X_MemoryFeedbackArchive",
)

BASELINE_SEED = 1
AUXILIARY_LAYERS = (6,)
FUTURE_TARGET_COUNT = 1


# ============================================================================
# Data structures
# ============================================================================

@dataclass(frozen=True)
class ParameterLeaf:
    path: str
    shape: list[int]
    dtype: str
    elements: int
    bytes: int
    component: str


# ============================================================================
# Parameter tree traversal
# ============================================================================

def path_to_string(path: tuple[Any, ...]) -> str:
    """
    Convert a JAX pytree path into a stable human-readable string.
    """

    parts: list[str] = []

    for entry in path:
        if hasattr(entry, "key"):
            parts.append(str(entry.key))
        elif hasattr(entry, "idx"):
            parts.append(str(entry.idx))
        elif hasattr(entry, "name"):
            parts.append(str(entry.name))
        else:
            parts.append(str(entry))

    return ".".join(parts)


def classify_parameter(path: str) -> str:
    """
    Classify every canonical parameter leaf into a B0 allocation component.

    Classification is based on the measured canonical parameter names exposed
    by language/models.py. Unknown parameters are deliberately assigned to
    'unclassified' so that no parameter can silently disappear from the B0
    accounting.
    """

    normalized = path.lower()
    leaf_name = normalized.split(".")[-1]

    # ------------------------------------------------------------------------
    # Global embedding
    # ------------------------------------------------------------------------

    if normalized == "embed":
        return "embedding"

    # ------------------------------------------------------------------------
    # Final language-model output head
    # ------------------------------------------------------------------------

    if normalized.startswith("head."):
        return "output_head"

    # ------------------------------------------------------------------------
    # Future prediction supervision heads
    # ------------------------------------------------------------------------

    if normalized.startswith("future_heads."):
        return "future_prediction_head"

    # ------------------------------------------------------------------------
    # Parameters outside the recurrent layer stack
    # ------------------------------------------------------------------------

    if not normalized.startswith("layers."):
        return "unclassified"

    # ------------------------------------------------------------------------
    # Pre-layer normalization
    #
    # Canonical parameter census exposed:
    #
    #   layers.pre_g
    #   layers.pre_b
    #
    # Both are 12 x 512 and belong to the layer normalization/preconditioning
    # stage rather than exclusively to matrix or vector recurrence.
    # ------------------------------------------------------------------------

    if leaf_name in {"pre_g", "pre_b"}:
        return "normalization"

    # ------------------------------------------------------------------------
    # Router pathway
    #
    # Canonical parameter census exposed:
    #
    #   layers.r_w
    #   layers.r_b
    #   layers.r_proj
    #   layers.r_proj_b
    #
    # The 32-wide dimension corresponds to the frozen router_hidden = 32
    # baseline configuration.
    # ------------------------------------------------------------------------

    if leaf_name in {
        "r_w",
        "r_b",
        "r_proj",
        "r_proj_b",
    }:
        return "router"

    # ------------------------------------------------------------------------
    # Memory-feedback bridge
    #
    # This check occurs before generic s_* vector classification because the
    # feedback bridge may use state-path projections.
    # ------------------------------------------------------------------------

    feedback_exact = {
        "s_memory_down",
        "s_memory_up",
        "s_w_memory_feedback",
        "s_b_memory_feedback",
    }

    if leaf_name in feedback_exact:
        return "feedback_bridge"

    # ------------------------------------------------------------------------
    # Archive-specific matrix memory controls
    # ------------------------------------------------------------------------

    if "archive" in normalized:
        return "archive_memory_control"

    # ------------------------------------------------------------------------
    # Matrix associative-memory pathway
    # ------------------------------------------------------------------------

    matrix_exact = {
        "m_wk",
        "m_wq",
        "m_wv",
        "m_w_write",
        "m_b_write",
        "m_w_ret",
        "m_b_ret",
        "m_ln_g",
        "m_ln_b",
        "m_proj_w",
        "m_proj_b",
    }

    if leaf_name in matrix_exact:
        return "matrix_memory"

    if leaf_name.startswith("m_"):
        return "matrix_memory"

    # ------------------------------------------------------------------------
    # Vector recurrent / Mamba-style pathway
    # ------------------------------------------------------------------------

    if leaf_name.startswith("s_"):
        return "vector_recurrence"

    if "mamba" in normalized:
        return "vector_recurrence"

    # ------------------------------------------------------------------------
    # Remaining normalization parameters
    # ------------------------------------------------------------------------

    if "norm" in leaf_name or "ln_" in leaf_name:
        return "normalization"

    return "unclassified"


def collect_parameter_leaves(params: Any) -> list[ParameterLeaf]:
    """
    Traverse the complete JAX parameter pytree and return one record per leaf.
    """

    leaves_with_paths, _ = jax.tree_util.tree_flatten_with_path(params)

    records: list[ParameterLeaf] = []

    for path_entries, leaf in leaves_with_paths:
        if not hasattr(leaf, "shape") or not hasattr(leaf, "size"):
            continue

        path = path_to_string(tuple(path_entries))
        shape = [int(dimension) for dimension in leaf.shape]
        elements = int(leaf.size)

        if hasattr(leaf, "dtype"):
            dtype = str(leaf.dtype)
            item_size = int(jnp.dtype(leaf.dtype).itemsize)
        else:
            dtype = type(leaf).__name__
            item_size = 0

        records.append(
            ParameterLeaf(
                path=path,
                shape=shape,
                dtype=dtype,
                elements=elements,
                bytes=elements * item_size,
                component=classify_parameter(path),
            )
        )

    records.sort(key=lambda record: record.path)

    if not records:
        raise RuntimeError(
            "Parameter census found zero parameter leaves. "
            "The canonical model initialization did not return a valid parameter tree."
        )

    return records


# ============================================================================
# Model construction
# ============================================================================

def build_baseline_config() -> ModelConfig:
    """
    Construct the frozen historical 47.44M baseline configuration.
    """

    return ModelConfig(**BASELINE_CONFIG)


def build_baseline_model() -> tuple[dict[str, Any], ModelConfig, str]:
    """
    Construct the canonical MemoryFeedbackArchive model.

    The repository has historically used model-selection strings in make_model.
    Because release variants may differ in exact accepted aliases, this function
    tries only explicit MemoryFeedbackArchive candidates and verifies the exact
    historical parameter count afterward.
    """

    cfg = build_baseline_config()
    key = jax.random.key(BASELINE_SEED)

    failures: list[str] = []

    for model_name in BASELINE_MODEL_KEY_CANDIDATES:
        try:
            result = make_model(
                model_name,
                key,
                cfg,
                auxiliary_layers=AUXILIARY_LAYERS,
                future_target_count=FUTURE_TARGET_COUNT,
                dropout_rate=0.0,
            )
        except TypeError:
            try:
                result = make_model(
                    model_name,
                    key,
                    cfg,
                    auxiliary_layers=AUXILIARY_LAYERS,
                    future_target_count=FUTURE_TARGET_COUNT,
                )
            except Exception as exc:
                failures.append(f"{model_name}: {type(exc).__name__}: {exc}")
                continue
        except Exception as exc:
            failures.append(f"{model_name}: {type(exc).__name__}: {exc}")
            continue

        if not isinstance(result, tuple) or len(result) < 1:
            failures.append(
                f"{model_name}: make_model returned an unexpected result "
                f"of type {type(result).__name__}"
            )
            continue

        params = result[0]
        actual_count = int(count_params(params))

        if actual_count == EXPECTED_PARAMETER_COUNT:
            return params, cfg, model_name

        failures.append(
            f"{model_name}: initialized successfully but produced "
            f"{actual_count:,} parameters instead of "
            f"{EXPECTED_PARAMETER_COUNT:,}"
        )

    failure_text = "\n".join(f"  - {failure}" for failure in failures)

    raise RuntimeError(
        "Could not instantiate the exact historical 47,437,768-parameter "
        "MemoryFeedbackArchive baseline.\n\n"
        "Attempted model keys:\n"
        f"{failure_text}\n\n"
        "Do not modify the expected parameter count to make this pass. "
        "The B0 census must first identify the exact canonical constructor "
        "used by this repository."
    )


# ============================================================================
# Aggregation
# ============================================================================

def aggregate_by_component(
    records: list[ParameterLeaf],
) -> dict[str, dict[str, int | float]]:
    """
    Aggregate parameter leaves into B0 allocation categories.
    """

    totals: dict[str, dict[str, int | float]] = defaultdict(
        lambda: {
            "parameters": 0,
            "bytes": 0,
            "leaf_count": 0,
        }
    )

    total_parameters = sum(record.elements for record in records)

    for record in records:
        bucket = totals[record.component]
        bucket["parameters"] = int(bucket["parameters"]) + record.elements
        bucket["bytes"] = int(bucket["bytes"]) + record.bytes
        bucket["leaf_count"] = int(bucket["leaf_count"]) + 1

    result: dict[str, dict[str, int | float]] = {}

    for component in sorted(totals):
        parameters = int(totals[component]["parameters"])

        result[component] = {
            "parameters": parameters,
            "percentage": (
                (parameters / total_parameters) * 100.0
                if total_parameters > 0
                else 0.0
            ),
            "bytes": int(totals[component]["bytes"]),
            "leaf_count": int(totals[component]["leaf_count"]),
        }

    return result


# ============================================================================
# Validation
# ============================================================================

def validate_census(
    records: list[ParameterLeaf],
    model_parameter_count: int,
) -> None:
    """
    Enforce B0 accounting invariants.
    """

    leaf_parameter_count = sum(record.elements for record in records)

    if leaf_parameter_count != model_parameter_count:
        raise AssertionError(
            "Parameter-tree accounting mismatch.\n"
            f"count_params(params): {model_parameter_count:,}\n"
            f"sum(census leaves):  {leaf_parameter_count:,}"
        )

    if model_parameter_count != EXPECTED_PARAMETER_COUNT:
        raise AssertionError(
            "Canonical baseline parameter count mismatch.\n"
            f"Expected: {EXPECTED_PARAMETER_COUNT:,}\n"
            f"Actual:   {model_parameter_count:,}\n\n"
            "B0 intentionally stops here because parameter allocation analysis "
            "must not be performed on a different model."
        )

    unclassified = [
        record
        for record in records
        if record.component == "unclassified"
    ]

    if unclassified:
        paths = "\n".join(
            f"  - {record.path} | shape={record.shape} | "
            f"parameters={record.elements:,}"
            for record in unclassified
        )

        raise AssertionError(
            "B0 found unclassified parameter leaves.\n"
            "Every parameter must be explicitly assigned before allocation "
            "percentages are treated as evidence.\n\n"
            f"{paths}"
        )


# ============================================================================
# Output writers
# ============================================================================

def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """
    Write deterministic UTF-8 JSON.
    """

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(
    path: Path,
    records: list[ParameterLeaf],
) -> None:
    """
    Write complete parameter-leaf census.
    """

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "path",
                "shape",
                "dtype",
                "elements",
                "bytes",
                "component",
            ),
        )

        writer.writeheader()

        for record in records:
            writer.writerow(
                {
                    "path": record.path,
                    "shape": json.dumps(record.shape),
                    "dtype": record.dtype,
                    "elements": record.elements,
                    "bytes": record.bytes,
                    "component": record.component,
                }
            )


def write_markdown_report(
    path: Path,
    model_name: str,
    cfg: ModelConfig,
    records: list[ParameterLeaf],
    summary: dict[str, dict[str, int | float]],
) -> None:
    """
    Write the human-readable B0 allocation report.
    """

    total_parameters = sum(record.elements for record in records)
    total_bytes = sum(record.bytes for record in records)

    lines: list[str] = []

    lines.append("# B0 Parameter Allocation Census")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append("**B0 PARAMETER ACCOUNTING PASS**")
    lines.append("")
    lines.append("## Canonical baseline")
    lines.append("")
    lines.append(f"- Model: `{BASELINE_MODEL_NAME}`")
    lines.append(f"- Constructor key: `{model_name}`")
    lines.append(f"- Expected parameters: `{EXPECTED_PARAMETER_COUNT:,}`")
    lines.append(f"- Actual parameters: `{total_parameters:,}`")
    lines.append(f"- Parameter storage bytes: `{total_bytes:,}`")
    lines.append("")
    lines.append("## Frozen architecture configuration")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---:|")

    for field_name, field_value in asdict(cfg).items():
        lines.append(f"| `{field_name}` | `{field_value}` |")

    lines.append("")
    lines.append("## Parameter allocation")
    lines.append("")
    lines.append(
        "| Component | Parameters | Share of total | Leaves | Parameter bytes |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|"
    )

    for component, values in sorted(
        summary.items(),
        key=lambda item: int(item[1]["parameters"]),
        reverse=True,
    ):
        lines.append(
            f"| `{component}` | "
            f"{int(values['parameters']):,} | "
            f"{float(values['percentage']):.4f}% | "
            f"{int(values['leaf_count']):,} | "
            f"{int(values['bytes']):,} |"
        )

    lines.append("")
    lines.append("## Accounting invariants")
    lines.append("")
    lines.append(
        f"- Sum of parameter census leaves: `{total_parameters:,}`"
    )
    lines.append(
        f"- Canonical expected total: `{EXPECTED_PARAMETER_COUNT:,}`"
    )
    lines.append(
        f"- Number of parameter leaves: `{len(records):,}`"
    )
    lines.append(
        "- Unclassified parameters: `0`"
    )
    lines.append("")
    lines.append("## Interpretation boundary")
    lines.append("")
    lines.append(
        "This B0 report is a static allocation census. It does not claim that "
        "a component with more parameters is inefficient or that reducing it "
        "will improve BPC. BPC conclusions require subsequent matched-budget "
        "training experiments."
    )
    lines.append("")

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("MODUS_X B0 PARAMETER ALLOCATION CENSUS")
    print("=" * 80)
    print()
    print(f"Repository root: {REPOSITORY_ROOT}")
    print(f"Canonical model: {MODELS_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    params, cfg, model_name = build_baseline_model()

    model_parameter_count = int(count_params(params))

    print("Baseline instantiated successfully.")
    print(f"Constructor key: {model_name}")
    print(f"Parameter count: {model_parameter_count:,}")
    print()

    records = collect_parameter_leaves(params)

    validate_census(
        records=records,
        model_parameter_count=model_parameter_count,
    )

    summary = aggregate_by_component(records)

    timestamp = datetime.now(timezone.utc).isoformat()

    json_payload: dict[str, Any] = {
        "report": "B0 Parameter Allocation Census",
        "generated_at_utc": timestamp,
        "baseline_model_name": BASELINE_MODEL_NAME,
        "constructor_key": model_name,
        "expected_parameter_count": EXPECTED_PARAMETER_COUNT,
        "actual_parameter_count": model_parameter_count,
        "config": asdict(cfg),
        "component_summary": summary,
        "parameter_leaves": [
            asdict(record)
            for record in records
        ],
    }

    json_path = OUTPUT_DIR / "b0_parameter_census.json"
    csv_path = OUTPUT_DIR / "b0_parameter_leaves.csv"
    markdown_path = OUTPUT_DIR / "B0_PARAMETER_ALLOCATION_REPORT.md"

    write_json(
        json_path,
        json_payload,
    )

    write_csv(
        csv_path,
        records,
    )

    write_markdown_report(
        markdown_path,
        model_name,
        cfg,
        records,
        summary,
    )

    print("=" * 80)
    print("B0 PARAMETER ACCOUNTING PASS")
    print("=" * 80)
    print()
    print(f"Expected parameters: {EXPECTED_PARAMETER_COUNT:,}")
    print(f"Actual parameters:   {model_parameter_count:,}")
    print(f"Parameter leaves:    {len(records):,}")
    print()

    print("Allocation summary:")

    for component, values in sorted(
        summary.items(),
        key=lambda item: int(item[1]["parameters"]),
        reverse=True,
    ):
        print(
            f"  {component:28s} "
            f"{int(values['parameters']):12,} "
            f"{float(values['percentage']):8.4f}%"
        )

    print()
    print("Output files:")
    print(f"  JSON:     {json_path}")
    print(f"  CSV:      {csv_path}")
    print(f"  Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
