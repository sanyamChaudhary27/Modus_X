from __future__ import annotations

import csv
import inspect
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
        "Canonical model file was not found.\n"
        f"Expected: {MODELS_PATH}\n"
        f"Resolved repository root: {REPOSITORY_ROOT}"
    )

if str(LANGUAGE_DIR) not in sys.path:
    sys.path.insert(0, str(LANGUAGE_DIR))


try:
    from models import ModelConfig
    from models import count_params
    from models import make_model
except ImportError as exc:
    raise ImportError(
        "Failed to import the canonical language/models.py module."
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

SEQUENCE_LENGTH = 512
BATCH_SIZE = 8


# ============================================================================
# Data structures
# ============================================================================

@dataclass(frozen=True)
class StateLeaf:
    path: str
    shape: list[int]
    dtype: str
    elements: int
    bytes: int
    component: str
    layer: int | None


# ============================================================================
# Utility functions
# ============================================================================

def path_to_string(path: tuple[Any, ...]) -> str:
    """
    Convert a JAX pytree path into a stable human-readable path.
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


def extract_layer_number(path: str) -> int | None:
    """
    Extract the first integer following a 'layers' path component.

    Supports paths such as:
        layers.0.current
        layers.11.archive
        state.layers.3.vector
    """

    parts = path.replace("[", ".").replace("]", "").split(".")

    for index, part in enumerate(parts):
        if part.lower() == "layers" and index + 1 < len(parts):
            candidate = parts[index + 1]

            if candidate.isdigit():
                return int(candidate)

    return None


def classify_state(path: str, shape: list[int]) -> str:
    """
    Classify recurrent-state leaves.

    The classification is intentionally conservative. Unknown state leaves are
    explicitly assigned to 'unclassified' so that B1 never silently hides state.
    """

    normalized = path.lower()

    # Current associative matrix memory.
    current_markers = (
        "current",
        "h_current",
        "current_state",
    )

    # Archive associative matrix memory.
    archive_markers = (
        "archive",
        "h_archive",
        "archive_state",
    )

    # Vector/selective recurrent state.
    vector_markers = (
        "vector",
        "mamba",
        "ssm",
        "selective",
        "recurrent",
    )

    if any(marker in normalized for marker in archive_markers):
        return "archive_matrix_state"

    if any(marker in normalized for marker in current_markers):
        return "current_matrix_state"

    if any(marker in normalized for marker in vector_markers):
        return "vector_recurrent_state"

    # Shape-based fallback for canonical Modus_X architecture.
    #
    # A matrix state is expected to have at least two trailing dimensions.
    # We do not automatically classify all 2D arrays because batch/vector
    # states may also be 2D. This fallback is only used after explicit path
    # markers have failed.
    if len(shape) >= 3 and shape[-1] == 512 and shape[-2] == 512:
        return "matrix_state_unresolved"

    if len(shape) >= 2 and shape[-1] == 512:
        return "vector_state_unresolved"

    return "unclassified"


def collect_state_leaves(state: Any) -> list[StateLeaf]:
    """
    Traverse the full recurrent state pytree.
    """

    leaves_with_paths, _ = jax.tree_util.tree_flatten_with_path(state)

    records: list[StateLeaf] = []

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

        component = classify_state(
            path=path,
            shape=shape,
        )

        records.append(
            StateLeaf(
                path=path,
                shape=shape,
                dtype=dtype,
                elements=elements,
                bytes=elements * item_size,
                component=component,
                layer=extract_layer_number(path),
            )
        )

    records.sort(key=lambda record: record.path)

    if not records:
        raise RuntimeError(
            "The state census found zero array leaves. "
            "The model did not expose a recurrent state in the inspected result."
        )

    return records


# ============================================================================
# Model construction
# ============================================================================

def build_baseline_config() -> ModelConfig:
    """
    Construct the frozen canonical B0 configuration.
    """

    return ModelConfig(**BASELINE_CONFIG)


def build_baseline_model() -> tuple[Any, ModelConfig, str]:
    """
    Construct the exact canonical 47,437,768 parameter model.

    The function accepts the first return value from make_model because the
    B1 script subsequently inspects the model object's callable interface.
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
                failures.append(
                    f"{model_name}: {type(exc).__name__}: {exc}"
                )
                continue
        except Exception as exc:
            failures.append(
                f"{model_name}: {type(exc).__name__}: {exc}"
            )
            continue

        if not isinstance(result, tuple) or len(result) < 2:
            failures.append(
                f"{model_name}: make_model returned unexpected result "
                f"of type {type(result).__name__}"
            )
            continue

        params = result[0]
        model = result[1]

        actual_parameter_count = int(count_params(params))

        if actual_parameter_count == EXPECTED_PARAMETER_COUNT:
            return (params, model), cfg, model_name

        failures.append(
            f"{model_name}: parameter count "
            f"{actual_parameter_count:,}, expected "
            f"{EXPECTED_PARAMETER_COUNT:,}"
        )

    failure_text = "\n".join(
        f"  - {failure}"
        for failure in failures
    )

    raise RuntimeError(
        "Could not instantiate the canonical baseline.\n\n"
        f"{failure_text}"
    )


# ============================================================================
# Stateful API discovery
# ============================================================================

def describe_model_api(model: Any) -> list[str]:
    """
    Return callable public attributes that may expose recurrent-state execution.
    """

    candidates: list[str] = []

    for name in dir(model):
        if name.startswith("_"):
            continue

        try:
            attribute = getattr(model, name)
        except Exception:
            continue

        if callable(attribute):
            candidates.append(name)

    return sorted(candidates)


def inspect_signature(callable_object: Any) -> str:
    """
    Return a safe textual signature for diagnostics.
    """

    try:
        return str(inspect.signature(callable_object))
    except (TypeError, ValueError):
        return "<signature unavailable>"


def build_dummy_inputs() -> jax.Array:
    """
    Create a valid byte-token batch for state initialization or forward calls.
    """

    return jnp.zeros(
        (
            BATCH_SIZE,
            SEQUENCE_LENGTH,
        ),
        dtype=jnp.int32,
    )


def try_state_initializers(
    model: Any,
    params: Any,
) -> tuple[Any, str] | None:
    """
    Probe explicit state-initialization APIs.

    Only methods with names strongly suggesting state initialization are tried.
    Every attempt is recorded by the caller if all fail.
    """

    method_names = (
        "init_state",
        "initial_state",
        "initialize_state",
        "make_initial_state",
        "get_initial_state",
    )

    inputs = build_dummy_inputs()

    argument_variants = (
        tuple(),
        (BATCH_SIZE,),
        (inputs,),
        (params,),
        (params, BATCH_SIZE),
        (params, inputs),
    )

    for method_name in method_names:
        if not hasattr(model, method_name):
            continue

        method = getattr(model, method_name)

        if not callable(method):
            continue

        for arguments in argument_variants:
            try:
                state = method(*arguments)

                if state is not None:
                    return state, method_name
            except Exception:
                continue

    return None


def extract_state_from_forward_result(result: Any) -> Any | None:
    """
    Identify a likely recurrent state from a model forward result.

    This intentionally handles common structures:
        logits
        (logits, state)
        (logits, auxiliary, state)
        dictionary containing state
    """

    if isinstance(result, dict):
        state_keys = (
            "state",
            "states",
            "recurrent_state",
            "recurrent_states",
            "next_state",
            "next_states",
        )

        for key in state_keys:
            if key in result:
                return result[key]

        return None

    if isinstance(result, tuple):
        for value in reversed(result):
            if isinstance(value, (dict, tuple, list)):
                return value

    return None


def try_forward_state_extraction(
    model: Any,
    params: Any,
) -> tuple[Any, str] | None:
    """
    Probe common model-call conventions for recurrent state output.

    This is diagnostic-only. The script never trains or mutates parameters.
    """

    inputs = build_dummy_inputs()

    callable_candidates: list[tuple[str, Any]] = []

    if callable(model):
        callable_candidates.append(("model.__call__", model))

    for method_name in (
        "forward",
        "apply",
        "run",
        "step",
    ):
        if hasattr(model, method_name):
            method = getattr(model, method_name)

            if callable(method):
                callable_candidates.append(
                    (f"model.{method_name}", method)
                )

    argument_variants = (
        (params, inputs),
        (inputs, params),
        (inputs,),
        (params, inputs, None),
        (inputs, None),
    )

    for callable_name, callable_object in callable_candidates:
        for arguments in argument_variants:
            try:
                result = callable_object(*arguments)
            except Exception:
                continue

            state = extract_state_from_forward_result(result)

            if state is not None:
                return state, callable_name

    return None


def discover_recurrent_state(
    model: Any,
    params: Any,
) -> tuple[Any, str]:
    """
    Discover the canonical recurrent-state representation without modifying the
    model implementation.

    Preference order:
        1. explicit state initializer;
        2. state returned by a forward call.

    If the repository uses a different API, the script prints the available
    model callables and fails rather than guessing a state representation.
    """

    initialized = try_state_initializers(
        model=model,
        params=params,
    )

    if initialized is not None:
        state, source = initialized
        return state, source

    extracted = try_forward_state_extraction(
        model=model,
        params=params,
    )

    if extracted is not None:
        state, source = extracted
        return state, source

    api = describe_model_api(model)

    api_text = "\n".join(
        f"  - {name}: {inspect_signature(getattr(model, name))}"
        for name in api
    )

    raise RuntimeError(
        "Could not automatically discover the recurrent-state API.\n\n"
        "Available public callable model attributes:\n"
        f"{api_text}\n\n"
        "Do not manually invent a state tensor. The B1 census must inspect "
        "the exact canonical recurrent state used by language/models.py."
    )


# ============================================================================
# Aggregation
# ============================================================================

def aggregate_by_component(
    records: list[StateLeaf],
) -> dict[str, dict[str, int | float]]:
    """
    Aggregate state resources by component.
    """

    totals: dict[str, dict[str, int | float]] = defaultdict(
        lambda: {
            "elements": 0,
            "bytes": 0,
            "leaf_count": 0,
        }
    )

    total_elements = sum(
        record.elements
        for record in records
    )

    for record in records:
        bucket = totals[record.component]

        bucket["elements"] = (
            int(bucket["elements"])
            + record.elements
        )

        bucket["bytes"] = (
            int(bucket["bytes"])
            + record.bytes
        )

        bucket["leaf_count"] = (
            int(bucket["leaf_count"])
            + 1
        )

    summary: dict[str, dict[str, int | float]] = {}

    for component, values in totals.items():
        elements = int(values["elements"])

        summary[component] = {
            "elements": elements,
            "percentage": (
                (elements / total_elements) * 100.0
                if total_elements > 0
                else 0.0
            ),
            "bytes": int(values["bytes"]),
            "leaf_count": int(values["leaf_count"]),
        }

    return summary


def aggregate_by_layer(
    records: list[StateLeaf],
) -> dict[str, dict[str, Any]]:
    """
    Aggregate state resources by layer.

    Global state entries without a layer identifier are kept under 'global'.
    """

    totals: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "elements": 0,
            "bytes": 0,
            "components": defaultdict(int),
        }
    )

    for record in records:
        layer_key = (
            str(record.layer)
            if record.layer is not None
            else "global"
        )

        bucket = totals[layer_key]

        bucket["elements"] += record.elements
        bucket["bytes"] += record.bytes
        bucket["components"][record.component] += record.elements

    normalized: dict[str, dict[str, Any]] = {}

    for layer_key, values in sorted(
        totals.items(),
        key=lambda item: (
            item[0] == "global",
            int(item[0]) if item[0].isdigit() else -1,
        ),
    ):
        normalized[layer_key] = {
            "elements": int(values["elements"]),
            "bytes": int(values["bytes"]),
            "components": dict(
                sorted(
                    values["components"].items()
                )
            ),
        }

    return normalized


# ============================================================================
# Validation
# ============================================================================

def validate_state_census(
    records: list[StateLeaf],
) -> None:
    """
    Enforce accounting invariants.
    """

    total_elements = sum(
        record.elements
        for record in records
    )

    if total_elements <= 0:
        raise AssertionError(
            "State census contains no state elements."
        )

    unresolved = [
        record
        for record in records
        if record.component in {
            "matrix_state_unresolved",
            "vector_state_unresolved",
            "unclassified",
        }
    ]

    if unresolved:
        details = "\n".join(
            f"  - {record.path} | "
            f"shape={record.shape} | "
            f"component={record.component} | "
            f"elements={record.elements:,}"
            for record in unresolved
        )

        raise AssertionError(
            "State classification is incomplete.\n"
            "B1 refuses to publish a resource allocation summary with "
            "unresolved recurrent-state leaves.\n\n"
            f"{details}"
        )


# ============================================================================
# Output writers
# ============================================================================

def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """
    Write UTF-8 JSON.
    """

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


def write_csv(
    path: Path,
    records: list[StateLeaf],
) -> None:
    """
    Write complete state-leaf census.
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
                "layer",
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
                    "layer": (
                        record.layer
                        if record.layer is not None
                        else ""
                    ),
                }
            )


def write_markdown_report(
    path: Path,
    model_name: str,
    cfg: ModelConfig,
    state_source: str,
    records: list[StateLeaf],
    component_summary: dict[str, dict[str, int | float]],
    layer_summary: dict[str, dict[str, Any]],
) -> None:
    """
    Write human-readable B1 state report.
    """

    total_elements = sum(
        record.elements
        for record in records
    )

    total_bytes = sum(
        record.bytes
        for record in records
    )

    bytes_per_lane = total_bytes // BATCH_SIZE
    elements_per_lane = total_elements // BATCH_SIZE

    lines: list[str] = []

    lines.append("# B1 Recurrent State Census")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append("**B1 STATE ACCOUNTING PASS**")
    lines.append("")

    lines.append("## Canonical baseline")
    lines.append("")
    lines.append(f"- Model: `{BASELINE_MODEL_NAME}`")
    lines.append(f"- Constructor key: `{model_name}`")
    lines.append(
        f"- Parameter count: `{EXPECTED_PARAMETER_COUNT:,}`"
    )
    lines.append(f"- State discovery source: `{state_source}`")
    lines.append("")

    lines.append("## Runtime census configuration")
    lines.append("")
    lines.append(f"- Batch lanes: `{BATCH_SIZE}`")
    lines.append(
        f"- Segment length: `{SEQUENCE_LENGTH}` bytes"
    )
    lines.append(f"- Layers: `{BASELINE_CONFIG['n_layers']}`")
    lines.append("")

    lines.append("## State totals")
    lines.append("")
    lines.append(f"- Total state elements: `{total_elements:,}`")
    lines.append(f"- Total state bytes: `{total_bytes:,}`")
    lines.append(
        f"- Approximate elements per lane: `{elements_per_lane:,}`"
    )
    lines.append(
        f"- Approximate bytes per lane: `{bytes_per_lane:,}`"
    )
    lines.append("")

    lines.append("## State allocation by component")
    lines.append("")
    lines.append(
        "| Component | Elements | Share of state | Leaves | Bytes |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|"
    )

    for component, values in sorted(
        component_summary.items(),
        key=lambda item: int(item[1]["elements"]),
        reverse=True,
    ):
        lines.append(
            f"| `{component}` | "
            f"{int(values['elements']):,} | "
            f"{float(values['percentage']):.4f}% | "
            f"{int(values['leaf_count']):,} | "
            f"{int(values['bytes']):,} |"
        )

    lines.append("")
    lines.append("## State allocation by layer")
    lines.append("")
    lines.append(
        "| Layer | Total elements | Total bytes | Component elements |"
    )
    lines.append(
        "|---|---:|---:|---|"
    )

    for layer_key, values in layer_summary.items():
        component_text = "<br>".join(
            f"`{component}`: {elements:,}"
            for component, elements in sorted(
                values["components"].items()
            )
        )

        lines.append(
            f"| `{layer_key}` | "
            f"{values['elements']:,} | "
            f"{values['bytes']:,} | "
            f"{component_text} |"
        )

    lines.append("")
    lines.append("## Frozen model configuration")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---:|")

    for field_name, field_value in asdict(cfg).items():
        lines.append(
            f"| `{field_name}` | `{field_value}` |"
        )

    lines.append("")
    lines.append("## Interpretation boundary")
    lines.append("")
    lines.append(
        "This report measures runtime recurrent-state resources. "
        "It does not establish that a larger state component is inefficient "
        "or that shrinking it improves BPC. Those claims require matched "
        "parameter-budget experiments."
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
    print("MODUS_X B1 RECURRENT STATE CENSUS")
    print("=" * 80)
    print()
    print(f"Repository root: {REPOSITORY_ROOT}")
    print(f"Canonical model: {MODELS_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    (params, model), cfg, model_name = build_baseline_model()

    parameter_count = int(
        count_params(params)
    )

    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise AssertionError(
            "Canonical parameter count changed during B1.\n"
            f"Expected: {EXPECTED_PARAMETER_COUNT:,}\n"
            f"Actual:   {parameter_count:,}"
        )

    print("Canonical baseline instantiated successfully.")
    print(f"Constructor key: {model_name}")
    print(f"Parameter count: {parameter_count:,}")
    print()

    print("Discovering canonical recurrent-state representation...")

    state, state_source = discover_recurrent_state(
        model=model,
        params=params,
    )

    print(f"State discovery source: {state_source}")

    records = collect_state_leaves(state)

    validate_state_census(records)

    component_summary = aggregate_by_component(records)
    layer_summary = aggregate_by_layer(records)

    total_elements = sum(
        record.elements
        for record in records
    )

    total_bytes = sum(
        record.bytes
        for record in records
    )

    generated_at = datetime.now(
        timezone.utc
    ).isoformat()

    json_payload: dict[str, Any] = {
        "report": "B1 Recurrent State Census",
        "generated_at_utc": generated_at,
        "baseline_model_name": BASELINE_MODEL_NAME,
        "constructor_key": model_name,
        "expected_parameter_count": EXPECTED_PARAMETER_COUNT,
        "actual_parameter_count": parameter_count,
        "state_discovery_source": state_source,
        "runtime_configuration": {
            "batch_size": BATCH_SIZE,
            "sequence_length": SEQUENCE_LENGTH,
            "n_layers": BASELINE_CONFIG["n_layers"],
        },
        "config": asdict(cfg),
        "total_state_elements": total_elements,
        "total_state_bytes": total_bytes,
        "component_summary": component_summary,
        "layer_summary": layer_summary,
        "state_leaves": [
            asdict(record)
            for record in records
        ],
    }

    json_path = (
        OUTPUT_DIR
        / "b1_state_census.json"
    )

    csv_path = (
        OUTPUT_DIR
        / "b1_state_leaves.csv"
    )

    markdown_path = (
        OUTPUT_DIR
        / "B1_STATE_CENSUS_REPORT.md"
    )

    write_json(
        json_path,
        json_payload,
    )

    write_csv(
        csv_path,
        records,
    )

    write_markdown_report(
        path=markdown_path,
        model_name=model_name,
        cfg=cfg,
        state_source=state_source,
        records=records,
        component_summary=component_summary,
        layer_summary=layer_summary,
    )

    print()
    print("=" * 80)
    print("B1 STATE ACCOUNTING PASS")
    print("=" * 80)
    print()

    print(
        f"Total state elements: {total_elements:,}"
    )

    print(
        f"Total state bytes:    {total_bytes:,}"
    )

    print(
        f"State leaves:         {len(records):,}"
    )

    print()

    print("State allocation summary:")

    for component, values in sorted(
        component_summary.items(),
        key=lambda item: int(item[1]["elements"]),
        reverse=True,
    ):
        print(
            f"  {component:28s} "
            f"{int(values['elements']):12,} "
            f"{float(values['percentage']):8.4f}%"
        )

    print()
    print("Output files:")
    print(f"  JSON:     {json_path}")
    print(f"  CSV:      {csv_path}")
    print(f"  Markdown: {markdown_path}")


if __name__ == "__main__":
    main()