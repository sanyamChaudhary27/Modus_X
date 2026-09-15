from __future__ import annotations

import csv
import importlib
import json
import pickle
import sys
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

CANONICAL_MODEL_PATH = PROJECT_ROOT / "language" / "models.py"
OUTPUT_DIRECTORY = PROJECT_ROOT / "research" / "parameter_allocation" / "outputs"

SEED_DIRECTORIES: dict[str, Path] = {
    "seed1": Path(r"E:\seed1"),
    "seed2": Path(r"E:\seed2"),
}

CHECKPOINT_FILENAME = "checkpoint.pkl"
CONFIG_FILENAME = "config.json"
PROGRESS_FILENAME = "progress.json"
ENDPOINT_DECISION_FILENAME = "ENDPOINT_DECISION.json"

EXPECTED_CANONICAL_PARAMETER_COUNT = 47_437_768

OUTPUT_JSON = OUTPUT_DIRECTORY / "PB0_CHECKPOINT_INSPECTION.json"
OUTPUT_MARKDOWN = OUTPUT_DIRECTORY / "PB0_CHECKPOINT_INSPECTION.md"
OUTPUT_PARAMETER_LEAVES = OUTPUT_DIRECTORY / "PB0_CHECKPOINT_PARAMETER_LEAVES.csv"


@dataclass(frozen=True)
class ParameterLeaf:
    path: str
    shape: tuple[int, ...]
    dtype: str
    elements: int
    nbytes: int


def fail(message: str) -> None:
    raise RuntimeError(message)


def ensure_output_directory() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)


def format_integer(value: int) -> str:
    return f"{value:,}"


def format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")

    numeric_value = float(value)
    unit_index = 0

    while numeric_value >= 1024.0 and unit_index < len(units) - 1:
        numeric_value /= 1024.0
        unit_index += 1

    if unit_index == 0:
        return f"{int(numeric_value)} {units[unit_index]}"

    return f"{numeric_value:.2f} {units[unit_index]}"


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, set):
        return sorted(value)

    return str(value)


def write_json(path: Path, payload: Any) -> None:
    try:
        with path.open("w", encoding="utf-8") as file:
            json.dump(
                payload,
                file,
                indent=2,
                sort_keys=True,
                default=json_default,
            )
            file.write("\n")
    except OSError as exc:
        fail(f"Could not write JSON output {path}: {exc}")


def load_json_if_present(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        fail(f"Could not parse JSON file {path}: {exc}")
    except OSError as exc:
        fail(f"Could not read JSON file {path}: {exc}")


def safe_type_name(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def load_checkpoint(path: Path) -> Any:
    if not path.exists():
        fail(
            f"Checkpoint does not exist: {path}\n"
            "Verify the seed directory and checkpoint.pkl location."
        )

    try:
        with path.open("rb") as file:
            return pickle.load(file)
    except ModuleNotFoundError as exc:
        fail(
            f"Checkpoint {path} requires a Python module that is not installed: "
            f"{exc.name!r}. The checkpoint is a pickle and must be inspected in "
            "an environment compatible with the training environment."
        )
    except ImportError as exc:
        fail(
            f"Checkpoint {path} could not import one of its required classes or "
            f"modules: {exc}"
        )
    except pickle.UnpicklingError as exc:
        fail(f"Checkpoint {path} is not a readable Python pickle: {exc}")
    except EOFError as exc:
        fail(f"Checkpoint {path} appears truncated or incomplete: {exc}")
    except OSError as exc:
        fail(f"Could not read checkpoint {path}: {exc}")


def flatten_tree(
    value: Any,
    prefix: str = "",
) -> list[tuple[str, Any]]:
    flattened: list[tuple[str, Any]] = []

    if isinstance(value, Mapping):
        if not value:
            flattened.append((prefix, value))
            return flattened

        for key in sorted(value.keys(), key=lambda item: str(item)):
            key_text = str(key)
            child_prefix = key_text if not prefix else f"{prefix}.{key_text}"
            flattened.extend(flatten_tree(value[key], child_prefix))

        return flattened

    if isinstance(value, tuple):
        if not value:
            flattened.append((prefix, value))
            return flattened

        for index, child in enumerate(value):
            child_prefix = str(index) if not prefix else f"{prefix}.{index}"
            flattened.extend(flatten_tree(child, child_prefix))

        return flattened

    if isinstance(value, list):
        if not value:
            flattened.append((prefix, value))
            return flattened

        for index, child in enumerate(value):
            child_prefix = str(index) if not prefix else f"{prefix}.{index}"
            flattened.extend(flatten_tree(child, child_prefix))

        return flattened

    flattened.append((prefix, value))
    return flattened


def is_array_like(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return True

    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return True

    return False


def to_numpy_array(value: Any) -> np.ndarray:
    try:
        array = np.asarray(value)
    except Exception as exc:
        fail(
            "Could not convert checkpoint leaf to a NumPy array. "
            f"Leaf type: {safe_type_name(value)}. Error: {exc}"
        )

    return array


def collect_parameter_leaves(params: Any) -> list[ParameterLeaf]:
    leaves: list[ParameterLeaf] = []

    for path, value in flatten_tree(params):
        if not is_array_like(value):
            continue

        array = to_numpy_array(value)

        if array.dtype == np.dtype("O"):
            continue

        shape = tuple(int(dimension) for dimension in array.shape)

        if shape:
            elements = int(np.prod(shape, dtype=np.int64))
        else:
            elements = 1

        leaves.append(
            ParameterLeaf(
                path=path,
                shape=shape,
                dtype=str(array.dtype),
                elements=elements,
                nbytes=int(array.nbytes),
            )
        )

    leaves.sort(key=lambda leaf: leaf.path)

    if not leaves:
        fail(
            "No array-like parameter leaves were found in checkpoint['params']. "
            "The checkpoint structure is incompatible with the expected "
            "JAX/NumPy parameter-tree format."
        )

    return leaves


def parameter_summary(leaves: Sequence[ParameterLeaf]) -> dict[str, int]:
    return {
        "leaf_count": len(leaves),
        "parameter_count": sum(leaf.elements for leaf in leaves),
        "parameter_bytes": sum(leaf.nbytes for leaf in leaves),
    }


def extract_params(checkpoint: Any) -> Any:
    if not isinstance(checkpoint, Mapping):
        fail(
            "Checkpoint root is not a mapping/dictionary. "
            f"Actual type: {safe_type_name(checkpoint)}."
        )

    if "params" not in checkpoint:
        available_keys = sorted(str(key) for key in checkpoint.keys())

        fail(
            "Checkpoint does not contain the required top-level 'params' key. "
            f"Available keys: {available_keys}"
        )

    return checkpoint["params"]


def summarize_checkpoint_root(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, Mapping):
        return {
            "root_type": safe_type_name(checkpoint),
            "top_level_keys": sorted(str(key) for key in checkpoint.keys()),
        }

    return {
        "root_type": safe_type_name(checkpoint),
        "top_level_keys": [],
    }


def extract_scalar_metadata(
    checkpoint: Any,
) -> dict[str, Any]:
    if not isinstance(checkpoint, Mapping):
        return {}

    metadata: dict[str, Any] = {}

    interesting_keys = (
        "step",
        "epoch",
        "iteration",
        "global_step",
        "train_step",
    )

    for key in interesting_keys:
        if key not in checkpoint:
            continue

        value = checkpoint[key]

        if isinstance(value, np.generic):
            value = value.item()

        if isinstance(value, (str, int, float, bool)) or value is None:
            metadata[key] = value
        else:
            metadata[key] = safe_type_name(value)

    return metadata


def compare_parameter_sets(
    first: Sequence[ParameterLeaf],
    second: Sequence[ParameterLeaf],
) -> dict[str, Any]:
    first_by_path = {leaf.path: leaf for leaf in first}
    second_by_path = {leaf.path: leaf for leaf in second}

    first_paths = set(first_by_path)
    second_paths = set(second_by_path)

    only_in_first = sorted(first_paths - second_paths)
    only_in_second = sorted(second_paths - first_paths)

    shared_paths = sorted(first_paths & second_paths)

    shape_mismatches: list[dict[str, Any]] = []
    dtype_mismatches: list[dict[str, Any]] = []

    for path in shared_paths:
        first_leaf = first_by_path[path]
        second_leaf = second_by_path[path]

        if first_leaf.shape != second_leaf.shape:
            shape_mismatches.append(
                {
                    "path": path,
                    "first_shape": list(first_leaf.shape),
                    "second_shape": list(second_leaf.shape),
                }
            )

        if first_leaf.dtype != second_leaf.dtype:
            dtype_mismatches.append(
                {
                    "path": path,
                    "first_dtype": first_leaf.dtype,
                    "second_dtype": second_leaf.dtype,
                }
            )

    return {
        "first_leaf_count": len(first),
        "second_leaf_count": len(second),
        "first_parameter_count": sum(leaf.elements for leaf in first),
        "second_parameter_count": sum(leaf.elements for leaf in second),
        "paths_only_in_first": only_in_first,
        "paths_only_in_second": only_in_second,
        "shape_mismatches": shape_mismatches,
        "dtype_mismatches": dtype_mismatches,
        "exact_structure_match": (
            not only_in_first
            and not only_in_second
            and not shape_mismatches
        ),
    }


def find_model_parameter_paths(
    leaves: Sequence[ParameterLeaf],
) -> dict[str, list[ParameterLeaf]]:
    targets = (
        "m_wq",
        "m_wk",
        "m_wv",
        "m_w_read",
        "m_w_out",
        "m_proj_w",
    )

    result: dict[str, list[ParameterLeaf]] = {
        target: []
        for target in targets
    }

    for leaf in leaves:
        path_parts = leaf.path.split(".")

        for target in targets:
            if target in path_parts or leaf.path.endswith(target):
                result[target].append(leaf)

    return result


def inspect_seed(
    seed_name: str,
    seed_directory: Path,
) -> dict[str, Any]:
    checkpoint_path = seed_directory / CHECKPOINT_FILENAME
    config_path = seed_directory / CONFIG_FILENAME
    progress_path = seed_directory / PROGRESS_FILENAME
    endpoint_decision_path = seed_directory / ENDPOINT_DECISION_FILENAME

    print()
    print("=" * 80)
    print(f"LOADING {seed_name.upper()}")
    print("=" * 80)
    print(f"Seed directory: {seed_directory}")
    print(f"Checkpoint    : {checkpoint_path}")

    checkpoint_size = checkpoint_path.stat().st_size if checkpoint_path.exists() else 0

    if checkpoint_path.exists():
        print(f"Size          : {format_bytes(checkpoint_size)}")

    checkpoint = load_checkpoint(checkpoint_path)

    root_summary = summarize_checkpoint_root(checkpoint)
    metadata = extract_scalar_metadata(checkpoint)
    params = extract_params(checkpoint)

    leaves = collect_parameter_leaves(params)
    summary = parameter_summary(leaves)
    target_parameters = find_model_parameter_paths(leaves)

    print()
    print("Checkpoint loaded successfully.")
    print(f"Root type              : {root_summary['root_type']}")
    print(f"Top-level keys         : {root_summary['top_level_keys']}")
    print(f"Parameter leaves       : {format_integer(summary['leaf_count'])}")
    print(f"Parameter count        : {format_integer(summary['parameter_count'])}")
    print(f"Parameter array bytes  : {format_bytes(summary['parameter_bytes'])}")

    if metadata:
        print(f"Scalar metadata        : {metadata}")

    if summary["parameter_count"] == EXPECTED_CANONICAL_PARAMETER_COUNT:
        print(
            "Canonical parameter count: MATCH "
            f"({format_integer(EXPECTED_CANONICAL_PARAMETER_COUNT)})"
        )
    else:
        difference = (
            summary["parameter_count"]
            - EXPECTED_CANONICAL_PARAMETER_COUNT
        )

        print(
            "Canonical parameter count: DIFFERENT "
            f"(expected {format_integer(EXPECTED_CANONICAL_PARAMETER_COUNT)}, "
            f"difference {difference:+,})"
        )

    return {
        "seed_name": seed_name,
        "seed_directory": str(seed_directory),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_size,
        "checkpoint_size_human": format_bytes(checkpoint_size),
        "checkpoint_root": root_summary,
        "scalar_metadata": metadata,
        "parameter_summary": summary,
        "canonical_parameter_count": EXPECTED_CANONICAL_PARAMETER_COUNT,
        "canonical_parameter_count_difference": (
            summary["parameter_count"]
            - EXPECTED_CANONICAL_PARAMETER_COUNT
        ),
        "canonical_parameter_count_match": (
            summary["parameter_count"]
            == EXPECTED_CANONICAL_PARAMETER_COUNT
        ),
        "parameter_leaves": leaves,
        "target_parameters": target_parameters,
        "config": load_json_if_present(config_path),
        "progress": load_json_if_present(progress_path),
        "endpoint_decision": load_json_if_present(endpoint_decision_path),
    }


def parameter_leaf_to_row(
    seed_name: str,
    leaf: ParameterLeaf,
) -> dict[str, Any]:
    return {
        "seed": seed_name,
        "path": leaf.path,
        "shape": json.dumps(list(leaf.shape)),
        "dtype": leaf.dtype,
        "elements": leaf.elements,
        "bytes": leaf.nbytes,
    }


def write_parameter_leaves_csv(
    seed_results: Mapping[str, dict[str, Any]],
) -> None:
    fieldnames = (
        "seed",
        "path",
        "shape",
        "dtype",
        "elements",
        "bytes",
    )

    try:
        with OUTPUT_PARAMETER_LEAVES.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for seed_name in sorted(seed_results):
                leaves = seed_results[seed_name]["parameter_leaves"]

                for leaf in leaves:
                    writer.writerow(
                        parameter_leaf_to_row(
                            seed_name,
                            leaf,
                        )
                    )

    except OSError as exc:
        fail(
            f"Could not write parameter leaves CSV "
            f"{OUTPUT_PARAMETER_LEAVES}: {exc}"
        )


def compact_seed_payload(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    target_parameters = {
        target: [
            {
                "path": leaf.path,
                "shape": list(leaf.shape),
                "dtype": leaf.dtype,
                "elements": leaf.elements,
            }
            for leaf in leaves
        ]
        for target, leaves in result["target_parameters"].items()
    }

    return {
        "seed_name": result["seed_name"],
        "seed_directory": result["seed_directory"],
        "checkpoint_path": result["checkpoint_path"],
        "checkpoint_size_bytes": result["checkpoint_size_bytes"],
        "checkpoint_size_human": result["checkpoint_size_human"],
        "checkpoint_root": result["checkpoint_root"],
        "scalar_metadata": result["scalar_metadata"],
        "parameter_summary": result["parameter_summary"],
        "canonical_parameter_count": result["canonical_parameter_count"],
        "canonical_parameter_count_difference": (
            result["canonical_parameter_count_difference"]
        ),
        "canonical_parameter_count_match": (
            result["canonical_parameter_count_match"]
        ),
        "target_parameters": target_parameters,
        "config": result["config"],
        "progress": result["progress"],
        "endpoint_decision": result["endpoint_decision"],
    }


def markdown_json_block(value: Any) -> str:
    return json.dumps(
        value,
        indent=2,
        sort_keys=True,
        default=json_default,
    )


def write_markdown(
    seed_results: Mapping[str, dict[str, Any]],
    seed_comparison: dict[str, Any] | None,
) -> None:
    lines: list[str] = []

    lines.append("# PB0 Checkpoint Compatibility Inspection")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "This inspection reads the existing trained checkpoints without "
        "modifying the canonical architecture, checkpoint files, or training "
        "configuration."
    )
    lines.append("")
    lines.append("The checkpoint format is inspected as a Python pickle containing a JAX/NumPy-compatible parameter tree.")
    lines.append("")
    lines.append("## Canonical Reference")
    lines.append("")
    lines.append(
        f"- Canonical model file: `{CANONICAL_MODEL_PATH}`"
    )
    lines.append("")
    lines.append(
        f"- Expected canonical parameter count: **{format_integer(EXPECTED_CANONICAL_PARAMETER_COUNT)}**"
    )
    lines.append("")

    for seed_name in sorted(seed_results):
        result = seed_results[seed_name]
        summary = result["parameter_summary"]

        lines.append(f"## {seed_name.upper()}")
        lines.append("")
        lines.append(
            f"- Checkpoint: `{result['checkpoint_path']}`"
        )
        lines.append(
            f"- Checkpoint size: **{result['checkpoint_size_human']}**"
        )
        lines.append(
            f"- Parameter leaves: **{format_integer(summary['leaf_count'])}**"
        )
        lines.append(
            f"- Parameter count: **{format_integer(summary['parameter_count'])}**"
        )
        lines.append(
            f"- Parameter array bytes: **{format_bytes(summary['parameter_bytes'])}**"
        )
        lines.append(
            f"- Matches canonical 47,437,768 count: **{result['canonical_parameter_count_match']}**"
        )
        lines.append(
            f"- Difference from canonical count: **{result['canonical_parameter_count_difference']:+,}**"
        )
        lines.append("")

        if result["scalar_metadata"]:
            lines.append("### Scalar Checkpoint Metadata")
            lines.append("")
            lines.append("```json")
            lines.append(
                markdown_json_block(
                    result["scalar_metadata"]
                )
            )
            lines.append("```")
            lines.append("")

        lines.append("### Target Matrix Parameters")
        lines.append("")
        lines.append(
            "| Family | Checkpoint path | Shape | Parameters |"
        )
        lines.append("|---|---|---|---:|")

        for target, leaves in result["target_parameters"].items():
            if not leaves:
                lines.append(
                    f"| `{target}` | not found | — | — |"
                )
                continue

            for leaf in leaves:
                lines.append(
                    f"| `{target}` | `{leaf.path}` | "
                    f"`{list(leaf.shape)}` | "
                    f"{format_integer(leaf.elements)} |"
                )

        lines.append("")

        if result["config"] is not None:
            lines.append("### config.json")
            lines.append("")
            lines.append("```json")
            lines.append(markdown_json_block(result["config"]))
            lines.append("```")
            lines.append("")

        if result["progress"] is not None:
            lines.append("### progress.json")
            lines.append("")
            lines.append("```json")
            lines.append(markdown_json_block(result["progress"]))
            lines.append("```")
            lines.append("")

        if result["endpoint_decision"] is not None:
            lines.append("### ENDPOINT_DECISION.json")
            lines.append("")
            lines.append("```json")
            lines.append(
                markdown_json_block(
                    result["endpoint_decision"]
                )
            )
            lines.append("```")
            lines.append("")

    if seed_comparison is not None:
        lines.append("## Seed Comparison")
        lines.append("")

        lines.append(
            f"- Exact parameter-tree structure match: **{seed_comparison['exact_structure_match']}**"
        )
        lines.append(
            f"- Paths only in seed1: **{len(seed_comparison['paths_only_in_first'])}**"
        )
        lines.append(
            f"- Paths only in seed2: **{len(seed_comparison['paths_only_in_second'])}**"
        )
        lines.append(
            f"- Shape mismatches: **{len(seed_comparison['shape_mismatches'])}**"
        )
        lines.append(
            f"- Dtype mismatches: **{len(seed_comparison['dtype_mismatches'])}**"
        )
        lines.append("")

    lines.append("## PB0 Gate")
    lines.append("")
    lines.append(
        "PB0 checkpoint inspection establishes checkpoint structure and parameter "
        "availability. Passing this inspection does not by itself prove that a "
        "checkpoint can reproduce canonical BPC. The next gate is to load the "
        "parameter tree through the exact canonical evaluation path and verify "
        "the frozen validation baseline."
    )
    lines.append("")

    try:
        with OUTPUT_MARKDOWN.open(
            "w",
            encoding="utf-8",
        ) as file:
            file.write("\n".join(lines))
            file.write("\n")
    except OSError as exc:
        fail(
            f"Could not write Markdown output "
            f"{OUTPUT_MARKDOWN}: {exc}"
        )


def main() -> None:
    ensure_output_directory()

    print("=" * 80)
    print("MODUS_X PB0 CHECKPOINT COMPATIBILITY INSPECTION")
    print("=" * 80)
    print(f"Project root         : {PROJECT_ROOT}")
    print(f"Canonical model      : {CANONICAL_MODEL_PATH}")
    print(f"Output directory     : {OUTPUT_DIRECTORY}")
    print(f"Expected parameters  : {EXPECTED_CANONICAL_PARAMETER_COUNT:,}")

    if not CANONICAL_MODEL_PATH.exists():
        fail(
            f"Canonical model file does not exist: {CANONICAL_MODEL_PATH}"
        )

    seed_results: dict[str, dict[str, Any]] = {}

    for seed_name, seed_directory in SEED_DIRECTORIES.items():
        seed_results[seed_name] = inspect_seed(
            seed_name=seed_name,
            seed_directory=seed_directory,
        )

    seed_comparison: dict[str, Any] | None = None

    if "seed1" in seed_results and "seed2" in seed_results:
        seed_comparison = compare_parameter_sets(
            seed_results["seed1"]["parameter_leaves"],
            seed_results["seed2"]["parameter_leaves"],
        )

    write_parameter_leaves_csv(seed_results)

    json_payload = {
        "analysis": "PB0 checkpoint compatibility inspection",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "canonical_model": str(CANONICAL_MODEL_PATH),
        "expected_canonical_parameter_count": (
            EXPECTED_CANONICAL_PARAMETER_COUNT
        ),
        "seeds": {
            seed_name: compact_seed_payload(result)
            for seed_name, result in seed_results.items()
        },
        "seed_comparison": seed_comparison,
        "outputs": {
            "json": str(OUTPUT_JSON),
            "markdown": str(OUTPUT_MARKDOWN),
            "parameter_leaves_csv": str(
                OUTPUT_PARAMETER_LEAVES
            ),
        },
    }

    write_json(
        OUTPUT_JSON,
        json_payload,
    )

    write_markdown(
        seed_results=seed_results,
        seed_comparison=seed_comparison,
    )

    print()
    print("=" * 80)
    print("PB0 CHECKPOINT INSPECTION COMPLETE")
    print("=" * 80)

    for seed_name in sorted(seed_results):
        summary = seed_results[seed_name]["parameter_summary"]

        print()
        print(seed_name.upper())
        print(
            f"  Parameter leaves : "
            f"{summary['leaf_count']:,}"
        )
        print(
            f"  Parameter count  : "
            f"{summary['parameter_count']:,}"
        )
        print(
            f"  Canonical match  : "
            f"{seed_results[seed_name]['canonical_parameter_count_match']}"
        )

    if seed_comparison is not None:
        print()
        print("SEED COMPARISON")
        print(
            f"  Exact structure match : "
            f"{seed_comparison['exact_structure_match']}"
        )
        print(
            f"  Shape mismatches      : "
            f"{len(seed_comparison['shape_mismatches'])}"
        )
        print(
            f"  Paths only in seed1   : "
            f"{len(seed_comparison['paths_only_in_first'])}"
        )
        print(
            f"  Paths only in seed2   : "
            f"{len(seed_comparison['paths_only_in_second'])}"
        )

    print()
    print("Output files:")
    print(f"  {OUTPUT_JSON}")
    print(f"  {OUTPUT_MARKDOWN}")
    print(f"  {OUTPUT_PARAMETER_LEAVES}")

    print()
    print(
        "NEXT GATE: inspect PB0_CHECKPOINT_INSPECTION.md. "
        "If checkpoint parameter counts and target matrix paths are valid, "
        "proceed to canonical evaluation-path loading and frozen PB0 spectral analysis."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 80)
        print("PB0 CHECKPOINT INSPECTION FAILED")
        print("=" * 80)
        print()
        print(f"{type(exc).__name__}: {exc}")
        print()
        traceback.print_exc()
        sys.exit(1)