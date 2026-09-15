from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"

B0_JSON_PATH = OUTPUT_DIR / "b0_parameter_census.json"
B0_CSV_PATH = OUTPUT_DIR / "b0_parameter_leaves.csv"

OUTPUT_JSON_PATH = OUTPUT_DIR / "B3_MATRIX_ANALYSIS.json"
OUTPUT_MARKDOWN_PATH = OUTPUT_DIR / "B3_MATRIX_ANALYSIS.md"
OUTPUT_MATRIX_CSV_PATH = OUTPUT_DIR / "B3_MATRIX_LEAVES.csv"
OUTPUT_FAMILY_CSV_PATH = OUTPUT_DIR / "B3_MATRIX_FAMILIES.csv"


def fail(message: str) -> None:
    raise RuntimeError(message)


def format_integer(value: int) -> str:
    return f"{value:,}"


def format_percentage(value: float) -> str:
    return f"{value:.4f}%"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        fail(
            f"Required input file does not exist: {path}\n"
            "Run b0_parameter_census.py before running this analysis."
        )

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except OSError as exc:
        fail(f"Could not read JSON file {path}: {exc}")
    except json.JSONDecodeError as exc:
        fail(f"Could not parse JSON file {path}: {exc}")

    if not isinstance(data, dict):
        fail(
            f"Expected a JSON object at the root of {path}, "
            f"but found {type(data).__name__}."
        )

    return data


def load_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        fail(
            f"Required input file does not exist: {path}\n"
            "Run b0_parameter_census.py before running this analysis."
        )

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)

            if reader.fieldnames is None:
                fail(f"CSV file has no header: {path}")

            fieldnames = [
                fieldname
                for fieldname in reader.fieldnames
                if fieldname is not None
            ]

            rows = list(reader)

    except OSError as exc:
        fail(f"Could not read CSV file {path}: {exc}")

    if not rows:
        fail(
            f"Parameter CSV contains no rows: {path}\n"
            "The census output appears incomplete."
        )

    return rows, fieldnames


def identify_column(
    fieldnames: list[str],
    candidates: tuple[str, ...],
    description: str,
) -> str:
    if not fieldnames:
        fail(f"The B0 parameter-leaves CSV has no columns for {description}.")

    normalized_to_original: dict[str, str] = {}

    for column in fieldnames:
        normalized = column.strip().lower()

        if normalized:
            normalized_to_original[normalized] = column

    for candidate in candidates:
        if candidate in normalized_to_original:
            return normalized_to_original[candidate]

    fail(
        f"Could not identify the {description} column. "
        f"Available columns: {fieldnames}. "
        f"Expected one of: {', '.join(candidates)}."
    )


def identify_path_column(fieldnames: list[str]) -> str:
    return identify_column(
        fieldnames=fieldnames,
        candidates=(
            "path",
            "name",
            "parameter_path",
            "parameter_name",
        ),
        description="parameter path",
    )


def identify_parameter_count_column(fieldnames: list[str]) -> str:
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
        description="parameter count",
    )


def identify_component_column(fieldnames: list[str]) -> str:
    return identify_column(
        fieldnames=fieldnames,
        candidates=(
            "component",
            "component_name",
            "group",
            "category",
        ),
        description="component",
    )


def identify_shape_column(fieldnames: list[str]) -> str:
    return identify_column(
        fieldnames=fieldnames,
        candidates=(
            "shape",
            "tensor_shape",
            "dimensions",
        ),
        description="tensor shape",
    )


def parse_parameter_count(
    row: dict[str, str],
    parameter_count_column: str,
) -> int:
    raw_value = row.get(parameter_count_column)

    if raw_value is None:
        fail(
            "A row in the B0 parameter-leaves CSV is missing the identified "
            f"parameter-count column '{parameter_count_column}'. "
            f"Row contents: {row}"
        )

    cleaned = raw_value.replace(",", "").strip()

    if not cleaned:
        fail(
            "A row in the B0 parameter-leaves CSV has an empty parameter "
            f"count in column '{parameter_count_column}'. "
            f"Row contents: {row}"
        )

    try:
        value = int(cleaned)
    except ValueError as exc:
        raise RuntimeError(
            "Could not parse parameter count from B0 parameter-leaves CSV. "
            f"Column: '{parameter_count_column}', "
            f"value: '{raw_value}', "
            f"row: {row}"
        ) from exc

    if value < 0:
        fail(
            "Parameter count cannot be negative. "
            f"Column: '{parameter_count_column}', "
            f"value: {value}, "
            f"row: {row}"
        )

    return value


def parse_shape(raw_shape: str) -> list[int]:
    cleaned = raw_shape.strip()

    if not cleaned:
        return []

    if not cleaned.startswith("[") or not cleaned.endswith("]"):
        return []

    inner = cleaned[1:-1].strip()

    if not inner:
        return []

    parts = [part.strip() for part in inner.split(",")]

    dimensions: list[int] = []

    for part in parts:
        if not part:
            return []

        try:
            dimension = int(part)
        except ValueError:
            return []

        if dimension < 0:
            return []

        dimensions.append(dimension)

    return dimensions


def classify_matrix_role(path: str) -> str:
    normalized = path.lower()

    if "m_wq" in normalized:
        return "query_projection"

    if "m_wk" in normalized:
        return "key_projection"

    if "m_wv" in normalized:
        return "value_projection"

    if "m_w_read" in normalized:
        return "memory_read_projection"

    if "m_proj_w" in normalized:
        return "memory_input_projection"

    if "m_w_out" in normalized:
        return "memory_output_projection"

    if "archive_mix" in normalized:
        return "archive_mixing"

    return "other_matrix_memory_parameter"


def derive_family(path: str) -> str:
    normalized = path.strip()

    if normalized.startswith("layers."):
        parts = normalized.split(".")

        if len(parts) >= 2:
            return parts[1]

    return normalized


def shape_signature(shape: list[int]) -> str:
    if not shape:
        return "unknown"

    return "x".join(str(dimension) for dimension in shape)


def calculate_matrix_density(
    parameter_count: int,
    shape: list[int],
) -> str:
    if len(shape) == 3:
        return "layer_stacked_matrix"

    if len(shape) == 2:
        return "matrix"

    if len(shape) == 1:
        return "vector"

    if len(shape) == 0:
        return "unknown"

    return "higher_rank_tensor"


def get_reported_total(data: dict[str, Any]) -> int | None:
    candidate_keys = (
        "total_parameters",
        "parameter_count",
        "total_parameter_count",
        "canonical_parameter_count",
        "params",
    )

    for key in candidate_keys:
        value = data.get(key)

        if isinstance(value, bool):
            continue

        if isinstance(value, int):
            return value

        if isinstance(value, float) and value.is_integer():
            return int(value)

        if isinstance(value, str):
            cleaned = value.replace(",", "").strip()

            if cleaned.isdigit():
                return int(cleaned)

    return None


def extract_component_totals(data: dict[str, Any]) -> dict[str, int]:
    candidate_containers = (
        "allocation_summary",
        "component_totals",
        "components",
        "summary",
    )

    for container_key in candidate_containers:
        container = data.get(container_key)

        if not isinstance(container, dict):
            continue

        result: dict[str, int] = {}

        for key, value in container.items():
            if isinstance(value, bool):
                continue

            if isinstance(value, int):
                result[str(key)] = value
                continue

            if isinstance(value, float) and value.is_integer():
                result[str(key)] = int(value)
                continue

            if isinstance(value, dict):
                for nested_key in (
                    "parameters",
                    "parameter_count",
                    "elements",
                    "count",
                    "total_parameters",
                ):
                    nested_value = value.get(nested_key)

                    if isinstance(nested_value, bool):
                        continue

                    if isinstance(nested_value, int):
                        result[str(key)] = nested_value
                        break

                    if (
                        isinstance(nested_value, float)
                        and nested_value.is_integer()
                    ):
                        result[str(key)] = int(nested_value)
                        break

        if result:
            return result

    return {}


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    try:
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for row in rows:
                writer.writerow(row)

    except OSError as exc:
        fail(f"Could not write CSV file {path}: {exc}")


def write_json(path: Path, data: dict[str, Any]) -> None:
    try:
        with path.open("w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                indent=2,
                sort_keys=False,
            )
            file.write("\n")

    except OSError as exc:
        fail(f"Could not write JSON file {path}: {exc}")


def write_markdown(
    path: Path,
    report: dict[str, Any],
) -> None:
    total_parameters = report["total_model_parameters"]
    matrix_memory_parameters = report["matrix_memory_parameters"]
    matrix_memory_share = report["matrix_memory_share_percent"]

    role_rows = report["role_summary"]
    shape_rows = report["shape_summary"]
    largest_rows = report["largest_matrix_leaves"]
    duplicate_rows = report["same_shape_groups"]

    lines: list[str] = []

    lines.append("# B3 Matrix Memory Structural Analysis")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "This analysis inspects the canonical parameter census and isolates "
        "the parameters classified as `matrix_memory`."
    )
    lines.append("")
    lines.append(
        "No model architecture, dimensions, parameter budget, or checkpoint "
        "has been modified by this analysis."
    )
    lines.append("")

    lines.append("## Global Totals")
    lines.append("")
    lines.append(
        f"- Total canonical model parameters: "
        f"**{format_integer(total_parameters)}**"
    )
    lines.append(
        f"- Matrix-memory parameters: "
        f"**{format_integer(matrix_memory_parameters)}**"
    )
    lines.append(
        f"- Matrix-memory share of model: "
        f"**{format_percentage(matrix_memory_share)}**"
    )
    lines.append(
        f"- Matrix-memory parameter leaves: "
        f"**{report['matrix_memory_leaf_count']}**"
    )
    lines.append("")

    lines.append("## Matrix Roles")
    lines.append("")
    lines.append("| Role | Parameters | Share of matrix memory |")
    lines.append("|---|---:|---:|")

    for row in role_rows:
        lines.append(
            f"| {row['role']} | "
            f"{format_integer(row['parameters'])} | "
            f"{format_percentage(row['share_of_matrix_memory_percent'])} |"
        )

    lines.append("")

    lines.append("## Shape Groups")
    lines.append("")
    lines.append("| Shape signature | Leaves | Parameters | Share of matrix memory |")
    lines.append("|---|---:|---:|---:|")

    for row in shape_rows:
        lines.append(
            f"| {row['shape_signature']} | "
            f"{row['leaf_count']} | "
            f"{format_integer(row['parameters'])} | "
            f"{format_percentage(row['share_of_matrix_memory_percent'])} |"
        )

    lines.append("")

    lines.append("## Largest Matrix-Memory Leaves")
    lines.append("")
    lines.append(
        "| Parameter path | Role | Shape | Parameters | Share of matrix memory |"
    )
    lines.append("|---|---|---|---:|---:|")

    for row in largest_rows:
        lines.append(
            f"| `{row['path']}` | "
            f"{row['role']} | "
            f"`{row['shape']}` | "
            f"{format_integer(row['parameters'])} | "
            f"{format_percentage(row['share_of_matrix_memory_percent'])} |"
        )

    lines.append("")

    lines.append("## Same-Shape Groups")
    lines.append("")
    lines.append(
        "These groups identify matrix-memory leaves with identical tensor "
        "shape signatures. Identical shape does not prove functional "
        "redundancy; it only identifies candidates for later controlled "
        "sharing, factorization, or width-allocation experiments."
    )
    lines.append("")

    lines.append(
        "| Shape signature | Leaves | Parameters | Parameter paths |"
    )
    lines.append("|---|---:|---:|---|")

    for row in duplicate_rows:
        path_list = ", ".join(f"`{value}`" for value in row["paths"])

        lines.append(
            f"| {row['shape_signature']} | "
            f"{row['leaf_count']} | "
            f"{format_integer(row['parameters'])} | "
            f"{path_list} |"
        )

    lines.append("")

    lines.append("## Structural Interpretation")
    lines.append("")

    if matrix_memory_share >= 40.0:
        lines.append(
            f"- Matrix memory consumes **{format_percentage(matrix_memory_share)}** "
            "of the canonical parameter budget, making it a primary target for "
            "controlled parameter-allocation experiments."
        )
    else:
        lines.append(
            f"- Matrix memory consumes **{format_percentage(matrix_memory_share)}** "
            "of the canonical parameter budget."
        )

    if duplicate_rows:
        largest_duplicate = duplicate_rows[0]

        lines.append(
            f"- The largest repeated-shape group is "
            f"`{largest_duplicate['shape_signature']}`, containing "
            f"{largest_duplicate['leaf_count']} leaves and "
            f"{format_integer(largest_duplicate['parameters'])} parameters."
        )

    lines.append(
        "- These results do not establish that any matrix can be removed. "
        "They establish the quantitative search space for PB1-A and PB1-B."
    )
    lines.append("")

    lines.append("## Next Experimental Gates")
    lines.append("")
    lines.append(
        "1. PB1-A: controlled matrix-memory reduction while holding the "
        "remaining architecture fixed."
    )
    lines.append(
        "2. PB1-B: approximately fixed total parameter budget with saved "
        "parameters reallocated to alternative architectural components."
    )
    lines.append(
        "3. Validate parameter counts and recurrent-state behavior before "
        "any long training run."
    )
    lines.append("")

    try:
        with path.open("w", encoding="utf-8") as file:
            file.write("\n".join(lines))
            file.write("\n")

    except OSError as exc:
        fail(f"Could not write Markdown report {path}: {exc}")


def main() -> None:
    print("=" * 80)
    print("MODUS_X B3 MATRIX MEMORY STRUCTURAL ANALYSIS")
    print("=" * 80)
    print()

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Input JSON   : {B0_JSON_PATH}")
    print(f"Input CSV    : {B0_CSV_PATH}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    b0_data = load_json(B0_JSON_PATH)
    rows, fieldnames = load_csv(B0_CSV_PATH)

    path_column = identify_path_column(fieldnames)
    parameter_count_column = identify_parameter_count_column(fieldnames)
    component_column = identify_component_column(fieldnames)
    shape_column = identify_shape_column(fieldnames)

    print(f"Detected path column      : {path_column}")
    print(f"Detected parameter column : {parameter_count_column}")
    print(f"Detected component column : {component_column}")
    print(f"Detected shape column     : {shape_column}")
    print()

    normalized_rows: list[dict[str, Any]] = []

    total_parameters_from_csv = 0
    matrix_memory_parameters = 0

    for row in rows:
        path_value = row.get(path_column)

        if path_value is None:
            fail(
                f"A B0 CSV row is missing path column '{path_column}'. "
                f"Row contents: {row}"
            )

        component_value = row.get(component_column)

        if component_value is None:
            fail(
                f"A B0 CSV row is missing component column '{component_column}'. "
                f"Row contents: {row}"
            )

        shape_value = row.get(shape_column)

        if shape_value is None:
            fail(
                f"A B0 CSV row is missing shape column '{shape_column}'. "
                f"Row contents: {row}"
            )

        parameter_count = parse_parameter_count(
            row=row,
            parameter_count_column=parameter_count_column,
        )

        path_text = path_value.strip()
        component_text = component_value.strip()
        shape_text = shape_value.strip()

        total_parameters_from_csv += parameter_count

        if component_text != "matrix_memory":
            continue

        matrix_memory_parameters += parameter_count

        parsed_shape = parse_shape(shape_text)

        normalized_rows.append(
            {
                "path": path_text,
                "component": component_text,
                "shape": shape_text,
                "shape_dimensions": parsed_shape,
                "shape_signature": shape_signature(parsed_shape),
                "parameters": parameter_count,
                "role": classify_matrix_role(path_text),
                "family": derive_family(path_text),
                "tensor_kind": calculate_matrix_density(
                    parameter_count=parameter_count,
                    shape=parsed_shape,
                ),
            }
        )

    if not normalized_rows:
        fail(
            "No rows were classified as 'matrix_memory'. "
            "The canonical B0 component classification appears inconsistent."
        )

    reported_total = get_reported_total(b0_data)
    component_totals = extract_component_totals(b0_data)

    reported_matrix_memory = component_totals.get("matrix_memory")

    if reported_total is not None and reported_total != total_parameters_from_csv:
        fail(
            "The B0 JSON total does not match the sum of the B0 CSV leaves. "
            f"JSON total: {format_integer(reported_total)}; "
            f"CSV total: {format_integer(total_parameters_from_csv)}."
        )

    if (
        reported_matrix_memory is not None
        and reported_matrix_memory != matrix_memory_parameters
    ):
        fail(
            "The B0 JSON matrix-memory total does not match the matrix-memory "
            f"rows in the B0 CSV. JSON total: "
            f"{format_integer(reported_matrix_memory)}; CSV total: "
            f"{format_integer(matrix_memory_parameters)}."
        )

    if total_parameters_from_csv <= 0:
        fail(
            "The measured canonical parameter total is not positive. "
            "Cannot compute allocation percentages."
        )

    matrix_memory_share_percent = (
        matrix_memory_parameters / total_parameters_from_csv
    ) * 100.0

    for row in normalized_rows:
        row["share_of_matrix_memory_percent"] = (
            row["parameters"] / matrix_memory_parameters
        ) * 100.0

        row["share_of_total_model_percent"] = (
            row["parameters"] / total_parameters_from_csv
        ) * 100.0

    normalized_rows.sort(
        key=lambda item: (
            -int(item["parameters"]),
            str(item["path"]),
        )
    )

    role_totals: dict[str, int] = defaultdict(int)
    role_leaf_counts: dict[str, int] = defaultdict(int)

    for row in normalized_rows:
        role = str(row["role"])

        role_totals[role] += int(row["parameters"])
        role_leaf_counts[role] += 1

    role_summary: list[dict[str, Any]] = []

    for role in sorted(
        role_totals,
        key=lambda value: (-role_totals[value], value),
    ):
        parameters = role_totals[role]

        role_summary.append(
            {
                "role": role,
                "leaf_count": role_leaf_counts[role],
                "parameters": parameters,
                "share_of_matrix_memory_percent": (
                    parameters / matrix_memory_parameters
                ) * 100.0,
                "share_of_total_model_percent": (
                    parameters / total_parameters_from_csv
                ) * 100.0,
            }
        )

    shape_totals: dict[str, int] = defaultdict(int)
    shape_leaf_counts: dict[str, int] = defaultdict(int)
    shape_paths: dict[str, list[str]] = defaultdict(list)

    for row in normalized_rows:
        signature = str(row["shape_signature"])

        shape_totals[signature] += int(row["parameters"])
        shape_leaf_counts[signature] += 1
        shape_paths[signature].append(str(row["path"]))

    shape_summary: list[dict[str, Any]] = []

    for signature in sorted(
        shape_totals,
        key=lambda value: (-shape_totals[value], value),
    ):
        parameters = shape_totals[signature]

        shape_summary.append(
            {
                "shape_signature": signature,
                "leaf_count": shape_leaf_counts[signature],
                "parameters": parameters,
                "share_of_matrix_memory_percent": (
                    parameters / matrix_memory_parameters
                ) * 100.0,
                "share_of_total_model_percent": (
                    parameters / total_parameters_from_csv
                ) * 100.0,
            }
        )

    same_shape_groups: list[dict[str, Any]] = []

    for signature, paths in shape_paths.items():
        if len(paths) < 2:
            continue

        parameters = shape_totals[signature]

        same_shape_groups.append(
            {
                "shape_signature": signature,
                "leaf_count": len(paths),
                "parameters": parameters,
                "share_of_matrix_memory_percent": (
                    parameters / matrix_memory_parameters
                ) * 100.0,
                "paths": sorted(paths),
            }
        )

    same_shape_groups.sort(
        key=lambda item: (
            -int(item["parameters"]),
            str(item["shape_signature"]),
        )
    )

    largest_matrix_leaves = normalized_rows[:20]

    matrix_csv_rows: list[dict[str, Any]] = []

    for row in normalized_rows:
        matrix_csv_rows.append(
            {
                "path": row["path"],
                "role": row["role"],
                "family": row["family"],
                "shape": row["shape"],
                "shape_signature": row["shape_signature"],
                "tensor_kind": row["tensor_kind"],
                "parameters": row["parameters"],
                "share_of_matrix_memory_percent": (
                    f"{row['share_of_matrix_memory_percent']:.8f}"
                ),
                "share_of_total_model_percent": (
                    f"{row['share_of_total_model_percent']:.8f}"
                ),
            }
        )

    family_rows: list[dict[str, Any]] = []

    for row in role_summary:
        family_rows.append(
            {
                "group_type": "role",
                "group_name": row["role"],
                "leaf_count": row["leaf_count"],
                "parameters": row["parameters"],
                "share_of_matrix_memory_percent": (
                    f"{row['share_of_matrix_memory_percent']:.8f}"
                ),
                "share_of_total_model_percent": (
                    f"{row['share_of_total_model_percent']:.8f}"
                ),
                "paths": "",
            }
        )

    for row in shape_summary:
        family_rows.append(
            {
                "group_type": "shape",
                "group_name": row["shape_signature"],
                "leaf_count": row["leaf_count"],
                "parameters": row["parameters"],
                "share_of_matrix_memory_percent": (
                    f"{row['share_of_matrix_memory_percent']:.8f}"
                ),
                "share_of_total_model_percent": (
                    f"{row['share_of_total_model_percent']:.8f}"
                ),
                "paths": "; ".join(
                    sorted(shape_paths[row["shape_signature"]])
                ),
            }
        )

    report: dict[str, Any] = {
        "analysis": "B3 matrix memory structural analysis",
        "project_root": str(PROJECT_ROOT),
        "inputs": {
            "b0_parameter_census_json": str(B0_JSON_PATH),
            "b0_parameter_leaves_csv": str(B0_CSV_PATH),
        },
        "detected_columns": {
            "path": path_column,
            "parameter_count": parameter_count_column,
            "component": component_column,
            "shape": shape_column,
        },
        "total_model_parameters": total_parameters_from_csv,
        "reported_total_parameters": reported_total,
        "matrix_memory_parameters": matrix_memory_parameters,
        "reported_matrix_memory_parameters": reported_matrix_memory,
        "matrix_memory_share_percent": matrix_memory_share_percent,
        "matrix_memory_leaf_count": len(normalized_rows),
        "role_summary": role_summary,
        "shape_summary": shape_summary,
        "largest_matrix_leaves": largest_matrix_leaves,
        "same_shape_groups": same_shape_groups,
        "validation": {
            "csv_total_matches_b0_json": (
                reported_total is None
                or reported_total == total_parameters_from_csv
            ),
            "matrix_memory_matches_b0_json": (
                reported_matrix_memory is None
                or reported_matrix_memory == matrix_memory_parameters
            ),
        },
        "interpretation_boundary": (
            "This analysis identifies structural concentration and repeated "
            "shape groups. It does not claim that same-shape matrices are "
            "functionally redundant or safe to remove."
        ),
    }

    write_json(
        path=OUTPUT_JSON_PATH,
        data=report,
    )

    write_csv(
        path=OUTPUT_MATRIX_CSV_PATH,
        rows=matrix_csv_rows,
        fieldnames=[
            "path",
            "role",
            "family",
            "shape",
            "shape_signature",
            "tensor_kind",
            "parameters",
            "share_of_matrix_memory_percent",
            "share_of_total_model_percent",
        ],
    )

    write_csv(
        path=OUTPUT_FAMILY_CSV_PATH,
        rows=family_rows,
        fieldnames=[
            "group_type",
            "group_name",
            "leaf_count",
            "parameters",
            "share_of_matrix_memory_percent",
            "share_of_total_model_percent",
            "paths",
        ],
    )

    write_markdown(
        path=OUTPUT_MARKDOWN_PATH,
        report=report,
    )

    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print()

    print(
        f"Measured total model parameters : "
        f"{format_integer(total_parameters_from_csv)}"
    )
    print(
        f"Matrix-memory parameters        : "
        f"{format_integer(matrix_memory_parameters)}"
    )
    print(
        f"Matrix-memory share             : "
        f"{format_percentage(matrix_memory_share_percent)}"
    )
    print(
        f"Matrix-memory leaves            : "
        f"{len(normalized_rows)}"
    )
    print()

    print("Matrix roles:")

    for row in role_summary:
        print(
            f"  {row['role']:<32} "
            f"{format_integer(row['parameters']):>14}  "
            f"{format_percentage(row['share_of_matrix_memory_percent']):>10}"
        )

    print()
    print("Largest matrix-memory leaves:")

    for row in largest_matrix_leaves:
        print(
            f"  {row['path']:<32} "
            f"{format_integer(row['parameters']):>14}  "
            f"{row['shape']}"
        )

    print()
    print("Output files:")
    print(f"  {OUTPUT_JSON_PATH}")
    print(f"  {OUTPUT_MARKDOWN_PATH}")
    print(f"  {OUTPUT_MATRIX_CSV_PATH}")
    print(f"  {OUTPUT_FAMILY_CSV_PATH}")
    print()
    print(
        "NEXT GATE: inspect B3_MATRIX_ANALYSIS.md. "
        "Do not modify matrix dimensions until the structural concentration "
        "and repeated-shape groups are verified."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 80)
        print("B3 ANALYSIS FAILED")
        print("=" * 80)
        print()
        print(str(exc))
        sys.exit(1)