from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


SEPARATOR = "=" * 80


def fail(message: str) -> None:
    print(f"\nERROR: {message}\n", file=sys.stderr)
    raise SystemExit(1)


def find_project_root(start_path: Path) -> Path:
    """
    Locate the repository root by walking upward until the canonical
    language/models.py file is found.
    """
    current = start_path.resolve()

    for candidate in (current, *current.parents):
        canonical_model = candidate / "language" / "models.py"

        if canonical_model.exists():
            return candidate

    fail(
        "Could not locate the Modus_X project root.\n"
        "Expected to find language/models.py in this directory or one of "
        "its parent directories."
    )

    raise RuntimeError("Unreachable")


def load_json(path: Path) -> dict[str, Any]:
    """
    Load a JSON object with explicit validation.
    """
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
        fail(
            f"Could not parse JSON file {path}.\n"
            f"Line: {exc.lineno}, Column: {exc.colno}\n"
            f"Details: {exc.msg}"
        )

    if not isinstance(data, dict):
        fail(
            f"Expected the B0 JSON output to contain a JSON object, "
            f"but found {type(data).__name__}."
        )

    return data


def load_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """
    Load the B0 parameter-leaves CSV.

    Returns:
        A tuple containing:
        - the CSV rows
        - the CSV field names
    """
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
                if fieldname is not None and fieldname.strip()
            ]

            rows = list(reader)

    except OSError as exc:
        fail(f"Could not read {path}: {exc}")

    if not rows:
        fail(
            f"Parameter CSV contains no rows: {path}\n"
            "The census output appears incomplete."
        )

    if not fieldnames:
        fail(f"CSV file contains no valid column names: {path}")

    return rows, fieldnames


def identify_parameter_count_column(fieldnames: list[str]) -> str:
    """
    Identify the CSV column containing the number of scalar parameters
    represented by each parameter leaf.

    The canonical B0 census currently uses 'elements', where each element
    corresponds to one scalar trainable parameter.
    """
    if not fieldnames:
        fail("The B0 parameter-leaves CSV has no header row.")

    normalized_to_original: dict[str, str] = {
        column.strip().lower(): column
        for column in fieldnames
        if column.strip()
    }

    candidate_columns = (
        "elements",
        "parameter_count",
        "parameters",
        "num_parameters",
        "num_params",
        "params",
        "count",
        "size",
    )

    for candidate in candidate_columns:
        if candidate in normalized_to_original:
            return normalized_to_original[candidate]

    fail(
        "Could not identify a parameter-count column in the B0 "
        "parameter-leaves CSV.\n"
        f"Available columns: {fieldnames}\n"
        "Expected one of: "
        + ", ".join(candidate_columns)
    )

    raise RuntimeError("Unreachable")


def identify_path_column(fieldnames: list[str]) -> str:
    """
    Identify the CSV column containing the parameter path.
    """
    normalized_to_original: dict[str, str] = {
        column.strip().lower(): column
        for column in fieldnames
        if column.strip()
    }

    candidate_columns = (
        "path",
        "parameter_path",
        "name",
        "parameter_name",
    )

    for candidate in candidate_columns:
        if candidate in normalized_to_original:
            return normalized_to_original[candidate]

    fail(
        "Could not identify a parameter-path column in the B0 "
        "parameter-leaves CSV.\n"
        f"Available columns: {fieldnames}\n"
        "Expected one of: "
        + ", ".join(candidate_columns)
    )

    raise RuntimeError("Unreachable")


def identify_component_column(fieldnames: list[str]) -> str | None:
    """
    Identify the canonical component-classification column.

    The B0 census explicitly writes the component assignment for every
    parameter leaf. B2 must use this classification as the authoritative
    source instead of independently reclassifying parameter paths.
    """
    normalized_to_original: dict[str, str] = {
        column.strip().lower(): column
        for column in fieldnames
        if column.strip()
    }

    candidate_columns = (
        "component",
        "component_name",
        "category",
        "group",
    )

    for candidate in candidate_columns:
        if candidate in normalized_to_original:
            return normalized_to_original[candidate]

    return None


def extract_reported_total(data: dict[str, Any]) -> int | None:
    """
    Extract the total parameter count from the B0 JSON output.

    The B0 census schema may evolve, so several explicit key names are
    supported without guessing numeric values.
    """
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


def infer_component_from_path(parameter_path: str) -> str:
    """
    Conservative fallback classifier.

    This function is used only if the B0 CSV does not contain the canonical
    component column. The current expected B0 schema does contain 'component',
    so normal execution should not depend on this fallback.
    """
    path = parameter_path.strip()

    if path == "embed" or path.startswith("embed."):
        return "embedding"

    if path.startswith("head."):
        return "output_head"

    if path.startswith("future_heads."):
        return "future_prediction_head"

    if path.startswith("layers.m_w_archive_mix"):
        return "archive_memory_control"

    if path.startswith("layers.r_"):
        return "router"

    if (
        path.startswith("layers.s_memory_up")
        or path.startswith("layers.s_memory_down")
        or path.startswith("layers.s_w_memory_feedback")
    ):
        return "feedback_bridge"

    if (
        path.startswith("layers.pre_")
        or ".norm" in path
        or path.startswith("norm")
    ):
        return "normalization"

    if path.startswith("layers.m_"):
        return "matrix_memory"

    if path.startswith("layers.s_"):
        return "vector_recurrence"

    return "other"


def parse_parameter_count(
    row: dict[str, str],
    parameter_count_column: str,
) -> int:
    """
    Parse and validate the scalar parameter count for one CSV row.
    """
    raw_parameter_value = row.get(parameter_count_column)

    if raw_parameter_value is None:
        raise RuntimeError(
            "A row in the B0 parameter-leaves CSV is missing the identified "
            f"parameter-count column '{parameter_count_column}'.\n"
            f"Row contents: {row}"
        )

    cleaned_value = raw_parameter_value.replace(",", "").strip()

    if not cleaned_value:
        raise RuntimeError(
            "A row in the B0 parameter-leaves CSV contains an empty parameter "
            f"count in column '{parameter_count_column}'.\n"
            f"Row contents: {row}"
        )

    try:
        parameter_count = int(cleaned_value)

    except ValueError as exc:
        raise RuntimeError(
            "Could not parse parameter count from B0 parameter-leaves CSV.\n"
            f"Column: '{parameter_count_column}'\n"
            f"Value: '{raw_parameter_value}'\n"
            f"Row: {row}"
        ) from exc

    if parameter_count < 0:
        raise RuntimeError(
            "Encountered a negative parameter count in the B0 "
            "parameter-leaves CSV.\n"
            f"Column: '{parameter_count_column}'\n"
            f"Value: {parameter_count}\n"
            f"Row: {row}"
        )

    return parameter_count


def get_row_value(
    row: dict[str, str],
    column_name: str,
    description: str,
) -> str:
    """
    Retrieve a required CSV value with validation.
    """
    value = row.get(column_name)

    if value is None:
        raise RuntimeError(
            f"A CSV row is missing the required {description} column "
            f"'{column_name}'.\n"
            f"Row contents: {row}"
        )

    cleaned = value.strip()

    if not cleaned:
        raise RuntimeError(
            f"A CSV row contains an empty value for required {description} "
            f"column '{column_name}'.\n"
            f"Row contents: {row}"
        )

    return cleaned


def build_component_summary(
    rows: list[dict[str, str]],
    path_column: str,
    parameter_count_column: str,
    component_column: str | None,
) -> tuple[
    dict[str, int],
    dict[str, int],
    list[dict[str, Any]],
]:
    """
    Aggregate parameter leaves into component-level totals.

    Returns:
        component_totals:
            Total parameters per component.

        component_leaf_counts:
            Number of parameter leaves assigned to each component.

        leaf_records:
            Normalized per-leaf records used for validation and reporting.
    """
    component_totals: dict[str, int] = defaultdict(int)
    component_leaf_counts: dict[str, int] = defaultdict(int)
    leaf_records: list[dict[str, Any]] = []

    for row_index, row in enumerate(rows, start=2):
        parameter_path = get_row_value(
            row=row,
            column_name=path_column,
            description="parameter-path",
        )

        parameter_count = parse_parameter_count(
            row=row,
            parameter_count_column=parameter_count_column,
        )

        if component_column is not None:
            raw_component = row.get(component_column)

            if raw_component is None:
                raise RuntimeError(
                    f"Row {row_index} is missing the canonical component "
                    f"column '{component_column}'.\n"
                    f"Row contents: {row}"
                )

            component = raw_component.strip()

            if not component:
                raise RuntimeError(
                    f"Row {row_index} contains an empty canonical component "
                    f"classification in column '{component_column}'.\n"
                    f"Parameter path: {parameter_path}"
                )
        else:
            component = infer_component_from_path(parameter_path)

        component_totals[component] += parameter_count
        component_leaf_counts[component] += 1

        leaf_records.append(
            {
                "path": parameter_path,
                "parameter_count": parameter_count,
                "component": component,
            }
        )

    return (
        dict(component_totals),
        dict(component_leaf_counts),
        leaf_records,
    )


def build_sorted_component_rows(
    component_totals: dict[str, int],
    component_leaf_counts: dict[str, int],
    measured_total: int,
) -> list[dict[str, Any]]:
    """
    Build deterministic component summary rows sorted by descending
    parameter count.
    """
    if measured_total <= 0:
        fail(
            "Measured parameter total must be greater than zero in order "
            "to compute allocation percentages."
        )

    summary_rows: list[dict[str, Any]] = []

    for component, parameter_count in component_totals.items():
        percentage = (parameter_count / measured_total) * 100.0

        summary_rows.append(
            {
                "component": component,
                "parameter_count": parameter_count,
                "percentage": percentage,
                "leaf_count": component_leaf_counts.get(component, 0),
            }
        )

    summary_rows.sort(
        key=lambda item: (
            -int(item["parameter_count"]),
            str(item["component"]),
        )
    )

    return summary_rows


def write_component_csv(
    path: Path,
    summary_rows: list[dict[str, Any]],
) -> None:
    """
    Write the component-level allocation summary as CSV.
    """
    fieldnames = (
        "component",
        "parameter_count",
        "percentage",
        "leaf_count",
    )

    try:
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for row in summary_rows:
                writer.writerow(
                    {
                        "component": row["component"],
                        "parameter_count": row["parameter_count"],
                        "percentage": f"{float(row['percentage']):.6f}",
                        "leaf_count": row["leaf_count"],
                    }
                )

    except OSError as exc:
        fail(f"Could not write component summary CSV {path}: {exc}")


def write_analysis_json(
    path: Path,
    project_root: Path,
    b0_json_path: Path,
    b0_csv_path: Path,
    path_column: str,
    parameter_count_column: str,
    component_column: str | None,
    expected_total: int | None,
    measured_total: int,
    summary_rows: list[dict[str, Any]],
) -> None:
    """
    Write machine-readable B2 analysis output.
    """
    difference = (
        measured_total - expected_total
        if expected_total is not None
        else None
    )

    analysis = {
        "analysis": "Modus_X Parameter Allocation Stage 1 Structured Analysis",
        "project_root": str(project_root),
        "inputs": {
            "b0_parameter_census_json": str(b0_json_path),
            "b0_parameter_leaves_csv": str(b0_csv_path),
        },
        "detected_columns": {
            "path": path_column,
            "parameter_count": parameter_count_column,
            "component": component_column,
        },
        "classification_policy": (
            "canonical_b0_component_column"
            if component_column is not None
            else "fallback_path_inference"
        ),
        "parameter_accounting": {
            "expected_total_parameters": expected_total,
            "measured_total_parameters": measured_total,
            "difference": difference,
        },
        "components": summary_rows,
    }

    try:
        with path.open("w", encoding="utf-8") as file:
            json.dump(
                analysis,
                file,
                indent=2,
            )
            file.write("\n")

    except OSError as exc:
        fail(f"Could not write B2 analysis JSON {path}: {exc}")


def write_markdown_report(
    path: Path,
    project_root: Path,
    b0_json_path: Path,
    b0_csv_path: Path,
    path_column: str,
    parameter_count_column: str,
    component_column: str | None,
    expected_total: int | None,
    measured_total: int,
    summary_rows: list[dict[str, Any]],
) -> None:
    """
    Write a human-readable B2 allocation report.
    """
    difference = (
        measured_total - expected_total
        if expected_total is not None
        else None
    )

    classification_policy = (
        "Canonical B0 `component` column"
        if component_column is not None
        else "Fallback parameter-path inference"
    )

    lines: list[str] = []

    lines.append("# Modus_X B2 Parameter Allocation Analysis")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "This report aggregates the parameter leaves produced by the B0 "
        "parameter census into architecture components."
    )
    lines.append("")
    lines.append(
        "The B0 `component` classification is treated as authoritative when "
        "that column is present. B2 does not independently reclassify those "
        "parameter paths."
    )
    lines.append("")

    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Project root: `{project_root}`")
    lines.append(f"- B0 JSON: `{b0_json_path}`")
    lines.append(f"- B0 CSV: `{b0_csv_path}`")
    lines.append("")

    lines.append("## Detected CSV Columns")
    lines.append("")
    lines.append(f"- Parameter path: `{path_column}`")
    lines.append(f"- Parameter count: `{parameter_count_column}`")

    if component_column is not None:
        lines.append(f"- Canonical component: `{component_column}`")
    else:
        lines.append("- Canonical component: not present")

    lines.append("")
    lines.append(f"Classification policy: **{classification_policy}**")
    lines.append("")

    lines.append("## Parameter Accounting")
    lines.append("")

    if expected_total is None:
        lines.append("- Expected baseline total: not reported by B0 JSON")
    else:
        lines.append(
            f"- Expected baseline total: `{expected_total:,}`"
        )

    lines.append(
        f"- Measured CSV total: `{measured_total:,}`"
    )

    if difference is not None:
        lines.append(
            f"- Difference: `{difference:,}`"
        )

    lines.append("")

    lines.append("## Component Allocation")
    lines.append("")
    lines.append(
        "| Component | Parameters | Share | Parameter Leaves |"
    )
    lines.append(
        "|---|---:|---:|---:|"
    )

    for row in summary_rows:
        lines.append(
            f"| {row['component']} "
            f"| {int(row['parameter_count']):,} "
            f"| {float(row['percentage']):.4f}% "
            f"| {int(row['leaf_count']):,} |"
        )

    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "The largest components should be investigated at matrix level before "
        "any architecture modification is made. A large parameter allocation "
        "does not by itself demonstrate redundancy or justify removing a "
        "matrix."
    )
    lines.append("")
    lines.append(
        "The next analysis stage is B3 matrix allocation analysis, which "
        "should inspect individual parameter matrices, their shapes, per-layer "
        "cost, and architectural role."
    )
    lines.append("")

    try:
        with path.open("w", encoding="utf-8") as file:
            file.write("\n".join(lines))

    except OSError as exc:
        fail(f"Could not write B2 markdown report {path}: {exc}")


def main() -> None:
    script_path = Path(__file__).resolve()
    script_directory = script_path.parent
    project_root = find_project_root(script_directory)

    output_directory = script_directory / "outputs"

    b0_json_path = output_directory / "b0_parameter_census.json"
    b0_csv_path = output_directory / "b0_parameter_leaves.csv"

    b2_json_path = output_directory / "B2_ALLOCATION_ANALYSIS.json"
    b2_markdown_path = output_directory / "B2_ALLOCATION_ANALYSIS.md"
    b2_csv_path = output_directory / "B2_COMPONENT_SUMMARY.csv"

    print(SEPARATOR)
    print("MODUS_X PARAMETER ALLOCATION — STAGE 1 STRUCTURED ANALYSIS")
    print(SEPARATOR)
    print(f"Project root : {project_root}")
    print(f"Input JSON   : {b0_json_path}")
    print(f"Input CSV    : {b0_csv_path}")
    print()

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    b0_json = load_json(b0_json_path)

    rows, fieldnames = load_csv(b0_csv_path)

    path_column = identify_path_column(fieldnames)

    parameter_count_column = identify_parameter_count_column(
        fieldnames
    )

    component_column = identify_component_column(
        fieldnames
    )

    print(
        f"Detected path column      : {path_column}"
    )
    print(
        f"Detected parameter column : {parameter_count_column}"
    )

    if component_column is not None:
        print(
            f"Detected component column : {component_column}"
        )
        print(
            "Classification policy    : canonical B0 component values"
        )
    else:
        print(
            "Detected component column : none"
        )
        print(
            "Classification policy    : fallback path inference"
        )

    (
        component_totals,
        component_leaf_counts,
        _leaf_records,
    ) = build_component_summary(
        rows=rows,
        path_column=path_column,
        parameter_count_column=parameter_count_column,
        component_column=component_column,
    )

    measured_total = sum(
        component_totals.values()
    )

    expected_total = extract_reported_total(
        b0_json
    )

    summary_rows = build_sorted_component_rows(
        component_totals=component_totals,
        component_leaf_counts=component_leaf_counts,
        measured_total=measured_total,
    )

    write_component_csv(
        path=b2_csv_path,
        summary_rows=summary_rows,
    )

    write_analysis_json(
        path=b2_json_path,
        project_root=project_root,
        b0_json_path=b0_json_path,
        b0_csv_path=b0_csv_path,
        path_column=path_column,
        parameter_count_column=parameter_count_column,
        component_column=component_column,
        expected_total=expected_total,
        measured_total=measured_total,
        summary_rows=summary_rows,
    )

    write_markdown_report(
        path=b2_markdown_path,
        project_root=project_root,
        b0_json_path=b0_json_path,
        b0_csv_path=b0_csv_path,
        path_column=path_column,
        parameter_count_column=parameter_count_column,
        component_column=component_column,
        expected_total=expected_total,
        measured_total=measured_total,
        summary_rows=summary_rows,
    )

    print()
    print(SEPARATOR)
    print("ANALYSIS COMPLETE")
    print(SEPARATOR)

    print(
        f"Measured total parameters : {measured_total:,}"
    )

    if expected_total is not None:
        difference = measured_total - expected_total

        print(
            f"Expected baseline          : {expected_total:,}"
        )
        print(
            f"Difference                 : {difference:,}"
        )
    else:
        print(
            "Expected baseline          : not available in B0 JSON"
        )

    print()
    print("Largest components:")

    for row in summary_rows:
        print(
            f"  {str(row['component']):30s} "
            f"{int(row['parameter_count']):12,} "
            f"{float(row['percentage']):8.4f}%"
        )

    print()
    print("Output files:")
    print(f"  {b2_json_path}")
    print(f"  {b2_markdown_path}")
    print(f"  {b2_csv_path}")
    print()
    print(
        "NEXT GATE: verify that the B2 component totals match the "
        "canonical B0 census before proceeding to B3 matrix analysis."
    )


if __name__ == "__main__":
    main()