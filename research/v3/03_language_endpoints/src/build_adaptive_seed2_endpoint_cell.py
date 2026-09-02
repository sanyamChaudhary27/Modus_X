from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_base_builder():
    spec = importlib.util.spec_from_file_location(
        "segment_retention_endpoint_base", ROOT / "build_endpoint_cell.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_cell():
    cell = load_base_builder().build_cell()
    replacements = (
        ("seed-1", "seed-2"),
        ("seed1", "seed2"),
        ("!= 1: return None", "!= 2: return None"),
        ("'--seed', '1'", "'--seed', '2'"),
        ("control_dense_validation = 1.459723", "control_dense_validation = 1.441670000553131"),
        ("/ 22570.22", "/ 22540.821273477"),
        ("'dense_test_bpc_report_only': 1.465006, 'elapsed_s': 22570.22",
         "'dense_test_bpc_report_only': 1.4431464076042175, 'elapsed_s': 22540.821273477"),
        ("'replicate the 102.4M endpoint on seeds 2 and 3' if decision['endpoint_pass']\n"
         "                    else 'freeze segment-scale retention at mechanistic evidence only'",
         "'stop immediate campaign at 2/2; reserve seed 3 for publication closure' if "
         "(decision['endpoint_pass'] and decision['strong_language_win'])\n"
         "                    else 'run seed 3 as the adaptive tiebreaker'"),
    )
    for old, new in replacements:
        if old not in cell:
            raise RuntimeError(f"Endpoint template changed; missing replacement anchor: {old}")
        cell = cell.replace(old, new)
    marker = "decision['next'] = ("
    insertion = (
        "decision['adaptive_stop_pass'] = bool(decision['endpoint_pass'] and "
        "decision['strong_language_win'])\n"
    )
    if marker not in cell:
        raise RuntimeError("Endpoint template is missing the adaptive decision anchor")
    cell = cell.replace(marker, insertion + marker, 1)
    ast.parse(cell)
    return cell


def main():
    cell = build_cell()
    generated = ROOT / "generated_cells" / "KAGGLE_TPU_SEGMENT_RETENTION_SEED2_102P4M_ADAPTIVE.py"
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_text(cell, encoding="utf-8", newline="\n")
    print("SEGMENT_RETENTION_ADAPTIVE_SEED2_CELL", generated)


if __name__ == "__main__":
    main()
