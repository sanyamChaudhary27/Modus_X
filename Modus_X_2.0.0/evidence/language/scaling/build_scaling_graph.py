from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


MEMORY_FEEDBACK_ANCHOR = {
    "family": "MemoryFeedbackArchive",
    "label": "MemoryFeedbackArchive",
    "params": 47_437_768,
    "processed_characters": 102_400_000,
    "dense_validation_bpc": 1.459723,
    "dense_test_bpc": 1.465006,
    "schedule": "6e-4 to 81.92M; 3e-4 to 102.4M",
    "evidence": (
        "Modus_X_2.0.0/experiments/enwik8_current_archive/"
        "MEMORY_FEEDBACK_ARCHIVE_V0_2026-07-12.md"
    ),
    "comparison_class": "fixed_budget_scaling",
    "approximate": False,
    "metric_type": "dense_test_bpc",
}

HISTORICAL_ENDPOINTS = [
    {
        "family": "Modus X v1.1.1",
        "label": "Modus X v1.1.1 long run",
        "params": 42_692_772,
        "processed_characters": 500_000_000,
        "dense_validation_bpc": 1.389,
        "dense_test_bpc": 1.4106,
        "schedule": "historical long-run recipe",
        "evidence": "Modus_X_v1.0.1/TPU_optim/BPC_1_1_PUSH_MEMORY.md",
        "comparison_class": "historical_endpoint",
        "approximate": True,
        "metric_type": "dense_test_bpc",
    },
    {
        "family": "Modus X v1.1.1",
        "label": "Modus X v1.1.1",
        "params": 82_764_964,
        "processed_characters": 163_840_000,
        "dense_validation_bpc": 1.378681,
        "dense_test_bpc": 1.384180,
        "schedule": "published v1.1.1 recipe",
        "evidence": "Modus_X_v1.1.1/evidence/RESULTS_LEDGER.md",
        "comparison_class": "historical_endpoint",
        "approximate": False,
        "metric_type": "dense_test_bpc",
    },
    {
        "family": "Official Mamba",
        "label": "Official Mamba",
        "params": 81_462_656,
        "processed_characters": 163_840_000,
        "dense_validation_bpc": 1.350538,
        "dense_test_bpc": 1.345780,
        "schedule": "matched 80M-tier baseline protocol",
        "evidence": "Modus_X_v1.1.1/evidence/RESULTS_LEDGER.md",
        "comparison_class": "historical_endpoint",
        "approximate": False,
        "metric_type": "dense_test_bpc",
    },
    {
        "family": "Official xLSTM",
        "label": "Official xLSTM",
        "params": 76_649_664,
        "processed_characters": 163_840_000,
        "dense_validation_bpc": 1.435132,
        "dense_test_bpc": 1.419620,
        "schedule": "matched 80M-tier baseline protocol",
        "evidence": "Modus_X_v1.1.1/evidence/RESULTS_LEDGER.md",
        "comparison_class": "historical_endpoint",
        "approximate": False,
        "metric_type": "dense_test_bpc",
    },
]


def load_point(path: Path) -> dict:
    point = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "params",
        "processed_characters",
        "dense_validation_bpc",
        "dense_test_bpc",
    }
    missing = sorted(required - point.keys())
    if missing:
        raise ValueError(f"{path} is missing {missing}")
    point["comparison_class"] = "fixed_budget_scaling"
    point["approximate"] = False
    point["metric_type"] = "dense_test_bpc"
    point["evidence"] = str(path)
    point["label"] = "MemoryFeedbackArchive"
    point["schedule"] = "; ".join(
        f"{segment['lr']:g} through {segment['through_characters'] / 1e6:g}M"
        for segment in point["schedule"]
    )
    return point


def write_table(rows: list[dict], path: Path) -> None:
    fields = [
        "comparison_class",
        "approximate",
        "metric_type",
        "family",
        "label",
        "params",
        "processed_characters",
        "dense_validation_bpc",
        "dense_test_bpc",
        "schedule",
        "evidence",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=Path, nargs="+", required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    measured = [MEMORY_FEEDBACK_ANCHOR, *(load_point(path) for path in args.points)]
    measured.sort(key=lambda row: row["params"])
    if any(row["processed_characters"] != 102_400_000 for row in measured):
        raise ValueError("Every MemoryFeedback scaling point must use 102.4M characters")
    if len(measured) != 3:
        raise ValueError("The pre-registered scaling curve requires exactly three points")
    new_points = measured[1:]
    if len({point.get("dataset_sha256") for point in new_points}) != 1:
        raise ValueError("The two overnight points used different enwik8 files")
    expected_params = {81_486_728, 99_438_920}
    if {point["params"] for point in new_points} != expected_params:
        raise ValueError("Overnight parameter counts do not match the frozen protocol")

    rows = [*measured, *HISTORICAL_ENDPOINTS]
    (args.outdir / "scaling_evidence.json").write_text(
        json.dumps(
            {
                "interpretation": {
                    "fixed_budget_panel": (
                        "Three MemoryFeedbackArchive scales at 102.4M processed characters."
                    ),
                    "historical_panel": (
                        "Context only; points used different architectures and/or budgets."
                    ),
                    "claim_limit": (
                        "A monotonic three-point curve is scaling evidence, not a universal law."
                    ),
                },
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_table(rows, args.outdir / "scaling_points.csv")

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15.5, 7.2),
        gridspec_kw={"width_ratios": [1.0, 1.28]},
        constrained_layout=True,
    )

    axis = axes[0]
    x = [row["params"] / 1e6 for row in measured]
    test = [row["dense_test_bpc"] for row in measured]
    validation = [row["dense_validation_bpc"] for row in measured]
    axis.plot(x, test, marker="o", linewidth=2.3, label="Dense test")
    axis.plot(x, validation, marker="s", linewidth=1.8, label="Dense validation")
    for row in measured:
        axis.annotate(
            f"{row['params'] / 1e6:.1f}M",
            (row["params"] / 1e6, row["dense_test_bpc"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    axis.set_title("MemoryFeedbackArchive at 102.4M characters")
    axis.set_xlabel("Parameters (millions)")
    axis.set_ylabel("Bits per character (lower is better)")
    axis.legend()
    axis.text(
        90.5,
        (measured[1]["dense_test_bpc"] + measured[2]["dense_test_bpc"]) / 2,
        "near-flat",
        color="#4b5563",
        fontsize=9,
        ha="center",
        va="bottom",
    )

    axis = axes[1]
    colors = {
        "MemoryFeedbackArchive": "#7a5195",
        "Modus X v1.1.1": "#1877b8",
        "Official Mamba": "#d34a3a",
        "Official xLSTM": "#62752e",
    }
    comparison_rows = sorted(
        [*measured, *HISTORICAL_ENDPOINTS],
        key=lambda row: row["dense_test_bpc"],
    )
    y_positions = list(range(len(comparison_rows)))
    for y, row in zip(y_positions, comparison_rows):
        axis.scatter(
            row["dense_test_bpc"],
            y,
            s=95,
            color=colors[row["family"]],
            marker="X" if row["approximate"] else "o",
            zorder=3,
        )
        axis.text(
            row["dense_test_bpc"] + 0.0015,
            y,
            f"{row['dense_test_bpc']:.6f}" + (" approx." if row["approximate"] else ""),
            va="center",
            fontsize=9,
            color="#20252b",
        )
    labels = [
        (
            f"{row['label']}  |  {row['params'] / 1e6:.2f}M params"
            f"  |  {'~' if row['approximate'] else ''}{row['processed_characters'] / 1e6:g}M chars"
        )
        for row in comparison_rows
    ]
    axis.set_yticks(y_positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(
        min(row["dense_test_bpc"] for row in comparison_rows) - 0.006,
        max(row["dense_test_bpc"] for row in comparison_rows) + 0.018,
    )
    axis.set_title("All endpoints use dense test BPC")
    axis.set_xlabel("Dense test bits per character (lower is better)")
    axis.set_ylabel("")
    axis.grid(axis="y", visible=False)

    figure.suptitle("Measured Modus X language-model scaling evidence", fontsize=15)
    figure.savefig(args.outdir / "memory_feedback_scaling.png", dpi=180)
    print("SCALING_GRAPH_READY", args.outdir, flush=True)


if __name__ == "__main__":
    main()
