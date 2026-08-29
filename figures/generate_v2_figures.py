from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent


def label_bars(ax, bars, suffix="", digits=2):
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.8,
            f"{value:.{digits}f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#263238",
        )


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfcfd",
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))
    fig.suptitle(
        "Modus_X 2.1 measured evidence: language coordination and bounded memory",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    # Panel A: matched 47M language result.
    labels = ["Dense validation", "Dense test"]
    current = [1.485020, 1.492694]
    feedback = [1.459723, 1.465006]
    x = np.arange(len(labels))
    width = 0.34
    axes[0].bar(
        x - width / 2, current, width, color="#9aa6b2", label="CurrentArchive"
    )
    axes[0].bar(
        x + width / 2,
        feedback,
        width,
        color="#087e8b",
        label="MemoryFeedbackArchive",
    )
    axes[0].set_ylim(1.43, 1.51)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Bits per character (lower is better)")
    axes[0].set_title("A. Matched ~47M language result")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)
    for i, (base, better) in enumerate(zip(current, feedback)):
        axes[0].text(
            i,
            better - 0.006,
            f"-{base - better:.3f}",
            ha="center",
            color="#065a62",
            fontsize=8,
            fontweight="bold",
        )

    # Panel B: equal constrained state mixed clean/update result.
    labels = ["Overall", "Clean", "Updated"]
    archive = [77.9514, 77.9080, 77.9948]
    kv = [16.3845, 14.1927, 18.5764]
    x = np.arange(len(labels))
    bars1 = axes[1].bar(
        x - width / 2, archive, width, color="#087e8b", label="CurrentArchive"
    )
    bars2 = axes[1].bar(
        x + width / 2, kv, width, color="#f4a261", label="Transformer KV-32"
    )
    axes[1].set_ylim(0, 92)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("B. Equal 16,512-byte recurrent state")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=0.2)
    label_bars(axes[1], bars1, "%", 1)
    label_bars(axes[1], bars2, "%", 1)

    # Panel C: measured state-budget crossover.
    conditions = ["Constrained state", "Full context fits"]
    archive = [72.92, 71.03]
    kv = [17.71, 98.63]
    x = np.arange(len(conditions))
    bars1 = axes[2].bar(
        x - width / 2, archive, width, color="#087e8b", label="CurrentArchive"
    )
    bars2 = axes[2].bar(
        x + width / 2, kv, width, color="#355070", label="Transformer KV"
    )
    axes[2].set_ylim(0, 112)
    axes[2].set_xticks(x, conditions)
    axes[2].set_ylabel("Accuracy (%)")
    axes[2].set_title("C. State-budget crossover")
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].grid(axis="y", alpha=0.2)
    label_bars(axes[2], bars1, "%", 1)
    label_bars(axes[2], bars2, "%", 1)

    fig.text(
        0.5,
        -0.03,
        "Measured protocols are distinct. Panel A is dense enwik8 BPC; panels B-C are controlled versioned-memory tasks.",
        ha="center",
        fontsize=8,
        color="#455a64",
    )
    fig.tight_layout()
    output = OUT / "v2_measured_evidence.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
