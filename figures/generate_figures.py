from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "figure.dpi": 180,
        "savefig.dpi": 240,
    }
)

COLORS = {
    "modus": "#146C94",
    "mamba": "#B23A48",
    "xlstm": "#5B8C5A",
    "transformer": "#5B5F97",
    "muted": "#697386",
}


def save(name):
    plt.tight_layout()
    plt.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close()


def dense_bpc():
    models = ["Official Mamba", "Modus_X", "Official xLSTM"]
    val = [1.350538, 1.378681, 1.435132]
    test = [1.345780, 1.384180, 1.419620]
    x = np.arange(len(models))
    width = 0.34
    plt.figure(figsize=(8.2, 4.8))
    plt.bar(x - width / 2, val, width, label="Dense validation", color="#4C78A8")
    plt.bar(x + width / 2, test, width, label="Dense test", color="#F58518")
    plt.xticks(x, models)
    plt.ylabel("Bits per character (lower is better)")
    plt.ylim(1.30, 1.46)
    plt.title("Matched 80M-tier enwik8 evaluation")
    plt.suptitle("40,000 updates and 163.84M processed characters", y=0.94, fontsize=9, color=COLORS["muted"])
    plt.legend(frameon=False)
    for offset, values in [(-width / 2, val), (width / 2, test)]:
        for i, value in enumerate(values):
            plt.text(i + offset, value + 0.003, f"{value:.3f}", ha="center", fontsize=8)
    save("dense_bpc_comparison.png")


def recall():
    lengths = np.array([128, 256, 512, 1024, 2048])
    modus = np.array([96.8917, 96.9000, 96.8583, 96.6417, 96.7583])
    modus_std = np.array([0.2184, 0.2817, 0.3711, 0.2036, 0.3166])
    mamba = np.array([3.2833, 3.0667, 3.3750, 3.4750, 3.0250])
    mamba_std = np.array([0.2566, 0.4065, 0.1750, 0.1392, 0.2165])
    plt.figure(figsize=(8.2, 4.8))
    plt.errorbar(lengths, modus, yerr=modus_std, marker="o", linewidth=2.6, capsize=3, color=COLORS["modus"], label="Modus_X VectorLeanPM")
    plt.errorbar(lengths, mamba, yerr=mamba_std, marker="o", linewidth=2.2, capsize=3, color=COLORS["mamba"], label="Official Mamba")
    plt.axhline(3.125, color="#888", linestyle="--", linewidth=1.2, label="Chance (32 values)")
    plt.xscale("log", base=2)
    plt.xticks(lengths, [str(x) for x in lengths])
    plt.ylim(0, 102)
    plt.xlabel("Evaluation sequence length (trained at 128)")
    plt.ylabel("Held-out retrieval accuracy (%)")
    plt.title("Content-addressed recall length extrapolation")
    plt.suptitle("Mean +/- sample standard deviation across seeds 17, 27, and 37", y=0.94, fontsize=8, color=COLORS["muted"])
    plt.legend(frameon=False, loc="center right")
    save("associative_recall_comparison.png")


def overwrite():
    labels = ["No overwrite", "50% same-key overwrite"]
    modus = [96.8917, 87.7250]
    modus_std = [0.2184, 0.5879]
    mamba = [3.2833, 3.3417]
    mamba_std = [0.2566, 0.1607]
    x = np.arange(2)
    width = 0.34
    plt.figure(figsize=(7.6, 4.8))
    plt.bar(x - width / 2, modus, width, yerr=modus_std, capsize=3, color=COLORS["modus"], label="Modus_X")
    plt.bar(x + width / 2, mamba, width, yerr=mamba_std, capsize=3, color=COLORS["mamba"], label="Official Mamba")
    plt.axhline(3.125, color="#888", linestyle="--", linewidth=1.2, label="Chance")
    plt.xticks(x, labels)
    plt.ylim(0, 102)
    plt.ylabel("Held-out accuracy (%)")
    plt.title("Same-key overwrite stress test")
    plt.suptitle("Mean +/- sample standard deviation across three matched seeds", y=0.94, fontsize=8.5, color=COLORS["muted"])
    plt.legend(frameon=False)
    for i, values in enumerate(zip(modus, mamba)):
        for j, value in enumerate(values):
            plt.text(i + (-width / 2 if j == 0 else width / 2), value + 2, f"{value:.1f}%", ha="center", fontsize=9)
    save("overwrite_comparison.png")


def memory_projection():
    contexts = np.array([512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
    # Analytical example: 12 layers, 8 KV heads, head dim 64, bf16 K and V.
    transformer_mb = contexts * 12 * 8 * 64 * 2 * 2 / (1024**2)
    modus_mb = np.full_like(contexts, 12 * (512 * 512 + 512) * 2 / (1024**2), dtype=float)
    plt.figure(figsize=(8.2, 4.8))
    plt.plot(contexts, transformer_mb, linewidth=2.5, color=COLORS["transformer"], label="Transformer KV cache")
    plt.plot(contexts, modus_mb, linewidth=2.5, color=COLORS["modus"], label="Modus_X recurrent state")
    plt.xscale("log", base=2)
    plt.yscale("log", base=2)
    plt.xlabel("Context length")
    plt.ylabel("Inference state (MiB, analytical)")
    plt.title("Different inference-memory scaling laws")
    plt.suptitle("Illustrative bf16 configuration; projection is analytical, not a measured throughput result", y=0.94, fontsize=8.5, color=COLORS["muted"])
    plt.grid(alpha=0.18, which="both")
    plt.legend(frameon=False)
    save("memory_scaling_projection.png")


def observed_scaling():
    chars = np.array([20.48, 40.96, 81.92, 102.4, 122.88, 143.36, 163.84])
    val = np.array([1.654598, 1.482049, 1.440819, 1.379616, 1.345095, 1.325960, 1.318288])
    plt.figure(figsize=(8.2, 4.8))
    plt.plot(chars, val, marker="o", linewidth=2.5, color=COLORS["modus"])
    plt.xlabel("Processed characters (millions)")
    plt.ylabel("Sparse validation BPC")
    plt.title("Observed Modus_X 80M training curve")
    plt.suptitle("Measured checkpoints only; no unobserved extrapolation is plotted", y=0.94, fontsize=8.5, color=COLORS["muted"])
    plt.grid(alpha=0.2)
    for x, y in zip(chars, val):
        plt.annotate(f"{y:.3f}", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7)
    save("modus_x_observed_scaling.png")


def evidence_map():
    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    ax.axis("off")
    columns = [
        ("Predictive modeling", "Official Mamba\nwins enwik8 BPC", COLORS["mamba"]),
        ("Associative recall", "Modus_X\n~97% vs chance", COLORS["modus"]),
        ("Overwrite", "Modus_X\n88.85% vs 3.43%", COLORS["modus"]),
        ("Inference memory", "Modus_X state\nconstant in context", COLORS["modus"]),
    ]
    for i, (heading, body, color) in enumerate(columns):
        x = 0.02 + i * 0.245
        ax.add_patch(plt.Rectangle((x, 0.22), 0.22, 0.58, facecolor="#F7F9FC", edgecolor=color, linewidth=2))
        ax.text(x + 0.11, 0.68, heading, ha="center", va="center", weight="bold", fontsize=10)
        ax.text(x + 0.11, 0.46, body, ha="center", va="center", color=color, weight="bold", fontsize=12)
    ax.text(0.5, 0.08, "The evidence supports capability specialization, not a one-metric universal ranking.", ha="center", fontsize=10, color=COLORS["muted"])
    save("evidence_summary.png")


def component_ablation():
    variants = ["ScalarPM", "VectorLeanPM", "MatrixOnly", "VectorOnly"]
    no_overwrite = np.array([96.3833, 96.7583, 96.9917, 3.1000])
    no_overwrite_std = np.array([0.4964, 0.3166, 0.4274, 0.1090])
    overwrite_50 = np.array([88.1083, 87.7583, 87.6250, 3.3083])
    overwrite_50_std = np.array([0.4474, 0.7767, 0.7454, 0.5058])
    x = np.arange(len(variants))
    width = 0.36
    colors = ["#4C78A8", COLORS["modus"], "#2E8B57", COLORS["mamba"]]
    plt.figure(figsize=(9.0, 5.0))
    for index, color in enumerate(colors):
        plt.bar(x[index] - width / 2, no_overwrite[index], width, yerr=no_overwrite_std[index], capsize=3, color=color)
        plt.bar(x[index] + width / 2, overwrite_50[index], width, yerr=overwrite_50_std[index], capsize=3, color=color, alpha=0.58)
    plt.axhline(3.125, color="#888", linestyle="--", linewidth=1.2, label="Chance (32 values)")
    plt.xticks(x, variants)
    plt.ylabel("Length-2048 held-out accuracy (%)")
    plt.ylim(0, 103)
    plt.title("Matrix stream carries the controlled associative-recall result")
    plt.suptitle("Mean +/- sample standard deviation across seeds 17, 27, and 37; light bars are 50% same-key overwrite", y=0.94, fontsize=8, color=COLORS["muted"])
    plt.legend(frameon=False, loc="lower left")
    save("component_ablation.png")


if __name__ == "__main__":
    dense_bpc()
    recall()
    overwrite()
    memory_projection()
    observed_scaling()
    evidence_map()
    component_ablation()
    print("Generated figures in", OUT)
