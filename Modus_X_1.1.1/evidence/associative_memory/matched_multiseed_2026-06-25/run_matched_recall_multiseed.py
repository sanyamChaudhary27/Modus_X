from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modus-script", required=True)
    parser.add_argument("--mamba-script", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seeds", default="17,27,37")
    parser.add_argument("--epochs", type=int, default=12)
    return parser.parse_args()


def run(command):
    print("RUN", " ".join(map(str, command)), flush=True)
    subprocess.run(command, check=True)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in args.seeds.split(",") if value]

    jobs = []
    for overwrite in (0.0, 0.5):
        for seed in seeds:
            jobs.extend(
                [
                    ("Modus_X_VectorLeanPM", seed, overwrite),
                    ("Official_Mamba", seed, overwrite),
                ]
            )

    for model, seed, overwrite in jobs:
        slug = f"{model.lower()}_seed{seed}_overwrite{overwrite:g}".replace(".", "p")
        run_dir = outdir / slug
        if (run_dir / "results.json").exists():
            print("SKIP_EXISTING", run_dir, flush=True)
            continue
        if model == "Modus_X_VectorLeanPM":
            command = [
                sys.executable,
                "-u",
                args.modus_script,
                "--models",
                "VectorLeanPM",
                "--router-hidden",
                "16",
                "--seed",
                str(seed),
                "--epochs",
                str(args.epochs),
                "--overwrite-rate",
                str(overwrite),
                "--outdir",
                str(run_dir),
            ]
        else:
            command = [
                sys.executable,
                "-u",
                args.mamba_script,
                "--seed",
                str(seed),
                "--overwrite-rate",
                str(overwrite),
                "--test-lens",
                "128,256,512,1024,2048",
                "--outdir",
                str(run_dir),
            ]
        run(command)

    rows = []
    for model, seed, overwrite in jobs:
        slug = f"{model.lower()}_seed{seed}_overwrite{overwrite:g}".replace(".", "p")
        result = json.loads((outdir / slug / "results.json").read_text(encoding="utf-8"))
        if model == "Modus_X_VectorLeanPM":
            payload = result["VectorLeanPM"]
            row = {
                "model": model,
                "seed": seed,
                "overwrite_rate": overwrite,
                "params": payload["params"],
                "train_best": payload["train_best"],
                **{f"length_{length}": payload[str(length)] for length in (128, 256, 512, 1024, 2048)},
            }
        else:
            row = {
                "model": model,
                "seed": seed,
                "overwrite_rate": overwrite,
                "params": result["params"],
                "train_best": max(epoch["accuracy"] for epoch in result["epochs"]),
                **{
                    f"length_{length}": result["length_accuracy"][str(length)]
                    for length in (128, 256, 512, 1024, 2048)
                },
            }
        rows.append(row)

    fields = list(rows[0])
    with (outdir / "seed_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for model in ("Modus_X_VectorLeanPM", "Official_Mamba"):
        for overwrite in (0.0, 0.5):
            group = [
                row for row in rows
                if row["model"] == model and row["overwrite_rate"] == overwrite
            ]
            aggregate = {
                "model": model,
                "overwrite_rate": overwrite,
                "seeds": seeds,
                "runs": len(group),
                "params": group[0]["params"],
            }
            for metric in ("train_best", "length_128", "length_256", "length_512", "length_1024", "length_2048"):
                values = [row[metric] for row in group]
                aggregate[f"{metric}_mean"] = statistics.mean(values)
                aggregate[f"{metric}_stdev"] = statistics.stdev(values)
            summary.append(aggregate)

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("MATCHED_RECALL_MULTISEED_SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
