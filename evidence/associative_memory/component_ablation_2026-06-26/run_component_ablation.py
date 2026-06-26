from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


MODELS = ("ScalarPM", "VectorLeanPM", "MatrixOnly", "VectorOnly")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seeds", default="17,27,37")
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in args.seeds.split(",") if value]

    for overwrite in (0.0, 0.5):
        for seed in seeds:
            for model in MODELS:
                slug = f"{model.lower()}_seed{seed}_overwrite{overwrite:g}".replace(".", "p")
                run_dir = outdir / slug
                if (run_dir / "results.json").exists():
                    print("SKIP_EXISTING", run_dir, flush=True)
                    continue
                router_hidden = "128" if model == "ScalarPM" else "16"
                command = [
                    sys.executable,
                    "-u",
                    args.runner,
                    "--models",
                    model,
                    "--seed",
                    str(seed),
                    "--epochs",
                    str(args.epochs),
                    "--router-hidden",
                    router_hidden,
                    "--overwrite-rate",
                    str(overwrite),
                    "--outdir",
                    str(run_dir),
                ]
                print("RUN_COMPONENT_ABLATION", " ".join(command), flush=True)
                subprocess.run(command, check=True)

    rows = []
    for overwrite in (0.0, 0.5):
        for seed in seeds:
            for model in MODELS:
                slug = f"{model.lower()}_seed{seed}_overwrite{overwrite:g}".replace(".", "p")
                result = json.loads((outdir / slug / "results.json").read_text(encoding="utf-8"))
                payload = result[model]
                rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "overwrite_rate": overwrite,
                        "params": payload["params"],
                        "train_best": payload["train_best"],
                        **{
                            f"length_{length}": payload[str(length)]
                            for length in (128, 256, 512, 1024, 2048)
                        },
                    }
                )

    fields = list(rows[0])
    with (outdir / "seed_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for overwrite in (0.0, 0.5):
        for model in MODELS:
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
            for metric in (
                "train_best",
                "length_128",
                "length_256",
                "length_512",
                "length_1024",
                "length_2048",
            ):
                values = [row[metric] for row in group]
                aggregate[f"{metric}_mean"] = statistics.mean(values)
                aggregate[f"{metric}_stdev"] = statistics.stdev(values)
            summary.append(aggregate)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("COMPONENT_ABLATION_SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
