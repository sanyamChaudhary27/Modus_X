from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--target-chars", type=int, default=20_480_000)
    parser.add_argument("--checkpoint-chars", type=int, default=4_096_000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--eval-chunks", type=int, default=128)
    parser.add_argument("--eval-batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lr", default="6e-4")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    command = [
        sys.executable, "-u", str(root / "tpu_lm_train.py"),
        "--data-path", args.data_path,
        "--outdir", args.outdir,
        "--model", "Modus_X_MemoryFeedbackArchive_DeepSupervision",
        "--batch", str(args.batch),
        "--target-chars", str(args.target_chars),
        "--checkpoint-chars", str(args.checkpoint_chars),
        "--eval-batch", str(args.eval_batch),
        "--eval-chunks", str(args.eval_chunks),
        "--embed-dim", "512",
        "--hidden-dim", "1536",
        "--state-dim", "512",
        "--n-layers", "12",
        "--router-hidden", "32",
        "--lr", str(args.lr),
        "--auxiliary-weight", "0.05",
        "--weight-decay", "1e-4",
        "--schedule", "constant",
        "--input-seq-len", "512",
        "--loss-tail", "512",
        "--auxiliary-layers", "6",
        "--future-targets", "2",
        "--future-target-weight", "0.5",
        "--seed", str(args.seed),
    ]
    print("RUN_MEMORY_FEEDBACK_ARCHIVE_SMOKE", " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
