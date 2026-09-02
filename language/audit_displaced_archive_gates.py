from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from models import (
    ModelConfig,
    count_params,
    modus_x_adaptive_preconditioned_archive_diagnostics,
    modus_x_attention_to_write_archive_diagnostics,
    modus_x_displaced_archive_diagnostics,
    modus_x_memory_feedback_archive_diagnostics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--windows", type=int, default=128)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument(
        "--variant",
        choices=(
            "displaced",
            "memory_feedback",
            "adaptive_preconditioned",
            "attention_to_write",
        ),
        default="displaced",
    )
    return parser.parse_args()


def batch_at(data: np.ndarray, starts: np.ndarray, seq_len: int) -> np.ndarray:
    offsets = np.arange(seq_len)
    return data[starts[:, None] + offsets[None, :]].astype(np.int32)


def main() -> None:
    args = parse_args()
    if args.batch % jax.device_count():
        raise ValueError("--batch must divide the TPU device count")
    args.outdir.mkdir(parents=True, exist_ok=True)
    with args.checkpoint.open("rb") as handle:
        state = pickle.load(handle)

    cfg = ModelConfig(
        vocab_size=256,
        embed_dim=512,
        hidden_dim=1536,
        ax_res=512,
        n_layers=12,
        n_heads_attn=8,
        seq_len=512,
        mamba_state_dim=512,
        vector_router=False,
        router_hidden=32,
    )
    expected_params = {
        "displaced": 47_044_600,
        "memory_feedback": 47_437_768,
        "adaptive_preconditioned": 47_038_396,
        "attention_to_write": 47_431_807,
    }[args.variant]
    actual_params = count_params(state["params"])
    if actual_params != expected_params:
        raise ValueError(f"Expected {expected_params} params, found {actual_params}")

    mesh = Mesh(np.array(jax.devices(), dtype=object), ("data",))
    replicated = NamedSharding(mesh, P())
    batch_sharding = NamedSharding(mesh, P("data", None))
    params = jax.device_put(state["params"], replicated)

    diagnostic_fn = {
        "displaced": modus_x_displaced_archive_diagnostics,
        "memory_feedback": modus_x_memory_feedback_archive_diagnostics,
        "adaptive_preconditioned": modus_x_adaptive_preconditioned_archive_diagnostics,
        "attention_to_write": modus_x_attention_to_write_archive_diagnostics,
    }[args.variant]

    @jax.jit
    def diagnose(batch):
        return jax.vmap(lambda tokens: diagnostic_fn(params, tokens, cfg))(batch)

    raw = np.memmap(args.data_path, dtype=np.uint8, mode="r")
    splits = {
        "train_tail": raw[85_000_000:90_000_000],
        "validation": raw[90_000_000:95_000_000],
        "test": raw[95_000_000:100_000_000],
    }
    rng = np.random.default_rng(args.seed)
    report = {
        "checkpoint_step": int(state["step"]),
        "variant": args.variant,
        "params": actual_params,
        "windows": args.windows,
        "results": {},
    }
    for split_name, split in splits.items():
        max_start = len(split) - cfg.seq_len
        starts = np.sort(rng.choice(max_start + 1, size=args.windows, replace=False))
        accumulators = None
        count = 0
        for offset in range(0, len(starts), args.batch):
            selected = starts[offset : offset + args.batch]
            real_count = len(selected)
            if real_count < args.batch:
                selected = np.pad(selected, (0, args.batch - real_count), mode="edge")
            batch = jax.device_put(batch_at(split, selected, cfg.seq_len), batch_sharding)
            values = jax.device_get(diagnose(batch))
            if accumulators is None:
                accumulators = {
                    name: {
                        # Most variants report all model layers. Sparse
                        # mechanisms such as AttentionToWrite report only the
                        # layers that contain that mechanism (3 of 12).
                        "sum": np.zeros(np.asarray(value).shape[2], dtype=np.float64),
                        "sumsq": np.zeros(np.asarray(value).shape[2], dtype=np.float64),
                        "above_half": np.zeros(
                            np.asarray(value).shape[2], dtype=np.float64
                        ),
                    }
                    for name, value in values.items()
                }
            for name, value in values.items():
                value = np.asarray(value[:real_count])
                accumulators[name]["sum"] += value.sum(axis=(0, 1))
                accumulators[name]["sumsq"] += np.square(value).sum(axis=(0, 1))
                accumulators[name]["above_half"] += (value > 0.5).sum(axis=(0, 1))
            count += real_count * cfg.seq_len

        split_report = {}
        for name, accumulator in accumulators.items():
            mean = accumulator["sum"] / count
            variance = np.maximum(accumulator["sumsq"] / count - np.square(mean), 0.0)
            split_report[name] = {
                "global_mean": float(mean.mean()),
                "mean_by_layer": mean.tolist(),
                "std_by_layer": np.sqrt(variance).tolist(),
                "fraction_above_0p5_by_layer": (
                    accumulator["above_half"] / count
                ).tolist(),
            }
        report["results"][split_name] = split_report
        print(
            "DISPLACED_ARCHIVE_DIAGNOSTICS",
            split_name,
            json.dumps(
                {
                    name: round(values["global_mean"], 6)
                    for name, values in split_report.items()
                }
            ),
            flush=True,
        )

    output = args.outdir / f"{args.variant}_gate_diagnostics.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("DISPLACED_ARCHIVE_DIAGNOSTICS_SAVED", output, flush=True)


if __name__ == "__main__":
    main()
