from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from models import ModelConfig, count_params, make_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--eval-batch", type=int, default=8)
    parser.add_argument("--sample-windows", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument(
        "--variant",
        choices=(
            "current_archive",
            "displaced",
            "memory_feedback",
            "feedback_attention_to_write",
        ),
        default="current_archive",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def batch_at(data: np.ndarray, starts: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.arange(seq_len + 1)
    chunks = data[starts[:, None] + offsets[None, :]]
    return chunks[:, :-1].astype(np.int32), chunks[:, 1:].astype(np.int32)


def make_starts(length: int, seq_len: int, count: int, seed: int) -> dict[str, np.ndarray]:
    max_start = length - seq_len - 1
    rng = np.random.default_rng(seed)
    count = min(count, max_start + 1)
    return {
        "linspace": np.linspace(0, max_start, count, dtype=np.int64),
        "random": np.sort(rng.choice(max_start + 1, size=count, replace=False)),
        "dense_offset_0": np.arange(0, max_start + 1, seq_len, dtype=np.int64),
        "dense_offset_half": np.arange(seq_len // 2, max_start + 1, seq_len, dtype=np.int64),
    }


def summarize(values: np.ndarray) -> dict:
    return {
        "windows": int(values.size),
        "bpc": float(values.mean()),
        "std_window_bpc": float(values.std()),
        "sem_bpc": float(values.std() / math.sqrt(values.size)),
        "min_window_bpc": float(values.min()),
        "max_window_bpc": float(values.max()),
    }


def main() -> None:
    args = parse_args()
    if args.eval_batch % jax.device_count():
        raise ValueError("--eval-batch must divide the device count")
    args.outdir.mkdir(parents=True, exist_ok=True)

    with args.checkpoint.open("rb") as handle:
        state = pickle.load(handle)

    expected_params = {
        "current_archive": 47_038_396,
        "displaced": 47_044_600,
        "memory_feedback": 47_437_768,
        "feedback_attention_to_write": 47_831_179,
    }[args.variant]
    actual_params = count_params(state["params"])
    if actual_params != expected_params:
        raise ValueError(f"Expected {expected_params} params, found {actual_params}")

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
    model_name = {
        "current_archive": "Modus_X_CurrentArchive_DeepSupervision",
        "displaced": "Modus_X_DisplacedArchive_DeepSupervision",
        "memory_feedback": "Modus_X_MemoryFeedbackArchive_DeepSupervision",
        "feedback_attention_to_write": (
            "Modus_X_FeedbackAttentionToWriteArchive_DeepSupervision"
        ),
    }[args.variant]
    _, fwd_fn = make_model(
        model_name,
        jax.random.key(1),
        cfg,
        auxiliary_layers=(6,),
        future_target_count=1,
    )
    mesh = Mesh(np.array(jax.devices(), dtype=object), ("data",))
    batch_sharding = NamedSharding(mesh, P("data", None))
    replicated = NamedSharding(mesh, P())
    params = jax.device_put(state["params"], replicated)

    @jax.jit
    def window_nll(x, y):
        outputs = jax.vmap(lambda sequence: fwd_fn(params, sequence))(x)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        logp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
        return -jnp.take_along_axis(logp, y[..., None], axis=-1).mean(axis=(1, 2))

    raw = np.memmap(args.data_path, dtype=np.uint8, mode="r")
    if len(raw) != 100_000_000:
        raise ValueError(f"Expected 100,000,000 enwik8 bytes, found {len(raw)}")
    splits = {
        "train_tail": raw[85_000_000:90_000_000],
        "validation": raw[90_000_000:95_000_000],
        "test": raw[95_000_000:100_000_000],
    }
    report = {
        "model": model_name,
        "variant": args.variant,
        "params": actual_params,
        "checkpoint_step": int(state["step"]),
        "processed_characters": int(state["step"]) * 4096,
        "dataset_bytes": int(len(raw)),
        "dataset_sha256": sha256(args.data_path),
        "devices": [str(device) for device in jax.devices()],
        "config": {
            "embed_dim": 512,
            "hidden_dim": 1536,
            "state_dim": 512,
            "n_layers": 12,
            "router_hidden": 32,
            "seq_len": 512,
            "matrix_states_per_layer": 2,
        },
        "results": {},
    }
    started = time.perf_counter()
    for split_index, (split_name, split) in enumerate(splits.items()):
        report["results"][split_name] = {}
        schemes = make_starts(len(split), cfg.seq_len, args.sample_windows, args.seed + split_index)
        for scheme_name, starts in schemes.items():
            values = []
            for offset in range(0, len(starts), args.eval_batch):
                selected = starts[offset : offset + args.eval_batch]
                real_count = len(selected)
                if real_count < args.eval_batch:
                    selected = np.pad(selected, (0, args.eval_batch - real_count), mode="edge")
                x, y = batch_at(split, selected, cfg.seq_len)
                losses = window_nll(
                    jax.device_put(x, batch_sharding),
                    jax.device_put(y, batch_sharding),
                )
                values.extend(np.asarray(losses[:real_count]) / math.log(2))
            summary = summarize(np.asarray(values))
            report["results"][split_name][scheme_name] = summary
            print("CURRENT_ARCHIVE_AUDIT", split_name, scheme_name, json.dumps(summary), flush=True)

    report["elapsed_s"] = time.perf_counter() - started
    output = args.outdir / "evaluation_audit.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"CURRENT_ARCHIVE_AUDIT_SAVED {output}", flush=True)


if __name__ == "__main__":
    main()
