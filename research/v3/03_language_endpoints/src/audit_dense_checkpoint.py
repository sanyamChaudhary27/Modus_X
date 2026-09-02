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

import models
from segment_scale_trainer import segment_scale_memory_feedback_stateful


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--eval-batch", type=int, default=8)
    parser.add_argument("--sample-windows", type=int, default=1024)
    return parser.parse_args()


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def batch_at(data, starts, seq_len):
    offsets = np.arange(seq_len + 1)
    chunks = data[starts[:, None] + offsets[None, :]]
    return chunks[:, :-1].astype(np.int32), chunks[:, 1:].astype(np.int32)


def starts_for(length, seq_len, count, seed):
    maximum = length - seq_len - 1
    rng = np.random.default_rng(seed)
    count = min(count, maximum + 1)
    return {
        "linspace": np.linspace(0, maximum, count, dtype=np.int64),
        "random": np.sort(rng.choice(maximum + 1, size=count, replace=False)),
        "dense_offset_0": np.arange(0, maximum + 1, seq_len, dtype=np.int64),
        "dense_offset_half": np.arange(seq_len // 2, maximum + 1, seq_len, dtype=np.int64),
    }


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "windows": int(values.size), "bpc": float(values.mean()),
        "std_window_bpc": float(values.std()),
        "sem_bpc": float(values.std() / math.sqrt(values.size)),
        "min_window_bpc": float(values.min()), "max_window_bpc": float(values.max()),
    }


def main():
    args = parse_args()
    if jax.default_backend() != "tpu" or jax.device_count() != 8:
        raise RuntimeError(f"Expected TPU v5e-8, found {jax.default_backend()} with {jax.device_count()} devices")
    if args.data_path.stat().st_size != 100_000_000:
        raise RuntimeError("Expected canonical 100,000,000-byte enwik8")
    if args.eval_batch % jax.device_count():
        raise ValueError("--eval-batch must divide the TPU device count")
    args.outdir.mkdir(parents=True, exist_ok=True)
    with args.checkpoint.open("rb") as handle:
        state = pickle.load(handle)
    if int(state.get("step", -1)) != 25_000:
        raise RuntimeError(f"Expected step 25,000 checkpoint, found {state.get('step')}")

    models.modus_x_memory_feedback_archive_layer_fwd_stateful = segment_scale_memory_feedback_stateful
    cfg = models.ModelConfig(
        vocab_size=256, embed_dim=512, hidden_dim=1536, ax_res=512,
        n_layers=12, n_heads_attn=8, seq_len=512, mamba_state_dim=512,
        vector_router=True, router_hidden=32,
    )
    _, forward = models.make_model(
        "Modus_X_MemoryFeedbackArchive_DeepSupervision", jax.random.key(1), cfg,
        auxiliary_layers=(6,), future_target_count=1,
    )
    mesh = Mesh(np.array(jax.devices(), dtype=object), ("data",))
    batch_sharding = NamedSharding(mesh, P("data", None))
    replicated = NamedSharding(mesh, P())
    params = jax.device_put(state["params"], replicated)

    @jax.jit
    def window_nll(x, y):
        outputs = jax.vmap(lambda sequence: forward(params, sequence))(x)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        logp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
        return -jnp.take_along_axis(logp, y[..., None], axis=-1).mean(axis=(1, 2))

    raw = np.memmap(args.data_path, dtype=np.uint8, mode="r")
    splits = {
        "train_tail": raw[85_000_000:90_000_000],
        "validation": raw[90_000_000:95_000_000],
        "test": raw[95_000_000:100_000_000],
    }
    report = {
        "audit": "segment_scale_retention_47m_step25000_dense_reset_windows",
        "checkpoint_step": 25_000, "dataset_sha256": sha256(args.data_path),
        "test_role": "report_only_after_frozen_endpoint", "results": {},
    }
    started = time.perf_counter()
    for split_index, (split_name, split) in enumerate(splits.items()):
        report["results"][split_name] = {}
        for scheme, starts in starts_for(len(split), 512, args.sample_windows, 20260828 + split_index).items():
            values = []
            for offset in range(0, len(starts), args.eval_batch):
                chosen = starts[offset:offset + args.eval_batch]
                real = len(chosen)
                if real < args.eval_batch:
                    chosen = np.pad(chosen, (0, args.eval_batch - real), mode="edge")
                x, y = batch_at(split, chosen, 512)
                losses = window_nll(jax.device_put(x, batch_sharding), jax.device_put(y, batch_sharding))
                values.extend(np.asarray(losses[:real]) / math.log(2.0))
            summary = summarize(values)
            report["results"][split_name][scheme] = summary
            print("SEGMENT_RETENTION_DENSE_AUDIT", split_name, scheme, json.dumps(summary), flush=True)
    report["elapsed_s"] = time.perf_counter() - started
    output = args.outdir / "evaluation_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("SEGMENT_RETENTION_DENSE_AUDIT_SAVED", output, flush=True)


if __name__ == "__main__":
    main()
