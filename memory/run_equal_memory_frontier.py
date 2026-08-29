"""Equal-inference-state-byte CurrentArchive versus Transformer frontier."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
LOCAL_RUNNER = THIS_DIR / "run_versioned_memory_ablation.py"
LOCAL_STAGE1G = THIS_DIR / "run_stage1g_versioned_memory.py"
LOCAL_BALANCED_KV = THIS_DIR / "balanced_kv.py"
LOCAL_MODUS = THIS_DIR / "modus_x2"
RUNNER = (
    LOCAL_RUNNER
    if LOCAL_RUNNER.exists()
    else REPO_ROOT / "Modus_X_2.0.0" / "experiments" / "memory" / "run_versioned_memory_ablation.py"
)
STAGE1G = (
    LOCAL_STAGE1G
    if LOCAL_STAGE1G.exists()
    else REPO_ROOT / "experiments" / "matrix_memory_capacity" / "run_stage1g_versioned_memory.py"
)
BALANCED_KV = (
    LOCAL_BALANCED_KV
    if LOCAL_BALANCED_KV.exists()
    else REPO_ROOT / "Modus_X_v1.1.1" / "benchmarks" / "modus_x" / "balanced_kv.py"
)
if LOCAL_MODUS.exists():
    sys.path.insert(0, str(THIS_DIR))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seeds", default="17,27,37")
    parser.add_argument("--matrix-widths", default="128,96,64,32,16")
    parser.add_argument("--bindings", type=int, default=32)
    parser.add_argument("--overwrite-rate", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--n-train", type=int, default=4096)
    parser.add_argument("--n-test", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--n-values", type=int, default=32)
    parser.add_argument("--transformer-d-model", type=int, default=64)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-ffn", type=int, default=512)
    parser.add_argument(
        "--transformer-qk-mode", choices=("independent", "tied"), default="tied"
    )
    parser.add_argument("--transformer-only", action="store_true")
    parser.add_argument("--state-dtype-bytes", type=int, default=2)
    parser.add_argument("--oracle-record-bytes", type=int, default=72)
    parser.add_argument("--min-full-context-accuracy", type=float, default=35.0)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def csv_ints(text: str) -> list[int]:
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def init_linear(kv, key, input_dim: int, output_dim: int, *, gain: float = 1.0):
    scale = gain * math.sqrt(2.0 / (input_dim + output_dim))
    return {
        "w": kv.random.normal(key, (input_dim, output_dim), dtype=kv.jnp.float32) * scale,
        "b": kv.jnp.zeros((output_dim,), dtype=kv.jnp.float32),
    }


def init_transformer(kv, key, input_dim: int, n_values: int, cfg: dict):
    count = 2 + 4 * cfg["layers"]
    keys = iter(kv.random.split(key, count))
    input_projection = init_linear(kv, next(keys), input_dim, cfg["d_model"])
    # Preserve the structured key/value/role coordinates at initialization.
    identity_width = min(input_dim, cfg["d_model"])
    input_projection["w"] = input_projection["w"].at[:identity_width, :identity_width].set(
        kv.jnp.eye(identity_width, dtype=kv.jnp.float32)
    )
    layers = []
    for _ in range(cfg["layers"]):
        qkv_key = next(keys)
        if cfg["qk_mode"] == "tied":
            qk_key, value_key = kv.random.split(qkv_key)
            qk = init_linear(kv, qk_key, cfg["d_model"], cfg["d_model"])
            value = init_linear(kv, value_key, cfg["d_model"], cfg["d_model"])
            qkv = {
                "w": kv.jnp.concatenate([qk["w"], qk["w"], value["w"]], axis=1),
                "b": kv.jnp.concatenate([qk["b"], qk["b"], value["b"]], axis=0),
            }
        else:
            qkv = init_linear(kv, qkv_key, cfg["d_model"], 3 * cfg["d_model"])
        layers.append(
            {
                "qkv": qkv,
                "out": init_linear(kv, next(keys), cfg["d_model"], cfg["d_model"]),
                "up": init_linear(kv, next(keys), cfg["d_model"], cfg["ffn"]),
                "down": init_linear(kv, next(keys), cfg["ffn"], cfg["d_model"]),
                "attn_scale": kv.jnp.ones((cfg["d_model"],), dtype=kv.jnp.float32),
                "ffn_scale": kv.jnp.ones((cfg["d_model"],), dtype=kv.jnp.float32),
            }
        )
    return {
        "input": input_projection,
        "layers": tuple(layers),
        "final_scale": kv.jnp.ones((cfg["d_model"],), dtype=kv.jnp.float32),
        "head": init_linear(kv, next(keys), cfg["d_model"], n_values),
    }


def linear(kv, params, x):
    return x @ params["w"] + params["b"]


def rms_norm(kv, x, scale):
    return x * kv.jax.lax.rsqrt(kv.jnp.mean(kv.jnp.square(x), axis=-1, keepdims=True) + 1e-6) * scale


def sinusoidal_positions(kv, length: int, width: int):
    positions = kv.jnp.arange(length, dtype=kv.jnp.float32)[:, None]
    frequencies = kv.jnp.exp(
        -math.log(10_000.0)
        * kv.jnp.arange(0, width, 2, dtype=kv.jnp.float32)
        / max(1, width)
    )
    angles = positions * frequencies[None]
    encoded = kv.jnp.zeros((length, width), dtype=kv.jnp.float32)
    encoded = encoded.at[:, 0::2].set(kv.jnp.sin(angles))
    encoded = encoded.at[:, 1::2].set(kv.jnp.cos(angles[:, : encoded[:, 1::2].shape[1]]))
    return encoded


def transformer_forward(kv, params, seq, cfg: dict, window: int):
    hidden = linear(kv, params["input"], seq)
    hidden = hidden + 0.1 * sinusoidal_positions(kv, hidden.shape[0], cfg["d_model"])
    head_dim = cfg["d_model"] // cfg["heads"]
    positions = kv.jnp.arange(hidden.shape[0])
    distance = positions[:, None] - positions[None, :]
    mask = (distance >= 0) & (distance <= window)
    for layer in params["layers"]:
        normed = rms_norm(kv, hidden, layer["attn_scale"])
        qkv = linear(kv, layer["qkv"], normed)
        query, key, value = kv.jnp.split(qkv, 3, axis=-1)
        query = query.reshape(hidden.shape[0], cfg["heads"], head_dim).transpose(1, 0, 2)
        key = key.reshape(hidden.shape[0], cfg["heads"], head_dim).transpose(1, 0, 2)
        value = value.reshape(hidden.shape[0], cfg["heads"], head_dim).transpose(1, 0, 2)
        scores = kv.jnp.einsum("hqd,hkd->hqk", query, key) / math.sqrt(head_dim)
        scores = kv.jnp.where(mask[None], scores, -1e30)
        weights = kv.jax.nn.softmax(scores, axis=-1)
        context = kv.jnp.einsum("hqk,hkd->hqd", weights, value)
        context = context.transpose(1, 0, 2).reshape(hidden.shape)
        hidden = hidden + linear(kv, layer["out"], context)
        normed = rms_norm(kv, hidden, layer["ffn_scale"])
        hidden = hidden + linear(kv, layer["down"], kv.jax.nn.gelu(linear(kv, layer["up"], normed)))
    hidden = rms_norm(kv, hidden, params["final_scale"])
    return linear(kv, params["head"], hidden[-1])


def oracle_accuracy(seqs: np.ndarray, labels: np.ndarray, cfg, budget_bytes: int, record_bytes: int):
    capacity = max(1, budget_bytes // record_bytes)
    fact_marker = cfg.key_dim + cfg.n_values
    value_offset = cfg.key_dim
    correct = 0
    answered = 0
    for seq, label in zip(np.asarray(seqs), np.asarray(labels)):
        fact_positions = np.flatnonzero(seq[:, fact_marker] > 0.5)
        retained = fact_positions[-capacity:]
        query = seq[-1, : cfg.key_dim]
        records = []
        for position in retained:
            key = seq[position, : cfg.key_dim]
            if float(np.dot(key, query)) > 0.999:
                value = int(np.argmax(seq[position, value_offset : value_offset + cfg.n_values]))
                records.append(value)
        if not records:
            prediction = -1
        elif seq[-1, fact_marker + 4] > 0.5:
            prediction = records[0]
        elif seq[-1, fact_marker + 3] > 0.5:
            prediction = records[-2] if len(records) >= 2 else records[0]
        else:
            prediction = records[-1]
        answered += int(bool(records))
        correct += int(prediction == int(label))
    return {
        "accuracy": 100.0 * correct / len(labels),
        "answer_coverage": 100.0 * answered / len(labels),
        "fact_capacity": capacity,
    }


def aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["model"], row["state_bytes"]), []).append(row)
    result = []
    for (model, state_bytes), members in sorted(grouped.items()):
        accuracies = [member["metrics"]["acc_all"] for member in members]
        result.append(
            {
                "model": model,
                "state_bytes": state_bytes,
                "seeds": sorted(member["seed"] for member in members),
                "accuracy_mean": statistics.mean(accuracies),
                "accuracy_stdev": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
                "params_mean": statistics.mean(member["params"] for member in members),
                "kv_window": members[0].get("kv_window"),
                "oracle_fact_capacity": members[0].get("oracle_fact_capacity"),
            }
        )
    return result


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.seeds = "17"
        args.matrix_widths = "64,16"
        args.epochs = 1
        args.n_train = 256
        args.n_test = 128
    args.outdir.mkdir(parents=True, exist_ok=True)
    runner = load_module(RUNNER, "equal_memory_runner")
    kv = load_module(BALANCED_KV, "equal_memory_balanced_kv")
    stage1g = load_module(STAGE1G, "equal_memory_stage1g")
    transformer_cfg = {
        "d_model": args.transformer_d_model,
        "layers": args.transformer_layers,
        "heads": args.transformer_heads,
        "ffn": args.transformer_ffn,
        "qk_mode": args.transformer_qk_mode,
    }
    if transformer_cfg["d_model"] % transformer_cfg["heads"]:
        raise ValueError("Transformer width must be divisible by head count")

    protocol_args = argparse.Namespace(
        n_values=args.n_values,
        d_model=args.d_model,
        n_train=args.n_train,
        n_test=args.n_test,
        batch=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        train_curriculum="role_balanced_overwritten",
        version_tag_facts=True,
        wrong_version_margin=0.5,
    )
    seeds = csv_ints(args.seeds)
    widths = csv_ints(args.matrix_widths)
    rows = []
    for seed in seeds:
        for matrix_width in widths:
            data_cfg, model_cfg = runner.make_configs(
                protocol_args,
                n_pairs=args.bindings,
                ax_res=matrix_width,
                overwrite_rate=args.overwrite_rate,
                router_bias=2.0,
                residual_scale=0.25,
            )
            key = kv.random.PRNGKey(seed)
            k_train, k_test, k_current, k_transformer = kv.random.split(key, 4)
            train = runner.make_train_set(kv, stage1g, k_train, data_cfg, protocol_args)
            test = stage1g.make_versioned_kv(
                kv,
                k_test,
                data_cfg.n_test,
                data_cfg.train_len,
                data_cfg,
                query_mode="mixed",
                target_mode="overwritten",
                version_tag_facts=True,
            )
            state_bytes = (2 * matrix_width * matrix_width + matrix_width) * args.state_dtype_bytes
            kv_bytes_per_token = (
                transformer_cfg["layers"]
                * 2
                * transformer_cfg["d_model"]
                * args.state_dtype_bytes
            )
            kv_window = max(1, min(data_cfg.train_len - 1, state_bytes // kv_bytes_per_token))
            print(
                "EQUAL_MEMORY_CASE",
                json.dumps(
                    {
                        "seed": seed,
                        "matrix_width": matrix_width,
                        "state_bytes": state_bytes,
                        "transformer_kv_bytes_per_token": kv_bytes_per_token,
                        "transformer_kv_window": kv_window,
                    }
                ),
                flush=True,
            )

            if not args.transformer_only:
                current_params, current_fwd, _, _ = runner.make_model_with_aux(
                    "CurrentArchiveDelta", k_current, model_cfg
                )
                _, current_trained, current_history = runner.train_model(
                    kv,
                    "CurrentArchiveDelta",
                    current_params,
                    current_fwd,
                    train,
                    (test[0], test[1]),
                    data_cfg,
                )
                current_metrics = stage1g.evaluate(
                    kv, current_trained, current_fwd, test[0], test[1], test[2], data_cfg
                )
                current_row = {
                    "model": "CurrentArchiveDelta",
                    "seed": seed,
                    "matrix_width": matrix_width,
                    "state_bytes": state_bytes,
                    "params": runner.count_params(current_trained),
                    "metrics": current_metrics,
                    "history": current_history,
                }
                rows.append(current_row)
                print("EQUAL_MEMORY_RESULT", json.dumps({k: v for k, v in current_row.items() if k != "history"}), flush=True)

            transformer_params = init_transformer(
                kv, k_transformer, data_cfg.d_model, data_cfg.n_values, transformer_cfg
            )
            transformer_fwd = lambda p, seq, window=kv_window: transformer_forward(
                kv, p, seq, transformer_cfg, window
            )
            _, transformer_trained, transformer_history = runner.train_model(
                kv,
                f"TransformerKV{kv_window}",
                transformer_params,
                transformer_fwd,
                train,
                (test[0], test[1]),
                data_cfg,
            )
            transformer_metrics = stage1g.evaluate(
                kv,
                transformer_trained,
                transformer_fwd,
                test[0],
                test[1],
                test[2],
                data_cfg,
            )
            transformer_row = {
                "model": "TransformerKV",
                "seed": seed,
                "matrix_width": matrix_width,
                "state_bytes": state_bytes,
                "kv_window": kv_window,
                "kv_bytes_per_token": kv_bytes_per_token,
                "params": runner.count_params(transformer_trained),
                "metrics": transformer_metrics,
                "history": transformer_history,
            }
            rows.append(transformer_row)
            print("EQUAL_MEMORY_RESULT", json.dumps({k: v for k, v in transformer_row.items() if k != "history"}), flush=True)

            if not args.transformer_only:
                oracle = oracle_accuracy(
                    test[0], test[1], data_cfg, state_bytes, args.oracle_record_bytes
                )
                oracle_row = {
                    "model": "ExactKeyRecentOracle",
                    "seed": seed,
                    "matrix_width": matrix_width,
                    "state_bytes": state_bytes,
                    "oracle_fact_capacity": oracle["fact_capacity"],
                    "params": 0,
                    "metrics": {
                        "acc_all": oracle["accuracy"],
                        "answer_coverage": oracle["answer_coverage"],
                    },
                }
                rows.append(oracle_row)
                print("EQUAL_MEMORY_RESULT", json.dumps(oracle_row), flush=True)

            if (
                matrix_width == max(widths)
                and seed == seeds[0]
                and transformer_metrics["acc_all"] < args.min_full_context_accuracy
                and not args.smoke
            ):
                raise RuntimeError(
                    "Full-context Transformer failed the pre-registered learnability gate: "
                    f"{transformer_metrics['acc_all']:.3f}% < {args.min_full_context_accuracy:.3f}%"
                )

            (args.outdir / "progress.json").write_text(
                json.dumps({"rows": rows, "aggregate": aggregate(rows)}, indent=2), encoding="utf-8"
            )

    report = {
        "protocol": {
            "seeds": seeds,
            "matrix_widths": widths,
            "bindings": args.bindings,
            "overwrite_rate": args.overwrite_rate,
            "state_dtype_bytes": args.state_dtype_bytes,
            "oracle_record_bytes": args.oracle_record_bytes,
            "transformer": transformer_cfg,
        },
        "rows": rows,
        "aggregate": aggregate(rows),
    }
    (args.outdir / "equal_memory_frontier.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("EQUAL_MEMORY_FRONTIER_READY", args.outdir, flush=True)


if __name__ == "__main__":
    main()
