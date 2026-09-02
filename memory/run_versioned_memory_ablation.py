"""
Run Modus_X 2.0.0 coordinated-memory ablations on the Stage1G versioned-memory task.

This compares:

- LateFusionV1Control: v1-style late router, no vector-controlled writes.
- CoordinatedWrite: vector state controls matrix write strength.
- PostReadRouter: router sees memory/vector reads before fusion.
- CoordinatedDualMemory: write control plus post-read router.

The goal is not a paper claim yet. The goal is to find out whether tighter
matrix/vector coordination improves the specific failure mode raised by the
bounded-memory objection: retrieving latest/previous/first versions.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3] if len(Path(__file__).resolve().parents) > 3 else THIS_DIR
LOCAL_BALANCED_KV = THIS_DIR / "balanced_kv.py"
LOCAL_STAGE1G = THIS_DIR / "run_stage1g_versioned_memory.py"
PACKAGE_SRC_ROOT = THIS_DIR.parent / "src"
LOCAL_SRC_ROOT = (
    PACKAGE_SRC_ROOT if (PACKAGE_SRC_ROOT / "modus_x2").exists() else THIS_DIR
)
BALANCED_KV = (
    LOCAL_BALANCED_KV
    if LOCAL_BALANCED_KV.exists()
    else REPO_ROOT / "Modus_X_v1.1.1" / "benchmarks" / "modus_x" / "balanced_kv.py"
)
STAGE1G = (
    LOCAL_STAGE1G
    if LOCAL_STAGE1G.exists()
    else REPO_ROOT / "experiments" / "matrix_memory_capacity" / "run_stage1g_versioned_memory.py"
)
SRC_ROOT = LOCAL_SRC_ROOT if (LOCAL_SRC_ROOT / "modus_x2").exists() else REPO_ROOT / "Modus_X_2.0.0" / "src"
sys.path.insert(0, str(SRC_ROOT))

from modus_x2.coordinated_dual_memory import CoordinatedMemoryConfig, count_params, make_model_with_aux


@dataclass(frozen=True)
class DataConfig:
    d_model: int = 96
    key_dim: int = 32
    n_values: int = 32
    n_pairs: int = 32
    train_len: int = 128
    n_train: int = 4096
    n_test: int = 1024
    batch: int = 64
    epochs: int = 24
    patience: int = 24
    lr: float = 3e-4
    ax_res: int = 128
    vector_state: int = 128
    router_hidden: int = 128
    router_bias: float = 2.0
    residual_scale: float = 0.25
    overwrite_rate: float = 0.5


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="Modus_X_2.0.0/experiments/memory/results_versioned_ablation")
    p.add_argument("--models", default="LateFusionV1Control,LateFusionPMControl,DisciplinedDeltaMemory")
    p.add_argument("--seeds", default="17,27,37")
    p.add_argument("--bindings", default="32,64")
    p.add_argument("--ax-res", default="64,128")
    p.add_argument("--overwrite-rates", default="0.5")
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--n-train", type=int, default=4096)
    p.add_argument("--n-test", type=int, default=1024)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-values", type=int, default=32)
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--router-biases", default="2.0")
    p.add_argument("--residual-scales", default="0.25")
    p.add_argument(
        "--wrong-version-weights",
        default="0.0",
        help=(
            "Comma-separated auxiliary weights for penalizing logits assigned "
            "to wrong versions of an overwritten key. 0.0 preserves the "
            "historical primary-loss-only protocol."
        ),
    )
    p.add_argument(
        "--wrong-version-margin",
        type=float,
        default=0.5,
        help="Soft margin used by the wrong-version penalty.",
    )
    p.add_argument(
        "--train-curriculum",
        default="mixed_any",
        choices=("mixed_any", "role_balanced_overwritten", "role_balanced_all"),
        help=(
            "Training data policy. mixed_any is the historical protocol. "
            "role_balanced_overwritten trains equal latest/previous/first "
            "query batches on overwritten keys. role_balanced_all trains equal "
            "query modes over arbitrary keys."
        ),
    )
    p.add_argument("--version-tag-facts", action="store_true")
    return p.parse_args()


def csv_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def csv_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def mean_stdev(values: list[float]) -> tuple[float, float]:
    values = [float(value) for value in values]
    if len(values) == 1:
        return values[0], 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


def make_configs(
    args: argparse.Namespace,
    *,
    n_pairs: int,
    ax_res: int,
    overwrite_rate: float,
    router_bias: float,
    residual_scale: float,
):
    seq_len = max(128, 4 * n_pairs + 1)
    min_width = 32 + args.n_values + 5
    d_model = max(args.d_model, min_width)
    data_cfg = DataConfig(
        d_model=d_model,
        n_values=args.n_values,
        n_pairs=n_pairs,
        train_len=seq_len,
        n_train=args.n_train,
        n_test=args.n_test,
        batch=args.batch,
        epochs=args.epochs,
        patience=args.epochs,
        lr=args.lr,
        ax_res=ax_res,
        vector_state=ax_res,
        overwrite_rate=overwrite_rate,
        router_bias=router_bias,
        residual_scale=residual_scale,
    )
    model_cfg = CoordinatedMemoryConfig(
        d_model=d_model,
        key_dim=data_cfg.key_dim,
        n_values=args.n_values,
        ax_res=ax_res,
        vector_state=ax_res,
        router_hidden=128,
        head_hidden=128,
        router_bias=router_bias,
        residual_scale=residual_scale,
    )
    return data_cfg, model_cfg


def train_model(
    kv,
    name: str,
    params: dict,
    fwd_fn,
    train,
    test,
    cfg: DataConfig,
    *,
    wrong_version_weight: float = 0.0,
    wrong_version_margin: float = 0.5,
):
    tr_s, tr_l, tr_meta = train
    te_s, te_l = test
    fwd_b = kv.jit(kv.vmap(fwd_fn, in_axes=(None, 0)))

    def loss_fn(p, s, y, latest_y, previous_y, first_y):
        logits = fwd_b(p, s)
        logp = kv.jax.nn.log_softmax(logits, axis=-1)
        row = kv.jnp.arange(len(y))
        primary = -kv.jnp.mean(logp[row, y])
        if wrong_version_weight == 0.0:
            return primary

        label_logits = logits[row, y]
        wrong_versions = kv.jnp.stack([latest_y, previous_y, first_y], axis=1)
        wrong_logits = logits[row[:, None], wrong_versions]
        is_first_occurrence = kv.jnp.stack(
            [
                kv.jnp.ones_like(y, dtype=bool),
                previous_y != latest_y,
                (first_y != latest_y) & (first_y != previous_y),
            ],
            axis=1,
        )
        distinct_wrong = (wrong_versions != y[:, None]) & is_first_occurrence
        penalties = kv.jax.nn.softplus(wrong_logits - label_logits[:, None] + wrong_version_margin)
        denom = kv.jnp.maximum(1.0, kv.jnp.sum(distinct_wrong))
        wrong_version_loss = kv.jnp.sum(kv.jnp.where(distinct_wrong, penalties, 0.0)) / denom
        return primary + wrong_version_weight * wrong_version_loss

    opt = kv.optax.chain(kv.optax.clip_by_global_norm(1.0), kv.optax.adamw(cfg.lr, weight_decay=1e-4))
    state = opt.init(params)

    @kv.jit
    def update(p, st, s, y, latest_y, previous_y, first_y):
        loss, grads = kv.jax.value_and_grad(loss_fn)(p, s, y, latest_y, previous_y, first_y)
        updates, st2 = opt.update(grads, st, p)
        return kv.optax.apply_updates(p, updates), st2, loss

    def eval_acc(p) -> float:
        correct = 0
        for start in range(0, len(te_l), cfg.batch):
            end = min(start + cfg.batch, len(te_l))
            logits = fwd_b(p, kv.jnp.array(te_s[start:end], dtype=kv.jnp.float32))
            pred = kv.jnp.argmax(logits, axis=-1)
            correct += int(kv.jnp.sum(pred == kv.jnp.array(te_l[start:end], dtype=kv.jnp.int32)))
        return 100.0 * correct / len(te_l)

    best, best_p, pat = 0.0, params, 0
    rows = []
    t0 = time.time()
    print(f"  params={count_params(params):,}", flush=True)
    for epoch in range(cfg.epochs):
        perm = np.random.default_rng(1000 + epoch).permutation(len(tr_l))
        losses = []
        for start in range(0, len(tr_l) - cfg.batch + 1, cfg.batch):
            idx = perm[start : start + cfg.batch]
            params, state, loss = update(
                params,
                state,
                kv.jnp.array(tr_s[idx], dtype=kv.jnp.float32),
                kv.jnp.array(tr_l[idx], dtype=kv.jnp.int32),
                kv.jnp.array(tr_meta["latest_values"][idx], dtype=kv.jnp.int32),
                kv.jnp.array(tr_meta["previous_values"][idx], dtype=kv.jnp.int32),
                kv.jnp.array(tr_meta["first_values"][idx], dtype=kv.jnp.int32),
            )
            losses.append(float(loss))
        acc = eval_acc(params)
        if acc > best:
            best, best_p, pat = acc, params, 0
        else:
            pat += 1
        row = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "acc": acc,
            "best": best,
            "elapsed_s": time.time() - t0,
        }
        rows.append(row)
        print(
            f"  [{name}] ep={epoch+1:03d} loss={row['loss']:.4f} "
            f"acc={acc:5.1f}% best={best:5.1f}% pat={pat}/{cfg.patience}",
            flush=True,
        )
        if pat >= cfg.patience:
            break
    return best, best_p, rows


def make_train_set(kv, stage1g, key, data_cfg: DataConfig, args: argparse.Namespace):
    if args.train_curriculum == "mixed_any":
        return stage1g.make_versioned_kv(
            kv,
            key,
            data_cfg.n_train,
            data_cfg.train_len,
            data_cfg,
            query_mode="mixed",
            target_mode="any",
            version_tag_facts=args.version_tag_facts,
        )

    per_mode = data_cfg.n_train // 3
    counts = [per_mode, per_mode, data_cfg.n_train - 2 * per_mode]
    keys = kv.random.split(key, 3)
    target_mode = "overwritten" if args.train_curriculum == "role_balanced_overwritten" else "any"
    parts = [
        stage1g.make_versioned_kv(
            kv,
            subkey,
            count,
            data_cfg.train_len,
            data_cfg,
            query_mode=mode,
            target_mode=target_mode,
            version_tag_facts=args.version_tag_facts,
        )
        for subkey, count, mode in zip(keys, counts, ("latest", "previous", "first"))
    ]
    seqs = np.concatenate([part[0] for part in parts], axis=0)
    labels = np.concatenate([part[1] for part in parts], axis=0)
    meta = {
        name: np.concatenate([np.asarray(part[2][name]) for part in parts], axis=0)
        for name in parts[0][2]
    }
    return seqs, labels, meta


def evaluate(kv, params, fwd_fn, seqs, labels, meta, cfg: DataConfig):
    stage1g = load_module(STAGE1G, "stage1g_eval_module")
    return stage1g.evaluate(kv, params, fwd_fn, seqs, labels, meta, cfg)


def router_diagnostics(kv, params, aux_fn, seqs, cfg: DataConfig):
    aux_b = kv.jit(kv.vmap(aux_fn, in_axes=(None, 0)))
    rows = []
    for start in range(0, len(seqs), cfg.batch):
        end = min(start + cfg.batch, len(seqs))
        batch_aux = aux_b(params, kv.jnp.array(seqs[start:end], dtype=kv.jnp.float32))
        rows.append({name: np.asarray(value) for name, value in batch_aux.items()})
    merged = {}
    for name in rows[0]:
        values = np.concatenate([np.ravel(row[name]) for row in rows])
        merged[name] = {
            "mean": float(np.mean(values)),
            "stdev": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return merged


def run_one(
    kv,
    stage1g,
    *,
    model_name: str,
    seed: int,
    n_pairs: int,
    ax_res: int,
    overwrite_rate: float,
    router_bias: float,
    residual_scale: float,
    wrong_version_weight: float,
    args,
):
    data_cfg, model_cfg = make_configs(
        args,
        n_pairs=n_pairs,
        ax_res=ax_res,
        overwrite_rate=overwrite_rate,
        router_bias=router_bias,
        residual_scale=residual_scale,
    )
    key = kv.random.PRNGKey(seed)
    k_train, k_mixed, k_latest, k_previous, k_first, k_model = kv.random.split(key, 6)

    train = make_train_set(kv, stage1g, k_train, data_cfg, args)
    tests = {
        "mixed_overwritten": stage1g.make_versioned_kv(
            kv,
            k_mixed,
            data_cfg.n_test,
            data_cfg.train_len,
            data_cfg,
            query_mode="mixed",
            target_mode="overwritten",
            version_tag_facts=args.version_tag_facts,
        ),
        "latest_overwritten": stage1g.make_versioned_kv(
            kv,
            k_latest,
            data_cfg.n_test,
            data_cfg.train_len,
            data_cfg,
            query_mode="latest",
            target_mode="overwritten",
            version_tag_facts=args.version_tag_facts,
        ),
        "previous_overwritten": stage1g.make_versioned_kv(
            kv,
            k_previous,
            data_cfg.n_test,
            data_cfg.train_len,
            data_cfg,
            query_mode="previous",
            target_mode="overwritten",
            version_tag_facts=args.version_tag_facts,
        ),
        "first_overwritten": stage1g.make_versioned_kv(
            kv,
            k_first,
            data_cfg.n_test,
            data_cfg.train_len,
            data_cfg,
            query_mode="first",
            target_mode="overwritten",
            version_tag_facts=args.version_tag_facts,
        ),
    }
    params, fwd_fn, aux_fn, actual_model_cfg = make_model_with_aux(model_name, k_model, model_cfg)
    best, trained, history = train_model(
        kv,
        model_name,
        params,
        fwd_fn,
        (train[0], train[1], train[2]),
        (tests["mixed_overwritten"][0], tests["mixed_overwritten"][1]),
        data_cfg,
        wrong_version_weight=wrong_version_weight,
        wrong_version_margin=args.wrong_version_margin,
    )
    diagnostics = {
        name: stage1g.evaluate(kv, trained, fwd_fn, seqs, labels, meta, data_cfg)
        for name, (seqs, labels, meta) in tests.items()
    }
    router_stats = {
        name: router_diagnostics(kv, trained, aux_fn, seqs, data_cfg)
        for name, (seqs, _, _) in tests.items()
    }
    return {
        "model": model_name,
        "seed": seed,
        "n_pairs": n_pairs,
        "ax_res": ax_res,
        "load_factor": n_pairs / ax_res,
        "overwrite_rate": overwrite_rate,
        "router_bias": router_bias,
        "residual_scale": residual_scale,
        "wrong_version_weight": wrong_version_weight,
        "wrong_version_margin": args.wrong_version_margin,
        "train_curriculum": args.train_curriculum,
        "version_tag_facts": args.version_tag_facts,
        "params": count_params(trained),
        "config": actual_model_cfg.__dict__,
        "best_mixed_overwritten_protocol_acc": best,
        "diagnostics": diagnostics,
        "router_diagnostics": router_stats,
        "history": history,
    }


def main() -> None:
    args = parse_args()
    kv = load_module(BALANCED_KV, "modus_x_balanced_kv_for_2")
    stage1g = load_module(STAGE1G, "stage1g_data_module")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_name in [x.strip() for x in args.models.split(",") if x.strip()]:
        for ax_res in csv_ints(args.ax_res):
            for n_pairs in csv_ints(args.bindings):
                for overwrite_rate in csv_floats(args.overwrite_rates):
                    for router_bias in csv_floats(args.router_biases):
                        for residual_scale in csv_floats(args.residual_scales):
                            for wrong_version_weight in csv_floats(args.wrong_version_weights):
                                for seed in csv_ints(args.seeds):
                                    print(
                                        "RUN_MODUS_X2_VERSIONED",
                                        json.dumps(
                                            {
                                                "model": model_name,
                                                "seed": seed,
                                                "n_pairs": n_pairs,
                                                "ax_res": ax_res,
                                                "overwrite_rate": overwrite_rate,
                                                "router_bias": router_bias,
                                                "residual_scale": residual_scale,
                                                "wrong_version_weight": wrong_version_weight,
                                            }
                                        ),
                                        flush=True,
                                    )
                                    rows.append(
                                        run_one(
                                            kv,
                                            stage1g,
                                            model_name=model_name,
                                            seed=seed,
                                            n_pairs=n_pairs,
                                            ax_res=ax_res,
                                            overwrite_rate=overwrite_rate,
                                            router_bias=router_bias,
                                            residual_scale=residual_scale,
                                            wrong_version_weight=wrong_version_weight,
                                            args=args,
                                        )
                                    )
                                    (outdir / "partial_results.json").write_text(json.dumps(rows, indent=2))

    grouped = {}
    for row in rows:
        key = (
            row["model"],
            row["n_pairs"],
            row["ax_res"],
            row["overwrite_rate"],
            row["router_bias"],
            row["residual_scale"],
            row["wrong_version_weight"],
            row["train_curriculum"],
        )
        grouped.setdefault(key, []).append(row)

    summary = []
    for (
        model,
        n_pairs,
        ax_res,
        overwrite_rate,
        router_bias,
        residual_scale,
        wrong_version_weight,
        train_curriculum,
    ), group in grouped.items():
        record = {
            "model": model,
            "n_pairs": n_pairs,
            "ax_res": ax_res,
            "load_factor": n_pairs / ax_res,
            "overwrite_rate": overwrite_rate,
            "router_bias": router_bias,
            "residual_scale": residual_scale,
            "wrong_version_weight": wrong_version_weight,
            "wrong_version_margin": args.wrong_version_margin,
            "train_curriculum": train_curriculum,
            "seeds": [row["seed"] for row in group],
            "params_mean": statistics.mean(row["params"] for row in group),
        }
        for eval_name in ("mixed_overwritten", "latest_overwritten", "previous_overwritten", "first_overwritten"):
            values = [row["diagnostics"][eval_name]["acc_all"] for row in group]
            mean, stdev = mean_stdev(values)
            record[f"{eval_name}_acc_mean"] = mean
            record[f"{eval_name}_acc_stdev"] = stdev
            for wrong_name in (
                "predicted_latest_when_not_label",
                "predicted_previous_when_not_label",
                "predicted_first_when_not_label",
            ):
                wrong_values = [row["diagnostics"][eval_name][wrong_name] for row in group]
                wrong_mean, wrong_stdev = mean_stdev(wrong_values)
                record[f"{eval_name}_{wrong_name}_mean"] = wrong_mean
                record[f"{eval_name}_{wrong_name}_stdev"] = wrong_stdev
            for stat_name in (
                "router_mean",
                "memory_norm",
                "vector_norm",
                "disagreement_norm",
                "current_read_norm",
                "history_read_norm",
                "current_history_disagreement_norm",
                "read_arbitration_current_weight",
                "read_arbitration_entropy",
                "attention_norm",
                "attention_gate",
                "attention_entropy",
            ):
                stat_values = [row["router_diagnostics"][eval_name][stat_name]["mean"] for row in group]
                stat_mean, stat_stdev = mean_stdev(stat_values)
                record[f"{eval_name}_{stat_name}_mean"] = stat_mean
                record[f"{eval_name}_{stat_name}_stdev"] = stat_stdev
        summary.append(record)

    (outdir / "results.json").write_text(json.dumps(rows, indent=2))
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("MODUS_X2_VERSIONED_ABLATION_READY", outdir, flush=True)
    print("SUMMARY_ROWS", len(summary), json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
