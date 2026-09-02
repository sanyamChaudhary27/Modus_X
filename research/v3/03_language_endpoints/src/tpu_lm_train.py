from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from models import ModelConfig, count_params, make_model


GPU_REFERENCES = {
    4_096_000: 2.506,
    20_480_000: 1.8638,
    40_960_000: 1.6918,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--target-chars", type=int, default=4_096_000)
    p.add_argument("--stop-chars", type=int, default=None)
    p.add_argument("--checkpoint-chars", type=int, default=4_096_000)
    p.add_argument("--eval-chunks", type=int, default=128)
    p.add_argument("--eval-batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--optimizer", choices=("adamw", "sgd_momentum"), default="adamw")
    p.add_argument("--momentum", type=float, default=0.99)
    p.add_argument("--auxiliary-weight", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Training-only uniform label smoothing for all supervised byte-prediction losses.",
    )
    p.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Training-only inverted dropout applied to Modus_X head inputs; evaluation is unchanged.",
    )
    p.add_argument("--schedule", choices=("constant", "warmup_cosine", "late_cosine"), default="constant")
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--decay-start-chars", type=int, default=100_000_000)
    p.add_argument("--end-lr-ratio", type=float, default=0.05)
    p.add_argument("--precision", choices=("float32", "bfloat16"), default="float32")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--model", default="Modus_X_Vector_Lean_DeepSupervision")
    p.add_argument("--embed-dim", type=int, default=352)
    p.add_argument("--hidden-dim", type=int, default=1056)
    p.add_argument("--state-dim", type=int, default=352)
    p.add_argument("--n-layers", type=int, default=12)
    p.add_argument("--router-hidden", type=int, default=32)
    p.add_argument(
        "--matrix-retain-bias",
        type=float,
        default=None,
        help="Optional Modus_X matrix retention bias override for initialization ablations.",
    )
    p.add_argument(
        "--matrix-write-bias",
        type=float,
        default=None,
        help="Optional Modus_X matrix write-gate bias override for initialization ablations.",
    )
    p.add_argument(
        "--vector-retain-bias",
        type=float,
        default=None,
        help="Optional Modus_X vector-state retention bias override for initialization ablations.",
    )
    p.add_argument("--input-seq-len", type=int, default=512)
    p.add_argument("--loss-tail", type=int, default=512)
    p.add_argument(
        "--auxiliary-layers",
        default="6",
        help="1-indexed comma-separated auxiliary layers for shared-head deep supervision.",
    )
    p.add_argument(
        "--input-corruption-rate",
        type=float,
        default=0.0,
        help="Training-only byte replacement probability for denoising-style next-byte prediction.",
    )
    p.add_argument(
        "--future-targets",
        default="",
        help="Comma-separated future target offsets beyond next byte, e.g. '2' or '2,3'.",
    )
    p.add_argument(
        "--future-target-weight",
        type=float,
        default=0.5,
        help="Weight applied to each additional future-target loss.",
    )
    p.add_argument(
        "--auxiliary-decay-chars",
        type=int,
        default=0,
        help="Linearly decay intermediate-layer auxiliary weight to zero by this processed-character count. 0 disables decay.",
    )
    p.add_argument(
        "--future-target-decay-chars",
        type=int,
        default=0,
        help="Linearly decay future-target weight to zero by this processed-character count. 0 disables decay.",
    )
    p.add_argument(
        "--paper-layer-decay",
        action="store_true",
        help=(
            "Use the Al-Rfou et al. intermediate-layer schedule: auxiliary "
            "layer l contributes until l/(2n) of training, instead of one "
            "shared auxiliary decay."
        ),
    )
    p.add_argument(
        "--aux-future-targets",
        action="store_true",
        help="Apply future-target losses to intermediate auxiliary layer outputs too.",
    )
    return p.parse_args()


def atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def atomic_pickle(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def block_tree(tree) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def non_embedding_params(params) -> int:
    total = count_params(params)
    return total - params["embed"].size


def load_split(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return data[:90_000_000], data[90_000_000:95_000_000], data[95_000_000:100_000_000]


def batch_at(data: np.ndarray, starts: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.arange(seq_len + 1)
    chunks = data[starts[:, None] + offsets[None, :]]
    return chunks[:, :-1].astype(np.int32), chunks[:, 1:].astype(np.int32)


def loss_fn(
    params,
    fwd_fn,
    x,
    y,
    auxiliary_weight: float,
    loss_tail: int | None = None,
    future_targets: tuple[int, ...] = (),
    future_target_weight: float = 0.5,
    auxiliary_layer_weights=None,
    aux_future_targets: bool = False,
    label_smoothing: float = 0.0,
    dropout_keys=None,
):
    def token_loss(logits, targets):
        logp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
        nll = -jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
        if label_smoothing <= 0:
            return nll
        smooth = -jnp.mean(logp, axis=-1)
        return (1.0 - label_smoothing) * nll + label_smoothing * smooth

    if dropout_keys is None:
        outputs = jax.vmap(lambda sequence: fwd_fn(params, sequence))(x)
    else:
        outputs = jax.vmap(lambda sequence, key: fwd_fn(params, sequence, key))(x, dropout_keys)
    future_logits = None
    auxiliary_future_logits = None
    if isinstance(outputs, tuple):
        if len(outputs) == 4:
            logits, auxiliary_logits, future_logits, auxiliary_future_logits = outputs
        elif len(outputs) == 3:
            logits, auxiliary_logits, future_logits = outputs
        else:
            logits, auxiliary_logits = outputs
    else:
        logits, auxiliary_logits = outputs, None
    logits = logits.astype(jnp.float32)
    full_logits = logits
    full_y = y
    if loss_tail is not None:
        logits = logits[:, -loss_tail:]
        y = y[:, -loss_tail:]
    loss = token_loss(logits, y).mean()
    for head_index, offset in enumerate(future_targets):
        if offset <= 1:
            raise ValueError("future target offsets must be greater than 1")
        if future_logits is None:
            raise ValueError("future target losses require separate future head logits")
        usable = future_logits[:, head_index, : -(offset - 1)]
        target = full_y[:, offset - 1 :]
        if loss_tail is not None:
            usable = usable[:, -loss_tail:]
            target = target[:, -loss_tail:]
        future_loss = token_loss(usable, target).mean()
        loss = loss + future_target_weight * future_loss
    if auxiliary_logits is not None:
        if auxiliary_logits.ndim == logits.ndim + 1 and auxiliary_logits.shape[1] == 1:
            auxiliary_logits = auxiliary_logits[:, 0]
        if auxiliary_logits.ndim == logits.ndim + 1:
            if loss_tail is not None:
                auxiliary_logits = auxiliary_logits[:, :, -loss_tail:]
            aux_targets = y[:, None, :, None]
            aux_logp = jax.nn.log_softmax(auxiliary_logits.astype(jnp.float32), axis=-1)
            aux_losses = -jnp.take_along_axis(aux_logp, aux_targets, axis=-1)[..., 0]
            if label_smoothing > 0:
                aux_smooth = -jnp.mean(aux_logp, axis=-1)
                aux_losses = (1.0 - label_smoothing) * aux_losses + label_smoothing * aux_smooth
            aux_losses = aux_losses.mean(axis=(0, 2))
            if auxiliary_layer_weights is not None:
                aux_losses = aux_losses * auxiliary_layer_weights
            aux_loss = aux_losses.mean()
        else:
            if loss_tail is not None:
                auxiliary_logits = auxiliary_logits[:, -loss_tail:]
            aux_loss = token_loss(auxiliary_logits, y).mean()
        loss = loss + auxiliary_weight * aux_loss
        if aux_future_targets and auxiliary_future_logits is not None:
            aux_future_total = 0.0
            for head_index, offset in enumerate(future_targets):
                usable = auxiliary_future_logits[:, head_index, :, : -(offset - 1)]
                target = full_y[:, offset - 1 :]
                if loss_tail is not None:
                    usable = usable[:, :, -loss_tail:]
                    target = target[:, -loss_tail:]
                aux_future_logp = jax.nn.log_softmax(usable.astype(jnp.float32), axis=-1)
                aux_future_losses = -jnp.take_along_axis(
                    aux_future_logp,
                    target[:, None, :, None],
                    axis=-1,
                )[..., 0]
                if label_smoothing > 0:
                    aux_future_smooth = -jnp.mean(aux_future_logp, axis=-1)
                    aux_future_losses = (
                        (1.0 - label_smoothing) * aux_future_losses
                        + label_smoothing * aux_future_smooth
                    )
                aux_future_losses = aux_future_losses.mean(axis=(0, 2))
                if auxiliary_layer_weights is not None:
                    aux_future_losses = aux_future_losses * auxiliary_layer_weights
                aux_future_total = aux_future_total + aux_future_losses.mean()
            loss = loss + future_target_weight * aux_future_total
    return loss


def evaluate(params, fwd_fn, data, seq_len, chunks, batch_size, batch_sharding, loss_tail=None):
    @jax.jit
    def eval_batch(x, y):
        outputs = jax.vmap(lambda sequence: fwd_fn(params, sequence))(x)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        logp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
        nll = -jnp.take_along_axis(logp, y[..., None], axis=-1)[..., 0]
        if loss_tail is not None:
            nll = nll[:, -loss_tail:]
        return nll.mean(axis=1)

    losses = []
    max_start = len(data) - seq_len - 1
    starts = np.linspace(0, max_start, chunks, dtype=np.int64)
    for offset in range(0, chunks, batch_size):
        selected = starts[offset : offset + batch_size]
        real = len(selected)
        if real < batch_size:
            selected = np.pad(selected, (0, batch_size - real), mode="edge")
        x, y = batch_at(data, selected, seq_len)
        values = eval_batch(jax.device_put(x, batch_sharding), jax.device_put(y, batch_sharding))
        losses.extend(np.asarray(values[:real]))
    return float(np.mean(losses) / math.log(2))


def linear_decay_weight(base_weight: float, step, decay_steps: int):
    if decay_steps <= 0:
        return base_weight
    progress = jnp.minimum(step / decay_steps, 1.0)
    return base_weight * (1.0 - progress)


def paper_auxiliary_layer_weights(
    auxiliary_layers: tuple[int, ...],
    n_layers: int,
    step,
    total_steps: int,
):
    cutoffs = jnp.array(
        [max(1, round((layer / (2 * n_layers)) * total_steps)) for layer in auxiliary_layers],
        dtype=jnp.float32,
    )
    return (step < cutoffs).astype(jnp.float32)


def main() -> None:
    args = parse_args()
    if jax.default_backend() != "tpu":
        raise RuntimeError(f"Expected TPU backend, found {jax.default_backend()}")
    if args.batch % jax.device_count():
        raise ValueError("Global batch must divide TPU device count")
    if args.eval_batch % jax.device_count():
        raise ValueError("Evaluation batch must divide TPU device count")
    if args.precision == "bfloat16":
        raise NotImplementedError(
            "bfloat16 requires dtype-stable Modus_X recurrent and layer-scan carries; "
            "the current architecture emits float32 carries. Use float32 until the "
            "model-level mixed-precision path is implemented and parity-tested."
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = outdir / "checkpoint.pkl"
    progress_path = outdir / "progress.json"
    config_path = outdir / "config.json"

    seq_len = args.input_seq_len
    if args.loss_tail <= 0 or args.loss_tail > seq_len:
        raise ValueError("--loss-tail must be between 1 and --input-seq-len")
    if not 0.0 <= args.input_corruption_rate < 1.0:
        raise ValueError("--input-corruption-rate must be in [0, 1)")
    future_targets = tuple(int(part) for part in args.future_targets.split(",") if part.strip())
    if any(offset <= 1 or offset > seq_len for offset in future_targets):
        raise ValueError("--future-targets must contain offsets in [2, input_seq_len]")
    if args.future_target_weight < 0:
        raise ValueError("--future-target-weight must be non-negative")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("--label-smoothing must be in [0, 1)")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1)")
    cfg = ModelConfig(
        vocab_size=256,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        ax_res=args.state_dim,
        n_layers=args.n_layers,
        n_heads_attn=8,
        seq_len=seq_len,
        mamba_state_dim=args.state_dim,
        vector_router=False,
        router_hidden=args.router_hidden,
    )
    auxiliary_layers = tuple(int(part) for part in args.auxiliary_layers.split(",") if part.strip())
    if not auxiliary_layers or any(layer < 1 or layer > cfg.n_layers for layer in auxiliary_layers):
        raise ValueError("--auxiliary-layers must contain 1-indexed layer numbers within model depth")
    params, fwd_fn = make_model(
        args.model,
        jax.random.key(args.seed),
        cfg,
        auxiliary_layers=auxiliary_layers,
        future_target_count=len(future_targets),
        dropout_rate=args.dropout,
    )
    if args.matrix_retain_bias is not None:
        params["layers"]["m_b_ret"] = jnp.ones_like(params["layers"]["m_b_ret"]) * args.matrix_retain_bias
    if args.matrix_write_bias is not None:
        params["layers"]["m_b_write"] = jnp.ones_like(params["layers"]["m_b_write"]) * args.matrix_write_bias
    if args.vector_retain_bias is not None:
        params["layers"]["s_b_ret"] = jnp.ones_like(params["layers"]["s_b_ret"]) * args.vector_retain_bias
    chars_per_step = args.batch * args.loss_tail
    total_steps = math.ceil(args.target_chars / chars_per_step)
    stop_steps = total_steps if args.stop_chars is None else min(
        total_steps, math.ceil(args.stop_chars / chars_per_step)
    )
    if stop_steps <= 0:
        raise ValueError("--stop-chars must produce at least one training step")
    checkpoint_steps = max(1, round(args.checkpoint_chars / chars_per_step))
    auxiliary_decay_steps = 0 if args.auxiliary_decay_chars <= 0 else max(
        1, round(args.auxiliary_decay_chars / chars_per_step)
    )
    future_target_decay_steps = 0 if args.future_target_decay_chars <= 0 else max(
        1, round(args.future_target_decay_chars / chars_per_step)
    )
    if not 0.0 <= args.end_lr_ratio <= 1.0:
        raise ValueError("--end-lr-ratio must be between 0 and 1")
    if args.schedule == "warmup_cosine":
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=args.lr * 0.05,
            peak_value=args.lr,
            warmup_steps=args.warmup_steps,
            decay_steps=total_steps,
            end_value=args.lr * args.end_lr_ratio,
        )
    elif args.schedule == "late_cosine":
        decay_start_step = min(total_steps - 1, max(0, round(args.decay_start_chars / chars_per_step)))
        decay_steps = max(1, total_steps - decay_start_step)
        def schedule(count):
            local_step = jnp.maximum(count - decay_start_step, 0)
            progress = jnp.minimum(local_step / decay_steps, 1.0)
            cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
            decayed = args.lr * (args.end_lr_ratio + (1.0 - args.end_lr_ratio) * cosine)
            return jnp.where(count < decay_start_step, args.lr, decayed)
    else:
        schedule = args.lr
    if args.optimizer == "adamw":
        optimizer = optax.adamw(schedule, weight_decay=args.weight_decay)
    else:
        optimizer = optax.sgd(schedule, momentum=args.momentum, nesterov=False)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optimizer)
    opt_state = tx.init(params)

    mesh = Mesh(np.array(jax.devices(), dtype=object), ("data",))
    batch_sharding = NamedSharding(mesh, P("data", None))
    replicated = NamedSharding(mesh, P())
    params = jax.device_put(params, replicated)
    opt_state = jax.device_put(opt_state, replicated)

    @jax.jit
    def update(p, state, x, y, step):
        dropout_keys = None
        if args.dropout > 0.0:
            dropout_keys = jax.random.split(jax.random.fold_in(jax.random.key(args.seed + 17_000), step), x.shape[0])
        auxiliary_weight = linear_decay_weight(args.auxiliary_weight, step, auxiliary_decay_steps)
        future_target_weight = linear_decay_weight(
            args.future_target_weight,
            step,
            future_target_decay_steps,
        )
        auxiliary_layer_weights = None
        if args.paper_layer_decay:
            auxiliary_layer_weights = paper_auxiliary_layer_weights(
                auxiliary_layers,
                cfg.n_layers,
                step,
                total_steps,
            )
        loss, grads = jax.value_and_grad(
            lambda pp: loss_fn(
                pp,
                fwd_fn,
                x,
                y,
                auxiliary_weight,
                args.loss_tail,
                future_targets,
                future_target_weight,
                auxiliary_layer_weights,
                args.aux_future_targets,
                args.label_smoothing,
                dropout_keys,
            )
        )(p)
        updates, state = tx.update(grads, state, p)
        return optax.apply_updates(p, updates), state, loss

    raw = np.memmap(args.data_path, dtype=np.uint8, mode="r")
    train, valid, test = load_split(raw)
    rng = np.random.default_rng(1000 + args.seed)
    rows = []
    start_step = 0
    elapsed_before = 0.0
    if args.resume and checkpoint_path.exists():
        with checkpoint_path.open("rb") as f:
            state = pickle.load(f)
        params = jax.device_put(state["params"], replicated)
        opt_state = jax.device_put(state["opt_state"], replicated)
        rng.bit_generator.state = state["rng_state"]
        rows = state["rows"]
        start_step = state["step"]
        elapsed_before = state["elapsed_s"]
        print(f"RESUME step={start_step}", flush=True)

    config = {
        "args": vars(args),
        "devices": [str(d) for d in jax.devices()],
        "params": count_params(params),
        "non_embedding_params": non_embedding_params(params),
        "chars_per_step": chars_per_step,
        "total_steps": total_steps,
        "stop_steps": stop_steps,
        "checkpoint_steps": checkpoint_steps,
        "auxiliary_decay_steps": auxiliary_decay_steps,
        "future_target_decay_steps": future_target_decay_steps,
    }
    atomic_json(config_path, config)
    print(json.dumps(config, indent=2), flush=True)

    started = time.perf_counter()
    for step in range(start_step + 1, stop_steps + 1):
        starts = rng.integers(0, len(train) - seq_len - 1, size=args.batch)
        x, y = batch_at(train, starts, seq_len)
        if args.input_corruption_rate > 0:
            mask = rng.random(x.shape) < args.input_corruption_rate
            replacement = rng.integers(0, cfg.vocab_size, size=x.shape, dtype=np.int32)
            x = np.where(mask, replacement, x).astype(np.int32)
        x = jax.device_put(x, batch_sharding)
        y = jax.device_put(y, batch_sharding)
        params, opt_state, loss = update(params, opt_state, x, y, step)
        if step == start_step + 1 or step % 10 == 0:
            block_tree(loss)
            elapsed = elapsed_before + time.perf_counter() - started
            print(
                f"PROGRESS step={step:,}/{total_steps:,} loss={float(loss):.4f} "
                f"chars_s={(step-start_step)*chars_per_step/max(elapsed-elapsed_before,1e-9):.0f}",
                flush=True,
            )
        if step % checkpoint_steps == 0 or step == stop_steps:
            block_tree((params, opt_state))
            elapsed = elapsed_before + time.perf_counter() - started
            val_bpc = evaluate(
                params, fwd_fn, valid, seq_len, args.eval_chunks, args.eval_batch,
                batch_sharding, args.loss_tail
            )
            processed = step * chars_per_step
            reference = GPU_REFERENCES.get(processed)
            row = {
                "step": step,
                "processed_characters": processed,
                "loss": float(loss),
                "val_bpc": val_bpc,
                "gpu_reference_bpc": reference,
                "delta_to_gpu": None if reference is None else val_bpc - reference,
                "elapsed_s": elapsed,
            }
            rows.append(row)
            atomic_json(progress_path, {"config": config, "rows": rows})
            atomic_pickle(
                checkpoint_path,
                {
                    "step": step,
                    "params": jax.device_get(params),
                    "opt_state": jax.device_get(opt_state),
                    "rng_state": rng.bit_generator.state,
                    "rows": rows,
                    "elapsed_s": elapsed,
                },
            )
            print("CHECKPOINT", json.dumps(row), flush=True)
            if not math.isfinite(val_bpc):
                raise RuntimeError("Non-finite validation BPC")
            if reference is not None and val_bpc - reference > 0.10:
                raise RuntimeError(f"TPU parity failed: delta_to_gpu={val_bpc-reference:+.4f}")

    if stop_steps == total_steps:
        test_bpc = evaluate(
            params, fwd_fn, test, seq_len, args.eval_chunks, args.eval_batch,
            batch_sharding, args.loss_tail
        )
        print(f"FINAL_TEST_BPC {test_bpc:.4f}", flush=True)


if __name__ == "__main__":
    main()
