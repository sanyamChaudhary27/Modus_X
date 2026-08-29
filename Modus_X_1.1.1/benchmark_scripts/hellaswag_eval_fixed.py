from __future__ import annotations
import argparse
import sys
import jax
from jax import random
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "/home/HP/Modus_v2")
from modus_v2.models import ModelConfig, make_model

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Transformer")
    p.add_argument("--ckpt-dir", default=None)
    p.add_argument("--load-step", type=int, default=None)
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--max-seq-len", type=int, default=384)
    p.add_argument("--embed-dim", type=int, default=512)
    p.add_argument("--hidden-dim", type=int, default=2048)
    p.add_argument("--ax-res", type=int, default=384)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--mamba-state-dim", type=int, default=384)
    return p.parse_args()

def restore_if_available(params: dict, ckpt_dir: str | None, load_step: int | None):
    if not ckpt_dir:
        print("No checkpoint provided; evaluating random weights.")
        return params
    import orbax.checkpoint as ocp

    mngr = ocp.CheckpointManager(ckpt_dir, ocp.PyTreeCheckpointer())
    step = load_step if load_step is not None else mngr.latest_step()
    if step is None:
        print(f"No checkpoint found in {ckpt_dir}; evaluating random weights.")
        return params
    empty = {"model": params}
    restore_args = ocp.checkpoint_utils.construct_restore_args(empty)
    restored = mngr.restore(step, items=empty, restore_kwargs={"restore_args": restore_args, "partial_restore": True})
    print(f"Restored {ckpt_dir} step={step}")
    return restored["model"]

def score_choice(params: dict, fwd_fn_jit, ids: list[int], choice_start: int, max_seq_len: int) -> float:
    ids = ids[:max_seq_len]
    L = len(ids)
    if L <= 1:
        return -1e9
    
    # Pad input exactly to max_seq_len - 1 to maintain static shapes for JAX JIT
    pad_len = max_seq_len - 1
    x_pad = np.zeros(pad_len, dtype=np.int32)
    x_pad[:L-1] = ids[:-1]
    
    # Run the compiled forward pass
    logits = fwd_fn_jit(params, jnp.array(x_pad))
    
    # Slice back to the actual length and compute log-probabilities
    logits_unpadded = logits[:L-1]
    logp = jax.nn.log_softmax(logits_unpadded, axis=-1)
    
    y = np.array(ids[1:], dtype=np.int32)
    start = max(0, min(choice_start - 1, len(y) - 1))
    tok_lp = logp[jnp.arange(len(y)), y]
    choice_lp = tok_lp[start:]
    return float(jnp.mean(choice_lp)) if len(choice_lp) else -1e9

def main() -> None:
    args = parse_args()
    cfg = ModelConfig(
        vocab_size=50257,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        ax_res=args.ax_res,
        n_layers=args.n_layers,
        seq_len=args.max_seq_len, # For HellaSwag we evaluate up to max_seq_len
        mamba_state_dim=args.mamba_state_dim,
        n_heads_attn=8,
    )
    params, fwd_fn = make_model(args.model, random.PRNGKey(0), cfg)
    fwd_fn_jit = jax.jit(fwd_fn)
    params = restore_if_available(params, args.ckpt_dir, args.load_step)

    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset("Rowan/hellaswag", split="validation", streaming=True)
    correct = 0
    total = 0
    for row in ds:
        if total >= args.samples:
            break
        ctx_ids = enc.encode(row["ctx"])
        scores = []
        for ending in row["endings"]:
            choice_ids = enc.encode(" " + ending)
            scores.append(score_choice(params, fwd_fn_jit, ctx_ids + choice_ids, len(ctx_ids), args.max_seq_len))
        pred = int(np.argmax(scores))
        label = int(row["label"])
        correct += int(pred == label)
        total += 1
        if total % 10 == 0:
            print(f"{args.model} {total}/{args.samples} acc={100.0 * correct / total:.2f}%")
            sys.stdout.flush()
    print(f"FINAL {args.model} HellaSwag accuracy={100.0 * correct / max(1, total):.2f}% ({correct}/{total})")

if __name__ == "__main__":
    main()
