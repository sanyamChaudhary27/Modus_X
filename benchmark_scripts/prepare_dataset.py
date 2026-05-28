from __future__ import annotations

import argparse
import os

import numpy as np
import tiktoken
from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="/home/HP/fineweb_tokens_modus_v2")
    p.add_argument("--samples", type=int, default=100000)
    p.add_argument("--shard-tokens", type=int, default=50_000_000)
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--name", default="sample-10BT")
    p.add_argument("--split", default="train")
    return p.parse_args()


def flush(tokens: list[int], out_dir: str, shard_idx: int) -> list[int]:
    if not tokens:
        return []
    path = os.path.join(out_dir, f"tokens_{shard_idx:05d}.npy")
    arr = np.array(tokens, dtype=np.uint16)
    np.save(path, arr)
    print(f"saved {path} tokens={len(arr):,}")
    return []


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset(args.dataset, name=args.name, split=args.split, streaming=True)
    shard_idx = len([x for x in os.listdir(args.out_dir) if x.endswith(".npy")])
    tokens: list[int] = []
    docs = 0
    for row in ds:
        if docs >= args.samples:
            break
        toks = enc.encode_ordinary(row["text"])
        toks.append(enc.eot_token)
        tokens.extend(toks)
        docs += 1
        if docs % 1000 == 0:
            print(f"docs={docs:,}/{args.samples:,} buffered_tokens={len(tokens):,}")
        while len(tokens) >= args.shard_tokens:
            tokens = flush(tokens[: args.shard_tokens], args.out_dir, shard_idx) + tokens[args.shard_tokens:]
            shard_idx += 1
    flush(tokens, args.out_dir, shard_idx)


if __name__ == "__main__":
    main()
