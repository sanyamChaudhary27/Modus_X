from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


@dataclass
class Config:
    input_dim: int = 96
    key_dim: int = 32
    n_values: int = 32
    n_pairs: int = 32
    train_len: int = 128
    n_train: int = 40_000
    n_test: int = 4_000
    batch: int = 64
    epochs: int = 12
    lr: float = 3e-4
    overwrite_rate: float = 0.0
    seed: int = 17


def make_balanced_kv(n: int, seq_len: int, cfg: Config, seed: int):
    rng = np.random.default_rng(seed)
    seqs = rng.normal(0, 0.05, (n, seq_len, cfg.input_dim)).astype(np.float32)
    seqs[:, :, : cfg.key_dim + cfg.n_values + 2] = 0
    keys = rng.normal(size=(n, cfg.n_pairs, cfg.key_dim)).astype(np.float32)
    keys /= np.linalg.norm(keys, axis=-1, keepdims=True) + 1e-8
    value_offset = cfg.key_dim
    fact_marker = cfg.key_dim + cfg.n_values
    query_marker = fact_marker + 1
    labels = np.empty(n, dtype=np.int64)

    for i in range(n):
        values = rng.permutation(cfg.n_pairs).astype(np.int64)
        positions = np.sort(rng.choice(seq_len - 1, cfg.n_pairs, replace=False))
        occupied = set(map(int, positions))
        for pair, position in enumerate(positions):
            seqs[i, position] = 0
            seqs[i, position, : cfg.key_dim] = keys[i, pair]
            seqs[i, position, value_offset + values[pair]] = 1
            seqs[i, position, fact_marker] = 1

        overwrite_count = round(cfg.n_pairs * cfg.overwrite_rate)
        for source in rng.choice(cfg.n_pairs, overwrite_count, replace=False):
            available = [
                p for p in range(int(positions[source]) + 1, seq_len - 1)
                if p not in occupied
            ]
            if not available:
                continue
            position = int(rng.choice(available))
            occupied.add(position)
            new_value = int(rng.integers(cfg.n_values))
            seqs[i, position] = 0
            seqs[i, position, : cfg.key_dim] = keys[i, source]
            seqs[i, position, value_offset + new_value] = 1
            seqs[i, position, fact_marker] = 1
            values[source] = new_value

        target = int(rng.integers(cfg.n_pairs))
        seqs[i, -1] = 0
        seqs[i, -1, : cfg.key_dim] = keys[i, target]
        seqs[i, -1, query_marker] = 1
        labels[i] = values[target]
    return seqs, labels


class OfficialMambaRecall(nn.Module):
    def __init__(self, input_dim=96, d_model=96, n_layers=2, d_state=16):
        super().__init__()
        self.input = nn.Linear(input_dim, d_model)
        with torch.no_grad():
            self.input.weight.zero_()
            self.input.bias.zero_()
            self.input.weight[: min(input_dim, d_model), : min(input_dim, d_model)] = torch.eye(
                min(input_dim, d_model)
            )
        self.layers = nn.ModuleList(
            [Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2) for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
        )

    def forward(self, x):
        h = self.input(x)
        for norm, layer in zip(self.norms, self.layers):
            h = h + layer(norm(h))
        return self.head(h[:, -1])


def accuracy(model, x, y, batch, device):
    model.eval()
    correct = 0
    with torch.inference_mode():
        for start in range(0, len(y), batch):
            xb = torch.from_numpy(x[start : start + batch]).to(device)
            pred = model(xb).argmax(-1).cpu().numpy()
            correct += int((pred == y[start : start + batch]).sum())
    return 100 * correct / len(y)


def streamed_accuracy(model, cfg, seq_len, n_examples, seed, device):
    correct = 0
    seen = 0
    # Generate only one evaluation batch at a time. Long-context evaluation
    # must not materialize the entire dataset or retain recurrent activations.
    for start in range(0, n_examples, cfg.batch):
        size = min(cfg.batch, n_examples - start)
        x, y = make_balanced_kv(size, seq_len, cfg, seed + start)
        correct += round(accuracy(model, x, y, size, device) * size / 100)
        seen += size
    return 100 * correct / seen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--overwrite-rate", type=float, default=0.0)
    parser.add_argument("--test-lens", default="128,256,512,1024,2048,4096,8192,16384")
    args = parser.parse_args()
    cfg = Config(seed=args.seed, overwrite_rate=args.overwrite_rate)
    device = "cuda"
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = OfficialMambaRecall().to(device)
    params = sum(p.numel() for p in model.parameters())
    if not 140_000 <= params <= 165_000:
        raise RuntimeError(f"Parameter match failed: {params:,}")
    print("OFFICIAL_MAMBA_RECALL_CONFIG", json.dumps({"params": params, **cfg.__dict__}), flush=True)

    train_x, train_y = make_balanced_kv(cfg.n_train, cfg.train_len, cfg, cfg.seed)
    test_x, test_y = make_balanced_kv(cfg.n_test, cfg.train_len, cfg, cfg.seed + 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    rng = np.random.default_rng(cfg.seed)
    rows = []
    started = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        permutation = rng.permutation(cfg.n_train)
        losses = []
        for start in range(0, cfg.n_train - cfg.batch + 1, cfg.batch):
            indices = permutation[start : start + cfg.batch]
            xb = torch.from_numpy(train_x[indices]).to(device)
            yb = torch.from_numpy(train_y[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            losses.append(float(loss.detach()))
        acc = accuracy(model, test_x, test_y, cfg.batch, device)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "accuracy": acc}
        rows.append(row)
        print("OFFICIAL_MAMBA_RECALL_EPOCH", json.dumps(row), flush=True)

    lengths = {}
    for length in map(int, args.test_lens.split(",")):
        # Keep short-length estimates high precision while bounding the cost of
        # 4K-16K recurrent evaluation.
        n_examples = cfg.n_test if length <= 2048 else (1000 if length <= 4096 else 256)
        lengths[str(length)] = streamed_accuracy(
            model,
            cfg,
            length,
            n_examples,
            cfg.seed + 10_000 + length,
            device,
        )
        print(
            "OFFICIAL_MAMBA_RECALL_LENGTH",
            length,
            lengths[str(length)],
            "examples",
            n_examples,
            flush=True,
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    result = {
        "config": cfg.__dict__,
        "params": params,
        "epochs": rows,
        "length_accuracy": lengths,
        "elapsed_s": time.time() - started,
    }
    (outdir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save({"model": model.state_dict(), "result": result}, outdir / "checkpoint.pt")
    print("OFFICIAL_MAMBA_RECALL_READY", outdir, flush=True)


if __name__ == "__main__":
    main()
