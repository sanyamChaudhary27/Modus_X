from __future__ import annotations

import argparse
import json
import math
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--target-chars", type=int, default=20_480_000)
    parser.add_argument("--checkpoint-chars", type=int, default=4_096_000)
    parser.add_argument("--eval-chunks", type=int, default=256)
    parser.add_argument("--eval-batch", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--n-layers", type=int, default=24)
    parser.add_argument("--d-state", type=int, default=64)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--multi-gpu", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--min-params", type=int, default=0)
    parser.add_argument("--max-params", type=int, default=2**63 - 1)
    return parser.parse_args()


def ensure_enwik8(path: Path) -> None:
    if path.exists():
        return
    import urllib.request

    zip_path = path.with_suffix(".zip")
    urllib.request.urlretrieve("http://mattmahoney.net/dc/enwik8.zip", zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(path.parent)


def load_split(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = raw[:90_000_000]
    valid = raw[90_000_000:95_000_000]
    test = raw[95_000_000:100_000_000]
    return train, valid, test


def batch_at(data: np.ndarray, starts: np.ndarray, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = np.stack([data[s : s + seq_len] for s in starts]).astype(np.int64)
    y = np.stack([data[s + 1 : s + seq_len + 1] for s in starts]).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


class OfficialMambaByteLM(nn.Module):
    def __init__(self, d_model: int, n_layers: int, d_state: int, d_conv: int, expand: int) -> None:
        super().__init__()
        from mamba_ssm import Mamba

        self.embed = nn.Embedding(256, d_model)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm": nn.LayerNorm(d_model),
                        "mamba": Mamba(
                            d_model=d_model,
                            d_state=d_state,
                            d_conv=d_conv,
                            expand=expand,
                        ),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 256, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for layer in self.layers:
            h = h + layer["mamba"](layer["norm"](h))
        return self.head(self.final_norm(h))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def evaluate(model: nn.Module, data: np.ndarray, seq_len: int, chunks: int, batch: int, device: str) -> float:
    model.eval()
    max_start = len(data) - seq_len - 1
    starts = np.linspace(0, max_start, chunks, dtype=np.int64)
    losses: list[float] = []
    weights: list[int] = []
    for i in range(0, len(starts), batch):
        part = starts[i : i + batch]
        x, y = batch_at(data, part, seq_len)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1), reduction="sum")
        losses.append(float(loss.detach().cpu()))
        weights.append(int(y.numel()))
    model.train()
    return sum(losses) / sum(weights) / math.log(2.0)


def save_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data_path)
    ensure_enwik8(data_path)
    raw = np.memmap(data_path, dtype=np.uint8, mode="r")
    train, valid, test = load_split(raw)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("Official Mamba baseline requires CUDA GPU runtime.")
    torch.backends.cuda.matmul.allow_tf32 = True

    base_model = OfficialMambaByteLM(
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
    ).to(device)
    parameter_count = count_params(base_model)
    if not args.min_params <= parameter_count <= args.max_params:
        raise RuntimeError(
            f"Parameter guard failed: {parameter_count:,} is outside "
            f"[{args.min_params:,}, {args.max_params:,}]"
        )
    model = (
        nn.DataParallel(base_model, device_ids=list(range(torch.cuda.device_count())))
        if args.multi_gpu and torch.cuda.device_count() > 1
        else base_model
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    chars_per_step = args.batch * args.seq_len
    total_steps = math.ceil(args.target_chars / chars_per_step)
    checkpoint_steps = max(1, math.ceil(args.checkpoint_chars / chars_per_step))
    checkpoint_path = outdir / "checkpoint.pt"
    progress_path = outdir / "progress.json"
    rows: list[dict] = []
    start_step = 0
    elapsed_before = 0.0
    skipped_updates = 0
    rng = np.random.default_rng(1000 + args.seed)

    if args.resume and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device)
        base_model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
            group["weight_decay"] = args.weight_decay
        rows = state["rows"]
        start_step = int(state["step"])
        elapsed_before = float(state["elapsed_s"])
        skipped_updates = int(state.get("skipped_updates", 0))
        rng.bit_generator.state = state["rng_state"]
        print(f"RESUME step={start_step}", flush=True)

    config = {
        "args": vars(args),
        "runtime": device,
        "torch_version": torch.__version__,
        "params": parameter_count,
        "non_embedding_params": parameter_count - base_model.embed.weight.numel(),
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "chars_per_step": chars_per_step,
        "total_steps": total_steps,
        "checkpoint_steps": checkpoint_steps,
    }
    save_json(outdir / "config.json", config)
    print(json.dumps(config, indent=2), flush=True)

    started = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step in range(start_step + 1, total_steps + 1):
        step_loss = 0.0
        finite_update = True
        for _ in range(args.grad_accum):
            starts = rng.integers(0, len(train) - args.seq_len - 1, size=args.batch)
            x, y = batch_at(train, starts, args.seq_len)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1)) / args.grad_accum
            if not torch.isfinite(loss):
                finite_update = False
                break
            scaler.scale(loss).backward()
            step_loss += float(loss.detach().cpu()) * args.grad_accum
        if finite_update:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            finite_update = bool(torch.isfinite(grad_norm))
        if finite_update:
            scaler.step(optimizer)
            scaler.update()
        else:
            skipped_updates += 1
            step_loss = float("nan")
        optimizer.zero_grad(set_to_none=True)

        if step == start_step + 1 or step % 10 == 0:
            torch.cuda.synchronize()
            elapsed = elapsed_before + time.perf_counter() - started
            print(
                f"PROGRESS step={step:,}/{total_steps:,} loss={step_loss:.4f} "
                f"chars_s={(step-start_step)*chars_per_step/max(elapsed-elapsed_before,1e-9):.0f} "
                f"skipped={skipped_updates}",
                flush=True,
            )

        if step % checkpoint_steps == 0 or step == total_steps:
            torch.cuda.synchronize()
            elapsed = elapsed_before + time.perf_counter() - started
            val_bpc = evaluate(model, valid, args.seq_len, args.eval_chunks, args.eval_batch, device)
            row = {
                "step": step,
                "processed_characters": step * chars_per_step,
                "loss": step_loss,
                "val_bpc": val_bpc,
                "elapsed_s": elapsed,
                "skipped_updates": skipped_updates,
            }
            rows.append(row)
            save_json(progress_path, {"config": config, "rows": rows})
            torch.save(
                {
                    "step": step,
                    "model": base_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rng_state": rng.bit_generator.state,
                    "rows": rows,
                    "elapsed_s": elapsed,
                    "skipped_updates": skipped_updates,
                },
                checkpoint_path,
            )
            print("CHECKPOINT", json.dumps(row), flush=True)

    test_bpc = evaluate(model, test, args.seq_len, args.eval_chunks, args.eval_batch, device)
    print(f"FINAL_TEST_BPC {test_bpc:.4f}", flush=True)


if __name__ == "__main__":
    main()
