# AttentionToWriteArchive v0 Gate

## Question

Can a small, fixed-window causal attention controller improve what enters
CurrentArchive memory without becoming a third output expert or introducing an
unbounded KV cache?

This is an isolated mechanism. It contains no MemoryFeedback bridge and does
not inherit AdaptivePreconditionedArchive.

## Architecture

- CurrentArchive remains unchanged in all 12 layers.
- Layers 4, 8, and 12 receive a causal local-attention write controller.
- Window: `64` tokens.
- Heads: `4`.
- Total K/V width: `64`.
- Attention context changes only the matrix value and write gate, with fixed
  residual strength `0.25`.
- Attention is never mixed directly into the layer output.

The production candidate has `47,431,807` parameters: `393,411` more than
CurrentArchive (`+0.84%`) and nearly parameter-matched to MemoryFeedbackArchive
(`47,437,768`). Existing CurrentArchive parameters are seed-paired exactly.

At inference, the three BF16 controllers require:

```text
3 layers * 2 (K,V) * 64 tokens * 64 width * 2 bytes = 49,152 bytes
```

The KV state is fixed with total conversation length. It must nevertheless be
counted in every memory and serving comparison.

## First Gate

Train from scratch to `20.48M` processed characters with seed `1`, batch `8`,
LR `6e-4`, AdamW, future target `2` at weight `0.5`, and auxiliary layer `6`
at weight `0.05`.

References:

- CurrentArchive: `1.677799` validation, `1.7782` sparse test;
- MemoryFeedbackArchive: `1.673634` validation, `1.7667` sparse test.

Decision bands against CurrentArchive:

- validation `<=1.657799`: strong independent promotion;
- validation `1.657800-1.667799`: confirm unchanged at `40.96M` only if
  throughput and attention diagnostics are healthy;
- validation `>1.667799`: reject for generic LM.

Do not combine with MemoryFeedback unless this mechanism independently passes.

## 20.48M Result

The model completed step `5,000` / `20.48M` characters successfully:

| metric | CurrentArchive | AttentionToWrite | delta vs control |
|---|---:|---:|---:|
| sparse validation BPC | 1.677799 | **1.670520** | -0.007279 |
| sparse test BPC | **1.7782** | 1.7800 | +0.0018 |
| elapsed seconds | **4,016.86** | 4,060.26 | +1.1% |

The validation gain is below the pre-registered `0.01` confirmation floor and
the sparse test signal is slightly worse. The candidate therefore fails the
independent generic-LM promotion gate. Do not continue it to `40.96M` and do
not combine it with MemoryFeedbackArchive.

Training and checkpoint creation succeeded. The first diagnostics invocation
failed after training because the generic collector allocated 12 layer slots,
while AttentionToWrite correctly emits diagnostics only for controller layers
4, 8, and 12. The collector was repaired to infer diagnostic-layer count from
the returned tensor. This bookkeeping failure does not invalidate the model
checkpoint or BPC result and does not require retraining.

## Recovered Controller Diagnostics

| split | context norm | entropy | mean distance | value adjustment norm | abs write adjustment |
|---|---:|---:|---:|---:|---:|
| train tail | 16.9278 | 0.5892 | 29.2439 | 38.0468 | 0.7553 |
| validation | 16.8867 | 0.5881 | 29.1620 | 37.8428 | 0.7684 |
| test | 16.7791 | 0.6066 | 29.2103 | 37.1811 | 0.7737 |

The controller is active and split-stable. Maximum entropy over a full
64-token window is `ln(64)=4.159`; observed entropy near `0.59-0.61` indicates
sharp attention rather than uniform averaging. Mean attended distance near 29
tokens shows that it uses much of the bounded window rather than only the
immediately previous token.

The fixed `0.25` injection turns the learned value-adjustment norms into an
effective contribution near `9.30-9.51`, while the effective write-logit
adjustment is moderate near `0.19`. The path therefore neither collapsed nor
remained decorative. Its rejection is substantive: active local
attention-to-write did not improve held-out byte prediction enough to pass the
pre-registered gate.
