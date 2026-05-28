# Modus_X Paper Package

This directory contains the deadline-ready Modus_X evidence bundle: clean result
tables, architecture notes, benchmark scripts, and raw outputs.

## Headline

Modus_X is a causal, no-attention, constant-state sequence model that combines:

- a Modus-style matrix delta-rule memory stream for content-addressed writes,
- a Mamba-style selective vector recurrence stream for sequential dynamics,
- an input-dependent router that decides how much each token uses each stream.

The best audited language-model result so far is the 80k Modus_X checkpoint:

```json
{"checkpoint_step": 80000, "valid_loss": 4.148229327052832, "valid_bpc": 5.984629878609282}
```

## Verified Language Modeling Results

All values below use the same held-out shard and the same explicit 64-chunk
evaluation protocol:

- validation file: `/home/HP/fineweb_tokens_modus_v2_big/tokens_00006.npy`
- sequence length: `512`
- eval chunks: `64`
- eval batch: `4`
- config: embed `512`, hidden `2048`, ax/state `384`, layers `8`

| Model | Step | Params | Valid Loss | Valid BPC | Perplexity | State at Inference |
|---|---:|---:|---:|---:|---:|---|
| Mamba | 40k | 139.7M | 4.321912 | 6.235201 | 75.33 | O(1) vector state |
| Modus_X | 40k | 153.9M | 4.205961 | 6.067920 | 67.08 | O(1) matrix + vector state |
| Modus_X | 60k | 153.9M | 4.165989 | 6.010252 | 64.46 | O(1) matrix + vector state |
| Modus_X | 80k | 153.9M | 4.148229 | 5.984630 | 63.32 | O(1) matrix + vector state |
| Transformer | 40k | 155.2M | 4.080766 | 5.887301 | 59.19 | O(L) KV cache |

## Defensible Claims

- Modus_X beats the matched Mamba checkpoint by `0.173682` validation loss at
  80k vs 40k, with about 10.2% more parameters.
- Modus_X reduces perplexity vs Mamba from `75.33` to `63.32`, a relative
  reduction of about `15.94%`.
- Modus_X continued training improved from `4.205961` at 40k to `4.148229` at
  80k, a `0.057732` validation-loss reduction.
- Modus_X does not yet beat the audited Transformer 40k validation loss; the
  remaining gap is `0.067463` loss.
- Unlike the Transformer baseline, Modus_X has no KV cache and keeps a fixed
  recurrent state independent of context length.

## HellaSwag Probe

The HellaSwag numbers are only 1000-sample probes, about 10% of the dataset.
They are useful as a quick sanity check, not a full benchmark claim.

| Model | Samples | Accuracy |
|---|---:|---:|
| Modus_X 40k | 1000 | 27.70% |
| Mamba 40k | 1000 | 28.00% |
| Transformer 40k | 1000 | 31.80% |

## Files

- `architecture.md`: concise technical description and diagrams.
- `results.md`: detailed benchmark summary.
- `raw_outputs/`: copied JSONL/TXT outputs used in the tables.
- `benchmark_scripts/`: copied scripts used for local benchmark/eval work.

