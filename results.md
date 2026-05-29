# Modus_X Results

## Audited FineWeb-Edu Language Modeling

Protocol:

- checkpoint eval script: `benchmarks/eval_lm_checkpoint.py`
- validation shard: `/home/HP/fineweb_tokens_modus_v2_big/tokens_00006.npy`
- eval chunks: `64`
- eval batch: `4`
- sequence length: `512`
- vocabulary: GPT-2 tokenizer, `50257`
- Modus_X config: embed `512`, hidden `2048`, ax/state `384`, layers `8`

| Model | Step | Valid Loss | Valid BPC | Perplexity |
|---|---:|---:|---:|---:|
| Mamba | 40k | 4.321912 | 6.235201 | 75.33 |
| Modus_X | 40k | 4.205961 | 6.067920 | 67.08 |
| Modus_X | 58k | 4.175283 | 6.023660 | 65.06 |
| Modus_X | 60k | 4.165989 | 6.010252 | 64.46 |
| Modus_X | 76k | 4.152610 | 5.990950 | 63.60 |
| Modus_X | 80k | 4.148229 | 5.984630 | 63.32 |
| Transformer | 40k | 4.080766 | 5.887301 | 59.19 |

## Main Comparisons

| Comparison | Loss Delta | Perplexity Delta |
|---|---:|---:|
| Modus_X 80k vs Mamba 40k | -0.173682 | -12.01 |
| Modus_X 80k vs Modus_X 40k | -0.057732 | -3.76 |
| Modus_X 80k vs Transformer 40k | +0.067463 | +4.13 |

Interpretation:

- Modus_X is the strongest recurrent/no-attention model in this run.
- The Transformer baseline remains lower on the audited LM validation metric.
- The gap to Transformer is now small enough that scaling, optimizer state,
  longer training, and architecture tuning are plausible next levers.

## 80k Continuation Curve

The 80k continuation used 32-chunk in-loop evals, so these numbers are not
directly interchangeable with the 64-chunk audit table above. They are still
useful for trend:

| Step | In-Loop Valid Loss | In-Loop Valid BPC |
|---:|---:|---:|
| 60k | 4.161579 | 6.003890 |
| 62k | 4.153648 | 5.992447 |
| 64k | 4.154800 | 5.994109 |
| 66k | 4.156896 | 5.997134 |
| 68k | 4.149930 | 5.987084 |
| 70k | 4.151167 | 5.988868 |
| 72k | 4.154589 | 5.993806 |
| 74k | 4.149152 | 5.985961 |
| 76k | 4.148533 | 5.985068 |
| 78k | 4.150170 | 5.987430 |

The explicit 64-chunk audit found the 80k checkpoint better than 76k:

| Step | 64-Chunk Valid Loss | 64-Chunk Valid BPC |
|---:|---:|---:|
| 76k | 4.152610 | 5.990950 |
| 80k | 4.148229 | 5.984630 |

## Scaling Probe

The current prototype is slower than Transformer at short contexts, but its
state memory is constant:

| Context | Transformer State MB | Modus_X State MB |
|---:|---:|---:|
| 128 | 1.000 | 1.004 |
| 256 | 2.000 | 1.004 |
| 512 | 4.000 | 1.004 |
| 1024 | 8.000 | 1.004 |
| 2048 | 16.000 | 1.004 |
| 4096 | 32.000 | 1.004 |
