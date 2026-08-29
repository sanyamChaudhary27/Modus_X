# AdaptivePreconditionedArchive v0 Gate

## Question

Can CurrentArchive reduce matrix-write interference by adapting its write
direction to recent key occupancy, without adding learned parameters,
attention, or sequence-growing state?

This experiment is deliberately separate from MemoryFeedbackArchive. If it
independently wins, the two mechanisms may then be combined to test whether
better matrix writing and better matrix-to-vector communication are
complementary.

## Single Change

Each current and archive matrix receives a fixed-size diagonal usage trace:

```text
usage_t = usage_(t-1) + (1 - 0.99) * write * (key_t^2 - usage_(t-1))
scale_t = clip(mean(usage_t) / usage_t, 0.25, 4.0)
direction_t = (key_t * scale_t) / dot(key_t, key_t * scale_t)
H_t = retain * H_(t-1) + learning_rate * outer(residual, direction_t)
```

The dot-product normalization preserves unit response along the presented key.
Clipping bounds the preconditioner. Underused dimensions receive stronger
updates; frequently occupied dimensions receive weaker updates.

## Controlled Properties

- Same `47,038,396` learned parameters as CurrentArchive.
- Parameter tree is bit-identical at initialization for a matched seed.
- No MemoryFeedback bridge.
- No attention.
- No optimizer, data, objective, or evaluation changes.
- Added recurrent state: two `ax_res` vectors per layer, fixed with sequence
  length. At the production configuration this is `49,152` fp32 bytes, about
  `0.195%` of the two matrix states.

## First Gate

Run from scratch to `20.48M` processed characters with the frozen recipe:

- seed `1`;
- global batch `8`;
- LR `6e-4`;
- future target `2`, weight `0.5`;
- auxiliary layer `6`, weight `0.05`;
- AdamW weight decay `1e-4`.

Matched references at this gate:

- CurrentArchive: `1.677799` validation, `1.7782` sparse test;
- MemoryFeedbackArchive: `1.673634` validation, `1.7667` sparse test.

Promotion requires a credible independent improvement over CurrentArchive,
finite diagnostics, and acceptable throughput. Do not combine with
MemoryFeedback unless this isolated write mechanism passes first.

Pre-registered decision bands:

- validation `<=1.657799` (`>=0.02` gain): strong independent promotion;
- validation `1.657800-1.667799` (`0.01-0.02` gain): confirm unchanged at
  `40.96M` before promotion;
- validation `>1.667799`: reject for generic LM even if individual diagnostics
  are interesting.

## 20.48M Result

The candidate completed the independent gate at step `5,000` / `20.48M`
processed characters:

| metric | CurrentArchive | AdaptivePreconditioned | delta vs control |
|---|---:|---:|---:|
| sparse validation BPC | 1.677799 | **1.674264** | -0.003535 |
| sparse test BPC | **1.7782** | 1.7903 | +0.0121 |
| elapsed seconds | **4,016.86** | 4,869.30 | +21.2% |

The validation change is below the pre-registered `0.01` confirmation floor,
while sparse test and throughput both regress. The mechanism therefore fails
the generic-language-model promotion gate.

Diagnostics show that this is not a dead or broken path. Across train,
validation, and test:

- current write gate is `~0.897`;
- archive write strength is `~0.218`;
- current usage CV is `~0.90-0.92`;
- archive usage CV is `~0.51`;
- current preconditioner spread is `~10.23-10.28`;
- archive preconditioner spread is `~5.29-5.31`;
- update-direction norms remain bounded near `1.06-1.11`.

The statistics are finite and split-stable. The negative result is therefore
informative: diagonal key-frequency correction changes matrix writes as
intended, but the added work does not improve byte-LM prediction enough to
justify its cost.

## Decision

Reject AdaptivePreconditionedArchive v0 for generic LM. Preserve it as an
interference-control ablation, but do not continue it to `40.96M` and do not
combine it with MemoryFeedbackArchive. The pre-registered causal rule is
binding: only an independent win qualified for combination.
