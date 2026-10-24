# Feedback + Attention-to-Write Factorial v0

## Purpose

This post-hoc exploratory run completes a `2 x 2` mechanism table. It asks
whether matrix-to-vector feedback and bounded attention-to-write control are
complementary. It is not a fresh architecture search and it does not override
the independent rejection of AttentionToWriteArchive.

## Existing 20.48M-character cells

| Feedback | Attention-to-write | Validation BPC |
|---|---|---:|
| no | no | 1.677799237 |
| yes | no | 1.673634043 |
| no | yes | 1.670519851 |

The no-interaction additive prediction for the combined cell is:

`1.673634043 + 1.670519851 - 1.677799237 = 1.666354657 BPC`.

## Predeclared gate

The combined model runs from scratch with the same seed, data, optimizer,
objective, batch, and `6e-4` learning rate. It automatically continues from
`20.48M` to `40.96M` characters only if validation BPC is at most
`1.656354657`, a `0.01` positive interaction beyond the additive prediction.

The interaction statistic is:

`combined - feedback - attention + control`.

Lower is better. The continuation gate therefore requires interaction at or
below `-0.01 BPC`.

## Architecture accounting

- Base: MemoryFeedbackArchive.
- Attention: causal window `64`, four heads, total KV width `64`, active only
  at layers `4`, `8`, and `12` and used only to control matrix writes.
- Expected training parameters: `47,831,179`.
- Fixed BF16 attention KV state: `49,152` bytes.
- No direct attention residual is added to the language-model output.

## Claim boundary

Passing establishes a promising interaction at one early enwik8 gate, not a
general superiority result. Failure means the mechanisms do not earn a joint
long run under this recipe. Any promoted result still requires matched dense
validation/test evaluation and runtime reporting.

## Result

The combined model completed the `20.48M`-character gate and did not earn
continuation:

| Metric | Result |
|---|---:|
| Parameters | 47,831,179 |
| Validation BPC | 1.678507016 |
| Sparse test BPC | 1.7771 |
| Elapsed time | 4,362.68 s |
| Additive prediction | 1.666354657 |
| Continuation threshold | 1.656354657 |
| Factorial interaction | +0.012152359 BPC |

The positive interaction statistic is harmful because lower BPC is better.
The combination was `0.004873` BPC worse than MemoryFeedback alone,
`0.007987` worse than AttentionToWrite alone, and `0.000708` worse than the
CurrentArchive control at the same early gate. It missed the continuation
threshold by `0.022152` BPC.

The sparse test value does not override the validation decision. It is worse
than MemoryFeedback's independent `1.7667` sparse test and the experiment was
pre-registered to promote on validation interaction. Therefore this branch is
frozen at step `5,000`; no dense audit, continuation, or hyperparameter tuning
is justified.

## Decision

Reject the combined mechanism under this recipe. Retain MemoryFeedbackArchive
as the efficiency-qualified v2 language candidate and keep the two mechanisms
separate in the evidence record. This closes the post-hoc factorial question
without opening a combination sweep.
