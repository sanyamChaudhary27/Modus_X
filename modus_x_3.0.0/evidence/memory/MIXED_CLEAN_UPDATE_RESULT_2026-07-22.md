# Mixed clean/update equal-memory result (2026-07-22)

## Decision

At the near-parameter-matched width-64 point, CurrentArchiveDelta strongly
outperforms the tied-Q/K learned Transformer under exactly matched BF16
inference-state bytes on both clean and overwritten queries. The result closes
the narrow objection that the matrix result only works by treating every query
as an update.

It also exposes a real weakness: CurrentArchive has higher stale-value false
recall and is substantially weaker on latest values than on previous/first
values. The next architectural question is therefore version arbitration, not
whether the bounded matrix stores useful information.

## Frozen protocol

- seeds: `17,27,37`
- parameters: CurrentArchiveDelta `166,791`; TransformerKV `174,112`
- BF16 inference state: `16,512` bytes per sequence for both
- Transformer KV window: `32` tokens
- bindings: `32`; overwrite rate: `0.5`
- train/validation/test: `6,144 / 1,536 / 1,536` examples
- test composition per seed: `768` clean and `768` overwritten queries
- latest/previous/first roles balanced inside each target group
- checkpoint selection: validation only; final test was not used for selection

## Three-seed results

| Metric | CurrentArchiveDelta | TransformerKV | Difference |
|---|---:|---:|---:|
| Overall accuracy | **77.95 +/- 1.16** | 16.38 +/- 0.75 | +61.57 |
| Clean accuracy | **77.91 +/- 2.86** | 14.19 +/- 1.36 | +63.72 |
| Overwritten accuracy | **77.99 +/- 0.72** | 18.58 +/- 0.66 | +59.42 |
| Latest, clean | **68.49 +/- 1.26** | 11.07 +/- 2.60 | +57.42 |
| Latest, overwritten | **67.32 +/- 1.13** | 28.39 +/- 3.25 | +38.93 |
| Previous, clean | **83.46 +/- 3.32** | 16.93 +/- 2.29 | +66.54 |
| Previous, overwritten | **83.59 +/- 2.17** | 12.50 +/- 0.39 | +71.09 |
| First, clean | **81.77 +/- 4.58** | 14.58 +/- 0.98 | +67.19 |
| First, overwritten | **83.07 +/- 1.13** | 14.84 +/- 2.17 | +68.23 |
| Stale false recall on latest overwritten | 11.28 +/- 1.40 | **4.65 +/- 0.26** | +6.63 worse |

Values are mean +/- sample standard deviation. Stale false recall is evaluated
only where the previous or first value differs from the requested latest
value; there are approximately `251` eligible examples per seed.

## Interpretation

1. **The storage result is not update-only.** Clean and overwritten accuracy
   are essentially identical for CurrentArchive (`77.91%` and `77.99%`).
2. **The bounded-KV failure is durable-history loss.** At a 32-token window,
   Transformer performance is especially weak on previous and first values.
3. **CurrentArchive is historically biased.** Its latest accuracy is about
   `15-16` points below its previous/first accuracy, and it emits stale values
   more often than the Transformer when asked for the latest value.
4. **The next gate should diagnose operations.** Instrument write, current
   read, archive read, version role, and prediction success before changing
   the architecture. If the archive path dominates latest queries, test one
   pre-registered arbitration correction; otherwise stop guessing.

## Claim boundary

Allowed:

> On a controlled balanced clean/update protocol at equal 16.13 KiB BF16
> recurrent-state bytes and near-matched parameters, CurrentArchiveDelta
> reaches 77.95% accuracy versus 16.38% for a tied-Q/K learned causal
> Transformer with a 32-token KV window. Full-context KV remains superior in
> the separate frontier endpoint.

Not allowed:

- Modus_X universally beats Transformers.
- This result proves better language modeling or reasoning.
- The matrix has unlimited capacity.
- CurrentArchive solves update semantics perfectly; stale recall remains a
  measured weakness.

## Source

Runner:
`experiments/matrix_memory_capacity/run_equal_memory_mixed_update_gate.py`.

The Kaggle result archive should be preserved with the next evidence bundle so
the six per-seed rows and training histories remain auditable.
