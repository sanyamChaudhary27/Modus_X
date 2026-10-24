# Latest-heavy curriculum diagnostic

Date: 2026-07-23

## Question

Is CurrentArchiveDelta's preference for historical values mainly caused by the
training distribution, or does a latest-versus-history arbitration weakness
remain after latest queries are emphasized?

## Frozen protocol

- Model: `CurrentArchiveDelta`
- Parameters: `166,791`
- Recurrent state: `16,512` bytes
- Seeds: `17, 27, 37`
- Clean and overwritten test queries: `768` each per seed
- Control curriculum: balanced latest/previous/first roles
- Candidate curriculum: `50/25/25` latest/previous/first roles
- Paired initialization and validation-only checkpoint selection
- Runner:
  `experiments/matrix_memory_capacity/run_latest_heavy_curriculum_gate.py`

The candidate had to satisfy all five pre-registered checks:

1. improve latest-overwritten accuracy by at least `5` percentage points;
2. reduce stale false recall on latest-overwritten queries to at most `7%`;
3. lose no more than `2` points overall;
4. lose no more than `5` points on previous-overwritten queries;
5. lose no more than `5` points on first-overwritten queries.

## Results

| Metric | Balanced | Latest-heavy | Delta |
|---|---:|---:|---:|
| Overall accuracy | 77.734% | 79.536% | +1.801 pp |
| Clean accuracy | 77.604% | 79.340% | +1.736 pp |
| Overwritten accuracy | 77.865% | 79.731% | +1.866 pp |
| Latest-clean accuracy | 67.318% | 69.922% | +2.604 pp |
| Latest-overwritten accuracy | 68.750% | 74.349% | +5.599 pp |
| Previous-overwritten accuracy | 84.375% | 82.292% | -2.083 pp |
| First-overwritten accuracy | 80.469% | 82.552% | +2.083 pp |
| Stale false recall on latest-overwritten | 9.950% | 8.090% | -1.860 pp |

Latest-overwritten accuracy had substantial seed variation under the
latest-heavy curriculum: `74.349% +/- 6.056` percentage points. Stale false
recall was `8.090% +/- 2.590`.

## Decision

Four of five gates passed. The stale-recall ceiling did not:
`8.090% > 7.000%`.

Training distribution therefore **contributes** to the latest-value weakness.
The latest-heavy curriculum improved the target metric, overall accuracy, and
both clean and overwritten accuracy; this is not merely a redistribution that
damages the rest of the task. It is the best observed curriculum for this
protocol.

Training distribution does **not fully explain or solve** the weakness. Stale
historical recall remains above the pre-registered ceiling, and the
latest-overwritten result is less seed-stable than the balanced control.
Accordingly:

- freeze further curriculum-ratio tuning;
- preserve latest-heavy as the preferred diagnostic/training recipe;
- retain balanced training as the causal control;
- instrument both sets of saved checkpoints for current/archive operation use;
- permit one arbitration correction only if those diagnostics show a stable,
  seed-consistent failure mechanism.

## Claim boundary

This result supports: "Latest-heavy supervision improves CurrentArchive's
latest-value retrieval while preserving overall and historical retrieval."

It does not support: "Training data alone fixes version arbitration," or
"CurrentArchive has solved stale recall."

The Kaggle result archive, paired checkpoints, validation histories, raw
predictions, and metadata must be preserved before the next diagnostic.
