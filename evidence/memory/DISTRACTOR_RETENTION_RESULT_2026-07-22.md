# CurrentArchive distractor-retention result

Date: 2026-07-22

## Protocol

The frozen gate trained `TwoPathLatestShadowDelta` and
`CurrentArchiveDelta` at 32 bindings, matrix resolution 64, and 50% overwrite.
Each model used seeds 17, 27, and 37, 4,096 training examples, 1,024 test
examples, and 24 epochs. The trained checkpoint was evaluated without
retraining after inserting 0, 16, 32, 64, 128, or 256 distractors immediately
before the query.

The four regimes were random noise, orthogonal irrelevant bindings,
query-similar bindings, and irrelevant bindings carrying the post-update
marker. There are 144 raw evaluation rows.

## Result

At zero distractors, CurrentArchive achieved `72.917 +/- 1.833%` mixed-version
accuracy, compared with `53.418 +/- 1.367%` for the TwoPath control.

| Regime | CurrentArchive at 256 | Control at 256 | Current minus control | Interpretation |
|---|---:|---:|---:|---|
| Random noise | 69.564% | 51.367% | +18.197 pp | Strong retention; only 3.353 pp below its own baseline |
| Irrelevant bindings | 33.691% | 44.792% | -11.100 pp | Bounded state saturates; crossover occurs between 128 and 256 distractors |
| Similar-key bindings | 0.944% | 0.391% | +0.553 pp | Both fail; CurrentArchive already collapses to 2.962% at 16 |
| Post-update bindings | 49.544% | 37.858% | +11.686 pp | Archive path preserves historical versions while the current path is overwritten |

The post-update breakdown localizes the mechanism. CurrentArchive latest-value
accuracy falls from `70.058%` to `11.167%` between 0 and 256 distractors, but
previous-value accuracy remains `68.701%` and first-value accuracy remains
`71.092%`. The architecture therefore preserves historical information in its
archive while repeated current/update traffic overwrites the latest-value
channel. This directly answers why retaining both updated and non-updated
values is useful, while also exposing that the current path needs better
collision handling.

The cleanest professor-facing conclusion is:

> CurrentArchive is not unlimited memory. It robustly ignores random noise,
> preserves historical versions under heavy update traffic, and has a
> measurable saturation frontier under real bindings. Its principal failure
> is key collision: a small number of similar-key writes can destroy recall.

## Confusion caveat

Raw `distractor_confusion` is not comparable across distractor counts. Values
are sampled from 31 wrong labels, so the probability that an arbitrary wrong
prediction appears somewhere among the distractor values approaches 100% as
the count grows. The compact analysis therefore also reports confusion among
errors, expected chance coverage, and excess confusion over that coverage.
Accuracy and version-specific accuracy remain the primary metrics.

## Decision

The distractor-retention question is answered with a bounded positive result
and an explicit failure region. Do not tune this synthetic task to hide the
similar-key failure. The next experiment is the equal-inference-state-byte
Transformer/KV frontier. In parallel, operation diagnostics should test
whether similar-key traffic produces excessive current writes, insufficient
archive protection, or ambiguous readout.

## Artifacts

- Raw log:
  `raw_outputs/current_archive_distractor_gate_2026-07-22_raw.txt`
- Compact JSON and CSV:
  `results/current_archive_distractor_gate_2026-07-22/`
- Plot:
  `results/current_archive_distractor_gate_2026-07-22/distractor_retention_curves.png`
- Analysis command:

```powershell
python experiments/matrix_memory_capacity/analyze_distractor_results.py `
  --input experiments/matrix_memory_capacity/raw_outputs/current_archive_distractor_gate_2026-07-22_raw.txt `
  --outdir experiments/matrix_memory_capacity/results/current_archive_distractor_gate_2026-07-22
```
