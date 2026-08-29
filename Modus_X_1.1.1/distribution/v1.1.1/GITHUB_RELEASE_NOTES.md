# Modus_X v1.1.1: Matrix-Memory Attribution and Reproducible Evidence

Modus_X v1.1.1 is a research-evidence release for an attention-free,
constant-state causal sequence architecture combining selective vector
recurrence with delta-rule associative matrix memory.

## What Changed

- Added a complete three-seed component ablation covering ScalarPM,
  VectorLeanPM, MatrixOnly, and VectorOnly under no-overwrite and 50%
  same-key-overwrite conditions.
- Released all 24 raw seed-level result files, the canonical aggregate,
  exact runner scripts, and reproduction instructions.
- Added a publication figure attributing the controlled associative-recall
  result to the matrix stream.
- Rebuilt the whitepaper and figures with three-seed uncertainty reporting.

## Results, With Caveats

At length 2048 on the recovered balanced-KV protocol:

| Condition | Variant | Accuracy |
| --- | --- | ---: |
| No overwrite | MatrixOnly | `96.992 +/- 0.427%` |
| No overwrite | VectorLeanPM | `96.758 +/- 0.317%` |
| No overwrite | VectorOnly | `3.100 +/- 0.109%` |
| 50% overwrite | MatrixOnly | `87.625 +/- 0.745%` |
| 50% overwrite | VectorLeanPM | `87.750 +/- 0.763%` |
| 50% overwrite | VectorOnly | `3.308 +/- 0.506%` |

With 32 possible values, chance is `3.125%`. The bounded conclusion is that
the matrix stream carries the tested associative binding and overwrite
capability. This does not establish general natural-language recall,
universal architecture superiority, or a general vector-router advantage.

For language modeling, the release reports both directions of evidence: the
tested official Mamba configuration is stronger on the matched 80M enwik8
dense test (`1.34578` BPC versus Modus_X `1.38418`), while Modus_X exceeds the
tested official xLSTM configuration (`1.41962`).

## Included Assets

- `Modus_X_v1.1.1_release.zip`: paper, evidence, raw results, scripts,
  figures, and release documentation.
- `Modus_X_v1.1.1_whitepaper.pdf`: the rendered whitepaper.

## Links

- Zenodo v1.1.1: https://doi.org/10.5281/zenodo.20923248
- Repository: https://github.com/sanyamChaudhary27/Modus_X
