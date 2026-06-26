# Modus_X v1.1.1 Distribution Kit

Versioned Zenodo record: https://doi.org/10.5281/zenodo.20923248

This directory is the canonical public-distribution kit for the Zenodo and
GitHub v1.1.1 release.

## Evidence Boundary

- The tested official Mamba configuration is stronger on the matched 80M
  enwik8 dense-test comparison: `1.34578` versus Modus_X `1.38418` BPC.
- Modus_X exceeds the tested official xLSTM configuration on the same audit:
  `1.38418` versus `1.41962` BPC.
- On the recovered balanced-KV protocol, Modus_X VectorLeanPM reaches
  `96.758 +/- 0.317%` at length 2048 without overwrite, while the tested
  official Mamba is at `3.025 +/- 0.217%`.
- At 50% same-key overwrite, the figures are `87.750 +/- 0.763%` for
  Modus_X and `3.242 +/- 0.142%` for the tested official Mamba.
- The new component ablation attributes the controlled associative-memory
  behavior to the matrix stream: MatrixOnly `96.992 +/- 0.427%`, VectorOnly
  `3.100 +/- 0.109%` at length 2048 without overwrite.

The release does not claim universal superiority, a natural-language
long-context win, lower total compute, or 1B-scale validation.

## Hugging Face Note

Hugging Face Paper Pages are indexed from arXiv IDs. A Zenodo-only paper
cannot honestly be created as an HF Paper Page. The right action is an arXiv
submission first; then index/claim the corresponding HF Paper Page and link
the GitHub repository.
