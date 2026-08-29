# Modus_X v1.1.1 Research Release

This directory is the isolated publication workspace for the next Modus_X
paper release. It is derived from the published Modus_X package and the
verified v1.0.1 evidence campaign. The original release directories remain
unchanged.

## Release Thesis

Modus_X combines a selective recurrent stream with content-addressed
delta-rule matrix memory. The current evidence does not support claiming that
Modus_X wins every language-modeling benchmark:

- official Mamba is stronger on the matched enwik8 byte-prediction protocol;
- Modus_X is stronger than the tested official xLSTM configuration;
- Modus_X is dramatically stronger than official Mamba on the recovered
  associative-recall and same-key-overwrite stress protocol.

The v1.1.0 paper will explain this separation directly. Generic next-byte
prediction and explicit associative memory are complementary capabilities, not
interchangeable metrics.

## Directory Map

- `paper/`: expanded paper source and PDF build tooling.
- `docs/`: architecture, model card, claims, limitations, provenance, and
  reproducibility documentation.
- `benchmarks/modus_x/`: Modus_X evaluation implementations and configs.
- `benchmarks/official_baselines/`: externally sourced baseline harnesses.
- `evidence/language_modeling/`: compact language-modeling outputs.
- `evidence/associative_memory/`: recall and overwrite evidence.
- `figures/`: publication figures and diagrams.
- `release/`: release checklist, changelog, archive manifest, and validation.

## Current Headline Evidence

| Evaluation | Modus_X | Comparator | Outcome |
|---|---:|---:|---|
| enwik8 dense test, 80M tier | `1.38418` BPC | official Mamba `1.34578` | Mamba wins |
| enwik8 dense test, 80M tier | `1.38418` BPC | official xLSTM `1.41962` | Modus_X wins |
| Balanced-KV recall, seed 17 | `97.325%` | official Mamba near `3.1%` chance | Modus_X wins |
| 50% same-key overwrite | `88.850%` | official Mamba `3.425%` | Modus_X wins |

All claims must remain scoped to the exact protocols and configurations in
`docs/CLAIMS_AND_EVIDENCE.md`.

## v1.1.1 Evidence Addition

The completed three-seed component ablation separates the two streams on the
recovered associative-memory protocol. At length 2048, MatrixOnly achieves
`96.992 +/- 0.427%` without overwrite and `87.625 +/- 0.745%` with 50%
overwrite; VectorOnly remains near the `3.125%` chance level. VectorLeanPM
retains similarly strong performance with fewer parameters than the scalar
router control. This supports a bounded mechanism claim: the matrix stream is
necessary for the tested binding and overwrite behavior. See
`evidence/associative_memory/component_ablation_2026-06-26/`.
