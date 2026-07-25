# Modus_X

Modus_X is an open research program exploring how bounded associative matrix
memory can cooperate with recurrent computation. This repository preserves the
published v1.1.1 line and now includes the experimental v2.0.0 release
candidate as a separate, versioned package.

## Releases

### Modus_X 2.0.0

The complete v2 research package is in [`Modus_X_2.0.0/`](Modus_X_2.0.0/).
It introduces two distinct experimental leads:

- **MemoryFeedbackArchive**, where retrieved matrix context conditions the
  recurrent stream and improves the tested dense enwik8 result over the
  matched CurrentArchive control;
- **CurrentArchiveDelta**, which separates bounded current and historical
  storage and performs strongly on controlled mixed clean/update retrieval
  after a matched Transformer KV cache can no longer retain full context.

These strengths have not yet been demonstrated together in one model.
Official Mamba remains stronger on the tested matched generic-language
comparison, the `1.1` BPC target remains unmet, and the 1B artifacts establish
systems readiness rather than trained model quality.

The v2.0.0 Zenodo draft has reserved DOI
[`10.5281/zenodo.21538210`](https://doi.org/10.5281/zenodo.21538210). The DOI
becomes registered when that draft is published.

The v2.1.0 paper correction is pre-registered in
[`roadmap/V2_1_0_PAPER_RESTRUCTURE.md`](roadmap/V2_1_0_PAPER_RESTRUCTURE.md).
It will make the transition from v1.1.1 to the v2 mechanisms and evidence
visually and narratively explicit.

### Modus_X v1.1.1

The repository root below remains the published v1.1.1 research release. Its
Zenodo DOI is
[`10.5281/zenodo.20923248`](https://doi.org/10.5281/zenodo.20923248).

---

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
