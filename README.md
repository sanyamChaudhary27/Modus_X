# Modus_X v1.1.0

This repository contains the Modus_X v1.1 paper, benchmark implementations,
evidence ledgers, generated figures, and reproducibility documentation.

- Latest paper: `whitepaper.pdf`
- Source: `whitepaper.md`
- Versioned release archive: `release/Modus_X_v1.1.0_release.zip`
- Project DOI: https://doi.org/10.5281/zenodo.20443698

## Release Thesis

Modus_X combines a selective recurrent stream with content-addressed
delta-rule matrix memory. The current evidence does not support claiming that
Modus_X wins every language-modeling benchmark:

- official Mamba is stronger on the matched enwik8 byte-prediction protocol;
- Modus_X is stronger than the tested official xLSTM configuration;
- Modus_X is dramatically stronger than official Mamba on the recovered
  associative-recall and same-key-overwrite stress protocol.

The v1.1.0 paper explains this separation directly. Generic next-byte
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
