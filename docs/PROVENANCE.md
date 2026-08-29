# Provenance

## Lineage

- Published stable predecessor: Modus_X v1.1.1,
  `https://doi.org/10.5281/zenodo.20923248`.
- Published v2.0.0 predecessor:
  `https://doi.org/10.5281/zenodo.21538210`.
- Historical release source: `Modus_X_v1.1.1/`.
- Experimental v2 workspace: `Modus_X_2.0.0/`.
- Final source/evidence staging directory:
  `Modus_X_2.1.0/release_candidate/`.

The v1.1.1 paper, baseline scripts, figures, and evidence ledgers are retained
as historical context. New v2 claims are sourced from v2 experiment
directories and are not retroactively attributed to v1.1.1.

## Artifact classes

- **Raw**: direct run JSON, CSV, logs, checkpoint metadata, or archives.
- **Derived**: tables and figures generated from raw values.
- **Narrative**: result interpretations, claim registers, and limitations.
- **Historical context**: copied v1.1.1 evidence used to explain lineage.

Each promoted v2 result should record its original workspace path, run date,
configuration, processed data, seed, parameter/state accounting, evaluator,
and artifact hash. Large checkpoints are not embedded in the compact archive
unless explicitly listed in the manifest.

## Dataset identity

The new 81M and 99M MemoryFeedback scaling points used the 100,000,000-byte
enwik8 file with SHA-256:

`2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8`

## Matched-endpoint provenance

The 81M and 99M checkpoint-preserving annealing continuations were excluded
from v2.0.0. They enter v2.1.0 only as normalized reports, configs, schedules,
dataset identity, and dense-audit endpoints. The promoted 81M run completed
40,000 optimizer steps and 163.84M processed characters. Large checkpoints
remain external to the compact release.
