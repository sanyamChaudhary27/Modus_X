# Modus_X 2.0.0

Status: **v2.0.0 release candidate**

Modus_X 2.0.0 studies whether bounded associative matrix memory can complement
a recurrent vector stream without a sequence-length-growing inference cache.
It is an experimental successor to the published Modus_X v1.1.1 release, not a
claim of state of the art.

## What v2 establishes

Two models lead different evidence layers:

- **MemoryFeedbackArchive** is the language-modeling lead. At `102.4M`
  processed enwik8 characters it improves matched dense validation BPC by
  `0.025297` and dense test BPC by `0.027688` over CurrentArchive, with `0.85%`
  more parameters and about `12.5%` more runtime.
- The measured MemoryFeedback scaling curve improves from `47.44M` to
  `81.49M` parameters, then saturates from `81.49M` to `99.44M` at the fixed
  `102.4M`-character budget. Longer anneal continuations are excluded from
  v2.0.0 and reserved for v2.1.0 only after completion and dense audit.
- **CurrentArchiveDelta** is the bounded-memory evidence model. At equal
  `16,512`-byte recurrent state it substantially outperforms the tested
  truncated Transformer KV baseline on mixed clean/update retrieval. A
  full-context KV baseline wins when the state budget can retain the complete
  context.

The evidence does not show that one v2 model simultaneously owns both leads.
That integration remains future work.

## What v2 does not establish

- It does not reach the `1.1` enwik8 BPC target.
- It does not beat official Mamba on the matched generic-language result.
- It does not establish universal superiority over Transformers.
- It does not contain a quality-trained 1B model.
- It does not solve stale latest-value recall completely.

## Package map

- `src/`: experimental coordinated-memory implementation.
- `paper/`: full whitepaper source and PDF builder, retaining the v1.1.1
  scientific lineage and adding the v2 evidence.
- `docs/`: model card, architecture, protocol, provenance, limitations, and
  reproducibility material.
- `figures/`: inherited v1.1.1 figures and measured v2 scaling figures.
- `benchmarks/`: official baseline scripts retained from the published
  v1.1.1 package.
- `language/`: enwik8 model, trainer, audits, and MemoryFeedback tests.
- `memory/`: controlled versioned-memory implementation and closure gates.
- `evidence/`: canonical positive and negative result documents plus retained
  raw evidence.
- `systems/`: exact 1B readiness code and bounded systems
  documentation. These are readiness artifacts, not 1B quality evidence.
- `ESSENTIAL_FILES.md`: the minimum reading and reproduction path.
- `REPRODUCIBILITY.md`: environment, protocol, and execution map.
- `EVIDENCE_INDEX.md`: measured results and claim boundaries.
- `LIMITATIONS.md`: unresolved scientific and systems limitations.
- `release/`: changelog, gates, validator, Zenodo metadata, and deterministic
  archive builder.
- `MANIFEST.sha256`: file-level integrity manifest.

## Release position

The core v2 result is modest but real: matrix-to-vector feedback improves the
tested bounded recurrent language model, while a separate current/archive
design demonstrates useful versioned storage under constrained state. The
next serious study must test whether those strengths can coexist at matched
scale and compute.

The published predecessor is Modus_X v1.1.1:
`https://doi.org/10.5281/zenodo.20923248`. The reserved v2.0.0 version DOI is
`https://doi.org/10.5281/zenodo.21538210`; it becomes registered when the
Zenodo draft is published.
