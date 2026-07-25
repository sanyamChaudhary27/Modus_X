# Release notes

## Modus_X 2.0.0

This candidate consolidates the July 2026 experimental branch.

### Added

- Version-aware bounded current/archive memory.
- Disciplined erase/write delta updates.
- Matrix-to-vector MemoryFeedback.
- Dense enwik8 audits at matched character budgets.
- Measured 47M/81M/99M MemoryFeedback scaling at a `102.4M`-character
  endpoint, including the observed 81M-to-99M saturation.
- Equal-state Transformer KV crossover study.
- Clean/update, distractor, curriculum, and operation diagnostics.
- Exact 1B systems-readiness references.
- Full v1.1.1-derived publication structure: whitepaper, figures, model card,
  benchmark protocol, provenance, release validation, manifest, and Zenodo
  metadata.

### Promoted

- `MemoryFeedbackArchive` for the generic-language lane.
- `CurrentArchiveDelta` for controlled bounded-memory evidence.

### Rejected or frozen

- Adaptive preconditioning.
- Attention-to-write.
- MemoryFeedback plus attention-to-write.
- Official-Mamba residual matrix insertion.
- Official-Mamba true MemoryFeedback at the tested gate.
- Latest-shadow refresh inside CurrentArchive.
- Open-ended synthetic architecture correction.

### Compatibility

The package is an experimental research release and does not preserve a stable
production API. It is prepared as a new Zenodo version linked to the published
v1.1.1 record.
