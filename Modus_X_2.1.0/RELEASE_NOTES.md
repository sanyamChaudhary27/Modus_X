# Release notes

## Modus_X 2.1.0

This candidate promotes the completed matched MemoryFeedback endpoint and
restructures the paper so the v2 mechanisms, evidence lanes, and limitations
are distinguishable from inherited v1.1.1 results.

### Added

- Version-aware bounded current/archive memory.
- Disciplined erase/write delta updates.
- Matrix-to-vector MemoryFeedback.
- Dense enwik8 audits at matched character budgets.
- Measured 47M/81M/99M MemoryFeedback scaling at a `102.4M`-character
  endpoint, including the observed 81M-to-99M saturation.
- Matched 81M MemoryFeedback endpoint at `163.84M` characters:
  `1.375422` dense validation and `1.382445` dense test BPC.
- Publication-grade coordinated-dual-memory architecture figure.
- Explicit v1.1.1-to-v2 contribution map and provenance labels on primary
  result tables.
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
production API. It is prepared as a new Zenodo version linked to v2.0.0 and
the published v1.1.1 lineage.
