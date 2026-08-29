# Changelog

## 2.1.0 - matched endpoint and paper restructure

### Added

- coordinated matrix-to-vector MemoryFeedback;
- version-aware current/archive bounded memory;
- dense 47M, 81M, and 99M MemoryFeedback scaling evidence;
- mixed clean/update, equal-state, distractor, operation, and stale-recall
  diagnostics;
- exact 1B systems-readiness appendices;
- full paper, model card, benchmark protocol, provenance, reproducibility,
  release validation, and archival metadata.
- matched 81.49M MemoryFeedback endpoint at 163.84M characters;
- publication-grade coordinated-dual-memory architecture figure;
- explicit contribution lineage and result-provenance labels.

### Changed

- v1.1.1 late fusion is no longer the only coordination design;
- language and controlled-memory leads are explicitly separated;
- all competitor chart endpoints are labeled by dense versus sparse protocol;
- scaling claims now disclose the 81M-to-99M saturation and schedule mismatch.
- the paper now leads with the v2 mechanisms and separates generic language,
  controlled memory, systems readiness, and inherited v1.1.1 evidence.

### Rejected or frozen

- adaptive preconditioning;
- attention-to-write and its MemoryFeedback combination;
- independent erase/write;
- latest-shadow refresh;
- unmatched official-Mamba matrix insertion;
- further gate-scale tuning of capped official-Mamba feedback.
