# Essential Files

This is the shortest reliable path through Modus_X 2.1.0.

## Scientific position

1. `README.md`
2. `CLAIMS.md`
3. `LIMITATIONS.md`
4. `EVIDENCE_INDEX.md`
5. `ARTIFACT_GAPS.md`
6. `paper/whitepaper.md`

## Language result

1. `evidence/language/MEMORY_FEEDBACK_ARCHIVE_V0_2026-07-12.md`
2. `language/models.py`
3. `language/tpu_lm_train.py`
4. `language/test_memory_feedback_archive.py`
5. `language/audit_current_archive_checkpoint.py`
6. `evidence/language/scaling/MEMORY_FEEDBACK_SCALING_RESULT_2026-07-24.md`
7. `evidence/language/scaling/memory_feedback_81m_matched_endpoint.json`
8. `evidence/language/scaling/memory_feedback_scaling.png`

MemoryFeedbackArchive is the language lead. CurrentArchiveDelta is its matched
control in the recorded dense enwik8 comparison.

## Controlled-memory result

1. `evidence/memory/MIXED_CLEAN_UPDATE_RESULT_2026-07-22.md`
2. `evidence/memory/EQUAL_MEMORY_FRONTIER_RESULT_2026-07-22.md`
3. `evidence/memory/CURRENT_ARCHIVE_OPERATION_DIAGNOSTICS_2026-07-23.md`
4. `evidence/memory/CURRENT_ARCHIVE_LATEST_SHADOW_GATE_2026-07-23.md`
5. `memory/run_equal_memory_mixed_update_gate.py`
6. `memory/run_equal_memory_frontier.py`

CurrentArchiveDelta is the controlled bounded-memory lead. The rejected
latest-shadow correction is retained because it closes an important causal
branch.

## Publication and provenance

1. `docs/MODEL_CARD.md`
2. `docs/BENCHMARK_PROTOCOL.md`
3. `docs/PROVENANCE.md`
4. `release/RELEASE_GATES.md`
5. `release/ZENODO_RELEASE_CHECKLIST.md`

## Systems appendix

1. `systems/MEMORY_FEEDBACK_ARCHIVE_1B_CONFIG_2026-07-21.md`
2. `systems/08_full_1b_train_step_readiness.py`
3. `systems/09_full_1b_real_data_smoke.py`

These files establish bounded systems readiness only. They do not establish
trained 1B quality.

## Reproduction

Read `REPRODUCIBILITY.md` and `ENVIRONMENTS.md`, then verify
`MANIFEST.sha256`. Missing raw artifacts are listed in `ARTIFACT_GAPS.md`.
