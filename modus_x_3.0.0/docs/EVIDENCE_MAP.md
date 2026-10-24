# Evidence Map

All paths below are package-relative. Upstream files are pinned and hashed in
SOURCE_PROVENANCE.json. They are not rewritten to appear contemporaneous with
this release preparation.

| Paper claim | Authoritative record |
|---|---|
| Three paired K/V endpoints, audits and hashes | `research/v3/05_local_kv_filtering/RESULT.json` |
| K/V protocol and development chronology | `research/v3/05_local_kv_filtering/PROTOCOL.json` |
| Prospective seed-3 declaration | `research/v3/05_local_kv_filtering/src/SEED3_PROTOCOL.json` |
| K/V equations and exact histories | `research/v3/05_local_kv_filtering/src/local_kv.py`, `local_context.py` |
| Actual endpoint trainer/objective | `research/v3/05_local_kv_filtering/src/run_seed3_frozen_endpoint.py`, `tpu_lm_train.py` |
| Execution screen, 500 updates | `research/v3/04_two_stage_execution/RESULT.json` |
| Execution replication, 2,000 updates | `research/v3/04_two_stage_execution/REPLICATION_RESULT.json` |
| Retention-only endpoints | `research/v3/03_language_endpoints/RESULT.json` |
| Retention diagnostics | `research/v3/02_segment_scale_retention/RESULT.json` |
| Carry/intervention/simplification history | `research/v3/01_streaming_and_coordination/RESULT.json` |
| Prior-release 81M endpoint | `evidence/language/scaling/memory_feedback_81m_matched_endpoint.json` |
| Historical official Mamba/xLSTM and retrieval table | `evidence/v1_1_1_context/RESULTS_LEDGER.md` |
| Equal-state mixed clean/update task | `evidence/memory/MIXED_CLEAN_UPDATE_RESULT_2026-07-22.md` |
| Equal-state crossover | `evidence/memory/EQUAL_MEMORY_FRONTIER_RESULT_2026-07-22.md` |
| Calculated aggregate and accounting | `docs/AGGREGATE.json` |

The six reported local-K/V endpoint hashes are in track 05 RESULT.json. A hash
in a report is not verification that a hosting link works or the payload has
been downloaded. No checkpoint payload is embedded in this source/evidence ZIP.
Persistent-state and primary-head audits are reported runner results, not a new
independent trained-checkpoint evaluation performed by this packaging session.
