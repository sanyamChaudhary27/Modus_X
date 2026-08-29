# Current Modus_X research status

Snapshot: **2026-07-23**

## North star

Demonstrate that coordinated bounded matrix memory plus a strong recurrent
reasoning stream can deliver all three of the following:

1. competitive language-model quality;
2. useful durable/updateable memory at constant recurrent state;
3. a credible systems and serving advantage at larger scale.

The project has evidence for each ingredient separately. It has not yet shown
all three in one frontier model, and it has not achieved the `1.1` enwik8 BPC
target.

## Stable and experimental lines

- **Published stable line:** Modus_X v1.1.1, Zenodo DOI
  `10.5281/zenodo.20923248`. Do not rewrite its historical results.
- **Experimental line:** `Modus_X_2.0.0/`. This is where CurrentArchive,
  MemoryFeedback, attention, and official-Mamba integration are tested.
- **Long-horizon name:** Askio. Treat it as the proposed scaled system, not an
  already trained model.

## Strongest measured evidence

### Generic language modeling

- At the approximately 80M tier and `163.84M` enwik8 characters, Modus_X
  reaches `1.38418` dense test BPC.
- Official Mamba reaches `1.34578`; Mamba wins by about `0.0384` BPC.
- Official xLSTM reaches `1.41962`; Modus_X wins by about `0.0354` BPC.
- Therefore Modus_X is competitive in this protocol, not SOTA.

Canonical source: `Modus_X_v1.1.1/evidence/RESULTS_LEDGER.md`.

### Associative memory

- Published v1.1.1 balanced-KV recall and overwrite tests show a large Modus_X
  advantage over the tested small official-Mamba configuration.
- These are controlled memory tests, not substitutes for BPC or reasoning
  benchmarks.

Canonical source:
`Modus_X_v1.1.1/evidence/associative_memory/matched_multiseed_2026-06-25/`.

### Matrix-to-vector language coordination

- MemoryFeedbackArchive improves dense test BPC over CurrentArchive by
  `0.032989` at `81.92M` characters and `0.027688` at `102.4M` characters.
- Cost: approximately `0.85%` more parameters and `12.5%` runtime overhead.
- This is the strongest v2 language result and the current architecture lead.

Canonical source:
`Modus_X_2.0.0/experiments/enwik8_current_archive/MEMORY_FEEDBACK_ARCHIVE_V0_2026-07-12.md`.

### Equal-state-byte bounded memory

- Full-context Transformer KV wins at `64.25 KiB`: `98.63%` versus `71.03%`.
- CurrentArchive wins after the KV budget truncates context; at the
  near-parameter-matched `16.13 KiB` point it reaches `72.92%` versus
  `17.71%`.
- CurrentArchive collapses at its smallest matrix state, so its own capacity is
  also bounded.

Canonical source:
`experiments/matrix_memory_capacity/EQUAL_MEMORY_FRONTIER_RESULT_2026-07-22.md`.

### Clean/update behavior

- At `16.13 KiB`, CurrentArchive reaches `77.95%` overall, `77.91%` clean, and
  `77.99%` overwritten accuracy.
- TransformerKV reaches `16.38%`, `14.19%`, and `18.58%` respectively.
- CurrentArchive stale false recall is worse: `11.28%` versus `4.65%`.
- The next memory problem is latest-versus-history arbitration, not whether the
  matrix stores useful information.

Canonical source:
`experiments/matrix_memory_capacity/MIXED_CLEAN_UPDATE_RESULT_2026-07-22.md`.

### Latest-value curriculum diagnostic

- Changing only the latest/previous/first training mix from balanced to
  `50/25/25` raises latest-overwritten accuracy from `68.75%` to `74.35%`.
- Overall accuracy also rises from `77.73%` to `79.54%`; the gain is not
  explained by broad damage to clean or historical retrieval.
- Stale false recall improves from `9.95%` to `8.09%`, but misses the
  pre-registered `<=7%` ceiling and remains variable across seeds.
- Training distribution contributes to the bias but does not fully explain
  it. Freeze curriculum tuning and diagnose current/archive arbitration on the
  saved paired checkpoints.

Canonical source:
`experiments/matrix_memory_capacity/LATEST_HEAVY_CURRICULUM_RESULT_2026-07-23.md`.

### Current/archive operation diagnosis

- On stale latest-value failures, final predictions agree with the
  current-memory shared-head probe about `83%` of the time and almost never
  agree with the vector probe.
- Current/history disagreement is lower on stale failures in all six
  curriculum-seed pairs; history-read norm is higher in all six.
- Router direction is inconsistent. The remaining failure is localized to
  current-slot content/separation rather than vector correction or archive
  readout selection.
- One state-neutral latest-shadow refresh correction is authorized under a
  frozen paired gate. It failed: latest-overwritten accuracy fell `14.06`
  points, stale recall rose `5.86` points, and runtime increased `28.9%`.
- The synthetic architecture-correction lane is now frozen for v2 packaging.

Canonical source:
`experiments/matrix_memory_capacity/CURRENT_ARCHIVE_OPERATION_DIAGNOSTICS_2026-07-23.md`.

## 1B systems state

- The frozen v1-style `1,058,963,121` parameter model has passed exact TPU
  forward/backward, AdamW, real-data updates, Orbax checkpoint, and restore.
- Best measured Kaggle v5e-8 topology: `4 data x 2 model`, context `2,048`,
  about `539 tokens/s`, approximately `10.68 GB/device` in the readiness gate.
- Real enwik8 smoke completed 17 resumable optimizer updates.
- A parameter-matched `1,058,467,601` MemoryFeedbackArchive candidate exists,
  but increases recurrent state by `1.5307x` and has not passed the full exact
  systems ladder. It is not the frozen training configuration yet.

Canonical sources:

- `proposals/1B_scaling/system_validation/`
- `proposals/1B_scaling/MEMORY_FEEDBACK_ARCHIVE_1B_CONFIG_2026-07-21.md`

## Rejected or frozen branches

- Dropout, label smoothing, SGD/momentum, corruption, broad auxiliary-target
  sweeps, and shallow budget reshaping did not supply the missing BPC lever.
- AdaptivePreconditionedArchive: insufficient gain and high runtime overhead.
- AttentionToWriteArchive: below promotion threshold.
- MemoryFeedback plus AttentionToWrite: harmful interaction; frozen.
- Official-Mamba matrix residual insertion: not a language win versus the
  width control.
- Official-Mamba true MemoryFeedback: efficient and bounded, but did not beat
  the width control; freeze this branch.
- Do not resume these without a new causal hypothesis and pre-registered gate.

## Current blockers

1. Latest-heavy supervision reduces but does not eliminate CurrentArchive's
   stale-history bias on latest queries.
2. Operation diagnostics localize stale failures to current/history
   interference; the tested latest-shadow correction made update behavior
   worse and is rejected.
3. MemoryFeedback's 1B candidate lacks exact TPU memory/throughput/parity and
   accumulated-update validation.
4. The project still needs funded compute and a faculty/institutional ally for
   a serious 1B study.
5. The `1.1` BPC target remains unmet and should not dominate every decision.

## Current priority

Package the v2 evidence with CurrentArchiveDelta as the bounded-memory evidence
model and MemoryFeedbackArchive as the language lead. Then redirect research
effort toward source-backed architecture strategy, matched language/scaling
evidence, and compute access rather than further small synthetic corrections.
