# Modus_X decision log

Append-only. Numerical detail belongs in the linked experiment documents.

## 2026-07-23

- The latest-heavy `50/25/25` curriculum is a partial causal win: it improves
  latest-overwritten and overall accuracy, but stale false recall remains
  above the pre-registered ceiling.
- Freeze curriculum-ratio tuning. Preserve latest-heavy as the best observed
  training recipe and balanced training as its control; use their saved
  checkpoints for operation-level arbitration diagnostics.
- Do not change CurrentArchive arbitration until a stable cross-seed
  relationship between measured memory operations and retrieval failure is
  established.
- Operation diagnostics establish that relationship: stale failures have
  lower current/history disagreement in all six paired checkpoints, and the
  final stale answer usually agrees with the current-memory probe rather than
  the vector probe.
- Authorize one state-neutral current-slot refresh test by enabling
  `latest_shadow_write` inside CurrentArchive. Reject broader router, loss, or
  attention sweeps; failure closes the synthetic architecture lane for v2.
- The latest-shadow correction failed decisively: it improved clean latest
  retrieval but harmed overwritten latest retrieval, stale recall, overall
  accuracy, and runtime. Reject it without tuning and close the synthetic
  architecture-correction lane for v2 packaging.

## 2026-07-22

- The equal-memory learned frontier is valid after replacing the unlearnable
  independent-Q/K Transformer with a calibrated tied-Q/K retrieval baseline
  and restoring `version_tag_facts=True` for CurrentArchive.
- Full-context KV wins; CurrentArchive wins in the measured constrained-state
  region. Preserve this as a crossover result, not a universal win.
- The balanced clean/update gate confirms CurrentArchive works on both target
  groups. Its stale false recall makes operation specialization the next gate.
- Refactor project memory: `AGENTS.md` becomes a short operating index; this
  directory owns current status, roadmap, claims, and decisions.

## 2026-07-21

- Derived an exactly counted 1B MemoryFeedbackArchive configuration, but did
  not replace the frozen 1B model because recurrent state rises `1.5307x` and
  exact systems gates remain open.

## 2026-07-19

- Froze official-Mamba true MemoryFeedback after the capped candidate stayed
  efficient but failed to beat its width control. Preserve the JAX
  MemoryFeedback language result as the v2 lead.

## 2026-07-18

- Froze the official-Mamba matrix Pareto branch as a bounded candidate, not a
  language win, because it lost to the narrower pure-Mamba control.

## 2026-07-14

- Rejected the MemoryFeedback plus AttentionToWrite combination after a
  harmful factorial interaction. No open-ended combination tuning.

## 2026-07-13

- Rejected AdaptivePreconditionedArchive and standalone AttentionToWrite as
  independent language promotions; both missed pre-registered gain floors.

## 2026-07-12

- Promoted MemoryFeedbackArchive as the efficiency-qualified v2 language
  candidate after matched dense improvements with active bounded feedback.

## 2026-07-09

- Promoted CurrentArchiveDelta only for versioned-memory semantics. Its later
  language branch was frozen after the completed dense enwik8 run.
