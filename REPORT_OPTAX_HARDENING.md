# v3 Training & Model Hardening Report

## 1. Executive Summary

This report documents the v3 training and model-hardening work performed after the audit of the previous implementation.

The work had two goals:

1. Correct implementation defects that could alter training schedules, diagnostics, historical data distributions, frozen evidence, attention behavior, memory initialization, auxiliary-loss semantics, or model dispatch.
2. Add focused regression coverage and validate that the hardened trainer executes end-to-end on a TPU using Enwik8.

The hardening work was deliberately separated from the fixed-budget C1 parameter-allocation research. Historical/frozen experiment sources and protocols were not silently changed.

Final hardening commit:

```text
c70a147 Harden v3 training and model dispatch
```

Files in the commit:

```text
language/models.py
memory/run_versioned_memory_ablation.py
src/modus_x2/coordinated_dual_memory.py
tests/test_tpu_scheduler.py
```

The branch is:

```text
aryan-v3-hardening
```

A tiny TPU smoke test was subsequently completed successfully using the real Enwik8 dataset.

---

## 2. Context

The v3 work is part of the Modus_X recurrent-memory language-model research codebase.

The audit identified issues where apparently small implementation changes could silently alter optimization schedules, diagnostic tensor semantics, historical data distributions, frozen evidence sources, attention behavior, initialization, auxiliary objectives, model dispatch, and evaluation.

The objective of this hardening pass was therefore not to introduce a new research architecture or claim a new model-quality result. The objective was to make the existing v3 training infrastructure internally consistent, testable, and safe to run.

---

# 3. Critical Fix: Optax Warmup-Cosine Scheduling

## Problem

The previous warmup-cosine implementation shortened the Optax decay horizon by subtracting the warmup period from `decay_steps`.

The problematic form was:

```text
decay_steps = total_steps - warmup_steps
```

For the intended Optax schedule semantics, this causes the schedule to reach its terminal learning rate too early.

For example:

```text
total_steps = 100
warmup_steps = 10
```

would make the decay reach its terminal region around step 90 rather than using the complete 100-step schedule horizon.

## Resolution

The schedule was corrected to use the full training horizon:

```text
decay_steps = total_steps
```

Warmup is therefore part of the overall schedule rather than reducing the total schedule length.

## Regression Test

Added:

```text
tests/test_tpu_scheduler.py
```

Validation:

```text
PASS: warmup cosine scheduler regression tests
```

---

# 4. Diagnostic Axis Correction

## Problem

The displaced-archive gate diagnostic incorrectly interpreted a sequence-position axis as the layer axis.

The relevant tensors have semantic dimensions:

```text
batch × layer × sequence
```

Changing reductions from the layer dimension to the sequence dimension can produce numerically valid results while labeling token positions as layers.

Synthetic example:

```text
Input shape:                 (2, 3, 5)
Correct per-layer result:    (3,)
Incorrect result:            (5,)
```

## Resolution

The diagnostic reductions were restored to the layer dimension.

The intended diagnostic contract is preserved, including expected-layer/rank checks for sparse AttentionToWrite diagnostics.

---

# 5. Stage1G Historical Generator Protection

## Problem

A proposed change replaced the historical Stage1G one-overwrite sampling behavior with repeated-key / multi-update sampling.

This changes the task distribution and therefore constitutes a new experiment rather than a neutral refactor.

## Resolution

Historical Stage1G behavior was preserved.

The repeated-key / multi-update formulation should be treated as a separately named experiment with its own:

- generator,
- protocol,
- result namespace,
- `latest_marker` semantics,
- tests,
- and metrics.

This prevents historical Stage1G results from silently changing meaning.

---

# 6. Frozen Evidence Sources

## Problem

The audit identified changes to frozen v2.1/v3 model evidence sources while generated artifacts still contained older embedded source payloads.

That creates source/artifact divergence.

A proposed normalization modification was also demonstrated to be numerically non-neutral at tiny values, so it cannot be treated as a purely cosmetic stability change for historical evidence.

## Resolution

Frozen evidence sources were not silently rewritten as part of this hardening contribution.

Any future normalization change should be a new development implementation with an explicit numerical contract and dedicated tests.

---

# 7. DeepSupervision Model Dispatch

## Problem

Model names ending in:

```text
_DeepSupervision
```

must be normalized before base-model alias dispatch.

Otherwise a valid DeepSupervision model name can fail registry lookup.

## Resolution

DeepSupervision suffix handling now occurs before model alias dispatch.

Conceptually:

```text
Modus_X_<variant>_DeepSupervision
              |
              v
remove suffix
              |
              v
resolve base alias
              |
              v
enable DeepSupervision behavior
```

Existing aliases were preserved.

---

# 8. AttentionToWrite Shallow-Model Handling

## Problem

Reduced-depth configurations can have fewer layers than assumptions made by normal AttentionToWrite initialization.

## Resolution

Zero-module / shallow-model fallbacks were added to normal and memory-feedback AttentionToWrite paths, including DeepSupervision paths.

This allows shallow configurations to execute without assuming unavailable layers.

---

# 9. Generic DeepSupervision LM Loss

## Problem

DeepSupervision forwards can return multiple outputs rather than one logits tensor.

A loss implementation assuming a single tensor cannot safely handle those models.

## Resolution

The generic `lm_loss()` path was hardened to support arbitrary tuple outputs from DeepSupervision forwards.

---

# 10. Future-Target Head Wiring

When future targets are requested, future heads are now explicitly initialized and attached.

Conceptually:

```text
future_target_count > 0
        |
        v
future-head initialization key
        |
        v
add future heads
```

This keeps future-target training configuration and parameter construction explicit.

---

# 11. Versioned-Memory Auxiliary Loss

## Problem

Repeated wrong-version labels can occur in the versioned-memory auxiliary objective.

Counting the same wrong version multiple times can distort the intended objective.

## Resolution

Wrong-version candidates are deduplicated using occurrence semantics over latest, previous, and first versions.

Only distinct incorrect candidates are counted.

---

# 12. Coordinated-Memory Configuration Validation

## Problem

The coordinated-memory configuration requires:

```text
key_dim + n_values <= d_model
```

Without explicit validation, invalid configurations can fail later during initialization.

## Resolution

The configuration now rejects invalid settings early:

```python
if self.key_dim + self.n_values > self.d_model:
    raise ValueError(
        "Invalid CoordinatedMemoryConfig: "
        "key_dim + n_values must be <= d_model"
    )
```

Validation performed:

```text
VALID_CONFIG=PASS
INVALID_CONFIG=PASS
```

The existing initialization semantics were otherwise preserved.

---

# 13. Query-Token Local Attention

A proposed change removed the current query token from local K/V attention.

That can be a legitimate history-only attention design, but it changes the architecture. The previous query self-entry was not treated as target leakage because the query token is an observed input.

## Disposition

Historical behavior was retained.

Query-token exclusion should be evaluated as a separately named experiment with independent training and comparison.

---

# 14. Evaluation Hardening

The evaluation path was hardened so that:

- padded duplicate windows are excluded from primary accounting,
- primary BPC is computed in float32,
- non-divisible evaluation configurations remain valid.

A focused A5-window / batch-3 calculation matched direct calculation within approximately:

```text
5.4e-8 BPC
```

---

# 15. Regression and Static Validation

Completed before the TPU smoke:

## Python compilation

```text
language/models.py                         PASS
memory/run_versioned_memory_ablation.py    PASS
src/modus_x2/coordinated_dual_memory.py    PASS
```

## Existing memory-feedback regression

```text
status: PASS
params: 85212
control_params: 83610
extra_params: 1602
loss: 9.210529583469906e-08
feedback_gate_prior: 0.11920291930437088
```

## Scheduler regression

```text
PASS: warmup cosine scheduler regression tests
```

## Full compileall

```text
python -m compileall -q .\language .\memory .\src .\tests
```

completed successfully.

## Whitespace validation

```text
git diff --check
```

completed successfully.

---

# 16. TPU + Enwik8 Smoke Test

After CPU and regression validation, the hardened trainer was tested end-to-end on TPU using the actual Enwik8 dataset.

This was an integration smoke test, not a model-quality benchmark.

## Configuration

```text
Model:              Modus_X_Vector_Lean_DeepSupervision
Precision:          float32
Optimizer:          AdamW
Learning rate:      2e-4
Weight decay:       1e-4
Schedule:           constant
Seed:               1

Batch:              1
Target characters:  32,768
Stop characters:    32,768
Checkpoint chars:   32,768
Evaluation chunks:  4
Evaluation batch:   1
Input sequence:     512
Loss tail:          512
Auxiliary layers:   6
```

Trainer initialization reported:

```text
params = 21,896,156
non_embedding_params = 21,806,044
chars_per_step = 512
total_steps = 64
stop_steps = 64
checkpoint_steps = 64
```

Training completed all 64 steps.

Representative progress:

```text
step=1/64   loss=5.8233
step=10/64  loss=4.6197
step=20/64  loss=4.3948
step=30/64  loss=3.7264
step=40/64  loss=3.4008
step=50/64  loss=3.8085
step=60/64  loss=3.1444
```

Final checkpoint:

```text
step = 64
processed_characters = 32768
loss = 3.2306830883026123
val_bpc = 4.770802432561306
```

Final test:

```text
FINAL_TEST_BPC 4.7655
```

## Smoke-Test Conclusion

The complete path succeeded:

```text
Enwik8
  |
  v
data loading
  |
  v
model dispatch
  |
  v
TPU initialization
  |
  v
TPU compilation
  |
  v
Optax / AdamW
  |
  v
training
  |
  v
checkpoint
  |
  v
evaluation
  |
  v
validation + test BPC
```

Therefore the hardened training stack is operational on TPU with the real language-modeling dataset.

The 32K-character BPC values are integration evidence only and must not be interpreted as model-quality benchmark results.

---

# 17. TPU Transparent Hugepages Warning

The TPU runtime emitted a warning that transparent hugepages were not enabled.

This affected startup/performance optimization only.

It did not prevent:

- TPU initialization,
- compilation,
- training,
- checkpointing,
- or evaluation.

No correctness issue was attributed to this warning.

---

# 18. Final Commit

The hardening implementation was committed as:

```text
c70a147 Harden v3 training and model dispatch
```

Files:

```text
language/models.py
memory/run_versioned_memory_ablation.py
src/modus_x2/coordinated_dual_memory.py
tests/test_tpu_scheduler.py
```

Branch:

```text
aryan-v3-hardening
```

The branch was pushed to the remote repository.

---

# 19. What This Work Establishes

This hardening work establishes that:

1. The identified warmup-cosine scheduling defect was corrected.
2. The corrected schedule has dedicated regression coverage.
3. DeepSupervision model dispatch is handled before alias resolution.
4. Reduced-depth AttentionToWrite paths have safe fallbacks.
5. Generic LM loss handles multi-output DeepSupervision models.
6. Coordinated-memory configuration constraints are enforced explicitly.
7. Versioned-memory wrong-version candidates are deduplicated.
8. Historical/frozen experiment sources are protected from silent behavioral changes.
9. The hardened training stack passes CPU regression and compilation checks.
10. The hardened trainer executes a real Enwik8 training/evaluation smoke on TPU.

---

# 20. What This Work Does Not Establish

This PR does not establish:

- a new state-of-the-art BPC result,
- superiority of the v3 architecture,
- improvement from query-token exclusion,
- improvement from repeated-key data generation,
- improvement from normalization changes,
- or improvement from fixed-budget C1 Q/K reallocation over a matched dense baseline.

Those are separate research questions requiring independent experiments and matched protocols.

In particular:

```text
32K TPU smoke BPC != benchmark result
```

---

# 21. Relationship to Fixed-Budget C1 Research

The fixed-budget parameter-allocation research is a separate research track.

That work asks whether capacity can be removed from highly compressible Q/K projections and reallocated elsewhere while maintaining approximately 47.44M parameters, and whether that reallocation changes language-modeling BPC.

C1/C2/C3 training results and their research report should therefore remain separate from this hardening PR.

This PR validates engineering infrastructure and does not constitute a new C1 performance experiment.

---

# 22. Reproducibility Checklist

## Source

```text
Branch: aryan-v3-hardening
Commit: c70a147
```

## Validation

```text
[PASS] Python compilation
[PASS] Existing memory-feedback regression
[PASS] Optax scheduler regression
[PASS] compileall
[PASS] git diff --check
[PASS] TPU initialization
[PASS] Enwik8 loading
[PASS] TPU compilation
[PASS] AdamW / Optax training
[PASS] checkpoint
[PASS] evaluation
```

## TPU smoke

```text
Dataset: Enwik8
Characters: 32,768
Steps: 64
Batch: 1
Sequence length: 512
Optimizer: AdamW
LR: 2e-4
Weight decay: 1e-4
Schedule: constant
Precision: float32
```

---

# 23. Final Status

```text
v3 hardening audit              COMPLETE
Optax scheduling correction     COMPLETE
Regression coverage             COMPLETE
Model dispatch hardening        COMPLETE
Memory/config hardening         COMPLETE
CPU validation                  COMPLETE
TPU + Enwik8 smoke              COMPLETE
Hardening branch pushed         COMPLETE
Scientific benchmark claim      NOT MADE
C1 research                     SEPARATE TRACK
Matched dense C1 control        PENDING
```

The v3 hardening work is ready for review as an engineering/infrastructure contribution, with historical research protocols preserved and remaining scientific questions kept explicitly separate.
