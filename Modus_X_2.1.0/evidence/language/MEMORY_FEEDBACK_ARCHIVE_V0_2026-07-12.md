# MemoryFeedbackArchive v0 Gate

Date: 2026-07-12

## Decision

Do not promote `DisplacedArchive` as the generic language-model branch. At
`20.48M` characters it reached validation BPC `1.682110` and sparse test BPC
`1.7883`, versus CurrentArchive control `1.677799` / `1.7782`. It also took
approximately `5,031s` versus `4,017s`. Preserve it as a memory-semantics
ablation, but do not stack new mechanisms on it.

The next single-change candidate is `Modus_X_MemoryFeedbackArchive`.

## Hypothesis

CurrentArchive computes matrix recall and vector recurrence mostly
independently, then mixes their outputs. This makes matrix memory a passive
expert. The new candidate lets matrix context alter the vector recurrence
inside the same layer, enabling retrieved information to affect subsequent
sequential computation.

## Mechanism

Keep CurrentArchive matrix updates unchanged. Compress the mixed current and
archive context through a rank-32 bridge:

```text
z = tanh(W_down @ matrix_context)
feedback = W_up @ z
g = sigmoid(W_gate @ token + b_gate)
vector_input = layer_norm(token + g * feedback)
vector_state <- selective_recurrence(vector_input)
```

The feedback gate starts near `0.119`, giving the control path a conservative
prior while allowing training to increase coupling where useful.

## Cost

- CurrentArchive control: `47,038,396` parameters.
- MemoryFeedbackArchive: `47,437,768` parameters.
- Added parameters: `399,372` (`~0.85%`).
- Recurrent state bytes: unchanged.
- Context-length state growth: constant.
- Attention: none.

Existing CurrentArchive parameters remain seed-paired. Bridge keys use
`jax.random.fold_in` and do not perturb control initialization.

## Correctness And Diagnostics

`test_memory_feedback_archive.py` verifies finite loss/gradients, causality,
output shapes, exact parameter delta, gate prior, and finite diagnostics.

The first TPU correctness attempt exposed a shape bug because the tiny test
deliberately used `embed_dim=32` and `mamba_state_dim=16`. The feedback bridge
was initially projected to vector-state width and then added to an embedding-
width token. The corrected bridge projects to `embed_dim`, which is the space
in which the residual is formed. The production configuration had equal
widths (`512/512`), so this test prevented a latent configuration-dependent
failure.

The gate audit reports by layer:

- feedback gate;
- feedback norm;
- matrix-context norm;
- feedback-to-input norm ratio;
- current/archive mix.

This distinguishes a real coordinated-memory result from a model that learns
to suppress the bridge.

## First Gate

Use the exact CurrentArchive first-gate recipe:

- `20.48M` processed characters;
- LR `6e-4`;
- batch `8`, context `512`, seed `1`;
- future target `2`, weight `0.5`;
- auxiliary layer `6`, weight `0.05`;
- AdamW, weight decay `1e-4`.

Promotion against validation control `1.677799`:

- `<=1.6578`: promote to `40.96M`;
- `1.6578-1.6878`: inspect bridge diagnostics and confirm only if the bridge
  is active, non-saturated, and throughput remains defensible;
- `>1.6878`: reject;
- NaN, dead gate, or excessive feedback-to-input ratio: stop and diagnose.

Do not add attention or displaced-value transfer to this candidate before the
single-change gate resolves.

## First Result

The `20.48M` screen completed:

| processed characters | step | sparse validation BPC |
|---:|---:|---:|
| 4.096M | 1,000 | 2.052349 |
| 8.192M | 2,000 | 1.867862 |
| 12.288M | 3,000 | 1.778926 |
| 16.384M | 4,000 | 1.719169 |
| 20.480M | 5,000 | **1.673634** |

Final sparse test BPC was `1.7667`; elapsed time was `4,514.01s`.

Relative to CurrentArchive control:

- validation improves by `0.004165` BPC;
- sparse test improves by `0.0115` BPC;
- elapsed time increases by approximately `12.4%`.

This is a mild positive signal but below the `0.02` direct-promotion bar. It
falls in the pre-registered diagnostic-confirmation band. Do not continue to
`40.96M` until `memory_feedback_gate_diagnostics.json` confirms that the
feedback gate is active, non-saturated, and has a bounded feedback-to-input
ratio. If the bridge is dead or unstable, reject it despite the small BPC gain.

## Gate Diagnostics

Global means from 128 windows per split:

| split | context norm | current mix | feedback gate | feedback norm | feedback/input ratio |
|---|---:|---:|---:|---:|---:|
| train tail | 13.3986 | 0.6040 | 0.1098 | 7.2729 | 0.0555 |
| validation | 13.4082 | 0.6048 | 0.1087 | 7.2656 | 0.0548 |
| test | 13.4326 | 0.6019 | 0.1060 | 7.2207 | 0.0525 |

The bridge is active and conservative. Its gate remains close to the `0.119`
initial prior, is neither near zero nor saturated, and produces only a
`~5.2-5.6%` perturbation relative to the vector input. Diagnostics are stable
across all splits. This passes the diagnostic-confirmation condition.

Promote exactly once to `40.96M` at the same LR `6e-4`. Compare against the
CurrentArchive control `1.579020`. A result `<=1.5590` is a direct promotion;
`1.5590-1.5890` requires dense/diagnostic interpretation; `>1.5890` rejects
the bridge for the generic-LM lane. Do not tune the gate or combine mechanisms
before this confirmation completes.

## 40.96M Confirmation

The checkpoint continued to step `10,000`:

| processed characters | step | sparse validation BPC |
|---:|---:|---:|
| 28.672M | 7,000 | 1.618048 |
| 32.768M | 8,000 | 1.597045 |
| 36.864M | 9,000 | 1.582335 |
| 40.960M | 10,000 | **1.567589** |

Final sparse test BPC was `1.6730`; elapsed time was `9,032.76s`.

Against CurrentArchive at the same gate:

- sparse validation improves by `0.011431` BPC;
- sparse test improves by `0.0224` BPC;
- elapsed time increases by approximately `12.7%`.

Diagnostics remain stable across splits. The feedback gate decreased to
`~0.083-0.086`, feedback norm increased to `~8.32-8.37`, and the effective
feedback/input ratio stayed bounded at `~0.048-0.051`. The model therefore
learned a smaller gate over a stronger bridge signal rather than suppressing
the path.

This is a reproducible mild positive, not a direct promotion under the
pre-registered `1.5590` threshold. Run a dense audit of this exact checkpoint.
Do not continue training or add adaptive preconditioning until dense
validation/test confirm that the sparse gain survives full-split evaluation.

## Dense Audit At 40.96M

| split | dense offset 0 | dense offset half | mean |
|---|---:|---:|---:|
| train tail | 1.605033 | 1.605512 | **1.605272** |
| validation | 1.622782 | 1.622993 | **1.622888** |
| test | 1.622845 | 1.623129 | **1.622987** |

The dense offsets agree within `0.000212` validation BPC and `0.000284` test
BPC. Dense train-to-validation gap is only `0.01762`; train-to-test gap is
`0.01772`. The feedback candidate is therefore generalizing cleanly at this
budget.

Sparse validation (`1.567589`) was optimistic by approximately `0.0553` BPC,
while sparse test (`1.6730`) was pessimistic by approximately `0.0500` BPC.
Dense values remain authoritative.

No CurrentArchive dense checkpoint exists at exactly `40.96M`, so this audit
cannot establish a matched dense improvement. Continue unchanged to `81.92M`,
where the frozen CurrentArchive control is available:

- dense validation `1.554986`;
- dense test `1.566349`;
- dense train tail `1.516367`.

At `81.92M`, promote feedback only if its matched dense improvement justifies
the measured runtime cost. A `>=0.02` validation/test gain is strong; a smaller
gain remains an efficiency-qualified ablation.

The first `81.92M` relay attempt selected a checkpoint copy from the dense-
audit output. That folder intentionally lacked `config.json` and
`progress.json`, so resume stopped safely before training. Checkpoint discovery
was corrected to accept only candidates with both training metadata files
beside the checkpoint. The complete step-10,000 training directory remained
intact; no retraining or state loss occurred.

## Matched 81.92M Result

Sparse endpoint:

- validation BPC: `1.473947`;
- sparse test BPC: `1.5821`;
- elapsed time: `18,048.31s`.

Dense audit:

| split | dense offset 0 | dense offset half | mean |
|---|---:|---:|---:|
| train tail | 1.501020 | 1.501782 | **1.501401** |
| validation | 1.535676 | 1.535726 | **1.535701** |
| test | 1.533305 | 1.533414 | **1.533360** |

Against the frozen CurrentArchive control at the exact same character budget:

| metric | CurrentArchive | MemoryFeedback | gain |
|---|---:|---:|---:|
| dense validation | 1.554986 | **1.535701** | **0.019284** |
| dense test | 1.566349 | **1.533360** | **0.032989** |
| dense train tail | 1.516367 | **1.501401** | 0.014966 |

The MemoryFeedback train-to-validation gap is `0.03430` BPC and train-to-test
gap is `0.03196` BPC. Offset agreement is excellent. The improvement is not a
sparse-window artifact.

Bridge diagnostics remain conservative:

- feedback gate: `~0.056-0.058`;
- feedback norm: `~10.43-10.49`;
- feedback/input ratio: `~0.044-0.046`;
- current-memory mix: `~0.585-0.586`.

The gate becomes smaller as the learned feedback representation becomes
stronger, while effective input perturbation stays bounded. The path is active,
not saturated, and stable across splits.

### Promotion Decision

Promote MemoryFeedbackArchive as the current Modus_X 2.0.0 language-model
candidate. The promotion is efficiency-qualified: it adds `0.85%` parameters
and approximately `12.5%` runtime versus CurrentArchive. It is not yet a
universal architecture-superiority claim.

Run one matched LR-staging block from `81.92M` to `102.4M` at LR `3e-4`, then
dense-audit against the frozen CurrentArchive `102.4M` control (`1.485020`
validation, `1.492694` test). Do not extend beyond that gate before returning
to the separate adaptive/interference-aware matrix-update experiment.

## Matched 102.4M Result

The checkpoint resumed from step `20,000` / `81.92M` characters and completed
one bounded block to step `25,000` / `102.4M` characters at LR `3e-4`.

Sparse endpoint:

- validation BPC: `1.406226`;
- sparse test BPC: `1.5119`;
- elapsed time: `22,570.22s`.

The frozen CurrentArchive control at the same character budget reached
`1.423321` sparse validation and `1.5342` sparse test. MemoryFeedback therefore
improves the sparse signals by `0.017095` validation BPC and approximately
`0.0223` test BPC.

Dense audit:

| split | dense offset 0 | dense offset half | mean |
|---|---:|---:|---:|
| train tail | 1.407919 | 1.408502 | **1.408210** |
| validation | 1.459713 | 1.459734 | **1.459723** |
| test | 1.464939 | 1.465073 | **1.465006** |

Against the frozen CurrentArchive control at exactly `102.4M` characters:

| metric | CurrentArchive | MemoryFeedback | gain |
|---|---:|---:|---:|
| dense validation | 1.485020 | **1.459723** | **0.025297** |
| dense test | 1.492694 | **1.465006** | **0.027688** |
| dense train tail | 1.427770 | **1.408210** | 0.019560 |

Offset agreement is excellent: `0.000021` validation BPC and `0.000134` test
BPC. The MemoryFeedback train-to-validation gap is `0.05151` BPC and its
train-to-test gap is `0.05680` BPC. The gain is therefore not a sparse-window
artifact or a larger generalization gap.

Diagnostics remain bounded and split-stable:

- feedback gate: `~0.058-0.060`;
- feedback norm: `~10.51-10.57`;
- feedback/input ratio: `~0.046-0.048`;
- current-memory mix: `~0.583-0.585`.

Relative to the `81.92M` MemoryFeedback checkpoint, dense validation improved
by `0.07598` BPC and dense test by `0.06835` BPC. Relative to CurrentArchive,
the matched dense advantage changed from `0.01928` validation / `0.03299` test
at `81.92M` to `0.02530` validation / `0.02769` test at `102.4M`.

### Final Decision For This Gate

Freeze MemoryFeedbackArchive at `102.4M` as the current Modus_X 2.0.0
language-model candidate. It delivers a reproducible matched dense gain with
only `0.85%` more parameters, but costs approximately `12.5%` more runtime.
This is a meaningful efficiency-qualified architectural result, not a claim
of universal superiority.

Do not continue this exact branch merely to collect more BPC points. Preserve
the checkpoint and raw audit, then test adaptive/interference-aware matrix
updates as a separate seed-paired, constant-state candidate. Do not combine
the two mechanisms until the adaptive update passes its own early gate.
