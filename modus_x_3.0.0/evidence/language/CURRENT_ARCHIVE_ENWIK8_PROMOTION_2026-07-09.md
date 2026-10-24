# CurrentArchiveDelta enwik8 Promotion Gate

Date: 2026-07-09

## Decision

Promote `CurrentArchiveDelta` from controlled Stage1G memory tests to a small
enwik8 byte-language-model smoke, but do not promote it to a long BPC campaign
yet.

## Why This Is Reasonable

`CurrentArchiveDelta` is currently the strongest Modus_X 2.0.0 memory variant
on the Stage1G versioned-retrieval task. It separates bounded state into:

- a current/latest associative matrix;
- a historical/archive associative matrix.

This directly addresses the update objection: updating a fact is not always
equivalent to deleting the earlier fact. Some tasks need the latest value, while
others need a previous or first value.

The Stage1G capacity frontier did not show that larger `ax_res=128` obviously
earns its cost under the short training budget. That is not yet a memory wall:
the `ax_res=64` version kept a large latest-value advantage across load, and
the larger state may simply need different training or more examples. For now,
the default is the smallest useful current/archive state.

## Translation Caveat

Stage1G supplies explicit query roles such as `latest`, `previous`, and `first`.
enwik8 has no such labels. Therefore the language-model version cannot simply
copy the synthetic model. It must learn when to use current versus archive
memory from byte context alone.

The first byte-LM variant keeps the v1.1.1 Modus_X layer shape and adds:

- one current matrix updated like the original Modus_X matrix;
- one archive matrix with slower write and stronger retention;
- a learned per-channel current/archive read mix;
- the same vector recurrence and output router as v1.1.1.

This doubles matrix state bytes inside the Modus stream, but remains constant
in sequence length.

## Smoke Configuration

Script directory:

`Modus_X_2.0.0/experiments/enwik8_current_archive/`

Model name:

`Modus_X_CurrentArchive_DeepSupervision`

First gate:

- dataset: enwik8, standard `90M/5M/5M` split;
- target characters: `20.48M`;
- context: `512`;
- batch: `8`;
- parameters: 12 layers, `embed_dim=512`, `state_dim=512`,
  `hidden_dim=1536`, `router_hidden=32`;
- training method: future target `2`, future weight `0.5`, auxiliary layer `6`,
  auxiliary weight `0.05`, AdamW, weight decay `1e-4`, constant LR `8e-4`.

## Promotion Rules

Compare against the known v1.1.1 42.69M/12-layer first gate:

- baseline sparse validation BPC at `20.48M`: approximately `1.714218`.

Outcomes:

- If CurrentArchive is within `+0.03` BPC and finite/stable, continue to
  `40.96M`.
- If it improves BPC while retaining Stage1G memory gains, promote as the main
  Modus_X 2.0.0 language candidate.
- If it is worse by `0.03-0.08` BPC but stable, keep it as a memory-specialized
  branch and tune archive write/read gates before a longer run.
- If it is worse by more than `0.08` BPC, stop. Do not spend long-run TPU on
  this translation.

## Exact Command

```bash
python -u run_current_archive_smoke.py \
  --data-path /kaggle/working/enwik8 \
  --outdir /kaggle/working/current_archive_enwik8_smoke \
  --target-chars 20480000 \
  --checkpoint-chars 4096000 \
  --batch 8 \
  --eval-chunks 128 \
  --eval-batch 8
```

## Scientific Claim Boundary

This smoke can only show whether the current/archive mechanism is compatible
with byte-LM learning. It cannot claim:

- improved generic BPC over v1.1.1;
- progress toward `1.1 BPC`;
- superiority over Mamba or Transformer.

Those claims require equal-character dense validation/test audits and matched
baselines.

## First Result: LR 8e-4 Smoke

Pasted Kaggle TPU result:

| processed characters | step | sparse validation BPC |
|---:|---:|---:|
| 4.096M | 1,000 | 2.039806 |
| 8.192M | 2,000 | 1.861412 |
| 12.288M | 3,000 | 1.783891 |
| 16.384M | 4,000 | 1.728891 |
| 20.480M | 5,000 | **1.686276** |

Other run metadata:

- parameters: `47,038,396`;
- non-embedding parameters: `46,907,324`;
- chars per step: `4,096`;
- final sparse test BPC: `1.7963`;
- elapsed: `4,015.51s`;
- LR: `8e-4`;
- auxiliary/future recipe: v1.1.1 future target `2`, future weight `0.5`,
  auxiliary layer `6`, auxiliary weight `0.05`.

Interpretation:

- The result passes the first compatibility gate. It is finite and beats the
  42.69M v1.1-style first gate (`~1.714218`) by approximately `0.028` BPC.
- It should not be compared as a same-size win against the 55M and 100M scaling
  lanes. It has fewer parameters than the 55M run and far fewer than the
  108.67M run.
- Relative to the measured scaling ladder, it sits in a plausible place:
  better than the 42.69M first gate and worse than the 55M first gate
  (`1.670623`), which suggests the current/archive mechanism did not break
  byte-LM learning.
- Throughput/elapsed is slower than the comparable v1 lane, as expected from
  the second matrix state. Future claims must report state bytes and elapsed
  time alongside BPC.

Next gate:

Run a tiny LR screen from scratch at the same `20.48M` character budget:

- `6e-4`;
- `4e-4`.

Promote the best LR to `40.96M` only if it improves the `8e-4` result or keeps
similar BPC with a smoother curve.

## Learning-Rate Screen Result

The `6e-4` candidate completed the `20.48M`-character gate:

| processed characters | step | sparse validation BPC |
|---:|---:|---:|
| 4.096M | 1,000 | 2.051047 |
| 8.192M | 2,000 | 1.874774 |
| 12.288M | 3,000 | 1.787700 |
| 16.384M | 4,000 | 1.724595 |
| 20.480M | 5,000 | **1.677799** |

Other result fields:

- final sparse test BPC: `1.7782`;
- elapsed: `4,016.86s`;
- validation gain over `8e-4`: `0.008477` BPC;
- sparse-test gain over `8e-4`: `0.0181` BPC.

The `4e-4` candidate was interrupted after step `2,000`. Its validation BPC
was `2.102013` at step `1,000` and `1.907363` at step `2,000`. At the same
milestone it trailed both `6e-4` (`1.874774`) and `8e-4` (`1.861412`), so it
is not promoted and should not be rerun with the remaining TPU quota.

### Interpretation and Next Gate

The `6e-4` curve starts slightly behind `8e-4`, crosses it by step `4,000`,
and finishes ahead. This is consistent with a modestly lower optimum learning
rate for the two-matrix model, but the gain is below `0.01` BPC and therefore
is not a standalone architecture win.

Run `6e-4` from scratch to `40.96M` characters. Compare it against the
published v1.1.1 future-target-2 result of `1.603902` sparse validation BPC at
the same character budget. Because CurrentArchive has `47.04M` parameters
versus `42.69M`, use these decision thresholds:

- below `1.590`: meaningful promotion signal; run a dense audit and memory
  diagnostics;
- `1.590-1.610`: compatible but inconclusive after parameter/state cost;
- above `1.610`: do not continue the generic BPC branch without changing the
  current/archive controller.

The longer gate must report throughput and recurrent-state bytes. The
CurrentArchive language model keeps two `512 x 512` matrices per layer, so its
bounded state is approximately twice the matrix-state footprint of v1.1.1.

## 40.96M Promotion Result

The `6e-4` run continued successfully to `40.96M` processed characters:

| processed characters | step | sparse validation BPC |
|---:|---:|---:|
| 24.576M | 6,000 | 1.645820 |
| 28.672M | 7,000 | 1.628799 |
| 32.768M | 8,000 | 1.602320 |
| 36.864M | 9,000 | 1.592612 |
| 40.960M | 10,000 | **1.579020** |

Final sparse test BPC was `1.6954`. Elapsed time was `8,016.14s`.

This passes the pre-registered meaningful-promotion threshold of `1.590` and
beats the v1.1.1 future-target-2 result (`1.603902`) by `0.024882` validation
BPC at the same character budget. Sparse test improves from the cited v1.1.1
result `1.7145` to `1.6954` (`0.0191` BPC).

The result warrants continuation to `81.92M` characters at the same `6e-4`
learning rate because the curve remains descending. It does not yet establish
an architecture-level win: CurrentArchive has `47.04M` versus `42.69M`
parameters and approximately twice the matrix-state bytes. After `81.92M`, run
a dense validation/test audit and report throughput/state accounting before
any external claim.

## 81.92M Result

The promoted `6e-4` run reached step `20,000` / `81.92M` processed characters:

| processed characters | step | sparse validation BPC |
|---:|---:|---:|
| 45.056M | 11,000 | 1.555913 |
| 49.152M | 12,000 | 1.551661 |
| 53.248M | 13,000 | 1.546936 |
| 57.344M | 14,000 | 1.535207 |
| 61.440M | 15,000 | 1.522699 |
| 65.536M | 16,000 | 1.519245 |
| 73.728M | 18,000 | 1.500766 |
| 77.824M | 19,000 | 1.495521 |
| 81.920M | 20,000 | **1.489027** |

Final sparse test BPC was `1.6167`; elapsed time was `16,036.61s`.

At equal characters, this improves over the strongest cited 42.69M v1.1.1
future-target branch (`1.513070` validation, `1.6379` test) by approximately
`0.024043` validation BPC and `0.0212` sparse test BPC. The improvement has
therefore persisted across both the `40.96M` and `81.92M` gates.

The next mandatory action is a dense train-tail/validation/test audit of this
exact checkpoint. Do not continue training until the dense result confirms
that the sparse improvement is not caused by evaluation-window sampling.

## Dense Audit At 81.92M

The step-20,000 checkpoint passed the dense audit using two non-overlapping
offset schemes with `9,765` windows per split and offset:

| split | dense offset 0 | dense offset half | mean |
|---|---:|---:|---:|
| train tail | 1.516197 | 1.516538 | **1.516367** |
| validation | 1.554753 | 1.555219 | **1.554986** |
| test | 1.566425 | 1.566272 | **1.566349** |

Sampling diagnostics were consistent with the dense result:

- validation linspace/random: `1.545772` / `1.534946`;
- test linspace/random: `1.557558` / `1.561314`;
- dense standard errors were approximately `0.00355-0.00359` BPC.

Interpretation:

- sparse validation (`1.489027`) was optimistic by approximately `0.066` BPC;
- sparse test (`1.6167`) was pessimistic by approximately `0.050` BPC;
- the dense audit is the authoritative result;
- dense train-to-validation gap is approximately `0.03862` BPC;
- dense train-to-test gap is approximately `0.04998` BPC.

This confirms stable generalization at the current budget and does not show the
large overfitting gap seen in the old 500M-character T12 run. It does not by
itself prove an architecture win over v1.1.1 because no parameter/state-matched
v1.1.1 dense audit exists at exactly `81.92M` characters.

Next training gate: preserve the checkpoint, then resume for one `20.48M`
character block to `102.4M` with LR reduced from `6e-4` to `3e-4`. Re-audit
dense validation/test only if sparse validation improves materially. Do not
continue indefinitely at constant `6e-4`.

## 102.4M LR-Staging Result

The checkpoint resumed from `81.92M` to `102.4M` with LR reduced from `6e-4`
to `3e-4`:

| processed characters | step | sparse validation BPC |
|---:|---:|---:|
| 90.112M | 22,000 | 1.437214 |
| 94.208M | 23,000 | 1.429591 |
| 98.304M | 24,000 | 1.427536 |
| 102.400M | 25,000 | **1.423321** |

Final sparse test BPC was `1.5342`; elapsed time was `20,056.10s`.

The LR drop was effective: sparse validation improved by `0.065706` BPC over
the `20.48M`-character block. Against the strongest comparable 42.69M v1.1.1
scheduled branch (`1.430213` validation, `1.5361` test), CurrentArchive is ahead
by only `0.006892` validation BPC and `0.0019` sparse test BPC. This is a
near-tie after accounting for CurrentArchive's extra parameters and doubled
matrix state, not a generic-language-modeling architecture victory.

Run the dense audit on the step-25,000 checkpoint before continuing. If the
dense result improves materially over the step-20,000 audit, the next bounded
training block may use LR `1e-4` to `143.36M`, following the established v1.1.1
staging recipe. Otherwise stop BPC continuation and prioritize displaced-value
archive transfer plus attention-to-write controller experiments.

## Dense Audit At 102.4M

The step-25,000 checkpoint passed the dense continuation gate:

| split | dense offset 0 | dense offset half | mean |
|---|---:|---:|---:|
| train tail | 1.427541 | 1.427999 | **1.427770** |
| validation | 1.484878 | 1.485162 | **1.485020** |
| test | 1.492598 | 1.492790 | **1.492694** |

Relative to the step-20,000 / `81.92M` dense audit:

- dense validation improved by approximately `0.069966` BPC;
- dense test improved by approximately `0.073655` BPC;
- dense train tail improved by approximately `0.088597` BPC.

The offset agreement is excellent (`0.000284` validation and `0.000192` test),
so the improvement is not a window-sampling artifact. The train-to-validation
gap is approximately `0.05725` BPC and train-to-test gap approximately
`0.06492` BPC: wider than at `81.92M`, but still controlled.

Promote the checkpoint to one final bounded block from `102.4M` to `143.36M`
at LR `1e-4`. Preserve the step-25,000 checkpoint separately. Dense-audit the
step-35,000 result before deciding on any further training.

## 143.36M LR-Staging Result

The `1e-4` block completed at step `35,000`:

| processed characters | step | sparse validation BPC |
|---:|---:|---:|
| 106.496M | 26,000 | 1.398153 |
| 110.592M | 27,000 | 1.391926 |
| 114.688M | 28,000 | 1.387831 |
| 118.784M | 29,000 | 1.382823 |
| 122.880M | 30,000 | 1.381133 |
| 126.976M | 31,000 | 1.375724 |
| 131.072M | 32,000 | 1.375204 |
| 135.168M | 33,000 | 1.372546 |
| 139.264M | 34,000 | 1.372089 |
| 143.360M | 35,000 | **1.369512** |

Final sparse test BPC was `1.4858`; elapsed time was `28,080.43s`.

The block improved sparse validation by `0.053809` BPC from the step-25,000
checkpoint. Against the strongest comparable 42.69M v1.1.1 trajectory at this
budget (`~1.382191` validation and `~1.4933` test), CurrentArchive is ahead by
approximately `0.01268` validation and `0.0075` sparse test BPC. This remains a
modest advantage rather than a parameter/state-adjusted architecture victory.

Run the dense audit on step `35,000`. Do not schedule another training block
until dense validation/test are compared against the step-25,000 audit.

## Dense Audit At 143.36M

The step-35,000 checkpoint produced:

| split | dense offset 0 | dense offset half | mean |
|---|---:|---:|---:|
| train tail | 1.351666 | 1.351969 | **1.351818** |
| validation | 1.433245 | 1.433499 | **1.433372** |
| test | 1.441241 | 1.441404 | **1.441323** |

Relative to the `102.4M` dense audit:

- dense validation improved by approximately `0.051648` BPC;
- dense test improved by approximately `0.051371` BPC;
- dense train tail improved by approximately `0.075952` BPC.

The two offsets agree within `0.000254` validation BPC and `0.000163` test
BPC. The train-to-validation gap is now approximately `0.08155`, and the
train-to-test gap approximately `0.08950`, showing that generalization is still
improving but more slowly than training fit.

Allow one final short annealing block from `143.36M` to `163.84M` at LR
`5e-5`. Preserve the step-35,000 checkpoint. Dense-audit step `40,000`, then
stop this training branch regardless of outcome and return to the
displaced-value archive and bounded attention-to-write controller gates.

## 163.84M Final Annealing Result

The final `5e-5` block completed:

| processed characters | step | sparse validation BPC |
|---:|---:|---:|
| 147.456M | 36,000 | 1.365687 |
| 151.552M | 37,000 | 1.364549 |
| 155.648M | 38,000 | 1.363757 |
| 159.744M | 39,000 | 1.361421 |
| 163.840M | 40,000 | **1.360302** |

Final sparse test BPC was `1.4760`; elapsed time was `32,098.51s`.

The block improved sparse validation by only `0.009210` BPC and sparse test by
`0.0098` BPC. This is useful final polish but clear diminishing return. Run the
final dense audit, freeze the checkpoint and curve, and do not add another BPC
training stage to this branch.

## Final Dense Audit At 163.84M

The frozen step-40,000 checkpoint produced:

| split | dense offset 0 | dense offset half | mean |
|---|---:|---:|---:|
| train tail | 1.332473 | 1.332889 | **1.332681** |
| validation | 1.422926 | 1.423167 | **1.423046** |
| test | 1.432063 | 1.432181 | **1.432122** |

Relative to the step-35,000 / `143.36M` dense audit:

- dense validation improved by approximately `0.010325` BPC;
- dense test improved by approximately `0.009200` BPC;
- dense train tail improved by approximately `0.019137` BPC.

The final train-to-validation gap is approximately `0.09037` BPC and the
train-to-test gap approximately `0.09944` BPC. Offset agreement remains
excellent (`0.000241` validation and `0.000118` test), so the plateau is real
rather than evaluation noise.

### Frozen Decision

Stop training this exact CurrentArchive language branch. Its final measured
claim is:

> A `47.04M`-parameter, two-matrix CurrentArchive Modus_X model trains stably
> on enwik8 and reaches `1.42305` dense validation BPC and `1.43212` dense test
> BPC after `163.84M` processed characters under the documented staged-LR
> recipe.

This is useful v2 evidence but not a generic-BPC breakthrough. The next work is
architectural and must begin from fresh controlled screens: explicit transfer
of displaced current values into archive memory, followed separately by
bounded attention-to-write control. The frozen checkpoint remains a baseline,
not an initialization for those ablations.
