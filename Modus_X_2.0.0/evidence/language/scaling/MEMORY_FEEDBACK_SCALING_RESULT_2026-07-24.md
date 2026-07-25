# MemoryFeedbackArchive scaling result

Date: 2026-07-24

Status: completed at the pre-registered `102.4M`-character gate; raw Kaggle
compact archives still need local preservation.

## Result

All three points use enwik8, seed 1, 25,000 optimizer updates, batch 8,
context 512, future target 2, and the same dense evaluator. The two new points
used the identical dataset SHA-256:
`2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8`.

| Model | Parameters | Dense train-tail | Dense validation | Dense test | Training time |
|---|---:|---:|---:|---:|---:|
| MemoryFeedback 47M | `47,437,768` | `1.408210` | `1.459723` | `1.465006` | `22,570s` |
| MemoryFeedback 81M | `81,486,728` | `1.360165` | **`1.433138`** | `1.443873` | `26,163s` |
| MemoryFeedback 99M | `99,438,920` | **`1.355061`** | `1.434283` | **`1.442034`** | `27,224s` |

The 47M anchor used LR `6e-4` followed by `3e-4`. Both new points used
`4e-4` through `81.92M` characters and `2e-4` through `102.4M`. The schedule
difference is disclosed and prevents attributing the entire 47M-to-81M gain
to parameter count alone.

## Scaling deltas

From 47M to 81M:

- parameters: `+71.78%`;
- dense test improvement: `0.021133` BPC;
- dense validation improvement: `0.026585` BPC;
- dense train-tail improvement: `0.048045` BPC;
- training time: `+15.92%`.

From 81M to 99M:

- parameters: `+22.03%`;
- dense test improvement: only `0.001839` BPC;
- dense validation **regression**: `0.001145` BPC;
- dense train-tail improvement: `0.005104` BPC;
- training time: `+4.06%`.

The 81M point captures about `92.0%` of the total dense-test improvement from
47M to 99M. The measured curve therefore shows a strong first scaling gain
followed by saturation.

## Generalization diagnosis

The train-to-test gap grows with scale:

| Model | Train-to-validation gap | Train-to-test gap |
|---|---:|---:|
| 47M | `0.05151` | `0.05680` |
| 81M | `0.07297` | `0.08371` |
| 99M | `0.07922` | `0.08697` |

The 99M model fits the train tail better than the 81M model while failing to
improve validation. At this fixed data/update budget, the limiting factor is
generalization or sample efficiency, not raw fitting capacity.

The `0.00184` test difference between 81M and 99M is too small for a strong
architecture claim from one seed. It must be treated as a near-tie unless
replicated.

## Scale-configuration diagnosis

The 99M point is not a self-similar enlargement of the 81M architecture:

| Quantity | 81M | 99M | Change |
|---|---:|---:|---:|
| embedding width | `672` | `768` | `+14.29%` |
| hidden width | `2016` | `2304` | `+14.29%` |
| matrix/vector state width | `672` | `704` | only `+4.76%` |
| feedback compression rank | `32` | `32` | `0%` |
| total parameters | `81.49M` | `99.44M` | `+22.03%` |

MemoryFeedback's compression rank is hard-coded at 32 in the frozen
implementation. It is `6.25%` of state width in the 47M model, `4.76%` in the
81M model, and `4.55%` in the 99M model. The 99M configuration therefore
allocated most new capacity to backbone width while proportionally narrowing
the matrix-to-vector communication bottleneck.

The shared LR schedule is another unresolved scale hyperparameter. Both larger
models used `4e-4` then `2e-4`; no LR range test was performed at 99M. The 99M
train-tail gain alongside flat validation is compatible with under-annealing
or insufficient regularization/data, but does not prove either.

Therefore the result measures the frozen configurations faithfully, but is not
evidence that an optimally allocated 99M MemoryFeedback model must saturate.

## Pre-registered decision

Outcome: **partial pass**.

- Dense test improves monotonically.
- Dense validation improves strongly at 81M, then remains effectively flat
  with a `0.00115` regression at 99M.
- Integrity checks pass: parameter totals, step count, processed characters,
  dataset identity, and evaluation protocol match.

Permitted claim:

> At a fixed 102.4M-character enwik8 budget, MemoryFeedbackArchive improved
> substantially from 47.44M to 81.49M parameters, while scaling from 81.49M
> to 99.44M showed saturation rather than a further robust validation gain.

Do not call this a universal scaling law, do not extrapolate it to `1.1` BPC,
and do not promote a 200M run at the same character budget.

## Competitor context

The historical 80M-tier endpoints used `163.84M` characters:

| Model | Parameters | Characters | Dense test BPC |
|---|---:|---:|---:|
| Modus_X v1.1.1 | `82,764,964` | `163.84M` | `1.384180` |
| Official Mamba | `81,462,656` | `163.84M` | `1.345780` |
| Official xLSTM | `76,649,664` | `163.84M` | `1.419620` |

The new MemoryFeedback points received only `102.4M` characters, so this is
context rather than a matched endpoint comparison. At present, the result does
not show MemoryFeedback beating any of those `163.84M` endpoints.

## Next gate

First run a checkpoint-preserving annealing diagnosis:

1. continue both the 81M and 99M step-25,000 checkpoints to step 30,000 /
   `122.88M` characters at LR `1e-4`;
2. dense-audit both;
3. retain the 99M branch only if it beats the 81M dense validation result by
   at least `0.005` BPC and is no worse on dense test.

This tests the LR/data explanation without paying for two fresh runs.

Then continue the winning efficiency branch to the existing `163.84M`
matched endpoint:

1. step 30,000 to 35,000 / `143.36M` at LR `7e-5`;
2. step 35,000 to 40,000 / `163.84M` at LR `5e-5`;
3. run the same dense audit.

This reproduces the documented v1.1.1 80M-tail schedule and creates the
closest available parameter-, update-, data-, and evaluation-matched
comparison. Promote MemoryFeedback as the v2 language lead at the 80M tier
only if the final dense test beats `1.384180` or supplies a clearly justified
memory/efficiency tradeoff at competitive BPC.

If 99M fails the annealing gate, freeze that checkpoint and test at most one
fresh scale-corrected 99M candidate: proportional matrix/vector state width
and feedback rank 48, with parameter and runtime matching pre-registered
before training. Do not tune that candidate from final test data.

The pre-counted candidate is:

- embedding width: `744`;
- hidden width: `2232`;
- matrix/vector state width: `744`;
- router hidden width: `64`;
- feedback rank: `48`;
- estimated parameters: `99,823,832`, approximately `0.387%` above the frozen
  `99,438,920` point.

The rank-48 estimate adds `285,696` parameters to the locally counted
rank-32 `99,538,136` tree. The implementation must verify this exact total
before any TPU run.
