# Three-seed contiguous-training replication

Date: 2026-08-24  
Selection split: enwik8 validation only  
Test split read: no  
Frozen aggregate decision: **FAIL**

## Primary result

| Seed | Carry-eval gain | Reset-eval degradation | Carry interaction | Runtime ratio | Individual gate |
|---:|---:|---:|---:|---:|---|
| 1 | +0.019785 | +0.005496 | +0.025282 | 0.997269 | pass |
| 2 | -0.002478 | +0.029953 | +0.027476 | 0.996918 | fail |
| 3 | -0.002488 | +0.026549 | +0.024061 | 0.996444 | fail |
| **Mean** | **+0.004940** | **+0.020666** | **+0.025606** | **0.996877** | **1/3 pass** |

The preregistered aggregate required at least two individual seed wins, mean
carry-evaluation gain of at least `0.010` BPC, and mean reset-evaluation
degradation no greater than `0.010` BPC. All three conditions failed.

## What replicated

Persistent-state training increased the value of carrying state at evaluation
in every seed. The interaction was tightly grouped between `0.024061` and
`0.027476` BPC. Candidate carry benefits were `0.044361`, `0.050767`, and
`0.049016` BPC, approximately twice the corresponding reset-control carry
benefits.

This establishes that detached carry-training reliably changes the model into
one that depends more strongly on cross-segment state.

## What did not replicate

That increased dependence did not reliably improve final carry-evaluated BPC.
Seed 1 improved by `0.019785` BPC, while seeds 2 and 3 were worse than their
matched controls by `0.002478` and `0.002488` BPC. Carry-trained models also
became materially worse when evaluated without their history in seeds 2 and
3.

The bounded conclusion is therefore: the tested training protocol teaches
state use, but does not provide a stable language-quality improvement.

## Decision

Freeze this exact detached full-state carry-training protocol without tuning.
Proceed with the preregistered factorial attribution audit on the frozen
checkpoints to determine whether current matrix, archive matrix, vector state,
or their interaction caused the learned dependence. No test data is needed.
