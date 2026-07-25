# Mamba + True MemoryFeedback 47M Result

## Decision

The first true-MemoryFeedback transplant is **not promoted** past the 20.48M
character gate. It established a parameter-matched and runtime-efficient
near-tie, but did not improve validation BPC over the narrower pure-Mamba
control and did not preserve the conservative feedback regime observed in the
successful 47M JAX `MemoryFeedbackArchive` experiment.

## Matched Results

| Case | Parameters | Validation BPC | Sparse test BPC | Elapsed seconds |
|---|---:|---:|---:|---:|
| `mamba_528` budget control | 47,157,792 | 1.680624 | 1.650235 | 2,449.89 |
| `mamba_520` width control | 45,844,240 | 1.683463 | 1.658052 | 2,404.11 |
| `mamba_true_feedback_520` | 47,163,211 | 1.683495 | 1.652492 | 2,453.86 |
| `mamba_true_feedback_520_gate0` | 47,163,211 | 1.683190 | 1.659281 | 2,449.20 |

The live candidate was only 0.0115% larger than the budget control. Relative
to the narrow width control, it changed validation BPC by `+0.000032` (a
practical tie), improved sparse test BPC by `-0.005561`, and cost `2.07%` more
wall time. Relative to the budget control, it was `+0.002872` validation BPC
and `+0.002257` sparse test BPC.

The forced-zero implementation control differed from the width control by
only `-0.000273` validation BPC. This is small enough to treat the forward path
as operationally paired, but the original `1e-6` equality check was too strict
for independently trained CUDA runs with additional zero-gradient tensors.

## Mechanism Diagnostics

The live feedback path was active but not conservative:

- mean effective gate: `0.7058`;
- feedback/input RMS ratio: `0.1524`;
- per-layer gates: `0.5859`, `0.7215`, and `0.8099`;
- total recurrent state: `6,890,496` bytes per sequence, slightly below the
  `mamba_528` control's `6,893,568` bytes.

The earlier 47M JAX MemoryFeedbackArchive winner used a mature gate near
`0.06` and a feedback/input ratio near `0.04-0.05`. The transplant therefore
tested strong matrix conditioning, not the same conservative coordination
regime that previously won.

## Capped-Gate Completion

The single permitted cap-only test was completed with:

```text
effective_gate = 0.25 * sigmoid(gate_logits)
```

It passed the early gate and reached 20.48M characters, but failed the final
language promotion gate:

| Metric | Capped candidate | Delta versus control |
|---|---:|---:|
| Validation BPC | 1.688144 | `+0.007520` vs `mamba_528`; `+0.004681` vs `mamba_520` |
| Sparse test BPC | 1.660544 | `+0.010309` vs `mamba_528`; `+0.002492` vs `mamba_520` |
| Early runtime ratio | 1.0298 | `+2.98%` vs fresh width control |
| Effective gate mean | 0.0768 | active and conservative |
| Feedback/input RMS ratio | 0.0300 | active and bounded |

The cap successfully reproduced the intended conservative operating regime,
but performance became worse than both controls. The first layer carried most
of the feedback (`0.2039` effective gate), while layers 15 and 23 largely
closed (`0.0156` and `0.0109`). Therefore excessive average gate amplitude was
not the missing lever. The official-Mamba transplant is frozen after both the
uncapped near-tie and capped negative result.

Memory-specific evaluation remains deferred because the preregistered
language promotion gate did not pass.

## Final Interpretation

The successful 47M JAX MemoryFeedbackArchive result does not transfer merely
by placing the same broad feedback motif in front of official Mamba. Its gain
likely depends on the original CurrentArchive matrix semantics, the original
vector recurrence, or their joint optimization. Do not run another gate-scale,
layer-placement, or bridge-rank search on this official-Mamba branch without a
new mechanistic result.
