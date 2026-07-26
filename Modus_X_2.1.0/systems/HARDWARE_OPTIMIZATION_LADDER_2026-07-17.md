# Accuracy-Preserving Hardware Optimization Ladder

Date: 2026-07-17

## Boundary

This lane optimizes the frozen `1,058,963,121`-parameter Modus_X computation.
It must not change vocabulary, layers, widths, state dimensions, context,
router equations, recurrence, loss objective, or optimizer semantics.

The selected measured baseline is a `4 data x 2 model` v5e-8 mesh, context
`2,048`, global microbatch `4`, and exact 64-token chunked rematerialization.
It executes an 8,192-token microstep at `539.19 tokens/s`. True four-microstep
gradient accumulation is not yet part of that measured number.

## Exact and numerically equivalent ladder

1. **Canonical accumulated-update baseline.** Run four enwik8 microsteps,
   accumulate gradients in FP32, then apply AdamW once. Record microstep and
   optimizer-update throughput separately and save a sharded checkpoint.
2. **Stable timing baseline.** Measure at least two warmups plus 5-10 full
   accumulated updates. Report median, p95, HBM, and loss; one timed sample is
   not enough for optimization claims.
3. **Move diagnostics off the hot path.** Compute the full-tree gradient norm,
   router statistics, and stream diagnostics only at their scheduled cadence.
4. **Direct integer-label NLL.** Replace materialized FP32 `log_softmax` with
   `logsumexp(logits) - target_logit`. Promote only with one-step loss/gradient
   parity and a measured buffer or throughput improvement.
5. **Rematerialization sweep.** Compare chunk sizes `32`, `64`, `128`, and
   `256`, plus single versus nested checkpoint boundaries. Preserve recurrent
   state across chunks and require output parity before full-model timing.
6. **Vocabulary-parallel output and distributed cross-entropy.** Shard
   `head.w2` and `head.b2` over vocabulary, communicate the 4,608-wide head
   activation rather than replicated 50,257-class logits, and compute global
   FP32 max/sum/target reductions. Require loss/gradient parity and no
   short-trajectory regression. This is the leading expected speedup.
7. **Compile four microsteps inside one scan.** After reference accumulation
   passes, fuse host dispatch and optimizer traversal while preserving one
   AdamW update per four averaged gradients.
8. **Projection packing after HLO evidence.** If optimized HLO confirms many
   small token-scan collectives, concatenate compatible projections and split
   their outputs. Preserve a reversible checkpoint mapping.
9. **Data-sharded optimizer state only if memory blocks the ladder.** ZeRO-1
   moment sharding can save memory but adds resharding complexity. The current
   parameter tree is already correctly model-sharded.
10. **Alternative topologies last.** An `8 data x 1 model` probe removes
    model-axis collectives but may not fit full parameters and FP32 moments.

## First accumulated-update attempt

The first real-data attempt initialized successfully but failed while loading
the microstep executable: XLA requested `5.35 GB` with only `3.91 GB`
reservable. No optimizer update or checkpoint was produced. The revised exact
gate keeps FP32 accumulation but shards the temporary gradient accumulator over
the data axis, uses direct target NLL, and removes full-tree gradient norms from
the hot path. These changes preserve the loss and AdamW equations while
addressing more than the measured `1.44 GB` shortfall.

## Numerical lane kept separate

Only after the exact lane: BF16 gradient accumulation, BF16 vector state,
BF16 matrix state, and BF16 optimizer moments last. Each requires full-model
one-step parity and at least 100 accumulated updates without material loss,
gradient, or router drift.

## Explicitly excluded

Reduced vocabulary, sampled/adaptive softmax, tied embeddings, smaller heads,
lower state dimensions, fewer layers, shorter context, approximate recurrence,
and altered routing are model changes. They do not enter this optimization
lane even if they are faster.
