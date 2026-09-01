# Streaming state and coordination

The normal 512-byte evaluation resets Modus_X state at every window. That is a
useful compression protocol, but it hides the behavior the architecture was
built for. We evaluated the frozen 47M MemoryFeedbackArchive checkpoint while
carrying state across contiguous windows. BPC improved, and component
interventions showed that matrix and vector state each contributed.

Naively deleting half the archive or removing the direct matrix/router path
made language modeling worse. Counterfactual read-off, write-off, router, and
feedback interventions also increased next-byte NLL. The model is therefore
not two dormant memory branches joined for decoration. Its components affect
the prediction.

The stronger original story did not survive: vector-only carry was larger than
matrix-only carry in the frozen horizon audit, and controller activation did
not calibrate cleanly with segment regret. We retain a complementary-memory
claim, not a durable-matrix/local-vector claim.

## Contiguous-training record

The detailed README originally shared with the research team is preserved at
`references/Modus_X_Contiguous_Training_Reproducibility_README.md` with SHA-256
`e48d00a0b3d47ca29f9371290e5e1f23c66fa8d1ed24e2398352a9976d2eefd2`.
It records the preregistered design and the first two completed seeds. Seed 1
improved carry-evaluated BPC by `0.019785`, which is the approximately `0.02`
result from that experiment.

The later three-seed decision is preserved separately at
`references/FINAL_REPLICATION_RESULT_2026-08-24.md`. Seeds 2 and 3 did not
replicate the endpoint gain, so the aggregate quality gate failed even though
the increased dependence on carried state replicated tightly in all seeds.
Neither document should be read without the other.

## Seed-1 endpoint coordination closure

At matched 102.4M-character endpoints, the segment-retention candidate reached
`1.372293` full-stream validation BPC versus `1.410296` for canonical
MemoryFeedbackArchive. Instrumented and ordinary forward passes matched
exactly. Removing feedback, matrix read, matrix write, or learned routing made
the candidate worse, and neither matrix-only nor vector-only routing was
sufficient. Feedback activation also became more predictive of its causal
utility, with Pearson/Spearman correlations increasing from `0.2779/0.2615`
to `0.5296/0.5515`.

This supports a coordinated-memory interpretation of the candidate. It does
not establish semantic recall. Router-confidence calibration remained
negative, so controller calibration is still open even though learned routing
itself is necessary.
