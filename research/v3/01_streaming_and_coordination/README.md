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
