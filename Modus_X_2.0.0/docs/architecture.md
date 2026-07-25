# Modus_X 2.0.0 Architecture

## Design objective

Modus_X 2.0.0 asks whether a bounded associative matrix can supply durable,
content-addressed information to a strong recurrent computation without
growing a token-by-token inference cache.

## MemoryFeedbackArchive

MemoryFeedbackArchive is the language-model lead. Each layer maintains current
and archive matrix state. Retrieved matrix context is compressed through a
low-rank path and conservatively gated into the vector-stream input:

```text
token representation
    -> bounded matrix update and retrieval
    -> low-rank feedback projection
    -> learned bounded gate
    -> vector recurrence
    -> residual output
```

The important coordination change is causal: matrix retrieval modifies what
the recurrent stream reasons over. It is not merely added as a second large
output expert.

## CurrentArchiveDelta

CurrentArchiveDelta is the controlled-memory lead. It separates rapidly
updated current state from durable archive state and uses version-aware
addressing and disciplined delta writes. The design is evaluated on clean,
updated, conflicting, and distractor-heavy bindings.

## Bounded state

For fixed matrix and vector dimensions, recurrent state does not grow with
sequence length. Weight storage, activations, computation, batching, and
hardware efficiency remain separate costs.

## Research boundary

The two promoted variants demonstrate complementary strengths. This release
does not claim that their advantages have been merged into one final model.
Rejected attention-to-write, adaptive-preconditioning, erase/write, latest
shadow, and official-Mamba insertion variants remain documented negative
results.

