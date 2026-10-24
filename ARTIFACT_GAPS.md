# Artifact Scope and Remaining Work

This release includes frozen source/cells, protocols, normalized result and
audit records, checkpoint SHA-256 identifiers, the paper, figure/statistics
generator, reproduction guide, local-test record and integrity manifests.

It does not include the six large25k checkpoint payloads. A public checkpoint
index needs durable links, complete pair/recovery metadata, independent download
and hash verification, and read-only audit from those public artifacts. Compact
evidence ZIPs are not substitutes for weights. Only trusted pickles may be loaded.

The recorded v3 endpoints use JAX0.10.2/Optax0.2.8/highest precision on Kaggle
TPUv5e-8. Exact dependency snapshots are incomplete for some older records.
Current localCPU validation uses a different documented environment.

Older retention-only and later K/V-control endpoint values need source-checkpoint
metadata reconciliation before a direct aggregate full-v2-to-v3 comparison.
The present K/V result remains a within-recipe paired comparison and its
reported endpoint audits are retained.

Public release does not convert these missing artifacts or unrun comparisons
into completed experiments. See docs/RELEASE_REVIEW.md for the detailed review.
