# Two-stage execution of v3 MemoryFeedback

This is an opt-in execution optimization, not a new memory architecture.
The matrix recurrence is independent of vector state within each layer.
Compute its retrieved contexts first, batch feedback/vector projections, then
run the vector recurrence. Both recurrences remain causal. Chunk32 uses
recomputation for backward memory savings. Parameters, objectives and recurrent
state shapes are unchanged; floating-point training trajectories need not match.

One seed-2 500-update real-enwik8 continuation passed its frozen screen:
2.923x median update speedup, dense validation 1.432001 versus 1.431994.
Read RESULT.json for provenance limitations. No default backend was changed.

Independent seed-1 replication also passed: 2,000 paired updates, 2.923x
median update speedup, dense validation 1.421511 versus 1.421549 (candidate
minus control +0.00003736 BPC). Both backend checkpoint-reload checks reproduced
the next update exactly. See REPLICATION_RESULT.json for reported source and
endpoint hashes. Raw artifact retrieval remains separate from this result.

Two-stage chunk32 is now recommended for the tested 47M/highest-precision TPU
configuration; the code remains opt-in with canonical as reference/fallback.
This evidence does not generalize the recommendation to other configurations.

Contents follow the project evidence format: PROTOCOL.json, RESULT.json,
REPRODUCE.md, CLAIM_BOUNDARY.md, and src/. Historical source copies in adjacent
directories and all released versions remain untouched.

Lineage: based on b5b3b250c0ca7b0fda701926b84e907b57586bd4, not the subsequent
e915409 team-fix commit. The review of that commit remains in the original
checkout and is not a substitute for source provenance here.
