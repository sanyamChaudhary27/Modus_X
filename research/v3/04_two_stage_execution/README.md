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

Contents follow the project evidence format: PROTOCOL.json, RESULT.json,
REPRODUCE.md, CLAIM_BOUNDARY.md, and src/. Historical source copies in adjacent
directories and all released versions remain untouched.

Lineage: based on b5b3b250c0ca7b0fda701926b84e907b57586bd4, not the subsequent
e915409 team-fix commit. The review of that commit remains in the original
checkout and is not a substitute for source provenance here.
