# Local K/V filtering in MemoryFeedback

Three paired 47M enwik8 endpoints at 25,000 updates / 102.4M processed
characters favor two-lag causal K/V filtering over the same retention-scale512,
two-stage control. This is a research candidate, not a release/default change.

| Seed | Control dense validation | Local K/V | Gain |
| --- | ---: | ---: | ---: |
| 1 | 1.482739 | 1.395448 | 0.087291 |
| 2 | 1.442483 | 1.432288 | 0.010196 |
| 3 | 1.427985 | 1.406772 | 0.021213 |

Mean gain 0.039566 BPC; median 0.021213. Unequal seed effects must remain
visible. Measured median-update overhead is 0.10-0.13%, not whole-job overhead.
Candidate adds 24,576 parameters and 98,304 bytes of fp32 inference state per
sequence. Fixed-size lag histories preserve bounded inference state.

Keys and values receive two learned elementwise lag terms before key
normalization and value tanh. Lag weights initialize to zero. Query, gates,
router, objectives and optimizer are unchanged. Short causal filters are known
mechanisms; this is evidence for their integration, not a claim of invention.

Seeds 2 and 3 lost their 5k screens but won at 25k. Seed 2 continuation was
declared after seed 1's endpoint; seed 3 had a fixed endpoint before execution.
All three checkpoint audits reportedly reproduced validation within 7e-9 BPC
and passed tested causality/handoff and serial/chunk32 comparisons.

RESULT.json contains normalized user-reported numbers and checkpoint hashes.
Raw checkpoints were not retrieved by the packaging agent and are not in Git.
Read CLAIM_BOUNDARY.md before citing. See REPRODUCE.md for runnable cells.
No adjacent experiment or released source is changed by this directory.
