# Problem ownership

Owner: Sanyam Chaudhary
Branch: `research/sanyam-v3-retention-coordination`

This branch owns three rows from the ten-problem research review.

## Problem 1: language compression

The immediate objective is to establish whether the promoted memory mechanism
improves held-out compression, not merely memory diagnostics. Two matched
102.4M-character endpoints are complete. The next closure work is a second
corpus and, only when preparing the paper, seed-3 endpoint replication.

## Problem 2: matrix-vector coordination

The final frozen v3 endpoint audit passed exact parity and every coordination
gate. Read, write, router, and feedback operations were causal, neither path
was sufficient alone, and feedback calibration improved materially. This
closes the bounded coordination row. It does not show that the controller is
optimally calibrated or that matrix state is always the longer-lived
component; router-confidence calibration remains a separate open problem.

## Problem 6: archive interference

Source traces identified per-token global archive decay as a retention
bottleneck. Segment-scale retention fixed the measured trace, passed three
short screens, improved two language endpoints, and passed frozen natural
conflict audits on seeds 1 and 2. The final two-lane closure passed without a
test read. The bounded byte-level gate is closed. Semantic updates and
universal stale-memory resistance remain open.

## Working rule

Do not combine new mechanisms or tune against test data. Every new run must
name one changed variable, freeze a decision threshold before execution, and
preserve failed results in `negative_results`.
