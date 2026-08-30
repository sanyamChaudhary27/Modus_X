# Modus_X v3 retention and coordination research

This branch collects the three parts of the v3 work that currently matter most.
The model benefits from persistent state, both memory bodies do useful work,
and the archive was forgetting useful information too quickly. Changing the
archive retention clock improved that failure mode without adding parameters
or slowing the measured endpoint, and the change improved held-out language
modeling in two independent runs.

The work is divided by research question rather than by the order in which
notebooks happened:

1. `01_streaming_and_coordination` asks whether persistent state and the
   matrix/vector operations actually affect prediction.
2. `02_segment_scale_retention` diagnoses archive decay, changes its clock,
   and tests whether slower decay causes stale interference.
3. `03_language_endpoints` measures whether the mechanism improves dense
   enwik8 validation at a matched 102.4M-character endpoint.

This is a research branch, not a v3 release. Seed 3 remains reserved for final
publication replication. Second-corpus generalization and semantic fact
revision are still open. Negative experiments are indexed separately so they
cannot quietly return as new proposals.

Start with `PROBLEM_OWNERSHIP.md`, then read each track's `README.md`. Exact
numbers live in `RESULT.json`; prose is deliberately kept separate from the
machine-readable record.
