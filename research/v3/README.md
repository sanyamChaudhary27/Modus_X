# Modus_X v3 retention and coordination research

## Execution optimization branch

`research/v3-two-stage-execution` starts at Sanyam's `b5b3b25`, excluding
the reviewed `e915409` changes. New execution evidence is isolated under
`03_language_endpoints/two_stage_execution/`, using the same six-file evidence
layout. It reports a one-seed 2.923x update-speed screen, not a new architecture
or a default backend change. Raw endpoint archival and independent-seed
continuation remain open. Existing results and frozen source files are intact.

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

The closure run frozen in `CLOSURE_PROTOCOL_2026-08-31.json` completed and
passed both lanes. It replicated the over-retention audit on seed 1 and compared the matched
canonical and segment-retention endpoints under feedback, read, write, and
router interventions. All closure gates passed, the candidate improved matched
streaming validation by `0.038003 BPC`, and test data was not read. The runnable Kaggle TPU cell is
`03_language_endpoints/src/generated_cells/KAGGLE_TPU_V3_CLOSURE.py`; it reads
validation only and performs no training or tuning.

Start with `PROBLEM_OWNERSHIP.md`, then read each track's `README.md`. Exact
numbers live in `RESULT.json`; prose is deliberately kept separate from the
machine-readable record.
