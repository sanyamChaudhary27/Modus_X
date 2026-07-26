# Claims and Evidence

The canonical numeric register is `../CLAIMS.md`; the artifact map is
`../EVIDENCE_INDEX.md`.

## Language claim

At approximately 47M parameters and 102.4M processed enwik8 characters,
MemoryFeedbackArchive improves dense validation and dense test BPC relative to
the matched CurrentArchive control. At the same 102.4M-character endpoint,
scaling MemoryFeedbackArchive to 81.49M parameters produces a further measured
dense-test improvement, while the 99.44M point is effectively saturated.
Matched late annealing promotes the 81.49M checkpoint to a `163.84M`-character
endpoint of `1.375422` dense validation and `1.382445` dense test BPC. This
narrowly improves v1.1.1 with fewer parameters.

This is a single-seed, fixed-budget result. It is not a universal scaling law,
and the 47M schedule differs from the larger-model schedule.

## Baseline boundary

At the matched 163.84M-character endpoint, official Mamba remains better than
MemoryFeedbackArchive on dense enwik8 BPC, while MemoryFeedbackArchive is
better than the tested official xLSTM and narrowly better than v1.1.1. The
single-seed v2 margin over v1.1.1 is small and requires replication.

## Bounded-memory claim

At equal constrained recurrent-state bytes, CurrentArchiveDelta substantially
outperforms the tested Transformer with a truncated KV window on the measured
mixed clean/update protocol. When the same Transformer can retain the full
context, it wins. The supported conclusion is a state-budget crossover, not
universal Transformer inferiority.

## Systems claim

The exact frozen 1B v1-style configuration passes bounded systems and restore
smokes. The 1B MemoryFeedback candidate is parameter-counted but does not yet
own the complete systems ladder or trained-model evidence.
