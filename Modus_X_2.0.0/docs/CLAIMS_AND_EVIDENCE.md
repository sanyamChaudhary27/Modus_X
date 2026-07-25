# Claims and Evidence

The canonical numeric register is `../CLAIMS.md`; the artifact map is
`../EVIDENCE_INDEX.md`.

## Language claim

At approximately 47M parameters and 102.4M processed enwik8 characters,
MemoryFeedbackArchive improves dense validation and dense test BPC relative to
the matched CurrentArchive control. At the same 102.4M-character endpoint,
scaling MemoryFeedbackArchive to 81.49M parameters produces a further measured
dense-test improvement, while the 99.44M point is effectively saturated.

This is a single-seed, fixed-budget result. It is not a universal scaling law,
and the 47M schedule differs from the larger-model schedule.

## Baseline boundary

At the historical 163.84M-character endpoint, official Mamba is better than
the tested Modus_X v1.1.1 checkpoint on dense enwik8 BPC, while the tested
Modus_X checkpoint is better than official xLSTM. The new v2 scaling points
have not yet reached that matched endpoint.

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

