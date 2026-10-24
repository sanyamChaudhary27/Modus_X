# Claims and Comparisons

## Recommended Public Lead

> Modus_X 3 couples bounded associative memory to recurrent processing. Its
> dependency-preserving TPU implementation delivers a measured 2.923x update
> speedup. Local K/V context improves all three frozen language endpoints,
> with a median 0.0212-BPC validation gain at 47.46M parameters, adding 96 KiB
> of state per sequence and approximately 0.12% measured update overhead.

This is confident, quantitative positioning. It does not need a claim of
universal superiority to communicate the result.

## Comparison Classes

| Evidence | Fair conclusion | Do not infer |
|---|---|---|
| Three local-K/V endpoints | Candidate improves this paired recipe at the frozen 25k horizon | A precise population effect from three seeds; three fully preregistered runs |
| Two execution continuations | Approximately 2.923x median synchronized update speedup on the tested TPU shape | 2.923x decoding, training-job or universal hardware speedup |
| Retention-only development endpoints | Two recorded recipe-specific improvements | Gains can be added to K/V gains; an exact factorial decomposition |
| Three-seed equal-state CurrentArchive/Transformer test | Strong associative retrieval advantage when KV history is constrained to the specified state budget | Universal Transformer defeat; v3 checkpoint semantic memory capability |
| Historical Mamba comparison | Older Mamba better on recorded compression, older Modus_X far better on specified synthetic retrieval/update task | A v3-vs-Mamba-3 benchmark, equal-state memory comparison or modern Pareto frontier |
| Carry/intervention probes | State and paths influence predictions in the tested setting | Infinite memory, factual consistency, reasoning, or retrained ablation equivalence |

## Wording for the Mamba Question

> Compression and associative recall expose different trade-offs. The
> historical official-Mamba run compresses enwik8 better; the controlled
> length-128 lookup test strongly favors Modus_X, including overwritten
> bindings. We report both, and keep those older task-specific results separate
> from v3's new matched language experiments.

Do not call all Mamba models "way worse at updates." Mamba-3, KDA and Gated
DeltaNet-2 are current related work and have not been run as matched v3 controls.
The older recall comparison has one seed and different parameter counts.

## Novelty

The current/archive-to-vector coordination, exact specification, intervention
evidence, and execution reordering are the project's concrete research objects.
Constant state, scalar/vector recurrence, delta rules, retention gates and short
convolutions are prior art. The archive logit offset is absorbable into a learned
bias and changes optimization/initialization rather than expressivity. The
two-stage implementation is a dependency schedule, not a newly invented
sequence-parallel delta algorithm. A priority claim about being the first
combined design would require a wider full-text related-work search.

## Statistical Reporting

Report every seed. Validation gains: 0.087291, 0.010196, 0.021213. Mean 0.039566;
median 0.021213. Report the sample SD generated in AGGREGATE.json. No token-level
p-value from overlapping dense windows. Seed 2's failed screen and post-screen
follow-up remain explicit. Seed 3 prospectively fixes the endpoint after the
earlier observations. Test does not select configurations.
