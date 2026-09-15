# PB0-C Parameter Budget Audit

## Research Question

Given a fixed approximately 47M parameter budget, what allocation between matrix memory, vector/Mamba pathway, router, feedback bridge, embeddings, and output head gives the lowest validation/test BPC?

This audit establishes the baseline allocation before any parameter reallocation experiment.

## Baseline Verification

| Check | Result |
|---|---|
| Expected baseline parameter count | 47,437,768 |
| Baseline count verification | PASS |
| Seed topology verification | PASS |

## Checkpoints

| Seed | Checkpoint | Parameters | Parameter tensors | Step |
|---|---|---|---|---|
| seed1 | E:\seed1\checkpoint.pkl | 47,437,768 | 52 | 25000 |
| seed2 | E:\seed2\checkpoint.pkl | 47,437,768 | 52 | 25000 |

## Baseline Parameter Allocation

| Seed | Component | Parameters | % of total | Tensors | Layer params | Global params |
|---|---|---|---|---|---|---|
| seed1 | Matrix Memory | 25,233,468 | 53.1928% | 23 | 0 | 25,233,468 |
| seed1 | Vector / Mamba Pathway | 18,898,944 | 39.8394% | 10 | 0 | 18,898,944 |
| seed1 | Router | 399,744 | 0.8427% | 4 | 0 | 399,744 |
| seed1 | Feedback Bridge | 6,156 | 0.0130% | 2 | 0 | 6,156 |
| seed1 | Embeddings | 131,072 | 0.2763% | 1 | 0 | 131,072 |
| seed1 | Output Head | 1,181,440 | 2.4905% | 4 | 0 | 1,181,440 |
| seed1 | Normalization | 12,288 | 0.0259% | 2 | 0 | 12,288 |
| seed1 | Auxiliary | 1,181,440 | 2.4905% | 4 | 0 | 1,181,440 |
| seed1 | Other | 393,216 | 0.8289% | 2 | 0 | 393,216 |
| seed2 | Matrix Memory | 25,233,468 | 53.1928% | 23 | 0 | 25,233,468 |
| seed2 | Vector / Mamba Pathway | 18,898,944 | 39.8394% | 10 | 0 | 18,898,944 |
| seed2 | Router | 399,744 | 0.8427% | 4 | 0 | 399,744 |
| seed2 | Feedback Bridge | 6,156 | 0.0130% | 2 | 0 | 6,156 |
| seed2 | Embeddings | 131,072 | 0.2763% | 1 | 0 | 131,072 |
| seed2 | Output Head | 1,181,440 | 2.4905% | 4 | 0 | 1,181,440 |
| seed2 | Normalization | 12,288 | 0.0259% | 2 | 0 | 12,288 |
| seed2 | Auxiliary | 1,181,440 | 2.4905% | 4 | 0 | 1,181,440 |
| seed2 | Other | 393,216 | 0.8289% | 2 | 0 | 393,216 |

## Cross-Seed Component Consistency

| Component | Seed1 | Seed2 | Difference | Relative difference | Consistent |
|---|---|---|---|---|---|
| Matrix Memory | 25,233,468 | 25,233,468 | 0 | 0.000000% | PASS |
| Vector / Mamba Pathway | 18,898,944 | 18,898,944 | 0 | 0.000000% | PASS |
| Router | 399,744 | 399,744 | 0 | 0.000000% | PASS |
| Feedback Bridge | 6,156 | 6,156 | 0 | 0.000000% | PASS |
| Embeddings | 131,072 | 131,072 | 0 | 0.000000% | PASS |
| Output Head | 1,181,440 | 1,181,440 | 0 | 0.000000% | PASS |
| Normalization | 12,288 | 12,288 | 0 | 0.000000% | PASS |
| Auxiliary | 1,181,440 | 1,181,440 | 0 | 0.000000% | PASS |
| Other | 393,216 | 393,216 | 0 | 0.000000% | PASS |

## Allocation Vector

The baseline allocation vector is represented as:

`[matrix_memory, vector_mamba, router, feedback_bridge, embeddings, output_head, normalization, auxiliary, other]`

### SEED1

`25,233,468, 18,898,944, 399,744, 6,156, 131,072, 1,181,440, 12,288, 1,181,440, 393,216`

### SEED2

`25,233,468, 18,898,944, 399,744, 6,156, 131,072, 1,181,440, 12,288, 1,181,440, 393,216`

## Interpretation Boundary

PB0-C measures where parameters are allocated. It does not determine which component should receive more or fewer parameters.

That causal question requires the subsequent fixed-budget allocation matrix and empirical validation/test BPC experiments.

## Next Experimental Gate

PB0-D will use this exact baseline allocation to construct architecturally valid candidates under the fixed parameter budget of approximately 47,437,768 parameters.
