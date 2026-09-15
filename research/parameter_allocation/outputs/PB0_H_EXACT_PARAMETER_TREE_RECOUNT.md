# PB0-H Exact Parameter-Tree Recount

## Scope

PB0-H instantiates the selected PB0-G architecture through the actual `language.models.make_model` factory and performs an exact recursive recount of every parameter leaf.

No model source, checkpoint, dataset, or training run was modified.

## Selected PB0-G Candidate

- Matrix rank `r`: **512**
- Vector dimension `n`: **512**
- Router hidden width `h`: **32**
- PB0-G feedback rank `f`: **32**
- Actual initializer feedback rank: **32**
- PB0-G predicted parameters: **47,437,768**

## Exact Count Verification

| Measurement | Parameters | Difference |
|---|---:|---:|
| Frozen B0 budget | 47,437,768 | +0 |
| PB0-G prediction | 47,437,768 | +0 |
| PB0-H recursive tree recount | 47,437,768 | +0 |
| `models.count_params` | 47,437,768 | +0 |

Overall exact verification: **True**

## Parameter Tree

- Parameter leaves: **52**
- SHA-256 structural fingerprint: `ccd7c701c46de892f258d3abe73ef4d1adf15730b898fe6e1f7c1be550f2c7c0`

## Component Summary

| Component | Leaves | Parameters |
|---|---:|---:|
| `matrix_memory` | 17 | 22,069,284 |
| `vector_mamba_pathway` | 10 | 18,898,944 |
| `archive_memory_control` | 6 | 3,164,184 |
| `auxiliary` | 4 | 1,181,440 |
| `output_head` | 4 | 1,181,440 |
| `router` | 4 | 399,744 |
| `feedback_bridge` | 4 | 399,372 |
| `embeddings` | 1 | 131,072 |
| `normalization` | 2 | 12,288 |

## Gate Result

PB0-H **PASSED**. The selected PB0-G candidate was instantiated through the actual Modus_X initializer and the recursive parameter-tree recount exactly matches both the PB0-G prediction and the frozen B0 parameter budget.

## Important Interpretation

The exact selected candidate uses the same search dimensions as the canonical baseline. Therefore an exact parameter-budget match alone does not yet demonstrate a new architecture.

Any subsequent architecture change must therefore be evaluated under explicit research constraints rather than assuming that a different configuration exists within the current three-dimensional configuration space.

## Next Gate

Do not train yet. The next step is to decide whether the research objective requires a non-baseline architecture while preserving the fixed parameter budget. If so, the next analysis must explicitly search or design an architectural change that is not expressible solely through `ax_res`, `mamba_state_dim`, and `router_hidden`.
