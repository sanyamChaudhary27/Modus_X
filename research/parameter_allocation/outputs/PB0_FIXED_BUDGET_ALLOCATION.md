# PB0-D Fixed-Budget Parameter Allocation Matrix

## Objective

Given the exact fixed parameter budget established by PB0-C, construct controlled parameter-allocation hypotheses without changing the total parameter count.

PB0-D does not train any candidate, does not evaluate validation BPC, and does not read the test split.

## Frozen Budget

- Exact baseline: `47,437,768` parameters
- Reallocatable budget: `44,538,312` parameters
- Frozen non-reallocation components: `2,899,456` parameters

## Frozen Architecture Constraints

| Field | Value |
|---|---:|
| vocabulary_size | 256 |
| layer_count | 12 |
| embedding_width | 512 |
| hidden_width | 1,536 |
| matrix_vector_state_width | 512 |

## Dataset Guardrails

- Training range: `0:90,000,000`
- Validation range: `90,000,000:95,000,000`
- Test range: `95,000,000:100,000,000`
- Test status: **unread until a candidate winner is frozen**

## PB0-C Baseline Allocation

| Component | Parameters | Percent of Total |
|---|---:|---:|
| Matrix Memory | 25,233,468 | 53.193% |
| Vector / Mamba Pathway | 18,898,944 | 39.839% |
| Router | 399,744 | 0.843% |
| Feedback Bridge | 6,156 | 0.013% |
| Embeddings | 131,072 | 0.276% |
| Output Head | 1,181,440 | 2.491% |
| Normalization | 12,288 | 0.026% |
| Auxiliary | 1,181,440 | 2.491% |
| Other | 393,216 | 0.829% |

## Candidate Matrix

| Candidate | Status | Total | Matrix Δ | Vector Δ | Router Δ | Feedback Δ |
|---|---|---:|---:|---:|---:|---:|
| BASELINE (Baseline Allocation) | control | 47,437,768 | +0 | +0 | +0 | +0 |
| A (Matrix-to-Vector Rebalance) | selected_hypothesis | 47,437,768 | -1,785,068 | +1,717,309 | +64,588 | +3,171 |
| B (Matrix Capacity Reduction) | unselected_candidate | 47,437,768 | -4,018,684 | +3,965,854 | +51,096 | +1,734 |
| C (Feedback Expansion) | unselected_candidate | 47,437,768 | -914,255 | +849,083 | +46,347 | +18,825 |
| D (Router Capacity Expansion) | unselected_candidate | 47,437,768 | -1,216,930 | +413,959 | +801,726 | +1,245 |

## Candidate A

**Candidate A is the selected first controlled allocation hypothesis. It is not trained by PB0-D.**

First controlled hypothesis: modestly reduce the dominant matrix-memory allocation and reallocate capacity primarily into the vector/Mamba pathway, with a smaller targeted increase to the feedback bridge and router. The objective is to test whether the current 53.193% matrix allocation is over-provisioned relative to vector computation and matrix-to-vector information transfer.

| Component | Baseline | Candidate A | Difference |
|---|---:|---:|---:|
| Matrix Memory | 25,233,468 | 23,448,400 | -1,785,068 |
| Vector / Mamba Pathway | 18,898,944 | 20,616,253 | +1,717,309 |
| Router | 399,744 | 464,332 | +64,588 |
| Feedback Bridge | 6,156 | 9,327 | +3,171 |
| Embeddings | 131,072 | 131,072 | +0 |
| Output Head | 1,181,440 | 1,181,440 | +0 |
| Normalization | 12,288 | 12,288 | +0 |
| Auxiliary | 1,181,440 | 1,181,440 | +0 |
| Other | 393,216 | 393,216 | +0 |

## Next Gate

1. Freeze Candidate A allocation.
2. Map the allocation to actual `model.py` dimensions.
3. Instantiate the modified architecture.
4. Perform an exact parameter recount.
5. Reject the mapping if it cannot preserve the fixed budget.
6. Only after the recount passes, train Candidate A.
7. Evaluate validation BPC only.
8. Keep the test split unread until the candidate winner is frozen.

## Explicit Limitation

The allocations in PB0-D are parameter-budget targets. They are not yet evidence that every target is realizable by changing integer model dimensions. Architectural mapping and exact recount are mandatory before training.

