# B4 Matrix Budget Analysis

## Scope

This analysis converts the measured B0/B3 parameter census into explicit parameter-budget scenarios.

No architecture, checkpoint, training configuration, or canonical model file is modified.

## Canonical Model Budget

- Total canonical model parameters: **47,437,768**

## Canonical Matrix Targets

### Repeated 512x512 Projection Group

| Parameter path | Shape | Parameters |
|---|---|---:|
| `layers.m_wk` | `[12, 512, 512]` | 3,145,728 |
| `layers.m_wq` | `[12, 512, 512]` | 3,145,728 |
| `layers.m_wv` | `[12, 512, 512]` | 3,145,728 |
| `layers.m_w_read` | `[12, 512, 512]` | 3,145,728 |
| `layers.m_w_out` | `[12, 512, 512]` | 3,145,728 |

- Projection group total: **15,728,640**

### Memory Input Projection

- Path: `layers.m_proj_w`
- Shape: `[12, 512, 1024]`
- Parameters: **6,291,456**

## Component Budget Context

| Component | Parameters | Share of model |
|---|---:|---:|
| matrix_memory | 22,069,284 | 46.5226% |
| vector_recurrence | 18,898,944 | 39.8394% |
| archive_memory_control | 3,164,184 | 6.6702% |
| future_prediction_head | 1,181,440 | 2.4905% |
| output_head | 1,181,440 | 2.4905% |
| router | 399,744 | 0.8427% |
| feedback_bridge | 399,372 | 0.8419% |
| embedding | 131,072 | 0.2763% |
| normalization | 12,288 | 0.0259% |

## PB1-A Candidate Budgets

| Candidate | Saved Parameters | Target Reduction | Projected Model Parameters | Model Reduction |
|---|---:|---:|---:|---:|
| `PB1-A-factorized-512x512-rank-64__PB1-A-memory-input-width-256` | 16,515,072 | 75.00% | 30,922,696 | 34.81% |
| `PB1-A-factorized-512x512-rank-64__PB1-A-memory-input-width-384` | 15,728,640 | 71.43% | 31,709,128 | 33.16% |
| `PB1-A-factorized-512x512-rank-64__PB1-A-memory-input-width-512` | 14,942,208 | 67.86% | 32,495,560 | 31.50% |
| `PB1-A-factorized-512x512-rank-96__PB1-A-memory-input-width-256` | 14,548,992 | 66.07% | 32,888,776 | 30.67% |
| `PB1-A-factorized-512x512-rank-64__PB1-A-memory-input-width-640` | 14,155,776 | 64.29% | 33,281,992 | 29.84% |
| `PB1-A-factorized-512x512-rank-96__PB1-A-memory-input-width-384` | 13,762,560 | 62.50% | 33,675,208 | 29.01% |
| `PB1-A-factorized-512x512-rank-64__PB1-A-memory-input-width-768` | 13,369,344 | 60.71% | 34,068,424 | 28.18% |
| `PB1-A-factorized-512x512-rank-96__PB1-A-memory-input-width-512` | 12,976,128 | 58.93% | 34,461,640 | 27.35% |
| `PB1-A-factorized-512x512-rank-64__PB1-A-memory-input-width-896` | 12,582,912 | 57.14% | 34,854,856 | 26.53% |
| `PB1-A-factorized-512x512-rank-128__PB1-A-memory-input-width-256` | 12,582,912 | 57.14% | 34,854,856 | 26.53% |
| `PB1-A-factorized-512x512-rank-96__PB1-A-memory-input-width-640` | 12,189,696 | 55.36% | 35,248,072 | 25.70% |
| `PB1-A-factorized-512x512-rank-64` | 11,796,480 | 75.00% | 35,641,288 | 24.87% |
| `PB1-A-factorized-512x512-rank-128__PB1-A-memory-input-width-384` | 11,796,480 | 53.57% | 35,641,288 | 24.87% |
| `PB1-A-factorized-512x512-rank-96__PB1-A-memory-input-width-768` | 11,403,264 | 51.79% | 36,034,504 | 24.04% |
| `PB1-A-factorized-512x512-rank-128__PB1-A-memory-input-width-512` | 11,010,048 | 50.00% | 36,427,720 | 23.21% |
| `PB1-A-factorized-512x512-rank-96__PB1-A-memory-input-width-896` | 10,616,832 | 48.21% | 36,820,936 | 22.38% |
| `PB1-A-factorized-512x512-rank-160__PB1-A-memory-input-width-256` | 10,616,832 | 48.21% | 36,820,936 | 22.38% |
| `PB1-A-factorized-512x512-rank-128__PB1-A-memory-input-width-640` | 10,223,616 | 46.43% | 37,214,152 | 21.55% |
| `PB1-A-factorized-512x512-rank-96` | 9,830,400 | 62.50% | 37,607,368 | 20.72% |
| `PB1-A-factorized-512x512-rank-160__PB1-A-memory-input-width-384` | 9,830,400 | 44.64% | 37,607,368 | 20.72% |
| `PB1-A-factorized-512x512-rank-128__PB1-A-memory-input-width-768` | 9,437,184 | 42.86% | 38,000,584 | 19.89% |
| `PB1-A-factorized-512x512-rank-160__PB1-A-memory-input-width-512` | 9,043,968 | 41.07% | 38,393,800 | 19.06% |
| `PB1-A-factorized-512x512-rank-192__PB1-A-memory-input-width-256` | 8,650,752 | 39.29% | 38,787,016 | 18.24% |
| `PB1-A-factorized-512x512-rank-128__PB1-A-memory-input-width-896` | 8,650,752 | 39.29% | 38,787,016 | 18.24% |
| `PB1-A-factorized-512x512-rank-160__PB1-A-memory-input-width-640` | 8,257,536 | 37.50% | 39,180,232 | 17.41% |
| `PB1-A-factorized-512x512-rank-192__PB1-A-memory-input-width-384` | 7,864,320 | 35.71% | 39,573,448 | 16.58% |
| `PB1-A-factorized-512x512-rank-128` | 7,864,320 | 50.00% | 39,573,448 | 16.58% |
| `PB1-A-factorized-512x512-rank-160__PB1-A-memory-input-width-768` | 7,471,104 | 33.93% | 39,966,664 | 15.75% |
| `PB1-A-factorized-512x512-rank-192__PB1-A-memory-input-width-512` | 7,077,888 | 32.14% | 40,359,880 | 14.92% |
| `PB1-A-factorized-512x512-rank-160__PB1-A-memory-input-width-896` | 6,684,672 | 30.36% | 40,753,096 | 14.09% |
| `PB1-A-factorized-512x512-rank-192__PB1-A-memory-input-width-640` | 6,291,456 | 28.57% | 41,146,312 | 13.26% |
| `PB1-A-factorized-512x512-rank-160` | 5,898,240 | 37.50% | 41,539,528 | 12.43% |
| `PB1-A-factorized-512x512-rank-192__PB1-A-memory-input-width-768` | 5,505,024 | 25.00% | 41,932,744 | 11.60% |
| `PB1-A-memory-input-width-256` | 4,718,592 | 75.00% | 42,719,176 | 9.95% |
| `PB1-A-factorized-512x512-rank-192__PB1-A-memory-input-width-896` | 4,718,592 | 21.43% | 42,719,176 | 9.95% |
| `PB1-A-memory-input-width-384` | 3,932,160 | 62.50% | 43,505,608 | 8.29% |
| `PB1-A-factorized-512x512-rank-192` | 3,932,160 | 25.00% | 43,505,608 | 8.29% |
| `PB1-A-memory-input-width-512` | 3,145,728 | 50.00% | 44,292,040 | 6.63% |
| `PB1-A-memory-input-width-640` | 2,359,296 | 37.50% | 45,078,472 | 4.97% |
| `PB1-A-memory-input-width-768` | 1,572,864 | 25.00% | 45,864,904 | 3.32% |
| `PB1-A-memory-input-width-896` | 786,432 | 12.50% | 46,651,336 | 1.66% |

## PB1-B Reallocation Interpretation

For every PB1-A reduction candidate, the saved parameter count is the maximum first-order budget available for a fixed-budget PB1-B reallocation.

The later PB1-B experiment should compare:

1. Canonical Modus_X at the measured canonical parameter budget.
2. The PB1-A reduced matrix-memory candidate.
3. A PB1-B candidate that reallocates approximately the saved parameters elsewhere.

## Interpretation

B4 does not select a winning architecture. It defines the quantitative candidate search space before architecture modifications and training experiments.

