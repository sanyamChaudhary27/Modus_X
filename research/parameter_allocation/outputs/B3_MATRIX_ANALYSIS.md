# B3 Matrix Memory Structural Analysis

## Scope

This analysis inspects the canonical parameter census and isolates the parameters classified as `matrix_memory`.

No model architecture, dimensions, parameter budget, or checkpoint has been modified by this analysis.

## Global Totals

- Total canonical model parameters: **47,437,768**
- Matrix-memory parameters: **22,069,284**
- Matrix-memory share of model: **46.5226%**
- Matrix-memory parameter leaves: **17**

## Matrix Roles

| Role | Parameters | Share of matrix memory |
|---|---:|---:|
| memory_input_projection | 6,291,456 | 28.5077% |
| key_projection | 3,145,728 | 14.2539% |
| memory_output_projection | 3,145,728 | 14.2539% |
| memory_read_projection | 3,145,728 | 14.2539% |
| query_projection | 3,145,728 | 14.2539% |
| value_projection | 3,145,728 | 14.2539% |
| other_matrix_memory_parameter | 49,188 | 0.2229% |

## Shape Groups

| Shape signature | Leaves | Parameters | Share of matrix memory |
|---|---:|---:|---:|
| 12x512x512 | 5 | 15,728,640 | 71.2694% |
| 12x512x1024 | 1 | 6,291,456 | 28.5077% |
| 12x512 | 5 | 30,720 | 0.1392% |
| 12x1x512 | 3 | 18,432 | 0.0835% |
| 12x1 | 3 | 36 | 0.0002% |

## Largest Matrix-Memory Leaves

| Parameter path | Role | Shape | Parameters | Share of matrix memory |
|---|---|---|---:|---:|
| `layers.m_proj_w` | memory_input_projection | `[12, 512, 1024]` | 6,291,456 | 28.5077% |
| `layers.m_w_out` | memory_output_projection | `[12, 512, 512]` | 3,145,728 | 14.2539% |
| `layers.m_w_read` | memory_read_projection | `[12, 512, 512]` | 3,145,728 | 14.2539% |
| `layers.m_wk` | key_projection | `[12, 512, 512]` | 3,145,728 | 14.2539% |
| `layers.m_wq` | query_projection | `[12, 512, 512]` | 3,145,728 | 14.2539% |
| `layers.m_wv` | value_projection | `[12, 512, 512]` | 3,145,728 | 14.2539% |
| `layers.m_b_out` | other_matrix_memory_parameter | `[12, 512]` | 6,144 | 0.0278% |
| `layers.m_b_read` | other_matrix_memory_parameter | `[12, 512]` | 6,144 | 0.0278% |
| `layers.m_ln_b` | other_matrix_memory_parameter | `[12, 512]` | 6,144 | 0.0278% |
| `layers.m_ln_g` | other_matrix_memory_parameter | `[12, 512]` | 6,144 | 0.0278% |
| `layers.m_proj_b` | other_matrix_memory_parameter | `[12, 512]` | 6,144 | 0.0278% |
| `layers.m_w_eta` | other_matrix_memory_parameter | `[12, 1, 512]` | 6,144 | 0.0278% |
| `layers.m_w_ret` | other_matrix_memory_parameter | `[12, 1, 512]` | 6,144 | 0.0278% |
| `layers.m_w_write` | other_matrix_memory_parameter | `[12, 1, 512]` | 6,144 | 0.0278% |
| `layers.m_b_eta` | other_matrix_memory_parameter | `[12, 1]` | 12 | 0.0001% |
| `layers.m_b_ret` | other_matrix_memory_parameter | `[12, 1]` | 12 | 0.0001% |
| `layers.m_b_write` | other_matrix_memory_parameter | `[12, 1]` | 12 | 0.0001% |

## Same-Shape Groups

These groups identify matrix-memory leaves with identical tensor shape signatures. Identical shape does not prove functional redundancy; it only identifies candidates for later controlled sharing, factorization, or width-allocation experiments.

| Shape signature | Leaves | Parameters | Parameter paths |
|---|---:|---:|---|
| 12x512x512 | 5 | 15,728,640 | `layers.m_w_out`, `layers.m_w_read`, `layers.m_wk`, `layers.m_wq`, `layers.m_wv` |
| 12x512 | 5 | 30,720 | `layers.m_b_out`, `layers.m_b_read`, `layers.m_ln_b`, `layers.m_ln_g`, `layers.m_proj_b` |
| 12x1x512 | 3 | 18,432 | `layers.m_w_eta`, `layers.m_w_ret`, `layers.m_w_write` |
| 12x1 | 3 | 36 | `layers.m_b_eta`, `layers.m_b_ret`, `layers.m_b_write` |

## Structural Interpretation

- Matrix memory consumes **46.5226%** of the canonical parameter budget, making it a primary target for controlled parameter-allocation experiments.
- The largest repeated-shape group is `12x512x512`, containing 5 leaves and 15,728,640 parameters.
- These results do not establish that any matrix can be removed. They establish the quantitative search space for PB1-A and PB1-B.

## Next Experimental Gates

1. PB1-A: controlled matrix-memory reduction while holding the remaining architecture fixed.
2. PB1-B: approximately fixed total parameter budget with saved parameters reallocated to alternative architectural components.
3. Validate parameter counts and recurrent-state behavior before any long training run.

