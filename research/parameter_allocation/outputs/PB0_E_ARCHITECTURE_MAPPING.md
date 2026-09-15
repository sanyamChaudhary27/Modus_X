# PB0-E Candidate Architecture Mapping

## Scope

PB0-E translates the selected PB0 Candidate A integer dimensions into a concrete parameterization plan for the existing Modus_X architecture.

No canonical model source file is modified by this analysis.

## Selected Candidate

- Candidate ID: **A**
- Frozen target budget: **47,437,768**
- PB0 mapping reported total: **47,437,888**
- PB0-E computed parameter-plan total: **30,317,528**
- Computed difference from frozen target: **-17,120,240**

## Concrete Dimensions

| Dimension | Value |
|---|---:|
| Main width `d` | 512 |
| Matrix rank `r` | 168 |
| Vector expansion `n` | 384 |
| Router hidden width | 20 |
| Feedback bottleneck rank | 7 |
| Layers | 12 |
| Vocabulary size | 256 |

## Component Parameter Plan

| Component | Parameters | Share |
|---|---:|---:|
| vector_mamba_pathway | 14,963,712 | 49.3566% |
| matrix_memory | 11,489,220 | 37.8963% |
| auxiliary | 1,179,648 | 3.8910% |
| output_head | 1,179,648 | 3.8910% |
| archive_memory_control | 1,046,520 | 3.4519% |
| router | 252,144 | 0.8317% |
| embeddings | 131,072 | 0.4323% |
| feedback_bridge | 63,276 | 0.2087% |
| normalization | 12,288 | 0.0405% |

## Tensor Mapping

| Component | Path | Shape | Formula | Per Layer | Total |
|---|---|---|---|---:|---:|
| matrix_memory | `layers.m_wk` | `[168, 512]` | `r * d` | 86,016 | 1,032,192 |
| matrix_memory | `layers.m_wq` | `[168, 512]` | `r * d` | 86,016 | 1,032,192 |
| matrix_memory | `layers.m_wv` | `[168, 512]` | `r * d` | 86,016 | 1,032,192 |
| matrix_memory | `layers.m_w_read` | `[168, 512]` | `r * d` | 86,016 | 1,032,192 |
| matrix_memory | `layers.m_w_out` | `[512, 512]` | `d * d` | 262,144 | 3,145,728 |
| matrix_memory | `layers.m_proj_w` | `[512, 680]` | `d * (d + r)` | 348,160 | 4,177,920 |
| matrix_memory | `layers.m_b_read` | `[168]` | `r` | 168 | 2,016 |
| matrix_memory | `layers.m_b_out` | `[512]` | `d` | 512 | 6,144 |
| matrix_memory | `layers.m_proj_b` | `[512]` | `d` | 512 | 6,144 |
| matrix_memory | `layers.m_ln_g` | `[168]` | `r` | 168 | 2,016 |
| matrix_memory | `layers.m_ln_b` | `[168]` | `r` | 168 | 2,016 |
| matrix_memory | `layers.m_w_eta` | `[1, 512]` | `1 * d` | 512 | 6,144 |
| matrix_memory | `layers.m_b_eta` | `[1]` | `1` | 1 | 12 |
| matrix_memory | `layers.m_w_write` | `[1, 512]` | `1 * d` | 512 | 6,144 |
| matrix_memory | `layers.m_b_write` | `[1]` | `1` | 1 | 12 |
| matrix_memory | `layers.m_w_ret` | `[1, 512]` | `1 * d` | 512 | 6,144 |
| matrix_memory | `layers.m_b_ret` | `[1]` | `1` | 1 | 12 |
| archive_memory_control | `layers.m_w_archive_write` | `[1, 512]` | `1 * d` | 512 | 6,144 |
| archive_memory_control | `layers.m_b_archive_write` | `[1]` | `1` | 1 | 12 |
| archive_memory_control | `layers.m_w_archive_ret` | `[1, 512]` | `1 * d` | 512 | 6,144 |
| archive_memory_control | `layers.m_b_archive_ret` | `[1]` | `1` | 1 | 12 |
| archive_memory_control | `layers.m_w_archive_mix` | `[168, 512]` | `r * d` | 86,016 | 1,032,192 |
| archive_memory_control | `layers.m_b_archive_mix` | `[168]` | `r` | 168 | 2,016 |
| vector_mamba_pathway | `layers.s_wu` | `[384, 512]` | `n * d` | 196,608 | 2,359,296 |
| vector_mamba_pathway | `layers.s_w_delta` | `[384, 512]` | `n * d` | 196,608 | 2,359,296 |
| vector_mamba_pathway | `layers.s_b_delta` | `[384]` | `n` | 384 | 4,608 |
| vector_mamba_pathway | `layers.s_w_ret` | `[384, 512]` | `n * d` | 196,608 | 2,359,296 |
| vector_mamba_pathway | `layers.s_b_ret` | `[384]` | `n` | 384 | 4,608 |
| vector_mamba_pathway | `layers.s_w_c` | `[384, 512]` | `n * d` | 196,608 | 2,359,296 |
| vector_mamba_pathway | `layers.s_w_gate` | `[512, 512]` | `d * d` | 262,144 | 3,145,728 |
| vector_mamba_pathway | `layers.s_b_gate` | `[512]` | `d` | 512 | 6,144 |
| vector_mamba_pathway | `layers.s_proj_w` | `[512, 384]` | `d * n` | 196,608 | 2,359,296 |
| vector_mamba_pathway | `layers.s_proj_b` | `[512]` | `d` | 512 | 6,144 |
| feedback_bridge | `layers.s_w_memory_feedback` | `[1, 512]` | `1 * d` | 512 | 6,144 |
| feedback_bridge | `layers.s_b_memory_feedback` | `[1]` | `1` | 1 | 12 |
| feedback_bridge | `layers.s_memory_down` | `[7, 168]` | `feedback_rank * r` | 1,176 | 14,112 |
| feedback_bridge | `layers.s_memory_up` | `[512, 7]` | `d * feedback_rank` | 3,584 | 43,008 |
| router | `layers.r_w` | `[20, 512]` | `router_hidden * d` | 10,240 | 122,880 |
| router | `layers.r_b` | `[20]` | `router_hidden` | 20 | 240 |
| router | `layers.r_proj` | `[512, 20]` | `router_outputs * router_hidden` | 10,240 | 122,880 |
| router | `layers.r_proj_b` | `[512]` | `router_outputs` | 512 | 6,144 |
| normalization | `layers.pre_g` | `[512]` | `d` | 512 | 6,144 |
| normalization | `layers.pre_b` | `[512]` | `d` | 512 | 6,144 |
| embeddings | `embed` | `[256, 512]` | `vocab_size * d` | 131,072 | 131,072 |
| output_head | `head.w1` | `[1536, 512]` | `(3 * d) * d` | 786,432 | 786,432 |
| output_head | `head.w2` | `[256, 1536]` | `vocab_size * (3 * d)` | 393,216 | 393,216 |
| auxiliary | `future_heads.w1` | `[1, 1536, 512]` | `1 * (3 * d) * d` | 786,432 | 786,432 |
| auxiliary | `future_heads.w2` | `[1, 256, 1536]` | `1 * vocab_size * (3 * d)` | 393,216 | 393,216 |

## Architectural Mapping Decision

The selected PB0 dimensions are mapped onto the existing MemoryFeedbackArchive architecture as follows:

- `r` changes matrix-memory projection dimensions and recurrent matrix state dimensions.
- `n` changes the vector recurrent state and associated input/output projections.
- `router_hidden` changes only the router hidden representation.
- `feedback_rank` changes the matrix-to-vector bottleneck.
- `d = 512`, layer count, vocabulary, embedding, and output interfaces remain frozen.

## Important Gate

This parameter plan is an analytical mapping, not yet proof that the canonical initializer and forward implementation instantiate exactly this tree.

The next gate is to create an isolated experimental architecture configuration, instantiate it through the actual JAX model initializer, and perform an exact parameter-tree recount.
