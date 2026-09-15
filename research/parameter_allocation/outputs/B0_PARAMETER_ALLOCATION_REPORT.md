# B0 Parameter Allocation Census

## Status

**B0 PARAMETER ACCOUNTING PASS**

## Canonical baseline

- Model: `Modus_X_MemoryFeedbackArchive_DeepSupervision`
- Constructor key: `Modus_X_MemoryFeedbackArchive`
- Expected parameters: `47,437,768`
- Actual parameters: `47,437,768`
- Parameter storage bytes: `189,751,072`

## Frozen architecture configuration

| Field | Value |
|---|---:|
| `vocab_size` | `256` |
| `embed_dim` | `512` |
| `hidden_dim` | `1536` |
| `ax_res` | `512` |
| `n_layers` | `12` |
| `n_heads_attn` | `8` |
| `seq_len` | `512` |
| `mamba_state_dim` | `512` |
| `vector_router` | `False` |
| `router_hidden` | `32` |

## Parameter allocation

| Component | Parameters | Share of total | Leaves | Parameter bytes |
|---|---:|---:|---:|---:|
| `matrix_memory` | 22,069,284 | 46.5226% | 17 | 88,277,136 |
| `vector_recurrence` | 18,898,944 | 39.8394% | 10 | 75,595,776 |
| `archive_memory_control` | 3,164,184 | 6.6702% | 6 | 12,656,736 |
| `future_prediction_head` | 1,181,440 | 2.4905% | 4 | 4,725,760 |
| `output_head` | 1,181,440 | 2.4905% | 4 | 4,725,760 |
| `router` | 399,744 | 0.8427% | 4 | 1,598,976 |
| `feedback_bridge` | 399,372 | 0.8419% | 4 | 1,597,488 |
| `embedding` | 131,072 | 0.2763% | 1 | 524,288 |
| `normalization` | 12,288 | 0.0259% | 2 | 49,152 |

## Accounting invariants

- Sum of parameter census leaves: `47,437,768`
- Canonical expected total: `47,437,768`
- Number of parameter leaves: `52`
- Unclassified parameters: `0`

## Interpretation boundary

This B0 report is a static allocation census. It does not claim that a component with more parameters is inefficient or that reducing it will improve BPC. BPC conclusions require subsequent matched-budget training experiments.
