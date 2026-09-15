# PB0-F Actual Architecture Parameter Formula Calibration

## Scope

PB0-F calibrates parameter accounting against the actual `language/models.py` implementation.

No canonical architecture source file, checkpoint, dataset, or training configuration was modified.

## Canonical Architecture Verification

- B0 measured parameter count: **47,437,768**
- PB0-F actual instantiated count: **47,437,768**
- Difference: **+0**
- Exact canonical match: **True**

## Calibrated Formula

```text
P = 12011080 + (36912 * r) + (30744 * n) + (12300 * h) + (12 * f * r) + (6144 * f)

where:

- `r` = matrix rank / `ax_res`
- `n` = vector recurrent state dimension / `mamba_state_dim`
- `h` = router hidden width
- `f = min(32, r, n)` in the current canonical `MemoryFeedbackArchive` initializer
```

## Critical Architectural Finding

The current `Modus_X_MemoryFeedbackArchive` initializer does not expose feedback rank as an independent `ModelConfig` dimension.

The actual initializer derives feedback rank as:

```text
feedback_rank = min(32, ax_res, mamba_state_dim)
```

Therefore the PB0-E proposed value `feedback_rank = 7` cannot be realized through the current canonical configuration alone. Implementing that value would require an explicit architecture change to the initializer.

## Calibration Samples

| Sample | r | n | h | Actual feedback rank | Parameters | Source |
|---|---:|---:|---:|---:|---:|---|
| `S001` | 512 | 512 | 32 | 32 | 47,437,768 | `canonical_baseline` |
| `S002` | 96 | 512 | 32 | 32 | 31,922,632 | `matrix_rank_sweep` |
| `S003` | 128 | 512 | 32 | 32 | 33,116,104 | `matrix_rank_sweep` |
| `S004` | 168 | 512 | 32 | 32 | 34,607,944 | `matrix_rank_sweep` |
| `S005` | 192 | 512 | 32 | 32 | 35,503,048 | `matrix_rank_sweep` |
| `S006` | 256 | 512 | 32 | 32 | 37,889,992 | `matrix_rank_sweep` |
| `S007` | 384 | 512 | 32 | 32 | 42,663,880 | `matrix_rank_sweep` |
| `S008` | 512 | 512 | 32 | 32 | 47,437,768 | `matrix_rank_sweep` |
| `S009` | 512 | 96 | 32 | 32 | 34,648,264 | `vector_dimension_sweep` |
| `S010` | 512 | 128 | 32 | 32 | 35,632,072 | `vector_dimension_sweep` |
| `S011` | 512 | 192 | 32 | 32 | 37,599,688 | `vector_dimension_sweep` |
| `S012` | 512 | 256 | 32 | 32 | 39,567,304 | `vector_dimension_sweep` |
| `S013` | 512 | 384 | 32 | 32 | 43,502,536 | `vector_dimension_sweep` |
| `S014` | 512 | 512 | 32 | 32 | 47,437,768 | `vector_dimension_sweep` |
| `S015` | 512 | 512 | 8 | 32 | 47,142,568 | `router_width_sweep` |
| `S016` | 512 | 512 | 16 | 32 | 47,240,968 | `router_width_sweep` |
| `S017` | 512 | 512 | 20 | 32 | 47,290,168 | `router_width_sweep` |
| `S018` | 512 | 512 | 32 | 32 | 47,437,768 | `router_width_sweep` |
| `S019` | 512 | 512 | 48 | 32 | 47,634,568 | `router_width_sweep` |
| `S020` | 512 | 512 | 64 | 32 | 47,831,368 | `router_width_sweep` |
| `S021` | 16 | 512 | 32 | 16 | 28,837,576 | `feedback_rank_interaction` |
| `S022` | 24 | 512 | 32 | 24 | 29,185,864 | `feedback_rank_interaction` |
| `S023` | 32 | 512 | 32 | 32 | 29,535,688 | `feedback_rank_interaction` |
| `S024` | 48 | 512 | 32 | 32 | 30,132,424 | `feedback_rank_interaction` |
| `S025` | 512 | 16 | 32 | 16 | 31,992,136 | `feedback_rank_interaction` |
| `S026` | 512 | 24 | 32 | 24 | 32,336,392 | `feedback_rank_interaction` |
| `S027` | 512 | 32 | 32 | 32 | 32,680,648 | `feedback_rank_interaction` |
| `S028` | 512 | 48 | 32 | 32 | 33,172,552 | `feedback_rank_interaction` |
| `S029` | 16 | 16 | 32 | 16 | 13,588,552 | `feedback_rank_interaction` |
| `S030` | 24 | 24 | 32 | 24 | 14,182,792 | `feedback_rank_interaction` |
| `S031` | 32 | 32 | 32 | 32 | 14,778,568 | `feedback_rank_interaction` |
| `S032` | 64 | 64 | 32 | 32 | 16,955,848 | `feedback_rank_interaction` |

## Held-Out Formula Validation

| Sample | r | n | h | f | Actual | Predicted | Error | Exact |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `V001` | 80 | 144 | 12 | 32 | 19,766,104 | 19,766,104 | +0 | True |
| `V002` | 120 | 224 | 28 | 32 | 23,914,264 | 23,914,264 | +0 | True |
| `V003` | 168 | 384 | 20 | 32 | 30,525,112 | 30,525,112 | +0 | True |
| `V004` | 200 | 300 | 24 | 32 | 29,185,288 | 29,185,288 | +0 | True |
| `V005` | 288 | 160 | 40 | 32 | 28,359,976 | 28,359,976 | +0 | True |
| `V006` | 448 | 448 | 56 | 32 | 43,378,408 | 43,378,408 | +0 | True |
| `V007` | 512 | 320 | 18 | 32 | 41,362,720 | 41,362,720 | +0 | True |

## Calibration Gate

PB0-F **PASSED**. The calibrated formula reproduces both the canonical architecture and all held-out validation configurations exactly.

## Next Gate

Do not modify `language/models.py` and do not train Candidate A until the next search uses the PB0-F calibrated formula.

The next step is **PB0-G: calibrated Candidate A dimension search**, with two separate modes:

1. Search dimensions that are realizable without modifying the canonical architecture.
2. Separately identify which PB0-D targets require explicit architecture changes, including independently configurable feedback rank.
