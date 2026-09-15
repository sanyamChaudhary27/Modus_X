# Modus_X B2 Parameter Allocation Analysis

## Purpose

This report aggregates the parameter leaves produced by the B0 parameter census into architecture components.

The B0 `component` classification is treated as authoritative when that column is present. B2 does not independently reclassify those parameter paths.

## Inputs

- Project root: `E:\Modus_X`
- B0 JSON: `E:\Modus_X\research\parameter_allocation\outputs\b0_parameter_census.json`
- B0 CSV: `E:\Modus_X\research\parameter_allocation\outputs\b0_parameter_leaves.csv`

## Detected CSV Columns

- Parameter path: `path`
- Parameter count: `elements`
- Canonical component: `component`

Classification policy: **Canonical B0 `component` column**

## Parameter Accounting

- Expected baseline total: not reported by B0 JSON
- Measured CSV total: `47,437,768`

## Component Allocation

| Component | Parameters | Share | Parameter Leaves |
|---|---:|---:|---:|
| matrix_memory | 22,069,284 | 46.5226% | 17 |
| vector_recurrence | 18,898,944 | 39.8394% | 10 |
| archive_memory_control | 3,164,184 | 6.6702% | 6 |
| future_prediction_head | 1,181,440 | 2.4905% | 4 |
| output_head | 1,181,440 | 2.4905% | 4 |
| router | 399,744 | 0.8427% | 4 |
| feedback_bridge | 399,372 | 0.8419% | 4 |
| embedding | 131,072 | 0.2763% | 1 |
| normalization | 12,288 | 0.0259% | 2 |

## Interpretation

The largest components should be investigated at matrix level before any architecture modification is made. A large parameter allocation does not by itself demonstrate redundancy or justify removing a matrix.

The next analysis stage is B3 matrix allocation analysis, which should inspect individual parameter matrices, their shapes, per-layer cost, and architectural role.
