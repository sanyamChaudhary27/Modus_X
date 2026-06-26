# Results Ledger

## Language Modeling

| Model | Params | Updates | Characters | Dense Val BPC | Dense Test BPC |
|---|---:|---:|---:|---:|---:|
| Modus_X | `82,764,964` | `40,000` | `163.84M` | `1.378681` | `1.384180` |
| Official Mamba | `81,462,656` | `40,000` | `163.84M` | `1.350538` | `1.345780` |
| Official xLSTM | `76,649,664` | `40,000` | `163.84M` | `1.435132` | `1.419620` |

## Associative Memory

| Model | Params | Seed | Overwrite | Length 128 Accuracy |
|---|---:|---:|---:|---:|
| Modus_X VectorLeanPM | `145,674` | `17` | `0%` | `97.325%` |
| Official Mamba | `162,560` | `17` | `0%` | `2.850%` |
| Modus_X VectorLeanPM | `145,674` | `17` | `50%` | `88.850%` |
| Official Mamba | `162,560` | `17` | `50%` | `3.425%` |

These are current verified values, but release claims remain gated on raw
artifact promotion and multi-seed confirmation.

### Router/Component Ablation, Three Seeds, Length 2048

| Condition | Model | Params | Accuracy |
|---|---|---:|---:|
| No overwrite | ScalarPM | `156,584` | `96.383 +/- 0.496%` |
| No overwrite | VectorLeanPM | `145,674` | `96.758 +/- 0.317%` |
| No overwrite | MatrixOnly | `145,674` | `96.992 +/- 0.427%` |
| No overwrite | VectorOnly | `145,674` | `3.100 +/- 0.109%` |
| 50% overwrite | ScalarPM | `156,584` | `88.108 +/- 0.447%` |
| 50% overwrite | VectorLeanPM | `145,674` | `87.758 +/- 0.777%` |
| 50% overwrite | MatrixOnly | `145,674` | `87.625 +/- 0.745%` |
| 50% overwrite | VectorOnly | `145,674` | `3.308 +/- 0.506%` |

The trained vector-only intervention is at chance. The matrix stream is
therefore necessary for this protocol's observed recall behavior. Raw
seed-level outputs remain required for final release promotion.
