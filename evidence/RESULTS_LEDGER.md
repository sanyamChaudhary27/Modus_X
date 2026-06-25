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

