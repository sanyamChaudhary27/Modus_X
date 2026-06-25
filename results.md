# Modus_X v1.1 Results

This file summarizes the headline evidence in the v1.1 publication. Exact
claim boundaries and protocol details are recorded in
`docs/CLAIMS_AND_EVIDENCE.md` and `docs/BENCHMARK_PROTOCOL.md`.

## Matched enwik8 Dense Audit

All models were evaluated at 40,000 optimizer updates and 163.84M processed
characters with nearby parameter counts.

| Model | Parameters | Dense validation BPC | Dense test BPC |
| --- | ---: | ---: | ---: |
| Official Mamba | 81.46M | **1.3505** | **1.3458** |
| Modus_X | 82.76M | **1.3787** | **1.3842** |
| Official xLSTM | 76.65M | 1.4351 | 1.4196 |

Supported interpretation:

- Official Mamba is the strongest byte-language model in this comparison.
- Modus_X outperforms the tested official xLSTM configuration.
- All three models use constant recurrent inference state with respect to
  sequence length.

## Associative Recall

| Model | Params | Length 128 | Length 256 | Length 512 | Length 1024 | Length 2048 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Modus_X VectorLean | 152,436 | **95.1%** | **94.5%** | **94.8%** | **94.5%** | **94.6%** |
| Official Mamba recall model | 162,560 | 2.85% | 3.70% | 3.30% | 3.28% | 3.33% |
| Chance | - | 3.125% | 3.125% | 3.125% | 3.125% | 3.125% |

This is a controlled synthetic binding task. It demonstrates a strong
content-addressed-memory advantage under the tested protocol, not universal
long-context superiority.

## Same-Key Overwrite

| Model | No-overwrite recall | 50% overwrite recall |
| --- | ---: | ---: |
| Modus_X VectorLean | **97.325%** | **88.850%** |
| Official Mamba recall model | 2.850% | 3.425% |

## Current Claim

> Modus_X is a competitive constant-state language model with a demonstrated
> associative-memory advantage on controlled binding and overwrite tasks. It
> does not yet lead official Mamba on enwik8 compression.

See `whitepaper.pdf` for the full analysis, limitations, and roadmap.
