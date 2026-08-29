# Matched Modus_X Versus Official Mamba Multi-Seed Run

Run date: 2026-06-25

## Protocol

- Seeds: `17`, `27`, `37`
- Training examples: `40,000`
- Test examples: `4,000`
- Epochs: `12`
- Training length: `128`
- Evaluation lengths: `128`, `256`, `512`, `1024`, `2048`
- Values: `32`; chance accuracy: `3.125%`
- Conditions: no overwrite and `50%` same-key overwrite
- Modus_X: VectorLeanPM, router width `16`, `145,674` parameters
- Official Mamba: `mamba_ssm.Mamba`, `162,560` parameters

The models use their established implementation-specific batch sizes
(`256` for the JAX Modus_X implementation and `64` for the PyTorch Mamba
implementation). Dataset size, epochs, learning rate, weight decay, seed set,
task generator, and evaluation lengths are aligned.

## Aggregate Results

| Condition | Model | Best train-length accuracy | L=2048 accuracy |
| --- | --- | ---: | ---: |
| No overwrite | Modus_X | `97.033 +/- 0.347%` | `96.758 +/- 0.317%` |
| No overwrite | Official Mamba | `3.508 +/- 0.296%` | `3.025 +/- 0.217%` |
| 50% overwrite | Modus_X | `87.967 +/- 0.822%` | `87.750 +/- 0.763%` |
| 50% overwrite | Official Mamba | `3.708 +/- 0.123%` | `3.242 +/- 0.142%` |

Values are sample mean plus/minus sample standard deviation across three
seeds.

## Interpretation

The previously reported seed-17 separation reproduces across seeds 27 and 37.
Modus_X remains stable through a 16x evaluation-length increase and retains
strong overwrite behavior. The evaluated official Mamba configuration remains
near chance under both conditions.

This is protocol-specific evidence from small models. It does not establish
general factual recall, universal superiority over Mamba, or superiority on
language-model compression. Official Mamba remains stronger on the published
enwik8 BPC comparison.

## Reproduction

The directory preserves:

- `full_run_log.txt`
- `summary.json`
- `run_matched_recall_multiseed.py`
- `modus_x_router_balanced_kv.py`
- `official_mamba_balanced_kv.py`

The Kaggle run used official precompiled `mamba_ssm` and `causal_conv1d`
wheels matched to the active Torch/CUDA/Python ABI.
