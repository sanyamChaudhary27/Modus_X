# PB0 Weight Spectrum Analysis

## Scope

This analysis inspects frozen trained checkpoints and measures the spectral structure of the five repeated matrix-memory projection families before any architecture modification.

The analysis does not modify `language/models.py`, checkpoint files, parameter counts, recurrent-state dimensions, or compiled kernels.

## Checkpoints

| Seed | Checkpoint | Parameter count | Step |
|---|---|---|---|
| seed1 | E:\seed1\checkpoint.pkl | 47,437,768 | 25000 |
| seed2 | E:\seed2\checkpoint.pkl | 47,437,768 | 25000 |

## Target Families

- `m_wq`
- `m_wk`
- `m_wv`
- `m_w_read`
- `m_w_out`

## Candidate Ranks

384, 320, 256, 192, 160, 128, 96, 64

## Per-Family Seed Summary

| Seed | Family | Matrices | Mean stable rank | Median stable rank | Mean entropy effective rank |
|---|---|---|---|---|---|
| seed1 | m_wq | 12 | 7.807 | 5.813 | 45.943 |
| seed1 | m_wk | 12 | 8.518 | 7.403 | 49.724 |
| seed1 | m_wv | 12 | 30.864 | 34.874 | 185.582 |
| seed1 | m_w_read | 12 | 14.989 | 16.777 | 170.335 |
| seed1 | m_w_out | 12 | 7.031 | 7.085 | 155.162 |
| seed2 | m_wq | 12 | 7.551 | 5.951 | 45.655 |
| seed2 | m_wk | 12 | 9.219 | 8.607 | 50.246 |
| seed2 | m_wv | 12 | 29.949 | 32.550 | 184.898 |
| seed2 | m_w_read | 12 | 16.496 | 16.539 | 172.612 |
| seed2 | m_w_out | 12 | 7.283 | 6.942 | 156.975 |

## Mean Retained Spectral Energy

| Seed | Family | Rank 384 | Rank 320 | Rank 256 | Rank 192 | Rank 160 | Rank 128 | Rank 96 | Rank 64 |
|---|---|---|---|---|---|---|---|---|---|
| seed1 | m_wq | 99.9039% | 99.6644% | 99.1639% | 98.2323% | 97.5020% | 96.4589% | 94.8201% | 91.6885% |
| seed1 | m_wk | 99.9268% | 99.7296% | 99.2893% | 98.4165% | 97.6972% | 96.6343% | 94.9211% | 91.5984% |
| seed1 | m_wv | 99.4777% | 98.1642% | 95.3945% | 90.2495% | 86.2946% | 80.9368% | 73.4556% | 62.3803% |
| seed1 | m_w_read | 99.4594% | 98.1090% | 95.2630% | 89.9942% | 85.9743% | 80.5850% | 73.2081% | 62.6616% |
| seed1 | m_w_out | 99.3860% | 97.8522% | 94.6612% | 88.8752% | 84.5613% | 78.9017% | 71.3356% | 60.7906% |
| seed2 | m_wq | 99.9040% | 99.6651% | 99.1669% | 98.2493% | 97.5409% | 96.5503% | 95.0299% | 92.1353% |
| seed2 | m_wk | 99.9269% | 99.7308% | 99.2928% | 98.4275% | 97.7259% | 96.7096% | 95.0976% | 91.9508% |
| seed2 | m_wv | 99.4710% | 98.1498% | 95.3660% | 90.2018% | 86.2516% | 80.9285% | 73.5329% | 62.5728% |
| seed2 | m_w_read | 99.4564% | 98.0801% | 95.1889% | 89.8563% | 85.8066% | 80.3971% | 73.0086% | 62.4932% |
| seed2 | m_w_out | 99.3740% | 97.8181% | 94.5773% | 88.7188% | 84.3588% | 78.6462% | 71.0312% | 60.4749% |

## Worst-Layer Retained Spectral Energy

For each family and seed, this table shows the minimum retained energy across the 12 layers. This is important because a good average can hide a compression-sensitive layer.

| Seed | Family | Rank 384 | Rank 320 | Rank 256 | Rank 192 | Rank 160 | Rank 128 | Rank 96 | Rank 64 |
|---|---|---|---|---|---|---|---|---|---|
| seed1 | m_wq | 99.8959% | 99.6428% | 99.1166% | 97.9083% | 96.7041% | 94.4522% | 89.7855% | 78.8371% |
| seed1 | m_wk | 99.8998% | 99.6509% | 99.1208% | 97.7989% | 96.4578% | 93.9636% | 88.8886% | 77.5175% |
| seed1 | m_wv | 99.2987% | 97.5309% | 93.8362% | 87.1585% | 82.1552% | 75.5514% | 66.7473% | 54.6721% |
| seed1 | m_w_read | 99.2954% | 97.5769% | 93.9720% | 87.4166% | 82.4975% | 76.0405% | 67.4581% | 55.7462% |
| seed1 | m_w_out | 99.1916% | 97.1984% | 93.1002% | 85.8523% | 80.5645% | 73.7744% | 65.0289% | 53.6163% |
| seed2 | m_wq | 99.8931% | 99.6329% | 99.1012% | 98.0810% | 97.2540% | 95.7961% | 92.7871% | 85.0672% |
| seed2 | m_wk | 99.8976% | 99.6484% | 99.1418% | 98.0057% | 96.9835% | 95.2952% | 91.9206% | 83.5535% |
| seed2 | m_wv | 99.3161% | 97.5955% | 94.0144% | 87.4625% | 82.6022% | 76.1716% | 67.6202% | 55.8497% |
| seed2 | m_w_read | 99.2855% | 97.4703% | 93.7198% | 86.9842% | 82.0072% | 75.4813% | 66.8959% | 55.3260% |
| seed2 | m_w_out | 99.2071% | 97.2562% | 93.2187% | 85.9939% | 80.6745% | 73.8125% | 65.0172% | 53.4577% |

## Cross-Seed Consistency

| Family | Seed1 stable rank | Seed2 stable rank | Stable-rank difference | Entropy-effective-rank difference | Consistency |
|---|---|---|---|---|---|
| m_wq | 7.807 | 7.551 | 3.339% | 0.288 | high |
| m_wk | 8.518 | 9.219 | 7.899% | 0.522 | moderate |
| m_wv | 30.864 | 29.949 | 3.011% | 0.685 | high |
| m_w_read | 14.989 | 16.496 | 9.575% | 2.277 | moderate |
| m_w_out | 7.031 | 7.283 | 3.517% | 1.813 | high |

## Layer-Level Variation

Layer-level metrics are written to `PB0_WEIGHT_SPECTRUM_LAYERS.csv`.

### `m_wq`

- Matrices analyzed: **24**
- Stable rank range across both seeds: **3.277 → 28.685**

### `m_wk`

- Matrices analyzed: **24**
- Stable rank range across both seeds: **3.267 → 20.592**

### `m_wv`

- Matrices analyzed: **24**
- Stable rank range across both seeds: **14.298 → 40.013**

### `m_w_read`

- Matrices analyzed: **24**
- Stable rank range across both seeds: **2.848 → 31.048**

### `m_w_out`

- Matrices analyzed: **24**
- Stable rank range across both seeds: **4.238 → 13.119**

## Interpretation Rules

This analysis does not itself prove that a projection can be compressed without language-quality loss.

A candidate is stronger when all of the following are observed:

1. High retained energy at the candidate rank.
2. Low reconstruction error.
3. No severely sensitive individual layer.
4. Similar results across Seed1 and Seed2.

The next experimental gate is PB0-B: replace one projection family at a time with dense truncated-SVD approximations and evaluate the frozen checkpoint without changing architecture topology.
