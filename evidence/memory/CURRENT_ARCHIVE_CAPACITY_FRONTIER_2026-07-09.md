# CurrentArchiveDelta Capacity Frontier

Date: 2026-07-09

Raw output:

- `raw_outputs/current_archive_capacity_frontier_2026-07-09_raw.txt`

## Question

How far does the `CurrentArchiveDelta` Stage1G win survive as binding load
increases, and does a larger matrix resolution earn its memory cost?

## Protocol

- models: `TwoPathLatestShadowDelta`, `CurrentArchiveDelta`
- seeds: `17,27,37`
- bindings: `32,64,96,128`
- `ax_res`: `64,128`
- overwrite rate: `0.5`
- training curriculum: `role_balanced_overwritten`
- version-tagged facts enabled

Important comparability note:

- rows are parameter-matched **within the same `ax_res`**;
- `ax_res=64` rows have `166,791` parameters;
- `ax_res=128` rows have `338,119` parameters;
- therefore the clean comparison is `CurrentArchiveDelta` versus
  `TwoPathLatestShadowDelta` at the same `ax_res`, not `64` versus `128` as if
  they were the same-size model.

Estimated matrix-state bytes:

- `TwoPathLatestShadowDelta`: `ax_res * ax_res * 4`
- `CurrentArchiveDelta`: `2 * ax_res * ax_res * 4`

## Frontier Table

### `ax_res=64`

| bindings | model | state KB | mixed | latest | previous | first |
|---:|---|---:|---:|---:|---:|---:|
| 32 | TwoPathLatestShadowDelta | 16 | 55.01 | 22.82 | 71.71 | 69.53 |
| 32 | **CurrentArchiveDelta** | 32 | **70.02** | **69.08** | **73.80** | **72.20** |
| 64 | TwoPathLatestShadowDelta | 16 | 36.23 | 17.51 | 46.52 | 45.57 |
| 64 | **CurrentArchiveDelta** | 32 | **48.54** | **41.63** | **51.43** | **48.83** |
| 96 | TwoPathLatestShadowDelta | 16 | 27.80 | 14.32 | **33.92** | 33.82 |
| 96 | **CurrentArchiveDelta** | 32 | **32.98** | **29.36** | 33.53 | **36.33** |
| 128 | TwoPathLatestShadowDelta | 16 | 22.07 | 11.88 | **27.41** | **28.45** |
| 128 | **CurrentArchiveDelta** | 32 | **27.12** | **23.14** | 27.21 | 28.32 |

### `ax_res=128`

| bindings | model | state KB | mixed | latest | previous | first |
|---:|---|---:|---:|---:|---:|---:|
| 32 | TwoPathLatestShadowDelta | 64 | 54.72 | 24.19 | **72.01** | 70.93 |
| 32 | **CurrentArchiveDelta** | 128 | **70.35** | **67.94** | 71.81 | **71.39** |
| 64 | TwoPathLatestShadowDelta | 64 | 37.27 | 18.95 | **47.20** | **46.68** |
| 64 | **CurrentArchiveDelta** | 128 | **46.32** | **46.42** | 46.81 | 46.22 |
| 96 | TwoPathLatestShadowDelta | 64 | 27.86 | 15.62 | 32.78 | 33.46 |
| 96 | **CurrentArchiveDelta** | 128 | **34.47** | **34.60** | **33.53** | **35.35** |
| 128 | TwoPathLatestShadowDelta | 64 | 23.24 | 12.92 | **27.67** | **27.44** |
| 128 | **CurrentArchiveDelta** | 128 | **26.24** | **26.73** | 24.54 | 25.03 |

## Deltas Versus TwoPath

| ax_res | bindings | mixed | latest | previous | first |
|---:|---:|---:|---:|---:|---:|
| 64 | 32 | +15.01 | +46.26 | +2.08 | +2.67 |
| 64 | 64 | +12.30 | +24.12 | +4.92 | +3.26 |
| 64 | 96 | +5.18 | +15.04 | -0.39 | +2.51 |
| 64 | 128 | +5.05 | +11.26 | -0.20 | -0.13 |
| 128 | 32 | +15.62 | +43.75 | -0.20 | +0.46 |
| 128 | 64 | +9.05 | +27.47 | -0.39 | -0.46 |
| 128 | 96 | +6.61 | +18.98 | +0.75 | +1.89 |
| 128 | 128 | +2.99 | +13.80 | -3.12 | -2.41 |

## Error Profile

CurrentArchiveDelta keeps stale-version false recall much lower than the
single-matrix/two-path baseline.

For CurrentArchiveDelta:

| ax_res | bindings | stale on latest | history to latest | current-history disagreement |
|---:|---:|---:|---:|---:|
| 64 | 32 | 6.84 | 0.75 | 3.99 |
| 64 | 64 | 13.93 | 1.45 | 3.67 |
| 64 | 96 | 13.51 | 2.43 | 3.49 |
| 64 | 128 | 12.63 | 2.07 | 3.59 |
| 128 | 32 | 7.68 | 0.75 | 5.64 |
| 128 | 64 | 10.12 | 1.50 | 4.78 |
| 128 | 96 | 11.69 | 2.18 | 4.51 |
| 128 | 128 | 9.38 | 2.31 | 4.41 |

## Interpretation

This confirms the `CurrentArchiveDelta` mechanism:

- the latest/current path is dramatically stronger than the prior one-matrix
  design;
- previous/first history is mostly preserved at moderate load;
- stale-version false recall stays much lower;
- the advantage persists through `128` bindings, but weakens at high load.

The compute/memory caveat is important:

- `ax_res=128` uses more parameters and more state than `ax_res=64`;
- under this short screen, the larger model does not yet show a clean enough
  gain to justify promoting it as the default;
- this does **not** prove `ax_res=128` is worse. It means the larger setting
  needs a separate longer/tuned run before we spend larger-run budget on it.

## Decision

Promote `CurrentArchiveDelta` as the current Modus_X 2.0.0 memory candidate.
Use `ax_res=64` as the default for the next gates because it is the cheaper
validated setting.

Do not reject `ax_res=128`; park it until a longer/tuned run can test whether
the larger state earns its parameter and memory cost.

## Next Gate

Run distractor retention at `ax_res=64`:

- random distractors;
- irrelevant key-value distractors;
- similar-key distractors;
- post-update distractors after the latest value.

Then run a small enwik8 smoke to verify that the current/archive split does not
break language-model learning.
