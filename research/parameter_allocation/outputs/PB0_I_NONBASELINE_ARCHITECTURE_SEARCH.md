# PB0-I Exact Non-Baseline Architecture Verification

## Purpose

PB0-I is the final parameter-accounting gate before any candidate training. PB0-G generated calibrated fixed-budget candidates; PB0-I instantiates the actual architecture and verifies the real parameter tree.

## Architecture Source of Truth

- Model: `Modus_X_MemoryFeedbackArchive`
- Public factory call: `make_model(name, key, cfg, auxiliary_layers=..., future_target_count=...)`
- Feedback rank rule: `min(32, ax_res, mamba_state_dim)`

## Canonical Baseline Gate

- B0 count: **47,437,768**
- Actual instantiated count: **47,437,768**
- Difference: **+0**
- Exact: **True**
- Tree fingerprint: `27e2206566ba1dbfa7cf4438ae716dc2091d6bcc344a8fe89355d89077824662`

## PB0-G Candidate Pool

- Candidates loaded from PB0-G: **500**
- Non-baseline candidates verified here: **25**

## Exact Verification Results

| # | r | n | h | f | PB0-G params | Actual params | Delta | Exact | Status |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 512 | 520 | 12 | 32 | 47,437,720 | 47,437,720 | +0 | True | `PASS_EXACT_PB0_G` |
| 2 | 512 | 504 | 52 | 32 | 47,437,816 | 47,437,816 | +0 | True | `PASS_EXACT_PB0_G` |
| 3 | 512 | 496 | 72 | 32 | 47,437,864 | 47,437,864 | +0 | True | `PASS_EXACT_PB0_G` |
| 4 | 512 | 488 | 92 | 32 | 47,437,912 | 47,437,912 | +0 | True | `PASS_EXACT_PB0_G` |
| 5 | 360 | 704 | 13 | 32 | 47,437,924 | 47,437,924 | +0 | True | `PASS_EXACT_PB0_G` |
| 6 | 512 | 480 | 112 | 32 | 47,437,960 | 47,437,960 | +0 | True | `PASS_EXACT_PB0_G` |
| 7 | 360 | 696 | 33 | 32 | 47,437,972 | 47,437,972 | +0 | True | `PASS_EXACT_PB0_G` |
| 8 | 328 | 704 | 110 | 32 | 47,437,552 | 47,437,552 | +0 | True | `PASS_EXACT_PB0_G` |
| 9 | 360 | 688 | 53 | 32 | 47,438,020 | 47,438,020 | +0 | True | `PASS_EXACT_PB0_G` |
| 10 | 328 | 712 | 90 | 32 | 47,437,504 | 47,437,504 | +0 | True | `PASS_EXACT_PB0_G` |
| 11 | 360 | 680 | 73 | 32 | 47,438,068 | 47,438,068 | +0 | True | `PASS_EXACT_PB0_G` |
| 12 | 328 | 720 | 70 | 32 | 47,437,456 | 47,437,456 | +0 | True | `PASS_EXACT_PB0_G` |
| 13 | 360 | 672 | 93 | 32 | 47,438,116 | 47,438,116 | +0 | True | `PASS_EXACT_PB0_G` |
| 14 | 328 | 728 | 50 | 32 | 47,437,408 | 47,437,408 | +0 | True | `PASS_EXACT_PB0_G` |
| 15 | 360 | 664 | 113 | 32 | 47,438,164 | 47,438,164 | +0 | True | `PASS_EXACT_PB0_G` |
| 16 | 328 | 736 | 30 | 32 | 47,437,360 | 47,437,360 | +0 | True | `PASS_EXACT_PB0_G` |
| 17 | 480 | 520 | 109 | 32 | 47,437,348 | 47,437,348 | +0 | True | `PASS_EXACT_PB0_G` |
| 18 | 328 | 744 | 10 | 32 | 47,437,312 | 47,437,312 | +0 | True | `PASS_EXACT_PB0_G` |
| 19 | 480 | 528 | 89 | 32 | 47,437,300 | 47,437,300 | +0 | True | `PASS_EXACT_PB0_G` |
| 20 | 480 | 536 | 69 | 32 | 47,437,252 | 47,437,252 | +0 | True | `PASS_EXACT_PB0_G` |
| 21 | 480 | 544 | 49 | 32 | 47,437,204 | 47,437,204 | +0 | True | `PASS_EXACT_PB0_G` |
| 22 | 480 | 552 | 29 | 32 | 47,437,156 | 47,437,156 | +0 | True | `PASS_EXACT_PB0_G` |
| 23 | 480 | 560 | 9 | 32 | 47,437,108 | 47,437,108 | +0 | True | `PASS_EXACT_PB0_G` |
| 24 | 392 | 664 | 16 | 32 | 47,438,536 | 47,438,536 | +0 | True | `PASS_EXACT_PB0_G` |
| 25 | 296 | 736 | 127 | 32 | 47,436,988 | 47,436,988 | +0 | True | `PASS_EXACT_PB0_G` |

## Gate Decision

### PASSED — realizable fixed-budget candidates exist

**25** verified non-baseline candidate(s) reproduce the PB0-G parameter count exactly and remain within the fixed canonical budget window.

## Important Interpretation

PB0-I does not establish that any candidate improves BPC. It only establishes that the candidate is a real, reproducible architecture under the current implementation and parameter accounting.

A candidate must still pass an empirical training/validation screen before any architecture change is accepted.

## Next Gate

If PB0-I passes, train only the strongest verified candidate(s) under the same data, optimizer, schedule, seed policy, and evaluation protocol used for the canonical baseline.

Do not modify `language/models.py` merely because PB0-G found a different allocation. Let the empirical BPC result determine whether a parameter-allocation change is justified.
