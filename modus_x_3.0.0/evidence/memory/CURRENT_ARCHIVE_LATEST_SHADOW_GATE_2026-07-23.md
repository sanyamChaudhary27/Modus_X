# CurrentArchive latest-shadow refresh gate

Date: 2026-07-23

## Question

Does initializing a consistent current/latest shadow slot for first facts let
later updates refresh that slot and reduce stale latest-value recall?

## Frozen protocol

- Control: `CurrentArchiveDelta`
- Candidate: `CurrentArchiveLatestShadow`
- Seeds: `17, 27, 37`
- Parameters: `166,791` for both
- Recurrent state: `16,512` bytes for both
- Latest-heavy `50/25/25` curriculum
- Identical initialization, training examples, permutations, validation data,
  test data, optimizer, and checkpoint selection
- Only changed variable: `latest_shadow_write`
- Runner:
  `experiments/matrix_memory_capacity/run_current_archive_latest_shadow_gate.py`

## Results

| Metric | Control | Latest shadow | Delta |
|---|---:|---:|---:|
| Overall accuracy | 77.951% | 76.172% | -1.780 pp |
| Clean accuracy | 78.385% | 80.339% | +1.953 pp |
| Overwritten accuracy | 77.517% | 72.005% | -5.512 pp |
| Latest-clean accuracy | 69.271% | 76.953% | +7.682 pp |
| Latest-overwritten accuracy | 70.443% | 56.380% | -14.062 pp |
| Previous-overwritten accuracy | 83.073% | 81.380% | -1.693 pp |
| First-overwritten accuracy | 79.036% | 78.255% | -0.781 pp |
| Stale false recall | 9.155% | 15.012% | +5.858 pp |
| Mean elapsed time | 70.355 s | 90.690 s | `1.289x` |

## Gate decision

The candidate fails the causal gate:

- latest-overwritten accuracy declines instead of improving;
- stale false recall increases instead of reaching `<=7%`;
- overall loss exceeds the allowed `1` point;
- runtime overhead is approximately `28.9%`, above the `5%` ceiling.

The clean/latest gain shows that the shadow slot is active, but it does not
provide update-safe refresh. Duplicating the first fact into a latest-address
lane creates additional interference that harms the exact overwritten-latest
behavior the correction targeted.

## Decision

- Reject `CurrentArchiveLatestShadow`.
- Do not tune its gate, weight, curriculum, or runtime path.
- Freeze the synthetic architecture-correction lane for the v2 release.
- Preserve `CurrentArchiveDelta` as the bounded storage/update evidence model.
- Preserve `MemoryFeedbackArchive` as the v2 language-model lead.

## Claim boundary

This result rejects one specific refresh mechanism. It does not show that
current/history interference is impossible to solve, but further correction
work requires a new task-relevant hypothesis after the v2 evidence package,
not another sweep inside this closure campaign.
