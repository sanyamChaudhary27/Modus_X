# Evidence index

All metrics below are measured unless explicitly labeled otherwise.

## Language modeling

### Promoted result

MemoryFeedbackArchive versus CurrentArchive at `102.4M` processed enwik8
characters:

| Metric | CurrentArchive | MemoryFeedback | Improvement |
|---|---:|---:|---:|
| Dense validation BPC | 1.485020 | 1.459723 | 0.025297 |
| Dense test BPC | 1.492694 | 1.465006 | 0.027688 |
| Dense train-tail BPC | 1.427770 | 1.408210 | 0.019560 |

Cost: `0.85%` more parameters and approximately `12.5%` more runtime.

Canonical document:
`evidence/language/MEMORY_FEEDBACK_ARCHIVE_V0_2026-07-12.md`.

### Measured scaling

At the fixed `102.4M`-character endpoint:

| Model | Parameters | Dense validation BPC | Dense test BPC |
|---|---:|---:|---:|
| MemoryFeedback 47M | 47,437,768 | 1.459723 | 1.465006 |
| MemoryFeedback 81M | 81,486,728 | **1.433138** | 1.443873 |
| MemoryFeedback 99M | 99,438,920 | 1.434283 | **1.442034** |

The 47M-to-81M interval improves strongly. The 81M-to-99M interval is
saturated: test improves by only `0.001839` while validation regresses by
`0.001145`. These are single-seed points; the 47M schedule differs from the
larger runs.

Canonical document and derived evidence:
`evidence/language/scaling/MEMORY_FEEDBACK_SCALING_RESULT_2026-07-24.md`.

Matched late annealing improves the 81M and 99M dense test endpoints to
`1.408172` and `1.404835` at `122.88M` characters. The pre-registered
validation gate promotes the more efficient 81M branch. At `163.84M`
characters, that branch reaches `1.375422` dense validation and `1.382445`
dense test BPC, narrowly improving v1.1.1 with fewer parameters.

### External baseline boundary

At the approximately 80M tier and `163.84M` enwik8 characters, the published
Modus_X checkpoint reaches about `1.38418` dense test BPC. The tested official
Mamba reaches `1.34578`; Mamba wins. The tested official xLSTM reaches
`1.41962`; Modus_X wins that comparison.

These v1.1.1 baseline results contextualize v2. They are not re-created by this
package.

## Bounded memory

### Equal-state crossover

- Full-context Transformer KV at `64.25 KiB`: `98.63%`.
- CurrentArchive at the same state budget: `71.03%`.
- Near-parameter-matched `16.13 KiB` point:
  CurrentArchive `72.92%`, truncated Transformer KV `17.71%`.

Conclusion: full-context KV wins when it fits; CurrentArchive wins in the
measured constrained-state region. CurrentArchive also fails at sufficiently
small matrix state.

### Mixed clean/update retrieval

At `16,512` state bytes:

| Model | Overall | Clean | Overwritten |
|---|---:|---:|---:|
| CurrentArchive | 77.95% | 77.91% | 77.99% |
| Transformer KV | 16.38% | 14.19% | 18.58% |

CurrentArchive stale false recall is worse: `11.28%` versus `4.65%`.

### Latest-heavy diagnostic

Latest-heavy supervision improves latest-overwritten accuracy by `5.60` points
and overall accuracy by `1.80` points, but stale recall remains `8.09%`, above
the pre-registered `7%` ceiling.

### Operation diagnosis

Across six paired checkpoints, current/history disagreement is lower and
history-read norm is higher on stale failures. Final stale predictions agree
with the current-memory shared-head probe about `83%` of the time, not the
vector probe.

### Rejected correction

`CurrentArchiveLatestShadow` improves clean latest retrieval but reduces
overwritten latest retrieval by `14.06` points, raises stale recall by `5.86`
points, reduces overall accuracy by `1.78` points, and costs `28.9%` runtime.
It is rejected.

## Negative architecture results

- AdaptivePreconditionedArchive: insufficient gain and excessive overhead.
- AttentionToWriteArchive: below promotion threshold.
- MemoryFeedback plus AttentionToWrite: harmful interaction.
- Official-Mamba matrix insertion: loses to the width control.
- Official-Mamba true MemoryFeedback: bounded and efficient but does not beat
  the width control.

Negative results are included to prevent repeated sweeps and post-hoc claims.

## Systems

The frozen v1-style `1,058,963,121` parameter configuration has passed exact
TPU forward/backward, AdamW, real-data update, checkpoint, and restore smokes.
The `1,058,467,601` parameter MemoryFeedback candidate is exactly counted but
has not passed the full systems ladder or quality training.

Systems readiness is not model-quality evidence.
