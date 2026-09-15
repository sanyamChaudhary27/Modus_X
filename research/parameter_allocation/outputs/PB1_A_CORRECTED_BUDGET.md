# PB1-A Actual Factorized Q/K Budget Correction

- Model: `Modus_X_MemoryFeedbackArchive_DeepSupervision`
- Canonical parameters: **47,437,768**
- Intervention: factorize only `m_wq` and `m_wk`.
- `vector_router=False`.
- Auxiliary layer: `(6,)`.
- Future target count: `1`.

## Exact verified candidates

| Rank | Candidate | Hidden | Mamba | ax_res | Q rank | K rank | Dense params | Factorized params | Budget error | Q/K savings |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | C1_q128_k128 | 1664 | 608 | 512 | 128 | 128 | 50,586,056 | 47,440,328 | +2,560 | 3,145,728 |
| 2 | C1_q128_k128 | 2304 | 576 | 512 | 128 | 128 | 50,586,568 | 47,440,840 | +3,072 | 3,145,728 |
| 3 | C3_q128_k64 | 1536 | 640 | 512 | 128 | 64 | 51,373,000 | 47,440,840 | +3,072 | 3,932,160 |
| 4 | C1_q128_k128 | 2944 | 544 | 512 | 128 | 128 | 50,587,080 | 47,441,352 | +3,584 | 3,145,728 |
| 5 | C3_q128_k64 | 2176 | 608 | 512 | 128 | 64 | 51,373,512 | 47,441,352 | +3,584 | 3,932,160 |
| 6 | C1_q128_k128 | 3584 | 512 | 512 | 128 | 128 | 50,587,592 | 47,441,864 | +4,096 | 3,145,728 |
| 7 | C2_q64_k64 | 2048 | 640 | 512 | 64 | 64 | 52,160,456 | 47,441,864 | +4,096 | 4,718,592 |
| 8 | C3_q128_k64 | 2816 | 576 | 512 | 128 | 64 | 51,374,024 | 47,441,864 | +4,096 | 3,932,160 |
| 9 | C2_q64_k64 | 2688 | 608 | 512 | 64 | 64 | 52,160,968 | 47,442,376 | +4,608 | 4,718,592 |
| 10 | C3_q128_k64 | 3456 | 544 | 512 | 128 | 64 | 51,374,536 | 47,442,376 | +4,608 | 3,932,160 |
| 11 | C2_q64_k64 | 2944 | 512 | 608 | 64 | 64 | 53,183,688 | 47,432,904 | -4,864 | 5,750,784 |
| 12 | C2_q64_k64 | 3328 | 576 | 512 | 64 | 64 | 52,161,480 | 47,442,888 | +5,120 | 4,718,592 |
| 13 | C3_q128_k64 | 4096 | 512 | 512 | 128 | 64 | 51,375,048 | 47,442,888 | +5,120 | 3,932,160 |
| 14 | C2_q64_k64 | 2304 | 544 | 608 | 64 | 64 | 53,183,176 | 47,432,392 | -5,376 | 5,750,784 |
| 15 | C2_q64_k64 | 3968 | 544 | 512 | 64 | 64 | 52,161,992 | 47,443,400 | +5,632 | 4,718,592 |

## Interpretation

PB0-C is retained as the completed dense allocation study.
PB1-A is correcting only the implementation-aware budget count.
No training candidate should be promoted until its factorized count
has been verified here.

The best candidate is the row with the smallest absolute
actual budget error. Prefer a slightly-under-budget candidate
over exceeding the fixed budget unless a separate explicit
budget policy is adopted.

This script does not establish that a factorized candidate is
functionally equivalent to the dense model. Functional validation
belongs to the subsequent PB1-A/PB1-B architecture smoke test and
matched training.
