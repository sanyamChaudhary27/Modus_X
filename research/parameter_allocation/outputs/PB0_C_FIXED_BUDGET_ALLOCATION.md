# PB0-C — Fixed-Budget Parameter Allocation

Canonical parameter budget: **47,437,768**

PB0-C is an accounting/search stage. No training is performed.

## Method

The full Cartesian allocation space is searched mathematically using measured one-dimensional parameter-count deltas.

Only the final promising candidates are constructed with the actual Modus_X model for exact verification.

## Matrix donor accounting

| Candidate | m_wq rank | m_wk rank | Recovered parameters | Model share |
|---|---:|---:|---:|---:|
| C1_q128_k128 | 128 | 128 | 3,145,728 | 6.631% |
| C2_q64_k64 | 64 | 64 | 4,718,592 | 9.947% |
| C3_q128_k64 | 128 | 64 | 3,932,160 | 8.289% |

## Verified candidates

### C1_q128_k128

| Rank | hidden_dim | mamba_state_dim | ax_res | Actual vector Δ | Actual total | Budget error | Exact | Estimator error |
|---:|---:|---:|---:|---:|---:|---:|:---:|---:|
| 1 | 2016 | 512 | 576 | +3,125,184 | 47,417,224 | -20,544 | NO | +0 |
| 2 | 2144 | 544 | 544 | +3,112,384 | 47,404,424 | -33,344 | NO | +0 |
| 3 | 2272 | 576 | 512 | +3,099,584 | 47,391,624 | -46,144 | NO | +0 |
| 4 | 1632 | 608 | 512 | +3,099,072 | 47,391,112 | -46,656 | NO | +0 |
| 5 | 1984 | 512 | 576 | +3,075,968 | 47,368,008 | -69,760 | NO | +0 |
| 6 | 2112 | 544 | 544 | +3,063,168 | 47,355,208 | -82,560 | NO | +0 |
| 7 | 2240 | 576 | 512 | +3,050,368 | 47,342,408 | -95,360 | NO | +0 |
| 8 | 1600 | 608 | 512 | +3,049,856 | 47,341,896 | -95,872 | NO | +0 |
| 9 | 1952 | 512 | 576 | +3,026,752 | 47,318,792 | -118,976 | NO | +0 |
| 10 | 2080 | 544 | 544 | +3,013,952 | 47,305,992 | -131,776 | NO | +0 |
| 11 | 2208 | 576 | 512 | +3,001,152 | 47,293,192 | -144,576 | NO | +0 |
| 12 | 1568 | 608 | 512 | +3,000,640 | 47,292,680 | -145,088 | NO | +0 |
| 13 | 1920 | 512 | 576 | +2,977,536 | 47,269,576 | -168,192 | NO | +0 |
| 14 | 2048 | 544 | 544 | +2,964,736 | 47,256,776 | -180,992 | NO | +0 |
| 15 | 2176 | 576 | 512 | +2,951,936 | 47,243,976 | -193,792 | NO | +0 |

### C2_q64_k64

| Rank | hidden_dim | mamba_state_dim | ax_res | Actual vector Δ | Actual total | Budget error | Exact | Estimator error |
|---:|---:|---:|---:|---:|---:|---:|:---:|---:|
| 1 | 2272 | 512 | 608 | +4,712,384 | 47,431,560 | -6,208 | NO | +0 |
| 2 | 1632 | 544 | 608 | +4,711,872 | 47,431,048 | -6,720 | NO | +0 |
| 3 | 1760 | 576 | 576 | +4,699,072 | 47,418,248 | -19,520 | NO | +0 |
| 4 | 1888 | 608 | 544 | +4,686,272 | 47,405,448 | -32,320 | NO | +0 |
| 5 | 2016 | 640 | 512 | +4,673,472 | 47,392,648 | -45,120 | NO | +0 |
| 6 | 2240 | 512 | 608 | +4,663,168 | 47,382,344 | -55,424 | NO | +0 |
| 7 | 1600 | 544 | 608 | +4,662,656 | 47,381,832 | -55,936 | NO | +0 |
| 8 | 1728 | 576 | 576 | +4,649,856 | 47,369,032 | -68,736 | NO | +0 |
| 9 | 1856 | 608 | 544 | +4,637,056 | 47,356,232 | -81,536 | NO | +0 |
| 10 | 1984 | 640 | 512 | +4,624,256 | 47,343,432 | -94,336 | NO | +0 |
| 11 | 2208 | 512 | 608 | +4,613,952 | 47,333,128 | -104,640 | NO | +0 |
| 12 | 1568 | 544 | 608 | +4,613,440 | 47,332,616 | -105,152 | NO | +0 |
| 13 | 1696 | 576 | 576 | +4,600,640 | 47,319,816 | -117,952 | NO | +0 |
| 14 | 1824 | 608 | 544 | +4,587,840 | 47,307,016 | -130,752 | NO | +0 |
| 15 | 1952 | 640 | 512 | +4,575,040 | 47,294,216 | -143,552 | NO | +0 |

### C3_q128_k64

| Rank | hidden_dim | mamba_state_dim | ax_res | Actual vector Δ | Actual total | Budget error | Exact | Estimator error |
|---:|---:|---:|---:|---:|---:|---:|:---:|---:|
| 1 | 1760 | 512 | 608 | +3,924,928 | 47,430,536 | -7,232 | NO | +0 |
| 2 | 1888 | 544 | 576 | +3,912,128 | 47,417,736 | -20,032 | NO | +0 |
| 3 | 2016 | 576 | 544 | +3,899,328 | 47,404,936 | -32,832 | NO | +0 |
| 4 | 2144 | 608 | 512 | +3,886,528 | 47,392,136 | -45,632 | NO | +0 |
| 5 | 1728 | 512 | 608 | +3,875,712 | 47,381,320 | -56,448 | NO | +0 |
| 6 | 1856 | 544 | 576 | +3,862,912 | 47,368,520 | -69,248 | NO | +0 |
| 7 | 1984 | 576 | 544 | +3,850,112 | 47,355,720 | -82,048 | NO | +0 |
| 8 | 2112 | 608 | 512 | +3,837,312 | 47,342,920 | -94,848 | NO | +0 |
| 9 | 1696 | 512 | 608 | +3,826,496 | 47,332,104 | -105,664 | NO | +0 |
| 10 | 1824 | 544 | 576 | +3,813,696 | 47,319,304 | -118,464 | NO | +0 |
| 11 | 1952 | 576 | 544 | +3,800,896 | 47,306,504 | -131,264 | NO | +0 |
| 12 | 2080 | 608 | 512 | +3,788,096 | 47,293,704 | -144,064 | NO | +0 |
| 13 | 1664 | 512 | 608 | +3,777,280 | 47,282,888 | -154,880 | NO | +0 |
| 14 | 1792 | 544 | 576 | +3,764,480 | 47,270,088 | -167,680 | NO | +0 |
| 15 | 1920 | 576 | 544 | +3,751,680 | 47,257,288 | -180,480 | NO | +0 |

## Interpretation

PB0-B provides the justification for treating m_wq and m_wk as candidate donor projections.

PB0-C does not claim that increased vector/Mamba capacity improves BPC. It only identifies parameter-valid allocations for training.

An exact 47,437,768-parameter allocation is preferred for the fixed-budget comparison.

The final architecture must be selected using matched training and validation/test BPC.

## Next experiment

PB1-A: implement the factorized m_wq/m_wk architecture and train the best verified fixed-budget candidates against the canonical 47M baseline.