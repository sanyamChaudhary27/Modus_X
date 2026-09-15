# PB0-G Calibrated Candidate Search

## Purpose

PB0-G searches the actual calibrated MemoryFeedbackArchive parameter space for integer dimension configurations near the frozen canonical parameter budget.

## Target

- Frozen canonical parameter budget: **47,437,768**

## Calibrated Formula

```text
P = 12011080 + (36912 * r) + (30744 * n) + (12300 * h) + (12 * f * r) + (6144 * f)

f = min(32, r, n)
```

## Search Result

- Stored candidates: **500**
- Exact parameter matches: **1**

## Best Candidate

- Matrix rank `r`: **512**
- Vector dimension `n`: **512**
- Router width `h`: **32**
- Feedback rank `f`: **32**
- Parameter count: **47,437,768**
- Parameter error: **+0**
- Exact match: **True**

## Top Candidates

| Rank r | Vector n | Router h | Feedback f | Parameters | Error | Exact |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 512 | 32 | 32 | 47,437,768 | +0 | True |
| 512 | 520 | 12 | 32 | 47,437,720 | -48 | False |
| 512 | 504 | 52 | 32 | 47,437,816 | +48 | False |
| 512 | 496 | 72 | 32 | 47,437,864 | +96 | False |
| 512 | 488 | 92 | 32 | 47,437,912 | +144 | False |
| 360 | 704 | 13 | 32 | 47,437,924 | +156 | False |
| 512 | 480 | 112 | 32 | 47,437,960 | +192 | False |
| 360 | 696 | 33 | 32 | 47,437,972 | +204 | False |
| 328 | 704 | 110 | 32 | 47,437,552 | -216 | False |
| 360 | 688 | 53 | 32 | 47,438,020 | +252 | False |
| 328 | 712 | 90 | 32 | 47,437,504 | -264 | False |
| 360 | 680 | 73 | 32 | 47,438,068 | +300 | False |
| 328 | 720 | 70 | 32 | 47,437,456 | -312 | False |
| 360 | 672 | 93 | 32 | 47,438,116 | +348 | False |
| 328 | 728 | 50 | 32 | 47,437,408 | -360 | False |
| 360 | 664 | 113 | 32 | 47,438,164 | +396 | False |
| 328 | 736 | 30 | 32 | 47,437,360 | -408 | False |
| 480 | 520 | 109 | 32 | 47,437,348 | -420 | False |
| 328 | 744 | 10 | 32 | 47,437,312 | -456 | False |
| 480 | 528 | 89 | 32 | 47,437,300 | -468 | False |
| 480 | 536 | 69 | 32 | 47,437,252 | -516 | False |
| 480 | 544 | 49 | 32 | 47,437,204 | -564 | False |
| 480 | 552 | 29 | 32 | 47,437,156 | -612 | False |
| 480 | 560 | 9 | 32 | 47,437,108 | -660 | False |
| 392 | 664 | 16 | 32 | 47,438,536 | +768 | False |

## Next Gate

Do not train any candidate yet. The selected candidate must be instantiated through the actual Modus_X initializer and verified with an exact parameter-tree recount.
