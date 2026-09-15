# PB0 Candidate A Dimension Mapping

## Purpose

Map the PB0-D Candidate A fixed-budget allocation to integer architecture dimensions without training a model or reading training, validation, or test data.

## Frozen Architecture

| Field | Value |
|---|---:|
| Vocabulary size | 256 |
| Layer count | 12 |
| Embedding width | 512 |
| Hidden width | 1,536 |
| Matrix/vector state width | 512 |

## Candidate A Parameter Targets

| Component | Baseline | Target | Delta |
|---|---:|---:|---:|
| matrix_memory | 23,448,400 | 23,448,400 | +0 |
| vector_mamba_pathway | 20,616,253 | 20,616,253 | +0 |
| router | 464,332 | 464,332 | +0 |
| feedback_bridge | 9,327 | 9,327 | +0 |
| embeddings | 131,072 | 131,072 | +0 |
| output_head | 1,181,440 | 1,181,440 | +0 |
| normalization | 12,288 | 12,288 | +0 |
| auxiliary | 1,181,440 | 1,181,440 | +0 |
| other | 393,216 | 393,216 | +0 |

## Selected Integer Mapping

| Dimension | Value |
|---|---:|
| Matrix rank | 168 |
| Vector expansion | 384 |
| Router hidden width | 20 |
| Feedback rank | 7 |

## Selected Parameter Counts

| Component | Target | Realized | Error |
|---|---:|---:|---:|
| matrix_memory | 23,448,400 | 20,643,840 | 2,804,560 |
| vector_mamba_pathway | 20,616,253 | 23,003,136 | 2,386,883 |
| router | 464,332 | 756,204 | 291,872 |
| feedback_bridge | 9,327 | 135,252 | 125,925 |
| Frozen components | 2,899,456 | 2,899,456 | 0 |
| **TOTAL** | **47,437,768** | **47,437,888** | **120** |

## Mapping Status

- Exact budget match: **False**
- Exact component-target match: **False**
- Candidate mappings searched: **3,600**
- Retained mappings: **50**

## Important Gate

This script is a mathematical dimension-mapping audit. Before Candidate A may train, the selected dimensions must be mapped into the actual model.py implementation and the instantiated parameter tree must be recounted exactly.

## Dataset Guardrail

- Training split: not read.
- Validation split: not read.
- Test split: not read.

## Top Candidate Mappings

| Rank | Matrix Rank | Vector Expansion | Router Width | Feedback Rank | Total Error | Budget Error |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 168 | 384 | 20 | 7 | 5,609,240 | 120 |
| 2 | 200 | 320 | 24 | 7 | 3,385,682 | 216 |
| 3 | 216 | 288 | 24 | 9 | 7,391,594 | 240 |
| 4 | 216 | 288 | 28 | 1 | 7,391,594 | 240 |
| 5 | 168 | 392 | 8 | 3 | 5,911,240 | 552 |
| 6 | 184 | 360 | 8 | 8 | 1,979,140 | 612 |
| 7 | 192 | 344 | 12 | 2 | 355,894 | 636 |
| 8 | 192 | 344 | 8 | 10 | 650,998 | 636 |
| 9 | 208 | 312 | 12 | 5 | 4,398,682 | 672 |
| 10 | 208 | 312 | 8 | 13 | 4,693,786 | 672 |
| 11 | 216 | 296 | 12 | 6 | 6,401,638 | 684 |
| 12 | 176 | 352 | 48 | 9 | 3,643,856 | 816 |
| 13 | 176 | 352 | 52 | 1 | 3,643,856 | 816 |
| 14 | 168 | 376 | 32 | 11 | 5,610,344 | 1,224 |
| 15 | 168 | 376 | 36 | 3 | 5,610,344 | 1,224 |
| 16 | 192 | 328 | 36 | 9 | 2,377,610 | 1,296 |
| 17 | 192 | 328 | 40 | 1 | 2,377,610 | 1,296 |
| 18 | 216 | 280 | 36 | 12 | 8,386,478 | 1,332 |
| 19 | 216 | 280 | 40 | 4 | 8,386,478 | 1,332 |
| 20 | 208 | 304 | 24 | 8 | 5,391,710 | 2,844 |
| 21 | 160 | 400 | 16 | 12 | 7,572,212 | 2,988 |
| 22 | 160 | 400 | 20 | 4 | 7,572,212 | 2,988 |
| 23 | 168 | 368 | 48 | 7 | 5,612,984 | 3,864 |
| 24 | 184 | 336 | 48 | 11 | 3,044,962 | 3,912 |
| 25 | 184 | 336 | 52 | 3 | 3,044,962 | 3,912 |
| 26 | 216 | 272 | 52 | 7 | 9,379,826 | 3,960 |
| 27 | 160 | 376 | 60 | 8 | 7,570,916 | 4,284 |
| 28 | 208 | 296 | 36 | 11 | 6,389,666 | 4,824 |
| 29 | 208 | 296 | 40 | 3 | 6,389,666 | 4,824 |
| 30 | 200 | 312 | 36 | 10 | 4,386,710 | 4,836 |
| 31 | 200 | 312 | 40 | 2 | 4,386,710 | 4,836 |
| 32 | 184 | 344 | 36 | 7 | 2,020,882 | 4,872 |
| 33 | 176 | 360 | 32 | 13 | 3,638,144 | 4,896 |
| 34 | 176 | 360 | 36 | 5 | 3,638,144 | 4,896 |
| 35 | 160 | 392 | 32 | 8 | 7,570,244 | 4,956 |
| 36 | 208 | 288 | 52 | 6 | 7,386,086 | 5,268 |
| 37 | 192 | 320 | 48 | 12 | 3,380,174 | 5,292 |
| 38 | 192 | 320 | 52 | 4 | 3,380,174 | 5,292 |
| 39 | 160 | 384 | 44 | 12 | 7,569,812 | 5,388 |
| 40 | 160 | 384 | 48 | 4 | 7,569,812 | 5,388 |
| 41 | 184 | 352 | 20 | 12 | 1,683,284 | 6,324 |
| 42 | 184 | 352 | 24 | 4 | 1,683,284 | 6,324 |
| 43 | 176 | 376 | 8 | 6 | 3,951,340 | 6,732 |
| 44 | 200 | 328 | 12 | 4 | 2,389,582 | 6,804 |
| 45 | 200 | 328 | 8 | 12 | 2,684,686 | 6,804 |
| 46 | 224 | 280 | 12 | 7 | 8,398,450 | 6,840 |
| 47 | 168 | 360 | 60 | 11 | 5,617,160 | 8,040 |
| 48 | 176 | 344 | 60 | 13 | 3,999,898 | 8,064 |
| 49 | 200 | 304 | 48 | 13 | 5,386,202 | 8,352 |
| 50 | 200 | 304 | 52 | 5 | 5,386,202 | 8,352 |
