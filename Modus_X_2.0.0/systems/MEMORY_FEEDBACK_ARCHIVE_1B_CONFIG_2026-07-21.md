# MemoryFeedbackArchive 1B Replacement Configuration

Date: 2026-07-21

## Status

This is a **candidate configuration for systems validation**, not a frozen
training configuration and not evidence that a 1B MemoryFeedbackArchive model
will outperform the published architecture.

## Matched Configuration

| Field | Frozen 1B | MFA candidate |
|---|---:|---:|
| Vocabulary | 50,257 | 50,257 |
| Embedding width | 1,536 | 1,536 |
| LM-head hidden width | 4,608 | 4,608 |
| Layers | 32 | 32 |
| Matrix state width | 1,024 | 896 |
| Vector state width | 1,024 | 960 |
| Current/archive matrices | 1 | 2 |
| Feedback rank | 0 | 32 |
| Parameters | 1,058,963,121 | **1,058,467,601** |

The candidate is 495,520 parameters (`0.0468%`) smaller than the frozen 1B
configuration. It preserves model width, depth, vocabulary, output head, and
router width. Capacity is reallocated from state width to the second bounded
archive and matrix-to-vector bridge.

## Explicit Cost

The BF16 recurrent state per sequence increases from approximately
`67.17 MB` to `102.82 MB`, a ratio of `1.5307x`. This remains constant in
sequence length, but constant does not mean free. Current/archive updates also
increase matrix work, and the rank-32 bridge adds a small projection cost.

The existing `539.19 tokens/s` and `10.68 GB/device` measurements do not apply
to this candidate. They belong to the one-matrix frozen configuration.

## Readiness Ladder

1. Catalog the exact stacked parameter tree and TPU partition specs.
2. Run reduced FP32/BF16 forward/backward parity for current, archive, and
   feedback operations.
3. Execute two exact AdamW updates at context 2,048 on the `4 data x 2 model`
   v5e-8 topology.
4. Measure per-device peak memory and steady-state throughput.
5. Validate four 8,192-token real-data microsteps, FP32 gradient accumulation,
   one 32,768-token update, Orbax save, restore, and the exact next batch.

Abort or redesign if:

- the exact model does not fit with at least 1 GB/device operating margin;
- throughput falls below 450 tokens/s without an accuracy-preserving recovery;
- archive or feedback operations lose reduced-precision parity;
- checkpoint/restore does not reproduce cursor and update state exactly.

Only after this ladder passes should the MFA configuration replace the frozen
1B proposal architecture.

Reproducible count:

```powershell
python proposals/1B_scaling/tools/count_memory_feedback_archive_parameters.py
```
