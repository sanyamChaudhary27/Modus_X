# Reproduction

`src/build_endpoint_cell.py` builds the self-contained seed-1 Kaggle cell.
`src/build_adaptive_seed2_endpoint_cell.py` derives the frozen seed-2 cell.
All source dependencies used by the builder are kept beside it in `src/`.

Use Python 3.12 to generate the cell, attach the matching 20.48M-character
candidate checkpoint to a fresh Kaggle TPU v5e-8 notebook, and execute the
generated cell. Keep the explicit constant-LR stages; do not replace the
optimizer with a scheduled optimizer while restoring an old constant-LR state.

Dense evaluation must cover validation offsets 0 and 256 with stride 512.
Sparse checkpoint validation is a progress signal, not the endpoint metric.

## Frozen closure audit

For `src/generated_cells/KAGGLE_TPU_V3_CLOSURE.py`, start a fresh Kaggle TPU
v5e-8 notebook and attach exactly these completed seed-1 endpoint outputs:

- canonical `Modus_X_MemoryFeedbackArchive_DeepSupervision`, step 25,000,
  102.4M processed characters, 47,437,768 parameters;
- `Modus_X_MemoryFeedbackSegmentRetention_DeepSupervision`, step 25,000,
  102.4M processed characters, 47,437,768 parameters.

Kaggle may expose either notebook output as unpacked directories. The launcher
discovers `checkpoint.pkl`, `config.json`, and `progress.json` together and
validates model, seed, step, characters, and parameter count before loading the
checkpoint. It runs the seed-1 over-retention replication first, then the
paired coordination audit. The expected compact output is
`/kaggle/working/memory_feedback_v3_closure_results.zip`.
