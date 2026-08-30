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
