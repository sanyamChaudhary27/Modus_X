# Reproducibility

The canonical reproduction guide is `../REPRODUCIBILITY.md`.

Start with:

1. `../ESSENTIAL_FILES.md`;
2. `../EVIDENCE_INDEX.md`;
3. `BENCHMARK_PROTOCOL.md`;
4. the result document beside the experiment being reproduced.

Validate package structure and hashes with the scripts under `../release/`.
Kaggle packages must be self-contained and Linux-safe. Do not initialize JAX
TPU in both a launcher and its subprocess.

