# Artifact gaps before final publication

This source/evidence package is a release candidate. Complete these items
before treating it as the final archival release:

1. Preserve the compact 81M and 99M MemoryFeedback scaling archives, including
   configs, progress records, and dense audit JSON.
2. Import the two pending checkpoint-annealing archives only after both runs
   complete and pass integrity checks; otherwise document them as incomplete.
3. Recover the dense 47M MemoryFeedback and CurrentArchive audit JSON files from
   their checkpoint archives, or publish the checkpoint archives separately
   with SHA256 hashes.
4. Record the exact Kaggle TPU Python, JAX, JAXlib, NumPy, and Optax versions
   from a rerun environment.
5. Split promoted implementations from rejected research variants before
   presenting `src/` as a stable public API.

The latest-heavy archive, mixed clean/update evidence, equal-memory rows,
operation diagnostics, and measured scaling summaries are included.
