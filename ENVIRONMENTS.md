# Environment snapshots

## Local release validation

- Python: `3.12.13`
- JAX: `0.10.2`
- JAXlib: `0.10.2`
- NumPy: `2.5.1`
- Optax: `0.2.8`
- Backend: CPU
- Validation: all packaged Python files compiled; promoted
  `test_memory_feedback_archive.py` passed.

## Kaggle TPU experiments

The experiments used Kaggle TPU v5e-8 sessions and the JAX packages available
in the corresponding Kaggle image. Historical logs preserve backend and device
lists but do not consistently preserve every package version. Exact TPU
package pinning is therefore an open archival item, not an inferred value.
