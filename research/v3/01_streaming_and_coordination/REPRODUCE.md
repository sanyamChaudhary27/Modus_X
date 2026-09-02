# Reproduction

Use Python 3.12, JAX, Optax, and a Kaggle TPU v5e-8. Attach canonical enwik8
and the matching 47,437,768-parameter checkpoint. The scripts require the
checkpoint's `config.json` and `progress.json` beside `checkpoint.pkl`.

Core entry points:

```text
src/run_contiguous_training_screen.py
src/run_state_component_attribution.py
src/run_counterfactual_operation_audit.py
```

Each entry point exposes `--help`. Preserve seed, checkpoint step, split, and
512-byte segment length from `PROTOCOL.json`. Do not substitute sparse test BPC
for the validation-only selection metrics.

The original long-form reproduction instructions, exact frozen configuration,
dataset hash, direct command, output schema, and environment caveat are kept in
`references/Modus_X_Contiguous_Training_Reproducibility_README.md`. It is an
interim two-seed snapshot; use `references/FINAL_REPLICATION_RESULT_2026-08-24.md`
for the final decision.
