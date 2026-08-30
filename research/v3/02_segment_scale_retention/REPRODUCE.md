# Reproduction

The implementation is in `src/segment_scale_trainer.py`. The source trace and
natural conflict interventions are in `src/run_source_trace.py` and
`src/run_over_retention_audit.py`. Their shared frozen data miner is
`src/run_natural_delayed_recall.py`.

Use a Kaggle TPU v5e-8 and canonical 100,000,000-byte enwik8. The published
over-retention result requires the seed-2 step-25,000 segment-retention
checkpoint with its original configuration and progress files. Run validation
bytes 90M:95M only. The script verifies backend, device count, parameter count,
checkpoint step, and dataset size before evaluation.
