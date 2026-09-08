# Reproduce the completed screen

## Independent-seed replication

Run `src/KAGGLE_REPLICATION_CELL.py` in a fresh eight-device Kaggle TPU
notebook with Internet enabled. Attach the ORIGINAL SEED-1 step-25,000 v3
SegmentRetention checkpoint and config.json. The previous seed-2 notebook
does not replace this input. The runner rejects other seeds and steps.

This cell performs 2,000 paired updates (8.192M added characters), dense
validation before/after, and a save/reload/next-update equality test at update
1,000. It writes complete endpoint checkpoints BEFORE final evaluation and
a checkpoint_manifest.json with their hashes. Preserve all notebook outputs;
the compact ZIP excludes these large checkpoint files. It does not currently
auto-resume an interrupted replication; do not substitute a partial endpoint
for the original checkpoint.

The completed seed-2 screen was saved by the user at Kaggle script version
348163508: https://www.kaggle.com/code/sanyamchoudhary27/notebook3cb7995c6e?scriptVersionId=348163508
This pointer is recorded, but its file inventory has not been independently
downloaded and verified here.

## Original 500-update screen

Use Kaggle TPU with eight devices and Internet enabled. Attach one original
47M v3 SegmentRetention trained notebook output with checkpoint.pkl and
config.json. Synthetic integration checkpoints are not compatible.

Run src/KAGGLE_REAL_DATA_CELL.py as a notebook cell. It embeds its frozen
sources, verifies package identity, downloads byte-identical enwik8 if absent,
and runs in the notebook process to avoid TPU ownership conflicts.
This cell reproduces the 500-update screen, NOT the planned 2000-update gate.

Preserve the notebook output directory, particularly real_data_result.json,
canonical_paired_checkpoint.pkl and two_stage_paired_checkpoint.pkl. The compact
ZIP excludes large checkpoints. Record SHA-256 for each and a durable download
location before claiming complete artifact reproducibility.

src/execution_backend.py provides a context-managed opt-in layer backend for
development. Create/trace separate JIT functions inside each context; JIT
executables already compiled elsewhere do not switch when the context changes.
The default is canonical v3 (segment-scale retention), not pre-v3 retention.

Local tests: python -m unittest discover -s src -p "test_*.py".
Generated code has syntax checks; training requires TPU. The integration hook
itself is not yet promoted as the default production trainer.
