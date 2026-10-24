# Reproduce

Use a fresh Kaggle TPU v5e-8 session with Internet enabled. Self-contained cells
are in src/cells/. They include their exact source snapshots; do not mix them
with arbitrary modules from another branch. They validate enwik8 bytes/SHA256
and run correctness checks before training. Attach raw enwik8 or enwik8.zip if
the network source is unavailable. Do not run these TPU jobs on a T4 notebook.

- KAGGLE_SEED1_RESUME_25K.py needs both original seed1 5k checkpoints and recovery
  manifests. They are external, not bundled. This cell does not start fresh.
- KAGGLE_SEED2_FRESH_FIXED_HORIZON.py starts both seed2 arms fresh through25k.
- KAGGLE_SEED3_FROZEN_ENDPOINT.py starts both seed3 arms fresh through25k.
- KAGGLE_SEED3_ENDPOINT_AUDIT.py reads attached full25k notebook outputs.
  Change its top-level SEED to1 or2 for those reported endpoints.

Use Save and Run All and preserve full notebook outputs. Compact ZIPs exclude
weights; paired checkpoints and recovery manifests are required for restore.
All TPU execution stays in the notebook process. Training data RNG and AdamW
moments/count are preserved by resume. Checkpoints are saved every1000 updates.

Local reduced checks from src/: `python -m unittest test_endpoint_recovery
test_endpoint_audit -v` (one line). Full model correctness is additionally in
test_local_kv.py. CPU checks do not certify TPU throughput.

Readable src/ contains the source used by these experiments. Cells are frozen
historical snapshots and can differ in launcher/downloader behavior; training
equations are unchanged. No new long TPU run was performed during packaging.
