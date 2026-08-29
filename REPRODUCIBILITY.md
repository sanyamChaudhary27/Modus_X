# Reproducibility

## Environment

The canonical experiments use Python, JAX, NumPy, and Optax. TPU experiments
were run on Kaggle TPU v5e-8 sessions. Controlled-memory CPU smokes use CPU
JAX; full gates use the same JAX model on TPU.

Install the minimum package set:

```bash
python -m pip install -r requirements.txt
```

The exact locally validated versions are recorded in `ENVIRONMENTS.md`.
Kaggle images may provide newer compatible JAX builds. Record the exact JAX,
JAXlib, backend, device list, and precision for every rerun.

## Language result

Primary files:

- `language/models.py`
- `language/tpu_lm_train.py`
- `language/run_memory_feedback_archive_smoke.py`
- `language/audit_current_archive_checkpoint.py`
- `language/audit_displaced_archive_gates.py`

The canonical MemoryFeedback result used:

- future target `2`;
- AdamW;
- fixed enwik8 split;
- `512` input sequence length;
- equal processed-character checkpoints;
- sparse checkpoint validation for progress;
- dense validation and dense test for conclusions.

Do not compare only the sparse endpoint with a dense baseline.

The v2 scaling evidence is under `evidence/language/scaling/`. The 81M and 99M
records include the exact dataset SHA-256 and dense protocol. The 47M point
uses a different LR schedule; reproduce each frozen configuration before
attempting a fitted scaling law.

## Controlled memory

Primary files:

- `memory/run_versioned_memory_ablation.py`
- `memory/run_equal_memory_frontier.py`
- `memory/run_equal_memory_mixed_update_gate.py`
- `memory/run_latest_heavy_curriculum_gate.py`
- `memory/run_current_archive_operation_diagnostics.py`
- `memory/run_current_archive_latest_shadow_gate.py`

Promoted small synthetic claims require three seeds, validation-only checkpoint
selection, fixed clean/update and role distributions, and explicit parameter
and recurrent-state accounting.

## Integrity

`MANIFEST.sha256` records every packaged file except the manifest itself.
Verify on Linux:

```bash
sha256sum -c MANIFEST.sha256
```

On PowerShell, recompute files with `Get-FileHash -Algorithm SHA256`.

## Missing artifacts

Large training checkpoints are intentionally not embedded in this compact
source/evidence package. The result documents identify the checkpoint step and
protocol. External sharing should pair this package with durable checkpoint
storage before claiming full artifact reproducibility.

The 122.88M-character anneal reports and promoted 163.84M endpoint are included
as normalized evidence. Large checkpoints remain external and must be paired
with hashes before claiming full artifact reproducibility.
