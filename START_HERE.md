# Start Here

This page is the shortest safe path from a fresh clone to contributing to
Modus_X research. You do not need to understand every release or experiment
before running the project.

## What Modus_X is

Modus_X studies recurrent language models with bounded matrix memory. The
current release has two separate research leads:

- **MemoryFeedbackArchive** retrieves information from matrix memory and feeds
  a compressed, gated signal into the vector recurrence. It is the primary
  language-modeling architecture.
- **CurrentArchiveDelta** separates rapidly changing current memory from slower
  archive memory. It is the primary controlled-memory architecture.

These mechanisms have not yet been combined into one proven model. Modus_X
does not currently beat every baseline, reach 1.1 enwik8 BPC, or demonstrate a
trained one-billion-parameter model. Read [CLAIMS.md](CLAIMS.md) and
[LIMITATIONS.md](LIMITATIONS.md) before describing results externally.

## Choose your path

### 1. Understand the project

Read these files in order:

1. [README.md](README.md), for the release-level summary.
2. [docs/architecture.md](docs/architecture.md), for the two v2 mechanisms.
3. [CLAIMS.md](CLAIMS.md), for statements supported by evidence.
4. [LIMITATIONS.md](LIMITATIONS.md), for statements the evidence does not support.
5. [ESSENTIAL_FILES.md](ESSENTIAL_FILES.md), when you need the full evidence map.

The complete paper is available as
[paper/Modus_X_2.1.0_whitepaper.pdf](paper/Modus_X_2.1.0_whitepaper.pdf). It is
useful context, but it is not required before the first local test.

### 2. Verify the code on your computer

Use Python 3.12 from the repository root. A virtual environment is strongly
recommended.

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python language/test_memory_feedback_archive.py
```

Linux or macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python language/test_memory_feedback_archive.py
```

The first JAX compilation can take a few minutes. A successful run ends with a
dictionary containing `"status": "PASS"`. This test checks a small
MemoryFeedbackArchive forward pass, gradients, causality, parameter accounting,
and feedback diagnostics. It does not reproduce a published BPC result.

### 3. Verify real training on a Kaggle TPU

Create a fresh Kaggle notebook, enable **TPU VM v5e-8**, enable Internet access,
and run this single cell:

```python
import pathlib, subprocess, sys

repo = pathlib.Path('/kaggle/working/Modus_X')
if not repo.exists():
    subprocess.run([
        'git', 'clone', '--depth', '1',
        'https://github.com/sanyamChaudhary27/Modus_X.git',
        str(repo),
    ], check=True)

subprocess.run([
    sys.executable, '-u', str(repo / 'onboarding/kaggle_tpu_smoke.py')
], check=True, cwd=repo)
```

The launcher downloads and verifies the canonical 100,000,000-byte enwik8
file, checks that JAX sees a TPU, and performs 100 optimizer updates using the
real approximately 47M-parameter MemoryFeedbackArchive model. It evaluates a
small validation sample, saves a resumable checkpoint under
`/kaggle/working/modus_x_onboarding_smoke`, and deliberately does not evaluate
the test split.

Expected completion markers include:

```text
ONBOARDING_DATA_READY
ONBOARDING_RUN
CHECKPOINT {"step": 100, ...}
ONBOARDING_TPU_SMOKE_COMPLETE
```

This is an accelerator integration test, not a benchmark result. Do not report
its validation BPC as a model comparison.

## Reproducing published evidence

Do not begin with a long training script. First choose one exact claim, then
match its model, seed, parameter count, processed characters, optimizer,
precision, data split, and evaluation protocol.

- Language-model evidence: [evidence/language](evidence/language)
- Controlled-memory evidence: [evidence/memory](evidence/memory)
- Unsuccessful candidates: [evidence/negative](evidence/negative)
- Benchmark rules: [docs/BENCHMARK_PROTOCOL.md](docs/BENCHMARK_PROTOCOL.md)
- Environment and artifact details: [REPRODUCIBILITY.md](REPRODUCIBILITY.md)

The repository does not include large trained checkpoints. Ask the experiment
owner for the checkpoint, its SHA-256 hash, and the frozen protocol before
attempting an endpoint reproduction.

## Starting a research contribution

Before writing code, create a short experiment note containing:

1. The question being tested.
2. One architectural or training change.
3. The unchanged control.
4. Parameter and recurrent-state accounting.
5. The validation metric and promotion threshold chosen before the run.
6. The compute budget and stopping rule.

Run local correctness checks before requesting TPU time. Use validation for
selection and read test metrics only after freezing the endpoint. Keep failed
experiments and their provenance. Do not silently tune a failed candidate or
compare sparse BPC with dense BPC.

For a first contribution, ask the project lead to assign one bounded task. A
good first task is reproducing an existing CPU test, validating an evidence
file, or implementing a pre-registered single-change smoke. Avoid starting with
the 1B systems scripts or an unreviewed architecture combination.

## Where files belong

- `language/`: language models, TPU trainer, and evaluation tools.
- `memory/`: controlled-memory experiments.
- `evidence/`: result records and raw evidence.
- `docs/`: architecture, benchmark, provenance, and reproducibility documents.
- `systems/`: large-model feasibility work, not the beginner entry point.
- `Modus_X_2.1.0/`: frozen release snapshot. Do not edit it for new research.

If a command, checkpoint, or protocol is unclear, stop and ask before spending
accelerator time. Include the command you ran, the last complete output block,
the active accelerator, and the relevant file path.
