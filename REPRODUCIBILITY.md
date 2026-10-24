# Reproduce Modus_X 3

At repository root, `python tools/validate_release.py` validates the canonical
versioned package and `python language/run_modus_x3.py smoke` runs its reduced
checks. You may also `cd modus_x_3.0.0` and use the package-relative instructions
below. Historical root trainers are not substituted for the frozen v3 runners.

## Inspect and Validate Without Training

Use Python with JAX, NumPy and Optax. The tested TPU environment is recorded in
ENVIRONMENTS.md; do not install CPU wheels over an initialized Kaggle TPU runtime.

From the package root:

```sh
python tools/validate_release.py
cd research/v3/05_local_kv_filtering/src
python -m unittest test_endpoint_recovery test_endpoint_audit test_local_kv -v
```

These are reduced CPU checks, not a throughput measurement or a new language
rerun. They cover identity initialization, lag gradients, causal prefixes,
streaming history, complete-objective gradients, AdamW/LR behavior, source
discovery, corruption rejection and audit-head selection.

## Reproduce a Frozen Language Endpoint

Use a **fresh Kaggle TPU v5e-8 notebook**, not T4 GPU. Enable Internet or attach
the verified raw enwik8 file or ZIP. Its uncompressed size must be 100,000,000
bytes and SHA-256 must be:

```text
2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8
```

The self-contained cells are in
`research/v3/05_local_kv_filtering/src/cells/`:

| Cell | Meaning |
|---|---|
| `KAGGLE_SEED3_FROZEN_ENDPOINT.py` | Fresh paired seed 3, fixed 25k horizon; no early quality stop |
| `KAGGLE_SEED2_FRESH_FIXED_HORIZON.py` | Fresh seed 2 follow-up, including the preserved failed early screen |
| `KAGGLE_SEED1_RESUME_25K.py` | Resume the original paired 5k seed-1 artifacts; requires both original checkpoints |
| `KAGGLE_SEED3_ENDPOINT_AUDIT.py` | Read-only audit of attached full endpoint output; seed selector documented in the cell |

Copy an entire cell into the notebook, not a Markdown-rendered fragment. Save
and Run All preserves output files. Do not run a second JAX subprocess after
the notebook owns TPU. Each self-contained cell has its source payload; do not
mix arbitrary files from other versions into that payload.

The readable runner for the prospective replication is
`src/run_seed3_frozen_endpoint.py`. `run_endpoint.py` retains the earlier
screen-and-stop behavior and is not a substitute for the fixed-horizon runner.

## Recovery and Audits

Keep **both arms** from the same `checkpoints/step_25000` generation, their
recovery JSONs and pair manifest, plus complete result JSON and notebook version.
Resume requires parameters, optimizer moments/count and data RNG, not weights
alone. Compact result ZIPs intentionally omit the large pickle payloads.
Discovery accepts unpacked notebook outputs and ZIPs and deduplicates complete
pairs by hash. Never load pickle files from an untrusted source.

The audit checks complete-window dense validation at offsets 0/256, primary-head
selection, serial/chunk32 agreement, suffix causality and state handoff. Its
65,536-target carry probe is diagnostic, not the five-million-byte dense metric.

## The Frozen Comparison

Batch 8; input length 512; 25,000 updates; 102.4M input characters per arm;
seed-paired initial shared parameters and identical data batches; archive scale
512 in both arms; two-stage chunk32 in both; highest matmul precision; FP32
backbone; AdamW weight decay 1e-4 and global norm clipping 1. LR is 6e-4 through
20k and 3e-4 through 25k, preserving optimizer state. Primary next-byte loss,
layer-6 auxiliary loss weight 0.05, and separate two-step-future head weight 0.5.
No future inputs are exposed to the model.

Train is bytes 0:90M; validation 90M:95M; test 95M:100M. Primary-head dense
validation decides at the frozen endpoint; dense test reports afterward.
Context resets per 512-input window; stride 512 at offsets 0 and 256. Do not
compare sparse progress validation to dense or carry evaluation.

No long TPU job is launched by package validation. Large checkpoints and a
public storage index remain separate artifacts; see docs/RELEASE_REVIEW.md.

## Rebuild the Paper Figures

Install `requirements-paper.txt`, then run `python tools/rebuild_evidence.py`
from the package root. This regenerates paired statistics, the endpoint table
and three figure PDFs from the recorded results. It does not run training or
change the frozen experiment records. Compile `paper/manuscript/main.tex` with
Tectonic or a compatible LaTeX distribution. A flattened paper-only source ZIP
is prepared separately for arXiv; do not upload the entire research-code ZIP as
the paper source.
