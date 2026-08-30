# Modus_X MemoryFeedbackArchive contiguous-training experiment

Reproducibility bundle, 2026-08-23  
Architecture: `Modus_X_MemoryFeedbackArchive_DeepSupervision`  
Scale: `47,437,768` parameters  
Status: two of three preregistered seeds complete; seed 3 pending

## 1. Research question

Previous evaluation showed that a frozen MemoryFeedbackArchive checkpoint
benefited when its recurrent state was carried across contiguous 512-byte
segments. This experiment asks the causal follow-up:

> Does training with persistent state teach the model to use cross-segment
> memory better than a model trained on the identical contiguous bytes while
> resetting state at every segment?

The experiment changes only recurrent-state handling. It does not change the
architecture, loss, optimizer, number of processed characters, data streams,
initial weights, or validation split within a seed.

## 2. Architecture under test

The tested model is the measured Modus_X v2 `MemoryFeedbackArchive` language
model. Each recurrent layer maintains three bounded states:

- a current matrix state;
- an archive matrix state;
- a vector recurrent state.

Matrix retrieval is fed back into the recurrent computation before the next
state update. The model also retains its direct matrix/vector fusion path. The
experiment does not introduce a new architecture. It changes whether all
three recurrent states are reset or carried between training segments.

The exact implementation is in `src/models.py`. It was recovered
byte-for-byte from the measured Modus_X v2 release archive. Its SHA-256 is:

```text
8536f19e61563fa0c71ffb687200b966f5315a383ae2d45e6eaf363e403a7a37
```

The source release ZIP had SHA-256:

```text
03fd868a805ad4c74e839445760d6013a245cb93a95da5c65df42e9c330cadd5
```

## 3. Experimental design

Two models are initialized identically for each seed and receive identical
ordered byte streams.

### Reset-trained control

`reset_contiguous` resets the current matrix, archive matrix, and vector state
before every 512-byte training segment.

### Carry-trained candidate

`carry_contiguous` carries all three states between adjacent 512-byte
segments. State is passed through `jax.lax.stop_gradient` at each boundary,
so gradients do not backpropagate across segments. State is reset after every
1,250 segments, giving a maximum carry horizon of 640,000 bytes per lane.

### Matched 2x2 evaluation

Each trained checkpoint is evaluated twice on contiguous validation streams:

| Training condition | Evaluation condition | Purpose |
|---|---|---|
| reset | reset | cold-state baseline |
| reset | carry | inference-only carry benefit |
| carry | reset | cost of relying on unavailable history |
| carry | carry | stateful training plus stateful inference |

This design separates three effects:

1. contiguous training data order;
2. carrying state at inference;
3. learning under persistent state.

The primary quantity is:

```text
carry_eval_gain =
  BPC(reset-trained, carry-eval)
  - BPC(carry-trained, carry-eval)
```

The interaction quantity is:

```text
carry_training_interaction =
  [BPC(carry-trained, reset-eval) - BPC(carry-trained, carry-eval)]
  - [BPC(reset-trained, reset-eval) - BPC(reset-trained, carry-eval)]
```

A positive interaction means persistent-state training increased the value of
persistent-state evaluation beyond the inference-only carry effect.

## 4. Frozen configuration

| Field | Value |
|---|---:|
| Parameters | 47,437,768 |
| Vocabulary | 256 bytes |
| Layers | 12 |
| Embedding width | 512 |
| Hidden width | 1,536 |
| Matrix/vector state width | 512 |
| Training segment length | 512 bytes |
| Batch | 8 contiguous lanes |
| Updates per condition | 5,000 |
| Processed targets per condition | 20,480,000 |
| Optimizer | AdamW |
| Learning rate | 0.0006, constant |
| Weight decay | 0.0001 |
| Global gradient clipping | 1.0 |
| Auxiliary layer | 6 |
| Auxiliary loss weight | 0.05 |
| Future target offset | 2 |
| Future target weight | 0.5 |
| Episode reset interval | 1,250 segments |
| Validation bytes | enwik8 90,000,000:95,000,000 |
| Test bytes | unread |

Both conditions together process 40.96M targets per seed. Parameter trees are
replicated across the eight TPU devices; the eight data lanes and recurrent
states are sharded over the data axis.

## 5. Dataset

The runner expects the canonical 100,000,000-byte `enwik8` file from:

```text
https://mattmahoney.net/dc/enwik8.zip
```

Expected uncompressed size:

```text
100000000 bytes
```

Expected SHA-256 used by the broader Modus_X enwik8 evidence suite:

```text
2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8
```

The runner records the actual dataset hash, lane starts, ordered-stream hash,
parameter-tree hash, device list, and validation range in `provenance.json`.

## 6. Current results

### Seed 1

| Training | Evaluation | Validation BPC |
|---|---|---:|
| reset | reset | 1.910136 |
| reset | carry | 1.891057 |
| carry | reset | 1.915632 |
| carry | carry | **1.871272** |

- primary carry-evaluation gain: `+0.019785 BPC`;
- reset-evaluation degradation: `+0.005496 BPC`;
- carry-training interaction: `+0.025282 BPC`;
- runtime ratio: `0.997269`;
- individual preregistered gate: **pass**.

### Seed 2

| Training | Evaluation | Validation BPC |
|---|---|---:|
| reset | reset | 1.904218 |
| reset | carry | **1.880927** |
| carry | reset | 1.934171 |
| carry | carry | 1.883405 |

- primary carry-evaluation gain: `-0.002478 BPC`;
- reset-evaluation degradation: `+0.029953 BPC`;
- carry-training interaction: `+0.027476 BPC`;
- runtime ratio: `0.996918`;
- individual preregistered gate: **fail**.

Seed 2 confirms that carry-trained weights use carried state strongly: their
carry benefit was `0.050767 BPC`, compared with `0.023291 BPC` for the reset
control. It does not confirm that this dependence improves the final
carry-evaluated BPC, because the reset-trained control remained better by
`0.002478 BPC` on that seed.

### Interim interpretation

Across the first two seeds, the mechanism-level interaction is consistent and
positive, but the end-quality improvement is not yet stable. This supports
the statement that detached persistent-state training changes how the model
uses cross-segment state. It does not yet support a replicated claim that it
improves validation BPC.

Absolute BPC in this experiment is not directly comparable with historical
random-window Modus_X runs. The data order and coverage protocol are different
because this experiment must preserve contiguous streams to isolate state
carry. All comparisons in this report are paired within the frozen protocol.

## 7. Preregistered decision rules

The seed-1 screen was promoted only if all of the following held:

- carry-evaluated gain at least `0.010 BPC`;
- reset-evaluation degradation at most `0.010 BPC`;
- carry-training interaction at least `0.005 BPC`;
- runtime overhead at most `15%`;
- all states and losses finite.

Seed 1 passed, so seeds 2 and 3 were frozen as exact replications. The
three-seed aggregate passes only if:

- at least two seeds pass the individual gate;
- mean carry-evaluation gain is at least `0.010 BPC`;
- mean reset-evaluation degradation is at most `0.010 BPC`;
- mean carry-training interaction is at least `0.005 BPC`;
- no seed has carry-evaluation gain below `-0.005 BPC`;
- mean runtime ratio is at most `1.15`;
- every state remains finite.

These thresholds are stored in `protocol/REPLICATION_PROTOCOL.json`. They
must not be changed after seeing seed 3.

## 8. Fastest reproduction: Kaggle TPU v5e-8

1. Create a fresh Kaggle notebook with a TPU v5e-8 accelerator.
2. Keep Internet enabled if `enwik8` is not attached as a dataset.
3. Paste one launcher from `kaggle_cells/` into the first cell.
4. Do not import JAX in another cell before starting the launcher. The runner
   intentionally owns the TPU from a child process.
5. Run the cell and retain both checkpoints until completion or resume.
6. Download the compact result ZIP printed at the end. It excludes the large
   checkpoint files but contains provenance, metrics, and decisions.

The three launchers differ only by the frozen seed. Running seeds in parallel
requires separate TPU notebooks.

## 9. Direct runner command

After extracting this bundle in a fresh TPU environment:

```bash
python -u run_contiguous_training_screen.py \
  --data-path /kaggle/working/enwik8 \
  --outdir /kaggle/working/contiguous_training_seed_1 \
  --seed 1 \
  --target-characters 20480000 \
  --checkpoint-every 1000 \
  --reset-interval 1250 \
  --resume
```

Change only `--seed` and `--outdir` for seeds 2 and 3. The runner rejects the
wrong TPU topology, data size, parameter count, target-character budget, or
reset interval.

## 10. Local CPU correctness test

The full experiment requires TPU v5e-8. The small parity test can run locally:

```bash
python -m pip install numpy "jax[cpu]" optax
python tests/test_stateful_training.py
```

Expected terminal marker:

```text
CONTIGUOUS_TRAINING_CPU_PARITY_PASS
```

The test verifies that zero-state execution matches the canonical forward
path, losses and gradients match, a real AdamW update changes parameters, and
carried states remain finite.

## 11. Output files

Each seed output contains:

- `provenance.json`: dataset, stream, parameter and device provenance;
- `parity.json`: canonical-versus-stateful zero-state parity check;
- `reset_contiguous/` and `carry_contiguous/`: resumable training outputs;
- four evaluation JSON files for the 2x2 design;
- `contiguous_training_screen.json`: complete structured report;
- `decision.json`: preregistered decision only.

Raw terminal excerpts and normalized summaries from completed seeds are in
`results/`. Checkpoints are intentionally excluded from this small sharing
bundle because each full training checkpoint is hundreds of megabytes.

## 12. Environment caveat

The original Kaggle runs used the platform-provided JAX, jaxlib, NumPy and
Optax environment on TPU v5e-8. Exact package versions were not printed in the
captured tail logs, so this bundle does not invent version pins. Reproduction
should report package versions from its own runtime and rely on the recorded
hashes, parity gate, parameter count, stream hash and numerical outputs to
detect drift.

## 13. Claims and limitations

Supported now:

- persistent inference state improves contiguous validation BPC for both
  trained models in both completed seeds;
- persistent-state training increases dependence on persistent state in both
  completed seeds;
- the end-quality advantage of persistent-state training is seed-sensitive.

Not supported now:

- a replicated BPC improvement from persistent-state training;
- superiority over Transformer, Mamba, xLSTM or other architectures;
- equivalence to standard random-window enwik8 BPC;
- generalization to natural-language tokenization or long-document tasks;
- claims about the enwik8 test split, which was not read.

## 14. File integrity

`MANIFEST.json` lists the SHA-256 and byte size of every file in the bundle.
Use `verify_bundle.py` from the bundle root:

```bash
python verify_bundle.py
```

Expected marker:

```text
REPRODUCIBILITY_BUNDLE_VERIFIED
```

