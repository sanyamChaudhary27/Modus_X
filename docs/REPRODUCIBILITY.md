# Reproducibility

## Environments

Record exact Python, framework, CUDA/JAX, accelerator, package, and source
commit versions in each run artifact. Do not rely only on this overview.

## Language Modeling

- Dataset: enwik8.
- Split: first `90M` bytes train, next `5M` validation, final `5M` test.
- Context: `512`.
- Dense audit: reset state for every window; evaluate offsets `0` and `256`
  with stride `512`.
- Matched 80M comparison: `40,000` updates, global batch `8`, and `163.84M`
  supervised characters.

## Associative Memory

- Protocol: balanced continuous key-value recall recovered from the older
  Modus experiment.
- Training length: `128`.
- Keys per sequence: `32`.
- Values: `32`; chance accuracy is `3.125%`.
- Standard training: `40,000` examples, `12` epochs.
- Overwrite evaluation: regenerate the dataset with overwrite rate `0.5`.
- Length evaluation must stream bounded batches instead of materializing the
  full long-context dataset.

## Commands

Modus_X:

```bash
python benchmarks/modus_x/balanced_kv.py \
  --models VectorLeanPM \
  --seed 17 \
  --epochs 12
```

Official Mamba recall:

```bash
python benchmarks/official_baselines/official_mamba_balanced_kv.py \
  --outdir evidence/associative_memory/mamba_no_overwrite_seed17 \
  --seed 17 \
  --overwrite-rate 0.0
```

```bash
python benchmarks/official_baselines/official_mamba_balanced_kv.py \
  --outdir evidence/associative_memory/mamba_overwrite_0p5_seed17 \
  --seed 17 \
  --overwrite-rate 0.5
```

## Missing Before Release

- source commit hashes;
- environment lock files for JAX and CUDA lanes;
- raw official Mamba recall JSON and checkpoints;
- raw 80M Modus_X audit JSON;
- raw official xLSTM and Mamba dense audit JSON;
- hashes for every promoted artifact.

