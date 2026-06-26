# Router and Component Ablation

Run date: 2026-06-26

## Question

On the recovered balanced continuous key-value protocol, which Modus_X
component carries the observed associative-recall capability, and does the
lean vector router retain the scalar-router control's performance?

## Fixed Protocol

- Seeds: `17`, `27`, `37`.
- Train examples: `40,000`; test examples: `4,000`.
- Train length: `128`; evaluated lengths: `128`, `256`, `512`, `1024`, `2048`.
- Values/classes: `32`, making chance accuracy `3.125%`.
- Epochs: `12`; optimizer: AdamW with LR `3e-4` and weight decay `1e-4`.
- Conditions: no overwrite and `50%` same-key overwrite.

All variants use the same task generator, seed set, data counts, learning
rate, optimizer, epoch budget, and length sweep. `VectorLeanPM`,
`MatrixOnly`, and `VectorOnly` have `145,674` parameters. `ScalarPM` has
`156,584` parameters because its scalar router's expanded classifier was
configured independently; this difference is reported rather than hidden.

`MatrixOnly` and `VectorOnly` retain the same lean-vector parameter allocation
but force the final fused output to the selected stream. They are output-stream
interventions, not physically pruned models.

## Aggregate Result

At length 2048 without overwrite, ScalarPM, VectorLeanPM, and MatrixOnly reach
`96.383 +/- 0.496%`, `96.758 +/- 0.317%`, and `96.992 +/- 0.427%` respectively;
VectorOnly reaches `3.100 +/- 0.109%`, near chance. With 50% overwrite, the
same variants reach `88.108 +/- 0.447%`, `87.758 +/- 0.777%`, and
`87.625 +/- 0.745%`; VectorOnly reaches `3.308 +/- 0.506%`.

The protocol supports one narrow conclusion: the matrix stream is necessary
for the tested associative-recall task, while the vector-only output is not
sufficient. It does *not* establish a general advantage for either router,
natural-language recall, or universal architecture superiority.

## Artifact Status

`aggregate_summary.json` is the canonical `results/summary.json` copied
byte-for-byte from the Kaggle archive. The verified
`raw_archive/modus_x_component_ablation_v2.zip` contains all 24 expected
seed-level `results.json` artifacts: four variants, three seeds, and two
overwrite conditions. Its SHA-256 is
`23E7ADDC26464E42D9E80B1039117C0B86A1A380B290562B2EB368DFCA0E7668`.

This experiment is eligible for bounded release claims. The raw archive does
not remove the protocol caveats above.

## Reproduction

```bash
python run_component_ablation.py \
  --runner modus_x_router_balanced_kv.py \
  --outdir results \
  --seeds 17,27,37 \
  --epochs 12
```
