# Modus_X v1.1.1 Release Manifest

This manifest identifies the reproducible v1.1.1 release artifacts.

| Path | Type | SHA-256 | Status |
|---|---|---|---|
| `paper/whitepaper.md` | paper source | `F4710C97B4F1B8748683E7005CFC98532C71B0ADD4FA16D157E00B3652968F31` | built |
| `paper/whitepaper.pdf` | publication PDF | `07FBF76601C5001B9D7119EC6B17E99AB1CD1C72BA4D067BF55C14700BB3E1DC` | rebuilt; first-page render inspected |
| `release/Modus_X_v1.1.1_whitepaper.pdf` | release PDF | `07FBF76601C5001B9D7119EC6B17E99AB1CD1C72BA4D067BF55C14700BB3E1DC` | included |
| `figures/generate_figures.py` | figure source | `07D73105EFCEEA07E71EA1FFFE40B77B174AD0585D951B32A51AE9ABC8C23B2B` | included |
| `figures/component_ablation.png` | measured aggregate figure | `7C0ED058C1BE9835C73A51DCFE2890147B5DB31A7176128D665558AD65E50351` | included |
| `evidence/associative_memory/component_ablation_2026-06-26/aggregate_summary.json` | three-seed aggregate | canonical archive summary | included |
| `evidence/associative_memory/component_ablation_2026-06-26/raw_archive/modus_x_component_ablation_v2.zip` | raw Kaggle output | `23E7ADDC26464E42D9E80B1039117C0B86A1A380B290562B2EB368DFCA0E7668` | included; 24 results verified |
| `release/Modus_X_v1.1.1_release.zip` | release package | recorded externally after packaging | validated archive; excludes itself to avoid a circular hash |
