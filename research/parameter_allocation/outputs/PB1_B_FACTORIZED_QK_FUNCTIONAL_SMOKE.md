# PB1-B Factorized Q/K Functional Smoke Test

- Model: `Modus_X_MemoryFeedbackArchive_DeepSupervision`
- Canonical parameters: **47,437,768**
- Sequence length: `512`
- Seed: `20260911`
- Intervention: factorize only `m_wq` and `m_wk`.
- Official `make_model` and DeepSupervision forward remain untouched.

## Results

| Candidate | Hidden | Mamba | ax_res | Q | K | Factorized params | Budget error | Q recon err | K recon err | Final logits rel L2 | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| C1_q128_k128 | 1664 | 608 | 512 | 128 | 128 | 47,440,328 | +2,560 | 0.61362884 | 0.61382307 | 1.08748608 | PASS |
| C2_q64_k64 | 2048 | 640 | 512 | 64 | 64 | 47,441,864 | +4,096 | 0.78987367 | 0.78962360 | 1.16542258 | PASS |
| C3_q128_k64 | 1536 | 640 | 512 | 128 | 64 | 47,440,840 | +3,072 | 0.61362884 | 0.78962360 | 1.12025157 | PASS |

## Gate

PB1-B PASS requires:

1. canonical parameter count matches exactly;
2. official DeepSupervision returns four outputs;
3. every candidate matches its PB1-A factorized parameter count;
4. dense and factorized output shapes match;
5. all factorized outputs are finite.

The reconstruction wrapper is a smoke-test mechanism, not the final
efficient training implementation. The next training implementation
must compute Wq/Wk through the A/B factors directly.