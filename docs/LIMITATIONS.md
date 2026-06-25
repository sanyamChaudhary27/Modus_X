# Limitations

1. enwik8 is a narrow byte-level benchmark, but it remains valid evidence of
   language-modeling quality. Modus_X currently loses to official Mamba on it.
2. The associative-memory task exposes explicit fact and query markers and
   uses task-aligned structure. It measures inductive bias, not complete
   language understanding.
3. Official Mamba recall evidence currently uses one main seed. Multi-seed
   confirmation is required.
4. Modus_X and external baselines use different frameworks and accelerators.
   Equal examples and updates support sample-efficiency comparisons, while
   wall-time comparisons require separate reporting.
5. Current Modus_X kernels are not optimized to the maturity of official
   Mamba CUDA kernels.
6. No 1B+ Modus_X language model has yet been trained.

