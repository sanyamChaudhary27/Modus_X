# Full 1B Train-Step Gate

Date: 2026-07-17

## Result

The exact frozen `1,058,963,121`-parameter Modus_X configuration completed
initialization, forward pass, backward pass, and two AdamW updates on a Kaggle
TPU v5e-8. The run used context `2,048`, global microbatch `2`, BF16
parameters and residual-stream activations, FP32 recurrent states, optimizer
moments, and loss reductions, and a `2 data x 4 model` mesh.

This is a **full-model correctness and memory pass**, but a **throughput
no-go** under the pre-registered 96-hour systems-smoke criterion.

## Measurements

- parameter count: `1,058,963,121`;
- initialization: `940.51 s`;
- compile plus first update: `32.62 s`;
- measured update: `16.77 s` for `4,096` tokens;
- throughput: `244.29 tokens/s`;
- finite loss: `9.85246`;
- finite gradient norm: `9.85410`;
- layer-zero router mean: `0.50005`;
- router saturation: `0.0`;
- persistent memory after training step: approximately `2.73 GB/device`;
- device memory limit: approximately `16.91 GB/device`;
- projected 100M-token time: `113.71 h` on one v5e-8 slice.

The apparent notebook `SystemExit: 2` was intentional fail-closed behavior:
the model trained correctly, but the projected 100M-token time exceeded the
96-hour threshold.

## Decision

Do not rerun the same topology. Memory headroom is large, so the next systems
gate changes only the mesh and global microbatch:

- mesh: `4 data x 2 model`;
- global microbatch: `4` sequences;
- context: `2,048`;
- tokens per microstep: `8,192`;
- all scientific architecture and optimizer settings unchanged.

This configuration reduces model-axis communication and uses the otherwise
idle memory. Promote it only if it remains finite and projects 100M tokens
within 96 hours. If it OOMs, test `2 x 4` with activation-aware batching or
optimizer-state refinements; do not change the Modus_X architecture to repair
a systems throughput problem.

## 4 x 2 follow-up and measured bottleneck

The first `4 data x 2 model` compilation exceeded HBM by `2.59 GB`:

- required HBM: `18.34 GB` versus `15.75 GB` compiler capacity;
- program memory: `13.41 GB`;
- parameter/optimizer arguments: `4.93 GB`;
- dominant allocations: three separate `4.00 GB` FP32 matrix-scan
  temporaries with shape `2048 x 1 x 512 x 1024`.

This identifies activation retention inside the matrix recurrence, not model
weights, as the limiting allocation. The next gate uses exact state-preserving
chunked rematerialization with 64-token chunks. A local correctness smoke
matched the unchunked recurrence with maximum absolute output difference
`0.0`. The architecture, context, states, parameter count, and optimizer stay
unchanged.

## Selected runtime result

The chunked `4 data x 2 model` follow-up passed every pre-registered gate:

- exact parameters: `1,058,963,121`;
- context: `2,048`;
- global microbatch: `4` (`8,192` tokens/microstep);
- planned training accumulation: four microsteps for `32,768` tokens/update;
- scan chunk: `64`, with recurrent state preserved across chunks;
- initialization: `949.66 s`;
- compile plus first update: `31.86 s`;
- measured update: `15.19 s`;
- throughput: `539.19 tokens/s`;
- finite loss: `9.86687`;
- finite gradient norm: `7.39316`;
- layer-zero router mean: `0.50005`;
- router saturation: `0.0`;
- memory after train step: approximately `10.68 GB/device` of
  `16.91 GB/device`;
- largest reported free block after the step: approximately `2.05 GB`;
- projected 100M-token time: `51.52 h`;
- abort reasons: none.

This is the selected measured v5e-8 runtime configuration. Relative to the
passing `2 x 4` control it improves measured throughput by approximately
`2.21x` and changes the 100M-token systems gate from no-go to go.

The readiness gate measured complete 8,192-token microsteps and applied AdamW
after each measured microstep. Therefore `539.19 tokens/s` is valid microstep
throughput, but it is not yet a measurement of the intended four-microstep
global-batch update. The separate real-data smoke must validate FP32 gradient
accumulation and one AdamW update per four microsteps before accumulated-update
readiness is closed.

## Claim boundary

This result proves that the exact frozen 1B parameterization can execute a
complete sharded training update on a free-tier v5e-8. It does not establish
language quality, convergence, the Llion `1.1 BPC` target, or superiority over
any baseline.
