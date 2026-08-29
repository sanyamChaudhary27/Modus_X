# Full 1B Real-Data Accumulation Smoke (2026-07-17)

## Status

**PASS: training execution, rolling checkpoint writes, restore, continuation,
and post-resume checkpointing all succeeded.**

The frozen `1,058,963,121`-parameter Modus_X configuration completed 16 real
enwik8 byte-ID optimizer updates on Kaggle TPU v5e-8. Each update accumulated
four `8,192`-token microsteps, so the run processed `524,288` tokens in total.
All reported losses and updates remained finite.

This is a systems smoke, not a BPC or convergence result. Raw enwik8 bytes
were used as valid token IDs for the frozen large-vocabulary system solely to
exercise the real data path, accumulated gradients, optimizer update, and
checkpoint state.

## Measured execution

- Parameters: `1,058,963,121`
- Mesh: `4 data x 2 model` across eight TPU v5e cores
- Context: `2,048`
- Global microbatch: `4`
- Gradient accumulation: `4` microsteps
- Tokens per optimizer update: `32,768`
- Completed optimizer updates: `16`
- Processed tokens: `524,288`
- Full initialization: `1,000.10 s`
- First update including compilation: `91.30 s`, `358.90 tokens/s`
- Steady updates 2-16: approximately `61.03 s/update`
- Steady accumulated throughput: approximately `536.9 tokens/s`
- Straight-line 100M-token execution projection excluding initialization and
  checkpoint/evaluation overhead: approximately `51.7 h`

The non-monotonic per-update losses are not interpreted as model quality. The
run is far too short, consumes contiguous heterogeneous byte segments, and
uses the byte IDs only as a systems-path input to the frozen vocabulary.

## Checkpoint incident and correction

All 16 updates completed before checkpoint serialization failed. Kaggle's
installed Orbax `StandardCheckpointer` expected the legacy positional state
argument, while the script used the newer `args=StandardSave(...)` API. The
TPU runtime was subsequently restarted, so the trained arrays held by the
failed Python traceback could no longer be recovered.

The corrected runner now:

1. supports both positional and newer Orbax save/restore APIs;
2. saves immediately after the first completed update;
3. checkpoints every four updates and at the final update;
4. waits for checkpoint completion before reporting success; and
5. retains only the newest completed checkpoint to bound Kaggle disk use.

The corrected rerun completed all 16 updates and wrote successful checkpoints
at steps `1`, `4`, `8`, `12`, and `16`. The final report recorded:

- status: `PASS`
- final processed tokens: `524,288`
- final data cursor: `524,544`
- total measured elapsed time: `3,284.14 s`
- steady accumulated throughput: approximately `536.9 tokens/s`
- per-device live memory: approximately `5.93 GB`
- per-device peak memory: approximately `7.25 GB`
- largest reported free block after training: approximately `4.11 GB`

The five synchronous full-state saves added substantial wall time compared with
the earlier no-checkpoint smoke. This four-update cadence is intentionally a
durability stress test, not the proposed production cadence.

The durability gate was closed by restoring the step-16 state, recovering its
data cursor, completing accumulated update 17, and writing a new step-17
checkpoint. The resumed report recorded `start_update=16`,
`target_update=17`, `processed_tokens=557,056`, and
`data_cursor=557,328`.

The first restore probe stopped before loading parameter buffers because this
Orbax release rejects `numpy.int64` scalar leaves in a restore target, despite
having accepted those leaves during save. The target metadata was changed to
zero-dimensional NumPy arrays with the same `int64` dtype. The valid step-16
checkpoint was not modified by this failed restore attempt.

The restore probe's wrapper printed an `AssertionError` only after the model
run and step-17 save had succeeded. Its hand-calculated expected cursor omitted
the extra next-token target byte in every sequence. The actual cursor advance
is `4 accumulation steps x 4 batch items x 2,049 input-plus-target bytes =
32,784`, so `524,544 + 32,784 = 557,328`. The wrapper assertion was corrected;
no additional TPU run is required.
