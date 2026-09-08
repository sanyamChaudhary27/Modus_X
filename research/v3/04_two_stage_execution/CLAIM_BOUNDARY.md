# Claim boundary

Supported by the reported screen: 2.923x median synchronized optimizer-update
speedup in this eight-device TPU configuration over 500 paired real-data
updates, with essentially unchanged reset-window dense validation BPC.

Not established: better BPC, long-run statistical equivalence, 2.923x total
training-job speedup, single-token decode gains, universal hardware gains,
lower total TPU memory, or a new learning algorithm. Temporary-memory compiler
estimates from separate probes are not measured whole-job peak memory.

Highest precision is mandatory. Default precision previously failed parity.
The failed synthetic trajectory tolerance is retained in RESULT.json.
Same-state diagnostics support numerical drift but do not prove equivalence
over every optimizer trajectory. Frozen canonical code remains the evaluator.

No raw Kaggle endpoint files were downloaded during this packaging work.
Source checkpoint hash is user-reported; new endpoint hashes remain pending.
