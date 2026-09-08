# Claim boundary

Supported by the reported screen: 2.923x median synchronized optimizer-update
speedup in this eight-device TPU configuration over 500 paired real-data
updates, with essentially unchanged reset-window dense validation BPC.
Independent seed-1 replication extended this to 2,000 paired updates at
2.923x speedup and +0.00003736 BPC candidate-minus-control, passing the frozen
+0.002 BPC tolerance. Do not present these unequal-length continuations as
full from-scratch training replications or a statistical equivalence test.

Not established: better BPC, long-run statistical equivalence, 2.923x total
training-job speedup, single-token decode gains, universal hardware gains,
lower total TPU memory, or a new learning algorithm. Temporary-memory compiler
estimates from separate probes are not measured whole-job peak memory.

Highest precision is mandatory. Default precision previously failed parity.
The failed synthetic trajectory tolerance is retained in RESULT.json.
Same-state diagnostics support numerical drift but do not prove equivalence
over every optimizer trajectory. Frozen canonical code remains the evaluator.

No raw Kaggle endpoint files were downloaded during this packaging work.
Source checkpoint hashes are user-reported. Seed-1 endpoint hashes are now
recorded from the runner; independent file verification and local storage
remain pending. The seed-2 saved notebook pointer is in RESULT.json.
