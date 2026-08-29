# Claims register

## Supported

1. At approximately 47M parameters and `102.4M` processed enwik8 characters,
   the single-seed MemoryFeedbackArchive configuration improves dense
   validation BPC from `1.485020` to `1.459723` and dense test BPC from
   `1.492694` to `1.465006` relative to its matched CurrentArchive control.
2. That language gain costs `0.85%` more parameters and approximately `12.5%`
   more measured runtime in the tested implementation.
3. On a three-seed synthetic versioned-memory protocol with `16,512` bytes of
   BF16 recurrent state, CurrentArchiveDelta reaches `77.95%` mean mixed
   clean/update accuracy versus `16.38%` for the tested tied-Q/K Transformer
   with a truncated 32-token KV window.
4. When the state budget retains the complete context, the tested Transformer
   wins `98.63%` to `71.03%`.
5. Latest-heavy supervision improves CurrentArchive's latest retrieval but
   does not eliminate stale recall.
6. The tested latest-shadow refresh correction is harmful and rejected.
7. At a fixed `102.4M`-character endpoint, scaling MemoryFeedbackArchive from
   `47,437,768` to `81,486,728` parameters improves dense test BPC from
   `1.465006` to `1.443873`. Scaling further to `99,438,920` parameters reaches
   `1.442034` dense test BPC but regresses dense validation from `1.433138` to
   `1.434283`; the measured second interval is saturated.
8. At `163.84M` processed characters, the promoted `81,486,728`-parameter
   MemoryFeedback checkpoint reaches `1.375422` dense validation and
   `1.382445` dense test BPC. This narrowly improves the published v1.1.1
   endpoint (`1.384180`) with `1.54%` fewer parameters.

## Unsupported

- Modus_X is state of the art.
- Modus_X universally beats Transformers, Mamba, RWKV, or xLSTM.
- Modus_X has reached `1.1` enwik8 BPC.
- The 1B configuration has demonstrated trained model quality.
- Constant recurrent state implies constant total serving cost.
- Synthetic retrieval establishes general reasoning capability.
- The current three single-seed MemoryFeedback points establish a universal
  scaling law or justify extrapolation to `1.1` BPC.
- The small single-seed MemoryFeedback improvement over v1.1.1 proves a
  variance-independent family-wide advantage.

Every external claim must name the tested configuration, matching axis,
processed data, metric, and relevant limitation.
