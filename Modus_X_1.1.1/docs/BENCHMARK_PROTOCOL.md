# Benchmark Protocol

## Comparison Axes

Results are reported on separate axes. No single metric substitutes for the
others.

1. Predictive quality: dense validation and test BPC.
2. Associative retrieval: held-out balanced-KV accuracy.
3. Overwrite: accuracy after repeated writes to the same key.
4. Length extrapolation: accuracy beyond the training length.
5. Parameter efficiency: total and non-embedding parameters.
6. Sample efficiency: updates and supervised examples/characters.
7. Systems efficiency: throughput, peak memory, and inference-state size.

## Fairness

- Match processed examples or characters and optimizer updates where possible.
- Keep context, vocabulary, split, and evaluation windows identical.
- Use official external implementations when available.
- Give baselines a reasonable learning-rate opportunity.
- Report framework, precision, accelerator, and source commit.
- Separate sample-efficiency conclusions from wall-clock conclusions.

## Statistical Standard

- Primary comparative claims require at least three seeds.
- Report mean, standard deviation, and per-seed results.
- Long-context evaluations must state sample count.
- Values based on fewer than `1,000` examples are diagnostic unless confidence
  intervals are supplied.

