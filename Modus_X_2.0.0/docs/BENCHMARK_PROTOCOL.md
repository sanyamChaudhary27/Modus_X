# Benchmark Protocol

## Evidence layers

The release keeps five evidence layers separate:

1. generic language modeling;
2. controlled associative memory;
3. downstream capability and reasoning;
4. systems behavior;
5. serving economics.

A win in one layer is not substituted for another.

## enwik8 language modeling

- Dataset: canonical 100,000,000-byte enwik8 file.
- Training region: first 90M bytes, with the final 5M of that region retained
  as the reported train-tail diagnostic.
- Validation: bytes 90M-95M.
- Test: bytes 95M-100M.
- Context: 512 bytes for the reported v2 language results.
- Sparse checkpoint validation: progress signal only.
- Dense evaluation: windows at offsets 0 and 256, stride 512.
- Final test data must not select a checkpoint or tune a configuration.

Comparisons must name parameter count, processed characters, optimizer steps,
seed, schedule, precision, and evaluator. The 47M MemoryFeedback point used a
different learning-rate schedule from the 81M and 99M points; the release
therefore does not fit a universal parameter-only scaling law.

## Controlled versioned memory

Promoted small-model claims use:

- at least three seeds;
- fixed clean/update/query-role distributions;
- explicit current, previous, and first-value metrics;
- stale-false-recall diagnostics;
- parameter and recurrent-state accounting;
- a protocol learnability check for every comparator.

For equal-memory comparisons, report whether a Transformer KV window contains
the complete context. Full-context and truncated-context results answer
different questions and must appear together.

## Systems

Systems readiness requires exact parameter counting, TPU mesh placement,
forward/backward execution, optimizer-state allocation, accumulated updates,
checkpoint creation, and independent restore. Passing those gates does not
establish language quality.

## Promotion rule

Pre-register the changed variable, control, matching axis, primary metric,
threshold, and stop rule. Preserve negative outcomes. Do not promote an
architecture from final-test improvement alone.

