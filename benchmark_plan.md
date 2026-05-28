# Modus_X Benchmark Plan

This file lists the benchmarks that matter most after the May 28 TPU sprint.

## Priority 1: Needle-in-a-Haystack

Goal: test content-addressed retrieval under long distractor contexts.

Protocol:

- Generate documents with random filler text.
- Insert one or more key-value facts:
  - `The passkey for orchid-17 is 492813.`
  - `The passkey for glacier-02 is 781204.`
- Query at the end:
  - `What is the passkey for orchid-17?`
- Sweep context lengths: `512`, `1k`, `2k`, `4k`, `8k`, `16k`.
- Sweep number of needles: `1`, `4`, `16`, `64`.
- Sweep overwrite mode:
  - no overwrite,
  - same key overwritten once,
  - same key overwritten many times.

Metrics:

- exact match accuracy,
- accuracy by needle position,
- accuracy by context length,
- memory usage by context length.

Expected Modus_X signal:

- better overwrite behavior than pure vector-state recurrence,
- constant memory while context grows.

## Priority 2: Associative Recall

Goal: isolate the matrix memory mechanism.

Protocol:

- Input sequence contains random key-value pairs.
- Final segment queries keys.
- Values are random symbols or short token strings.
- Include same-key overwrites.

Metrics:

- exact retrieval accuracy,
- overwrite accuracy,
- degradation as number of stored pairs grows.

Models:

- Modus_X 80k,
- Mamba 40k,
- Mamba 154M if audited,
- Transformer 40k,
- original Modus if compatible.

## Priority 3: Full HellaSwag

The current result is only `1000` samples. Run full HellaSwag before claiming
downstream reasoning performance.

Report:

- total samples,
- accuracy,
- confidence interval if possible,
- exact checkpoint used.

## Priority 4: Standard Commonsense Suite

Run:

- PIQA,
- WinoGrande,
- ARC-Easy,
- ARC-Challenge,
- OpenBookQA,
- LAMBADA.

These tests tell us whether the LM gain over Mamba transfers beyond perplexity.

## Priority 5: Long-Context LM

Evaluate perplexity at contexts beyond the training sequence length:

- `512`,
- `1024`,
- `2048`,
- `4096`,
- `8192`.

Important: report both loss and memory usage. Modus_X should not be sold as
faster before custom kernels; the first systems claim is memory flatness.

## Priority 6: Router Analysis

Analyze router values by:

- layer,
- token position,
- punctuation/word/subword categories,
- high-loss vs low-loss spans,
- retrieval tasks vs normal LM.

This can show whether the matrix stream behaves like an associative coprocessor
and when the model prefers vector recurrence.

