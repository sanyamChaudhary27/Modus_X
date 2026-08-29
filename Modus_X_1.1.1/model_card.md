# Modus_X Model Card

## Model

Modus_X is an experimental attention-free causal sequence architecture with a
selective recurrent stream, content-addressed matrix memory, and learned
routing between them.

## Intended Use

- research on efficient sequence architectures;
- associative memory and overwrite experiments;
- constant-state language-model research;
- scaling studies toward open attention-free language models.

## Out Of Scope

- production deployment;
- safety-critical decisions;
- claims of general superiority over established architectures;
- use of synthetic recall results as a substitute for natural-language
  evaluation.

## Current Strengths

- content-addressed retrieval;
- same-key overwrite;
- length extrapolation on the recovered synthetic protocol;
- fixed inference-state size with sequence length.

## Current Weaknesses

- lower enwik8 BPC than the tested official Mamba baseline;
- slower research implementation;
- incomplete natural-language long-context evaluation;
- limited large-scale training evidence.

