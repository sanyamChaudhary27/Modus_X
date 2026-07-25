# Modus_X 2.0.0 Model Card

## Model family

Modus_X 2.0.0 is an experimental family of causal recurrent language and
memory models. It combines a bounded vector recurrence with associative matrix
state. This release contains two promoted research configurations:

- **MemoryFeedbackArchive**, the generic-language lead, feeds compressed
  matrix retrieval back into the vector computation.
- **CurrentArchiveDelta**, the controlled-memory lead, separates current and
  archived matrix state for versioned binding and update experiments.

These are related research models, not one checkpoint that owns every reported
advantage.

## Intended use

- research on bounded-state sequence models;
- associative binding, update, conflict, and distractor studies;
- byte-level language-model scaling and dense evaluation;
- systems studies of recurrent-state memory and long-context serving.

## Out of scope

- production or safety-critical deployment;
- treating synthetic retrieval as general reasoning;
- claiming state of the art or universal superiority;
- representing the exact 1B systems smoke as a trained 1B model.

## Measured strengths

- MemoryFeedbackArchive improves the matched CurrentArchive dense enwik8 result
  at approximately 47M parameters.
- Scaling MemoryFeedbackArchive from 47.44M to 81.49M parameters improves dense
  test BPC at the measured 102.4M-character endpoint.
- CurrentArchiveDelta retains clean and versioned bindings under a constrained
  recurrent-state budget where the tested truncated Transformer KV baseline
  cannot retain the full context.
- The inference state is bounded in sequence length for a fixed configuration.

## Measured weaknesses

- Official Mamba remains better on the available matched dense enwik8 endpoint.
- The 81M-to-99M MemoryFeedback scaling interval is saturated at the measured
  data and schedule budget.
- CurrentArchiveDelta still exhibits stale latest-value recall.
- The research kernels are not production-optimized.
- Broad downstream capability, instruction following, and natural-language
  long-context memory are not established.

## Evaluation boundaries

Use dense validation and dense test BPC for language conclusions. Sparse
checkpoint BPC is a progress metric. Synthetic memory results must name the
task, recurrent-state budget, parameter count, seeds, and whether the
Transformer comparator retained the full context.

