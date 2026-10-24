# Modus_X 3.0.0 Model Card

## Identity and Distribution

Sanyam Chaudhary, Independent Researcher. Version3.0.0,2026-10-24.
DOI10.5281/zenodo.23621101. Research source, paper and evidence are distributed;
the six trained endpoint checkpoint payloads are not bundled. CodeMIT;
manuscript/new documentationCC BY4.0. No conference peer-review status is claimed.

## Architecture

MemoryFeedbackArchive with current/archive matrix state, segment-scale archive
retention512, rank32 memory-to-vector feedback, direct routed matrix/vector
outputs and two-lag K/V histories. Twelve512-wide layers, hidden_dim1536,
router_hidden32.47,462,344 training parameters. Persistent FP32 state25,288,704
bytes per sequence. The vector path is not the official Mamba library.

Two-stage chunk32 is the tested execution path; canonical recurrence is retained
for verification. Highest matmul precision is part of the reported protocol.

## Intended Use

Research on bounded-state sequence modeling, byte-level prediction, memory
coordination, numerical equivalence and accelerator execution. This is not an
instruction-tuned assistant, production API, safety-certified system or deployed
developer model. No downstream coding/reasoning benchmark is reported.

## Training and Evaluation

Raw enwik8,90M/5M/5M byte split;102.4M sampled input characters per arm;batch8,
length512,25k updates. AdamW,clip1,weight_decay1e-4;LR6e-4through20k then3e-4.
Primary loss+layer6auxiliary0.05+separate two-step future head0.5. Future targets
are labels, not future inputs. Training windows reset recurrent state.

Dense reset512 validation/test use offsets0/256,stride512,primary head. Validation
selects; test reports afterward. Three paired validation gains .087291,.010196,
.021213 BPC,median .021213,mean .039566. Seed effects are unequal. Seed2 was a
post-screen follow-up; seed3 prospectively fixed its endpoint after development.
Early seed2/3 screen losses remain recorded. The reported audit outputs reproduce
validation and test causality/handoff; they are not independently re-downloaded
weights in this release build.

## Cost and Limits

Two continuation experiments report2.923x synchronized median update speedup on
Kaggle TPUv5e-8. K/V filtering adds~0.115% mean update overhead across seeds,
24,576params and96KiB state/sequence. These are not whole-job or decoding cost
claims. Compiler temporary storage is not total peak HBM.

One corpus, one new endpoint model size, three seeds. No matched modern Mamba-3,
KDA or Gated DeltaNet-2 v3 benchmark. Historical low-state retrieval advantages
do not establish semantic correctness or infinite retention. Full public weight
reproducibility and cross-lane metadata reconciliation remain open.
