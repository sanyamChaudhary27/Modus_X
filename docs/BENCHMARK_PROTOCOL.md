# v3 Benchmark Protocol

The language result compares the retention-calibrated MemoryFeedbackArchive
control with identical training plus causal two-lag K/V filtering. Both use
two-stage chunk32 and highest precision. Three seeds reach25k updates,102.4M
sampled enwik8 input characters, batch8/context512. LR6e-4through20k then3e-4,
AdamW weight decay1e-4,clip1; primary+layer6auxiliary0.05+future-target2weight0.5.

Split raw bytes0:90Mtrain,90M:95Mvalidation,95M:100Mtest. Dense evaluation resets
state per512-input window, stride512 at offsets0/256, complete windows only,
primary head. Sparse progress and carry probes are separate protocols.
Validation decides at the frozen endpoint; test reports afterward.

Read `research/v3/05_local_kv_filtering/PROTOCOL.json` for full chronology:
seed2 followed a failed5k screen under a later fixed-horizon protocol; seed3
fixed25k prospectively after development. Failed early results are not removed.

Two-stage backend speed is measured in separate continuation experiments.
Historical Mamba/xLSTM and equal-state memory results are not new v3 paired
trials. Do not add gains or fit a scaling law across mismatched recipes.
