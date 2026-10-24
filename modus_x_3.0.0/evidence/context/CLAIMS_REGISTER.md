# Modus_X claims register

Snapshot: **2026-07-22**

| Claim | Status | Canonical evidence | Required caveat |
|---|---|---|---|
| Modus_X is a constant-recurrent-state attention-free language-model architecture in v1.1.1 | Supported for the published implementation | `Modus_X_v1.1.1/paper/whitepaper.md` | Does not imply constant total memory or compute in every implementation |
| Modus_X beats the tested official xLSTM on matched dense enwik8 test | Supported | `Modus_X_v1.1.1/evidence/RESULTS_LEDGER.md` | One configuration and protocol, not all xLSTM models |
| Modus_X beats official Mamba on generic enwik8 BPC | **False for tested models** | Same ledger | Mamba wins `1.34578` versus `1.38418` |
| Modus_X has a large associative recall/overwrite advantage over the tested small official-Mamba baseline | Supported in the controlled protocol | v1.1.1 associative-memory evidence | Not language modeling or general reasoning |
| MemoryFeedback improves CurrentArchive language modeling | Supported at 47M-class scale | `Modus_X_2.0.0/experiments/enwik8_current_archive/MEMORY_FEEDBACK_ARCHIVE_V0_2026-07-12.md` | About `12.5%` runtime overhead; not proven in official Mamba or at 1B |
| CurrentArchive beats bounded Transformer KV at equal state bytes after context truncation | Supported | `experiments/matrix_memory_capacity/EQUAL_MEMORY_FRONTIER_RESULT_2026-07-22.md` | Full-context KV wins; specialized tied-Q/K baseline; synthetic task |
| CurrentArchive handles both clean and updated values under constrained state | Supported | `experiments/matrix_memory_capacity/MIXED_CLEAN_UPDATE_RESULT_2026-07-22.md` | Stale false recall is `11.28%`; latest arbitration remains weak |
| The exact frozen 1B model can train and resume on Kaggle TPU v5e-8 | Supported as a systems smoke | `proposals/1B_scaling/system_validation/` | Seventeen updates are not a trained 1B model or quality result |
| The 1B MemoryFeedback model is ready to replace the frozen model | Not yet supported | `proposals/1B_scaling/MEMORY_FEEDBACK_ARCHIVE_1B_CONFIG_2026-07-21.md` | Exact systems gates remain open; recurrent state is `1.5307x` |
| Modus_X follows a favorable `3x model vs 1x Transformer` production crossover | Modeled hypothesis | Askio serving-economics files | Requires measured quality, throughput, hardware, batching, and workload data |
| Modus_X has achieved `1.1` BPC | **False** | BPC campaign memory | Treat `1.1` as an external benchmark target, not a completed result |
| Modus_X is SOTA or universally superior to Transformers/Mamba | **Not supported** | Entire evidence ledger | Never claim without direct matched measurements |

## External-writing rule

Every quantitative external claim must name the protocol, model size, state or
character budget, and comparison configuration. Use “the tested
configuration” rather than turning one result into a family-wide claim.
