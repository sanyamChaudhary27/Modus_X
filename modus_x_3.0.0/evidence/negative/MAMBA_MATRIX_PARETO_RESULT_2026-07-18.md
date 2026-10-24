# Official-Mamba Matrix Pareto Result

## Result

The three seed-paired cases completed `5,000` updates and `20.48M` processed
characters on Kaggle T4x2. These are sparse fixed-window enwik8 evaluations,
not dense audits and not multi-seed estimates.

| Case | Parameters | Recurrent state/sequence | Validation BPC | Test BPC | Elapsed |
|---|---:|---:|---:|---:|---:|
| `mamba_704` | 81,462,656 | 9,191,424 B | 1.667784 | 1.638328 | 3,685.66 s |
| `mamba_696` | 79,735,152 | 9,086,976 B | **1.642716** | **1.622739** | 3,650.18 s |
| `modus_x_mamba_696` | 81,907,668 | 9,222,144 B | 1.662533 | 1.631937 | 3,728.83 s |

Relative to the matched approximately-81M budget control, the hybrid used:

- `+445,012` parameters (`+0.546%`);
- `+30,720` recurrent-state bytes (`+0.334%`);
- `+1.171%` elapsed time;
- `-0.005251` validation BPC and `-0.006391` test BPC.

Relative to the narrower width control, the hybrid used `+2.725%` parameters
and `+1.487%` recurrent-state bytes, but was worse by `+0.019817` validation
BPC and `+0.009198` test BPC.

## Decision

The pre-registered language-promotion rule failed because the hybrid did not
beat `mamba_696`. Freeze this exact enwik8 branch: do not continue it, tune it,
or claim that matrix feedback improved generic language modeling.

The result nevertheless preserves one bounded hypothesis. Against the
parameter-budget control, the hybrid retained Mamba-class compression and
slightly improved both sparse splits for very small measured state and runtime
overheads. This is insufficient by itself because one seed and sparse windows
cannot distinguish a stable architectural gain from width sensitivity or
evaluation noise.

## Next Gate

Run one matched associative-recall and same-key overwrite experiment using
these exact three architecture cases and the same precision policies. Compare
accuracy, update accuracy, stale-value false recall, recurrent-state bytes,
parameters, and runtime. The hybrid advances only if it provides a material
memory advantage while retaining the measured language-cost envelope.

Do not spend another long enwik8 run before that memory gate. If the hybrid
does not win the memory task, reject this insertion strategy. If it wins, run
three memory seeds before considering a multi-seed or dense enwik8
confirmation.

## Evidence Boundary

The final Kaggle decision JSON is archived locally at
`raw/pareto_summary.json` with SHA-256
`2C937FDFD75B77CB281D26258EAD641B251AE8FC2479AC9D86F061D06C2C13C1`.
The values above match its `PARETO_CASE_COMPLETE` and
`PARETO_FINAL_DECISION` records. The notebook and checkpoints remain to be
archived before external use.
