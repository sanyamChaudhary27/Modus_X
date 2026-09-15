# PB0-B Dense Truncated-SVD Sensitivity

## Purpose

PB0-B measures whether the spectral redundancy observed in PB0-A is functionally removable. Each experiment replaces one complete projection family with rank-k truncated-SVD reconstructions while preserving the original dense tensor shape.

No architecture topology, tensor shape, recurrent state dimension, or checkpoint parameter count is changed.

## Parameter Inventory

| Category | Parameters | Share |
|---|---|---|
| vector_mamba | 19,298,316 | 40.6813% |
| matrix_memory | 18,905,148 | 39.8525% |
| other | 6,740,352 | 14.2088% |
| output_head | 2,362,880 | 4.9810% |
| embeddings | 131,072 | 0.2763% |

The category inventory is a bookkeeping input to the later fixed-budget allocation matrix. It does not itself determine where parameters should be moved.

## Experiment Results

| Seed | Family | Rank | Baseline BPC | Compressed BPC | Δ BPC | Mean energy | Min energy | Mean relative error |
|---|---|---|---|---|---|---|---|---|
| seed1 | m_wq | 256 | 1.516444 | 1.516471 | +0.000027 | 99.1639% | 99.1166% | 0.091411 |
| seed1 | m_wq | 128 | 1.516444 | 1.516504 | +0.000060 | 96.4589% | 94.4522% | 0.187501 |
| seed1 | m_wq | 64 | 1.516444 | 1.518744 | +0.002300 | 91.6885% | 78.8371% | 0.282052 |
| seed1 | m_wk | 256 | 1.516444 | 1.516457 | +0.000013 | 99.2893% | 99.1208% | 0.084067 |
| seed1 | m_wk | 128 | 1.516444 | 1.516579 | +0.000135 | 96.6343% | 93.9636% | 0.182121 |
| seed1 | m_wk | 64 | 1.516444 | 1.517218 | +0.000774 | 91.5984% | 77.5175% | 0.282321 |
| seed1 | m_w_out | 256 | 1.516444 | 1.518329 | +0.001885 | 94.6612% | 93.1002% | 0.225309 |
| seed1 | m_w_out | 128 | 1.516444 | 1.522904 | +0.006460 | 78.9017% | 73.7744% | 0.450098 |
| seed1 | m_w_out | 64 | 1.516444 | 1.544902 | +0.028459 | 60.7906% | 53.6163% | 0.620674 |
| seed2 | m_wq | 256 | 1.481373 | 1.481374 | +0.000001 | 99.1669% | 99.1012% | 0.091239 |
| seed2 | m_wq | 128 | 1.481373 | 1.481469 | +0.000096 | 96.5503% | 95.7961% | 0.185513 |
| seed2 | m_wq | 64 | 1.481373 | 1.481806 | +0.000433 | 92.1353% | 85.0672% | 0.277334 |
| seed2 | m_wk | 256 | 1.481373 | 1.481406 | +0.000033 | 99.2928% | 99.1418% | 0.083852 |
| seed2 | m_wk | 128 | 1.481373 | 1.481524 | +0.000151 | 96.7096% | 95.2952% | 0.180702 |
| seed2 | m_wk | 64 | 1.481373 | 1.482365 | +0.000992 | 91.9508% | 83.5535% | 0.279434 |
| seed2 | m_w_out | 256 | 1.481373 | 1.482583 | +0.001210 | 94.5773% | 93.2187% | 0.227251 |
| seed2 | m_w_out | 128 | 1.481373 | 1.495887 | +0.014514 | 78.6462% | 73.8125% | 0.453289 |
| seed2 | m_w_out | 64 | 1.481373 | 1.520552 | +0.039179 | 60.4749% | 53.4577% | 0.623410 |

## Cross-Seed Summary

| Family | Rank | Mean Δ BPC | Worst Δ BPC | Mean energy | Minimum energy | Recommendation |
|---|---|---|---|---|---|---|
| m_w_out | 64 | 0.033819 | 0.039179 | 60.6327% | 53.4577% | reject_at_this_rank |
| m_w_out | 128 | 0.010487 | 0.014514 | 78.7740% | 73.7744% | reject_at_this_rank |
| m_w_out | 256 | 0.001547 | 0.001885 | 94.6193% | 93.1002% | candidate |
| m_wk | 64 | 0.000883 | 0.000992 | 91.7746% | 77.5175% | strong_candidate |
| m_wk | 128 | 0.000143 | 0.000151 | 96.6719% | 93.9636% | strong_candidate |
| m_wk | 256 | 0.000023 | 0.000033 | 99.2911% | 99.1208% | strong_candidate |
| m_wq | 64 | 0.001366 | 0.002300 | 91.9119% | 78.8371% | candidate |
| m_wq | 128 | 0.000078 | 0.000096 | 96.5046% | 94.4522% | strong_candidate |
| m_wq | 256 | 0.000014 | 0.000027 | 99.1654% | 99.1012% | strong_candidate |

## Interpretation

PB0-A measured spectral structure. PB0-B measures functional sensitivity. A family should not be considered a parameter reallocation donor merely because it has a low stable rank.

A family becomes a stronger donor candidate when low-rank reconstruction preserves validation BPC consistently across both trained seeds.

The next research stage is fixed-budget PB0-C parameter allocation. PB0-C converts empirically compressible capacity into an explicit 47M allocation matrix spanning matrix memory, vector/Mamba capacity, router capacity, feedback bridge capacity, embeddings, and output capacity.
