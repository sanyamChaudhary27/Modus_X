# PB0 Checkpoint Compatibility Inspection

## Scope

This inspection reads the existing trained checkpoints without modifying the canonical architecture, checkpoint files, or training configuration.

The checkpoint format is inspected as a Python pickle containing a JAX/NumPy-compatible parameter tree.

## Canonical Reference

- Canonical model file: `E:\Modus_X\language\models.py`

- Expected canonical parameter count: **47,437,768**

## SEED1

- Checkpoint: `E:\seed1\checkpoint.pkl`
- Checkpoint size: **542.89 MiB**
- Parameter leaves: **52**
- Parameter count: **47,437,768**
- Parameter array bytes: **180.96 MiB**
- Matches canonical 47,437,768 count: **True**
- Difference from canonical count: **+0**

### Scalar Checkpoint Metadata

```json
{
  "step": 25000
}
```

### Target Matrix Parameters

| Family | Checkpoint path | Shape | Parameters |
|---|---|---|---:|
| `m_wq` | `layers.m_wq` | `[12, 512, 512]` | 3,145,728 |
| `m_wk` | `layers.m_wk` | `[12, 512, 512]` | 3,145,728 |
| `m_wv` | `layers.m_wv` | `[12, 512, 512]` | 3,145,728 |
| `m_w_read` | `layers.m_w_read` | `[12, 512, 512]` | 3,145,728 |
| `m_w_out` | `layers.m_w_out` | `[12, 512, 512]` | 3,145,728 |
| `m_proj_w` | `layers.m_proj_w` | `[12, 512, 1024]` | 6,291,456 |

### config.json

```json
{
  "args": {
    "aux_future_targets": false,
    "auxiliary_decay_chars": 0,
    "auxiliary_layers": "6",
    "auxiliary_weight": 0.05,
    "batch": 8,
    "checkpoint_chars": 4096000,
    "data_path": "/kaggle/input/notebooks/sanyamchoudhary27/notebookcbb2c4d051/enwik8",
    "decay_start_chars": 100000000,
    "dropout": 0.0,
    "embed_dim": 512,
    "end_lr_ratio": 0.05,
    "eval_batch": 8,
    "eval_chunks": 128,
    "future_target_decay_chars": 0,
    "future_target_weight": 0.5,
    "future_targets": "2",
    "hidden_dim": 1536,
    "input_corruption_rate": 0.0,
    "input_seq_len": 512,
    "label_smoothing": 0.0,
    "loss_tail": 512,
    "lr": 0.0003,
    "matrix_retain_bias": null,
    "matrix_write_bias": null,
    "model": "Modus_X_MemoryFeedbackSegmentRetention_DeepSupervision",
    "momentum": 0.99,
    "n_layers": 12,
    "optimizer": "adamw",
    "outdir": "/kaggle/working/segment_retention_seed1_102p4m",
    "paper_layer_decay": false,
    "precision": "float32",
    "resume": true,
    "router_hidden": 32,
    "schedule": "constant",
    "seed": 1,
    "state_dim": 512,
    "stop_chars": 102400000,
    "target_chars": 102400000,
    "vector_retain_bias": null,
    "warmup_steps": 100,
    "weight_decay": 0.0001
  },
  "auxiliary_decay_steps": 0,
  "chars_per_step": 4096,
  "checkpoint_steps": 1000,
  "devices": [
    "TPU_0(process=0,(0,0,0,0))",
    "TPU_1(process=0,(1,0,0,0))",
    "TPU_2(process=0,(0,1,0,0))",
    "TPU_3(process=0,(1,1,0,0))",
    "TPU_4(process=0,(0,2,0,0))",
    "TPU_5(process=0,(1,2,0,0))",
    "TPU_6(process=0,(0,3,0,0))",
    "TPU_7(process=0,(1,3,0,0))"
  ],
  "future_target_decay_steps": 0,
  "non_embedding_params": 47306696,
  "params": 47437768,
  "stop_steps": 25000,
  "total_steps": 25000
}
```

### progress.json

```json
{
  "config": {
    "args": {
      "aux_future_targets": false,
      "auxiliary_decay_chars": 0,
      "auxiliary_layers": "6",
      "auxiliary_weight": 0.05,
      "batch": 8,
      "checkpoint_chars": 4096000,
      "data_path": "/kaggle/input/notebooks/sanyamchoudhary27/notebookcbb2c4d051/enwik8",
      "decay_start_chars": 100000000,
      "dropout": 0.0,
      "embed_dim": 512,
      "end_lr_ratio": 0.05,
      "eval_batch": 8,
      "eval_chunks": 128,
      "future_target_decay_chars": 0,
      "future_target_weight": 0.5,
      "future_targets": "2",
      "hidden_dim": 1536,
      "input_corruption_rate": 0.0,
      "input_seq_len": 512,
      "label_smoothing": 0.0,
      "loss_tail": 512,
      "lr": 0.0003,
      "matrix_retain_bias": null,
      "matrix_write_bias": null,
      "model": "Modus_X_MemoryFeedbackSegmentRetention_DeepSupervision",
      "momentum": 0.99,
      "n_layers": 12,
      "optimizer": "adamw",
      "outdir": "/kaggle/working/segment_retention_seed1_102p4m",
      "paper_layer_decay": false,
      "precision": "float32",
      "resume": true,
      "router_hidden": 32,
      "schedule": "constant",
      "seed": 1,
      "state_dim": 512,
      "stop_chars": 102400000,
      "target_chars": 102400000,
      "vector_retain_bias": null,
      "warmup_steps": 100,
      "weight_decay": 0.0001
    },
    "auxiliary_decay_steps": 0,
    "chars_per_step": 4096,
    "checkpoint_steps": 1000,
    "devices": [
      "TPU_0(process=0,(0,0,0,0))",
      "TPU_1(process=0,(1,0,0,0))",
      "TPU_2(process=0,(0,1,0,0))",
      "TPU_3(process=0,(1,1,0,0))",
      "TPU_4(process=0,(0,2,0,0))",
      "TPU_5(process=0,(1,2,0,0))",
      "TPU_6(process=0,(0,3,0,0))",
      "TPU_7(process=0,(1,3,0,0))"
    ],
    "future_target_decay_steps": 0,
    "non_embedding_params": 47306696,
    "params": 47437768,
    "stop_steps": 25000,
    "total_steps": 25000
  },
  "rows": [
    {
      "delta_to_gpu": -0.4473703564011,
      "elapsed_s": 916.9324380410001,
      "gpu_reference_bpc": 2.506,
      "loss": 2.6199028491973877,
      "processed_characters": 4096000,
      "step": 1000,
      "val_bpc": 2.0586296435988998
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 1815.920643938,
      "gpu_reference_bpc": null,
      "loss": 2.291710615158081,
      "processed_characters": 8192000,
      "step": 2000,
      "val_bpc": 1.876472574353457
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 2715.05507482,
      "gpu_reference_bpc": null,
      "loss": 2.5472328662872314,
      "processed_characters": 12288000,
      "step": 3000,
      "val_bpc": 1.7922914523137092
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 3614.2950429820003,
      "gpu_reference_bpc": null,
      "loss": 2.2995903491973877,
      "processed_characters": 16384000,
      "step": 4000,
      "val_bpc": 1.7217306897895237
    },
    {
      "delta_to_gpu": -0.18905755554041637,
      "elapsed_s": 4513.440504258,
      "gpu_reference_bpc": 1.8638,
      "loss": 2.218491792678833,
      "processed_characters": 20480000,
      "step": 5000,
      "val_bpc": 1.6747424444595835
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 5431.924556657001,
      "gpu_reference_bpc": null,
      "loss": 2.2606654167175293,
      "processed_characters": 24576000,
      "step": 6000,
      "val_bpc": 1.6368138098460112
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 6332.219965025,
      "gpu_reference_bpc": null,
      "loss": 2.1467700004577637,
      "processed_characters": 28672000,
      "step": 7000,
      "val_bpc": 1.615954711304549
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 7231.679153701,
      "gpu_reference_bpc": null,
      "loss": 2.1190614700317383,
      "processed_characters": 32768000,
      "step": 8000,
      "val_bpc": 1.595975169660804
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 8131.199581963001,
      "gpu_reference_bpc": null,
      "loss": 2.0340206623077393,
      "processed_characters": 36864000,
      "step": 9000,
      "val_bpc": 1.5736372030534882
    },
    {
      "delta_to_gpu": -0.13410368074896128,
      "elapsed_s": 9030.730178485,
      "gpu_reference_bpc": 1.6918,
      "loss": 1.9488600492477417,
      "processed_characters": 40960000,
      "step": 10000,
      "val_bpc": 1.5576963192510387
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 9930.658975276,
      "gpu_reference_bpc": null,
      "loss": 2.084550380706787,
      "processed_characters": 45056000,
      "step": 11000,
      "val_bpc": 1.5323248205331057
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 10830.407844784,
      "gpu_reference_bpc": null,
      "loss": 2.0846803188323975,
      "processed_characters": 49152000,
      "step": 12000,
      "val_bpc": 1.522928080825384
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 11729.891756577,
      "gpu_reference_bpc": null,
      "loss": 1.945189356803894,
      "processed_characters": 53248000,
      "step": 13000,
      "val_bpc": 1.5100211160945163
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 12629.554912429001,
      "gpu_reference_bpc": null,
      "loss": 1.6871145963668823,
      "processed_characters": 57344000,
      "step": 14000,
      "val_bpc": 1.5074149608712932
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 13529.698134,
      "gpu_reference_bpc": null,
      "loss": 1.9169912338256836,
      "processed_characters": 61440000,
      "step": 15000,
      "val_bpc": 1.484158977745961
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 14429.168074339,
      "gpu_reference_bpc": null,
      "loss": 1.6512157917022705,
      "processed_characters": 65536000,
      "step": 16000,
      "val_bpc": 1.4881681835582963
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 15328.712019551,
      "gpu_reference_bpc": null,
      "loss": 1.9637178182601929,
      "processed_characters": 69632000,
      "step": 17000,
      "val_bpc": 1.474536908518862
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 16228.202009249,
      "gpu_reference_bpc": null,
      "loss": 1.753839135169983,
      "processed_characters": 73728000,
      "step": 18000,
      "val_bpc": 1.4676057228428252
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 17128.001021783,
      "gpu_reference_bpc": null,
      "loss": 1.5471577644348145,
      "processed_characters": 77824000,
      "step": 19000,
      "val_bpc": 1.4558139849859353
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 18028.173763984996,
      "gpu_reference_bpc": null,
      "loss": 2.0508837699890137,
      "processed_characters": 81920000,
      "step": 20000,
      "val_bpc": 1.4500228239323476
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 18945.210443105992,
      "gpu_reference_bpc": null,
      "loss": 1.6256343126296997,
      "processed_characters": 86016000,
      "step": 21000,
      "val_bpc": 1.4069904106739393
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 19845.265007386,
      "gpu_reference_bpc": null,
      "loss": 1.5302445888519287,
      "processed_characters": 90112000,
      "step": 22000,
      "val_bpc": 1.3936503828792832
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 20744.788007197996,
      "gpu_reference_bpc": null,
      "loss": 1.6990982294082642,
      "processed_characters": 94208000,
      "step": 23000,
      "val_bpc": 1.391102852242173
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 21644.258889785,
      "gpu_reference_bpc": null,
      "loss": 1.8229941129684448,
      "processed_characters": 98304000,
      "step": 24000,
      "val_bpc": 1.3876385143400602
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 22543.789300653996,
      "gpu_reference_bpc": null,
      "loss": 1.718469262123108,
      "processed_characters": 102400000,
      "step": 25000,
      "val_bpc": 1.3877832538641348
    }
  ]
}
```

### ENDPOINT_DECISION.json

```json
{
  "aggregate_all_state_patch_gain_bpc": 0.0028718168032355607,
  "archive_retention_improvement_ratio_vs_canonical_screen": 205.16816583763654,
  "candidate": {
    "dense_test_bpc_report_only": 1.4344876219722105,
    "dense_validation_bpc": 1.4328895986576113,
    "last_sparse_checkpoint": {
      "delta_to_gpu": null,
      "elapsed_s": 22543.789300653996,
      "gpu_reference_bpc": null,
      "loss": 1.718469262123108,
      "processed_characters": 102400000,
      "step": 25000,
      "val_bpc": 1.3877832538641348
    },
    "params": 47437768
  },
  "control_minus_candidate_dense_validation_bpc": 0.026833401342388763,
  "endpoint_pass": true,
  "frozen_control": {
    "dense_test_bpc_report_only": 1.465006,
    "dense_validation_bpc": 1.459723,
    "elapsed_s": 22570.22,
    "params": 47437768
  },
  "mean_long_range_archive_delta_ratio": 0.0008127649024988516,
  "next": "replicate the 102.4M endpoint on seeds 2 and 3",
  "promotion_checks": {
    "archive_retention_improves_at_least_100x": true,
    "dense_validation_regression_at_most_0p005": true,
    "runtime_at_most_1p05x": true,
    "source_patch_gain_at_least_0p0005": true
  },
  "runtime_ratio_vs_control": 0.9988289569465426,
  "stage": "seed-1 matched 102.4M endpoint, dense audit, and frozen source trace",
  "strong_language_win": true,
  "test_data_read": true,
  "test_role": "report_only_after_frozen_endpoint"
}
```

## SEED2

- Checkpoint: `E:\seed2\checkpoint.pkl`
- Checkpoint size: **542.89 MiB**
- Parameter leaves: **52**
- Parameter count: **47,437,768**
- Parameter array bytes: **180.96 MiB**
- Matches canonical 47,437,768 count: **True**
- Difference from canonical count: **+0**

### Scalar Checkpoint Metadata

```json
{
  "step": 25000
}
```

### Target Matrix Parameters

| Family | Checkpoint path | Shape | Parameters |
|---|---|---|---:|
| `m_wq` | `layers.m_wq` | `[12, 512, 512]` | 3,145,728 |
| `m_wk` | `layers.m_wk` | `[12, 512, 512]` | 3,145,728 |
| `m_wv` | `layers.m_wv` | `[12, 512, 512]` | 3,145,728 |
| `m_w_read` | `layers.m_w_read` | `[12, 512, 512]` | 3,145,728 |
| `m_w_out` | `layers.m_w_out` | `[12, 512, 512]` | 3,145,728 |
| `m_proj_w` | `layers.m_proj_w` | `[12, 512, 1024]` | 6,291,456 |

### config.json

```json
{
  "args": {
    "aux_future_targets": false,
    "auxiliary_decay_chars": 0,
    "auxiliary_layers": "6",
    "auxiliary_weight": 0.05,
    "batch": 8,
    "checkpoint_chars": 4096000,
    "data_path": "/kaggle/input/notebooks/sanyamchoudhary27/notebookd63b4783ea/enwik8",
    "decay_start_chars": 100000000,
    "dropout": 0.0,
    "embed_dim": 512,
    "end_lr_ratio": 0.05,
    "eval_batch": 8,
    "eval_chunks": 128,
    "future_target_decay_chars": 0,
    "future_target_weight": 0.5,
    "future_targets": "2",
    "hidden_dim": 1536,
    "input_corruption_rate": 0.0,
    "input_seq_len": 512,
    "label_smoothing": 0.0,
    "loss_tail": 512,
    "lr": 0.0003,
    "matrix_retain_bias": null,
    "matrix_write_bias": null,
    "model": "Modus_X_MemoryFeedbackSegmentRetention_DeepSupervision",
    "momentum": 0.99,
    "n_layers": 12,
    "optimizer": "adamw",
    "outdir": "/kaggle/working/segment_retention_seed2_102p4m",
    "paper_layer_decay": false,
    "precision": "float32",
    "resume": true,
    "router_hidden": 32,
    "schedule": "constant",
    "seed": 2,
    "state_dim": 512,
    "stop_chars": 102400000,
    "target_chars": 102400000,
    "vector_retain_bias": null,
    "warmup_steps": 100,
    "weight_decay": 0.0001
  },
  "auxiliary_decay_steps": 0,
  "chars_per_step": 4096,
  "checkpoint_steps": 1000,
  "devices": [
    "TPU_0(process=0,(0,0,0,0))",
    "TPU_1(process=0,(1,0,0,0))",
    "TPU_2(process=0,(0,1,0,0))",
    "TPU_3(process=0,(1,1,0,0))",
    "TPU_4(process=0,(0,2,0,0))",
    "TPU_5(process=0,(1,2,0,0))",
    "TPU_6(process=0,(0,3,0,0))",
    "TPU_7(process=0,(1,3,0,0))"
  ],
  "future_target_decay_steps": 0,
  "non_embedding_params": 47306696,
  "params": 47437768,
  "stop_steps": 25000,
  "total_steps": 25000
}
```

### progress.json

```json
{
  "config": {
    "args": {
      "aux_future_targets": false,
      "auxiliary_decay_chars": 0,
      "auxiliary_layers": "6",
      "auxiliary_weight": 0.05,
      "batch": 8,
      "checkpoint_chars": 4096000,
      "data_path": "/kaggle/input/notebooks/sanyamchoudhary27/notebookd63b4783ea/enwik8",
      "decay_start_chars": 100000000,
      "dropout": 0.0,
      "embed_dim": 512,
      "end_lr_ratio": 0.05,
      "eval_batch": 8,
      "eval_chunks": 128,
      "future_target_decay_chars": 0,
      "future_target_weight": 0.5,
      "future_targets": "2",
      "hidden_dim": 1536,
      "input_corruption_rate": 0.0,
      "input_seq_len": 512,
      "label_smoothing": 0.0,
      "loss_tail": 512,
      "lr": 0.0003,
      "matrix_retain_bias": null,
      "matrix_write_bias": null,
      "model": "Modus_X_MemoryFeedbackSegmentRetention_DeepSupervision",
      "momentum": 0.99,
      "n_layers": 12,
      "optimizer": "adamw",
      "outdir": "/kaggle/working/segment_retention_seed2_102p4m",
      "paper_layer_decay": false,
      "precision": "float32",
      "resume": true,
      "router_hidden": 32,
      "schedule": "constant",
      "seed": 2,
      "state_dim": 512,
      "stop_chars": 102400000,
      "target_chars": 102400000,
      "vector_retain_bias": null,
      "warmup_steps": 100,
      "weight_decay": 0.0001
    },
    "auxiliary_decay_steps": 0,
    "chars_per_step": 4096,
    "checkpoint_steps": 1000,
    "devices": [
      "TPU_0(process=0,(0,0,0,0))",
      "TPU_1(process=0,(1,0,0,0))",
      "TPU_2(process=0,(0,1,0,0))",
      "TPU_3(process=0,(1,1,0,0))",
      "TPU_4(process=0,(0,2,0,0))",
      "TPU_5(process=0,(1,2,0,0))",
      "TPU_6(process=0,(0,3,0,0))",
      "TPU_7(process=0,(1,3,0,0))"
    ],
    "future_target_decay_steps": 0,
    "non_embedding_params": 47306696,
    "params": 47437768,
    "stop_steps": 25000,
    "total_steps": 25000
  },
  "rows": [
    {
      "delta_to_gpu": -0.42152664169401843,
      "elapsed_s": 917.7564763669999,
      "gpu_reference_bpc": 2.506,
      "loss": 2.5107176303863525,
      "processed_characters": 4096000,
      "step": 1000,
      "val_bpc": 2.0844733583059813
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 1816.846015263,
      "gpu_reference_bpc": null,
      "loss": 2.3898987770080566,
      "processed_characters": 8192000,
      "step": 2000,
      "val_bpc": 1.8808576375003374
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 2716.280868129,
      "gpu_reference_bpc": null,
      "loss": 2.1978888511657715,
      "processed_characters": 12288000,
      "step": 3000,
      "val_bpc": 1.7778004843727322
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 3615.675966986,
      "gpu_reference_bpc": null,
      "loss": 1.987883448600769,
      "processed_characters": 16384000,
      "step": 4000,
      "val_bpc": 1.711153654646762
    },
    {
      "delta_to_gpu": -0.1809279410085891,
      "elapsed_s": 4514.999323057,
      "gpu_reference_bpc": 1.8638,
      "loss": 1.9912055730819702,
      "processed_characters": 20480000,
      "step": 5000,
      "val_bpc": 1.6828720589914108
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 5434.21768724,
      "gpu_reference_bpc": null,
      "loss": 1.5432357788085938,
      "processed_characters": 24576000,
      "step": 6000,
      "val_bpc": 1.6392333477696748
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 6334.494037112,
      "gpu_reference_bpc": null,
      "loss": 2.2994160652160645,
      "processed_characters": 28672000,
      "step": 7000,
      "val_bpc": 1.610610930870434
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 7234.0723476289995,
      "gpu_reference_bpc": null,
      "loss": 2.415189504623413,
      "processed_characters": 32768000,
      "step": 8000,
      "val_bpc": 1.5939883723334585
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 8134.165915822,
      "gpu_reference_bpc": null,
      "loss": 1.949455976486206,
      "processed_characters": 36864000,
      "step": 9000,
      "val_bpc": 1.5736762592386075
    },
    {
      "delta_to_gpu": -0.13710107500506585,
      "elapsed_s": 9034.240007905,
      "gpu_reference_bpc": 1.6918,
      "loss": 1.7945382595062256,
      "processed_characters": 40960000,
      "step": 10000,
      "val_bpc": 1.5546989249949341
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 9934.266715901002,
      "gpu_reference_bpc": null,
      "loss": 1.5529593229293823,
      "processed_characters": 45056000,
      "step": 11000,
      "val_bpc": 1.5504567256863424
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 10834.276024028,
      "gpu_reference_bpc": null,
      "loss": 1.9635237455368042,
      "processed_characters": 49152000,
      "step": 12000,
      "val_bpc": 1.5314942733164265
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 11734.278214136,
      "gpu_reference_bpc": null,
      "loss": 1.9360542297363281,
      "processed_characters": 53248000,
      "step": 13000,
      "val_bpc": 1.5186708760305045
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 12634.631924902,
      "gpu_reference_bpc": null,
      "loss": 1.966977596282959,
      "processed_characters": 57344000,
      "step": 14000,
      "val_bpc": 1.5027270416507013
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 13534.597116961,
      "gpu_reference_bpc": null,
      "loss": 1.879798173904419,
      "processed_characters": 61440000,
      "step": 15000,
      "val_bpc": 1.4897169034526894
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 14434.578691224,
      "gpu_reference_bpc": null,
      "loss": 1.9035369157791138,
      "processed_characters": 65536000,
      "step": 16000,
      "val_bpc": 1.483993928145711
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 15334.838065592001,
      "gpu_reference_bpc": null,
      "loss": 1.7054824829101562,
      "processed_characters": 69632000,
      "step": 17000,
      "val_bpc": 1.4746636382347165
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 16234.877173351,
      "gpu_reference_bpc": null,
      "loss": 1.8321866989135742,
      "processed_characters": 73728000,
      "step": 18000,
      "val_bpc": 1.4723162631564635
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 17135.022714964998,
      "gpu_reference_bpc": null,
      "loss": 1.9232642650604248,
      "processed_characters": 77824000,
      "step": 19000,
      "val_bpc": 1.4671385427206722
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 18034.630839393,
      "gpu_reference_bpc": null,
      "loss": 1.7738908529281616,
      "processed_characters": 81920000,
      "step": 20000,
      "val_bpc": 1.4469672511701308
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 18954.12158147,
      "gpu_reference_bpc": null,
      "loss": 1.8620141744613647,
      "processed_characters": 86016000,
      "step": 21000,
      "val_bpc": 1.4022655474301589
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 19853.661848957996,
      "gpu_reference_bpc": null,
      "loss": 1.715099573135376,
      "processed_characters": 90112000,
      "step": 22000,
      "val_bpc": 1.3971138662425995
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 20753.924422356995,
      "gpu_reference_bpc": null,
      "loss": 1.7326586246490479,
      "processed_characters": 94208000,
      "step": 23000,
      "val_bpc": 1.3879300840523088
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 21653.28236321,
      "gpu_reference_bpc": null,
      "loss": 1.5359036922454834,
      "processed_characters": 98304000,
      "step": 24000,
      "val_bpc": 1.3880857713470025
    },
    {
      "delta_to_gpu": null,
      "elapsed_s": 22552.616781417,
      "gpu_reference_bpc": null,
      "loss": 1.8809870481491089,
      "processed_characters": 102400000,
      "step": 25000,
      "val_bpc": 1.3798350219099316
    }
  ]
}
```

### ENDPOINT_DECISION.json

```json
{
  "adaptive_stop_pass": true,
  "aggregate_all_state_patch_gain_bpc": 0.001732041739160195,
  "archive_retention_improvement_ratio_vs_canonical_screen": 145.90178971997435,
  "candidate": {
    "dense_test_bpc_report_only": 1.4361443790926178,
    "dense_validation_bpc": 1.4346890949555928,
    "last_sparse_checkpoint": {
      "delta_to_gpu": null,
      "elapsed_s": 22552.616781417,
      "gpu_reference_bpc": null,
      "loss": 1.8809870481491089,
      "processed_characters": 102400000,
      "step": 25000,
      "val_bpc": 1.3798350219099316
    },
    "params": 47437768
  },
  "control_minus_candidate_dense_validation_bpc": 0.006980905597538278,
  "endpoint_pass": true,
  "frozen_control": {
    "dense_test_bpc_report_only": 1.4431464076042175,
    "dense_validation_bpc": 1.441670000553131,
    "elapsed_s": 22540.821273477,
    "params": 47437768
  },
  "mean_long_range_archive_delta_ratio": 0.0005779836916318018,
  "next": "stop immediate campaign at 2/2; reserve seed 3 for publication closure",
  "promotion_checks": {
    "archive_retention_improves_at_least_100x": true,
    "dense_validation_regression_at_most_0p005": true,
    "runtime_at_most_1p05x": true,
    "source_patch_gain_at_least_0p0005": true
  },
  "runtime_ratio_vs_control": 1.0005232953935836,
  "stage": "seed-2 matched 102.4M endpoint, dense audit, and frozen source trace",
  "strong_language_win": true,
  "test_data_read": true,
  "test_role": "report_only_after_frozen_endpoint"
}
```

## Seed Comparison

- Exact parameter-tree structure match: **True**
- Paths only in seed1: **0**
- Paths only in seed2: **0**
- Shape mismatches: **0**
- Dtype mismatches: **0**

## PB0 Gate

PB0 checkpoint inspection establishes checkpoint structure and parameter availability. Passing this inspection does not by itself prove that a checkpoint can reproduce canonical BPC. The next gate is to load the parameter tree through the exact canonical evaluation path and verify the frozen validation baseline.

