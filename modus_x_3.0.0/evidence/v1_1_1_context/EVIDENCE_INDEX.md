# Evidence Index

| ID | Result | Status | Required Artifact |
|---|---|---|---|
| LM-MX-80 | Modus_X 80M dense enwik8 audit | verified, raw JSON pending promotion | audit JSON and config |
| LM-MAMBA-80 | Official Mamba 80M dense enwik8 audit | verified, raw JSON pending promotion | audit JSON, config, checkpoint metadata |
| LM-XLSTM-80 | Official xLSTM 80M dense enwik8 audit | verified, raw JSON pending promotion | audit JSON and source commit |
| MEM-MX-STD | Modus_X balanced-KV recall | verified, seeds 17/27/37 | `associative_memory/matched_multiseed_2026-06-25/` |
| MEM-MAMBA-STD | Official Mamba balanced-KV recall | verified, seeds 17/27/37 | `associative_memory/matched_multiseed_2026-06-25/` |
| MEM-MX-OW50 | Modus_X 50% overwrite | verified, seeds 17/27/37 | `associative_memory/matched_multiseed_2026-06-25/` |
| MEM-MAMBA-OW50 | Official Mamba 50% overwrite | verified, seeds 17/27/37 | `associative_memory/matched_multiseed_2026-06-25/` |
| MEM-COMP-ABL | Modus_X router/component ablation | verified; three seeds, raw archive promoted | `associative_memory/component_ablation_2026-06-26/` |

No table may appear in the final paper unless its evidence row links to a
promoted raw artifact.
