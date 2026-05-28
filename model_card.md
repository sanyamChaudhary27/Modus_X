# Modus_X Model Card

## Model

- Name: Modus_X
- Type: causal language model
- Attention: none
- State: fixed matrix + fixed vector recurrent state
- Parameters: about 153.9M in the benchmark configuration
- Vocabulary: GPT-2 tokenizer, 50,257 tokens

## Benchmark Configuration

```text
seq_len=512
embed_dim=512
hidden_dim=2048
ax_res=384
n_layers=8
mamba_state_dim=384
batch_per_device=2
TPU devices=4
```

## Best Audited Checkpoint

```text
GCS path: gs://axiom-v5-checkpoints-p6-eu/modus_v2/runs/modus_x_80k_cold_continue/Modus_X_ckpt/80000
valid_loss: 4.148229327052832
valid_bpc: 5.984629878609282
```

## Intended Use

Research on attention-free, constant-state sequence modeling and long-context
memory architectures.

## Not Intended As

- a production language model,
- a fully optimized inference implementation,
- a claim of full benchmark superiority over Transformers.

