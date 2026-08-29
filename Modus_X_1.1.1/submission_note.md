# Submission Note

The strongest honest claim for the May 28 deadline:

> Modus_X is a no-attention, constant-state matrix-vector recurrent language
> model that decisively improves over a matched Mamba baseline and continues to
> close the gap to a same-scale Transformer, while avoiding KV-cache growth.

Avoid claiming:

- full HellaSwag superiority, because only 1000 examples were evaluated;
- Transformer validation-loss superiority, because the audited Transformer 40k
  checkpoint remains lower on this held-out shard;
- production speed superiority, because the current prototype is not optimized.

Use the result instead as an architecture discovery:

- matrix delta-rule memory gives content-addressed writes,
- Mamba-like gates give selective recurrence,
- the router makes both streams cooperate inside every layer,
- all of this runs without attention and with O(1) recurrent state memory.

