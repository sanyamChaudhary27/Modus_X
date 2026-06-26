# Claims And Evidence

## Supported

1. Modus_X uses constant-size recurrent and matrix state with respect to
   sequence length.
2. The tested 80M Modus_X checkpoint outperforms the tested official NXAI
   xLSTM configuration on the matched enwik8 dense audit.
3. The tested official Mamba configuration outperforms Modus_X on the matched
   enwik8 dense audit.
4. Modus_X strongly outperforms the tested official Mamba configuration on the
   recovered balanced-KV associative-recall protocol.
5. Modus_X strongly outperforms the tested official Mamba configuration under
   50% same-key overwrite.
6. On the recovered associative-memory protocol, forcing Modus_X to use the
   matrix stream preserves high recall, while forcing the vector stream alone
   reduces accuracy to near chance.

## Not Supported

- Modus_X universally outperforms Mamba, xLSTM, RWKV, or Transformers.
- Modus_X has reached `1.1 BPC`.
- Synthetic associative recall alone proves superior natural-language
  long-context ability.
- The current research implementation is throughput competitive with fused
  official Mamba kernels.
- Modus_X has demonstrated billion-parameter scaling.
- The lean vector router, by itself, is the cause of the associative-memory
  advantage or is generally superior to a scalar router.

## Required Claim Format

Every comparative statement must include:

- model parameter count;
- optimizer updates and processed characters or examples;
- dataset and split;
- evaluation mode;
- implementation source;
- precision and accelerator;
- uncertainty, seeds, and relevant caveats.
