import unittest
import jax
import jax.numpy as jnp
import numpy as np
import models
from execution_backend import execution_backend


class BackendTests(unittest.TestCase):
    def test_restore_and_invalid_name(self):
        original=models.modus_x_memory_feedback_archive_layer_fwd_stateful
        with self.assertRaisesRegex(RuntimeError,'intentional'):
            with execution_backend('two_stage_chunk32'):
                raise RuntimeError('intentional')
        self.assertIs(models.modus_x_memory_feedback_archive_layer_fwd_stateful,original)
        with self.assertRaises(ValueError):
            with execution_backend('invalid'): pass

    def test_small_forward(self):
        cfg=models.ModelConfig(vocab_size=256,embed_dim=16,hidden_dim=48,
            ax_res=16,mamba_state_dim=16,n_layers=1,router_hidden=8)
        p=models.init_modus_x_memory_feedback_archive_lm(jax.random.key(3),cfg)
        layer=jax.tree.map(lambda x:x[0],p['layers'])
        x=jax.random.normal(jax.random.key(4),(32,16))
        results=[]
        for name in ('canonical','two_stage_chunk32'):
            with execution_backend(name):
                fn=jax.jit(models.modus_x_memory_feedback_archive_layer_fwd_stateful)
                results.append(fn(layer,x,None))
        for a,b in zip(jax.tree.leaves(results[0]),jax.tree.leaves(results[1])):
            np.testing.assert_allclose(a,b,rtol=1e-4,atol=1e-5)


if __name__=='__main__': unittest.main()
