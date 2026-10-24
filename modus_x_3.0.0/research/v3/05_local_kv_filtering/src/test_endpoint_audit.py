import unittest
import jax
import jax.numpy as jnp
import models
from audit_endpoints import zeros,forward,error


class AuditTests(unittest.TestCase):
    def test_primary_head_and_handoff(self):
        cfg=models.ModelConfig(vocab_size=256,embed_dim=16,hidden_dim=48,ax_res=16,mamba_state_dim=16,n_layers=2,router_hidden=8)
        p=models.init_modus_x_memory_feedback_archive_lm(jax.random.key(1),cfg)
        original=models.modus_x_memory_feedback_archive_layer_fwd_stateful
        from run_endpoint import baseline
        import local_kv
        try:
            for candidate in (False,True):
                if candidate:
                    p=dict(p,layers=dict(p['layers'],kv_key_lags=jnp.ones((2,2,16))*.03,kv_value_lags=jnp.ones((2,2,16))*.02))
                models.modus_x_memory_feedback_archive_layer_fwd_stateful=(lambda p,x,s=None:local_kv.chunked(p,x,s,32,512.)) if candidate else baseline
                ids=jnp.arange(64,dtype=jnp.int32)
                f=jax.jit(lambda p,x,s:forward(p,x,s,candidate,True))
                state=zeros(p,candidate); s,y=f(p,ids,state)
                expected=models.modus_x_memory_feedback_archive_lm_fwd(p,ids,cfg)
                self.assertTrue(error(y,expected)['pass'])
                first,l=f(p,ids[:32],state); end,r=f(p,ids[32:],first)
                self.assertTrue(error((s,y),(end,jnp.concatenate((l,r))))['pass'])
        finally:
            models.modus_x_memory_feedback_archive_layer_fwd_stateful=original


if __name__=='__main__': unittest.main()
