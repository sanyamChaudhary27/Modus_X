import unittest
import jax
import jax.numpy as jnp
import numpy as np
import models
from two_stage import forward as baseline
from local_kv import forward,chunked


class LocalKVTests(unittest.TestCase):
    def test_full_objective(self):
        import optax
        from run_screen import baseline as baseline_chunked
        from tpu_lm_train import loss_fn
        cfg=models.ModelConfig(vocab_size=256,embed_dim=16,hidden_dim=48,ax_res=16,
            mamba_state_dim=16,n_layers=2,router_hidden=8)
        p,fwd=models.make_model('Modus_X_MemoryFeedbackArchive_DeepSupervision',jax.random.key(1),cfg,
            auxiliary_layers=(1,),future_target_count=1,dropout_rate=0.)
        q=dict(p); q['layers']=dict(p['layers'],kv_key_lags=jnp.zeros((2,2,16)),kv_value_lags=jnp.zeros((2,2,16)))
        x=jax.random.randint(jax.random.key(2),(1,32),0,256); y=jnp.roll(x,-1,axis=1)
        original=models.modus_x_memory_feedback_archive_layer_fwd_stateful
        results=[]
        try:
            for tree,impl in ((p,baseline_chunked),
                              (q,lambda p,x,s=None:chunked(p,x,s,32,512.))):
                models.modus_x_memory_feedback_archive_layer_fwd_stateful=impl
                value,g=jax.value_and_grad(lambda t:loss_fn(t,fwd,x,y,.05,32,(2,),.5))(tree)
                results.append(float(value))
                self.assertTrue(all(bool(jnp.all(jnp.isfinite(v))) for v in jax.tree.leaves(g)))
                tx=optax.adamw(6e-4,weight_decay=1e-4)
                updates,_=tx.update(g,tx.init(tree),tree)
                updated=optax.apply_updates(tree,updates)
                self.assertTrue(all(bool(jnp.all(jnp.isfinite(v))) for v in jax.tree.leaves(updated)))
            self.assertAlmostEqual(*results,places=5)
        finally:
            models.modus_x_memory_feedback_archive_layer_fwd_stateful=original

    def test_identity_gradients_and_history(self):
        cfg=models.ModelConfig(vocab_size=256,embed_dim=16,hidden_dim=48,ax_res=16,mamba_state_dim=16,n_layers=1,router_hidden=8)
        params=models.init_modus_x_memory_feedback_archive_lm(jax.random.key(1),cfg)
        p=jax.tree.map(lambda v:v[0],params['layers'])
        q=dict(p,kv_key_lags=jnp.zeros((2,16)),kv_value_lags=jnp.zeros((2,16)))
        x=jax.random.normal(jax.random.key(4),(64,16))
        with jax.default_matmul_precision('highest'):
            a,y=baseline(p,x,None,False,512.)
            b,z=forward(q,x,None,False,512.)
            np.testing.assert_allclose(y,z,atol=1e-6,rtol=1e-5)
            for u,v in zip(a,b[:3]): np.testing.assert_allclose(u,v,atol=1e-6,rtol=1e-5)
            grad=jax.grad(lambda p:jnp.sum(forward(p,x,None,False,512.)[1]**2))(q)
            for name in ('kv_key_lags','kv_value_lags'):
                self.assertTrue(bool(jnp.all(jnp.isfinite(grad[name]))))
                self.assertGreater(float(jnp.linalg.norm(grad[name])),0.)
            q['kv_key_lags']=jnp.ones((2,16))*.03
            q['kv_value_lags']=jnp.ones((2,16))*.02
            s,whole=forward(q,x,None,False,512.)
            s1,left=forward(q,x[:17],None,False,512.)
            s2,right=forward(q,x[17:],s1,False,512.)
            np.testing.assert_allclose(whole,jnp.concatenate((left,right)),atol=2e-6,rtol=2e-5)
            for u,v in zip(s,s2): np.testing.assert_allclose(u,v,atol=2e-6,rtol=2e-5)
            _,c=chunked(q,x,None,32,512.)
            np.testing.assert_allclose(whole,c,atol=2e-6,rtol=2e-5)
            changed=x.at[33:].set(9.)
            np.testing.assert_allclose(whole[:33],forward(q,changed,None,False,512.)[1][:33],atol=2e-6,rtol=2e-5)


if __name__=='__main__': unittest.main()
