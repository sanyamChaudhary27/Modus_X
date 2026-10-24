import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import jax.numpy as jnp
import optax
from endpoint_recovery import save_pair,inspect_pair,load_pair,discover

class RecoveryTests(unittest.TestCase):
    def test_pairs_directories_zip_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); rng=np.random.default_rng(1001)
            states={c:({'w':np.ones(2,np.float32)},()) for c in ('control','local_kv')}
            early=save_pair(d/'input',1,1000,states,rng,SimpleNamespace(width=2))
            self.assertEqual(inspect_pair(early,1)[0],1000)
            save_pair(d/'input',1,4000,states,rng,SimpleNamespace(width=2))
            root=save_pair(d/'input',1,5000,states,rng,SimpleNamespace(width=2))
            self.assertFalse(early.exists())
            self.assertEqual(inspect_pair(root,1)[0],5000)
            pair=load_pair(root)
            clone=np.random.default_rng(); clone.bit_generator.state=pair['control']['rng_state']
            np.testing.assert_array_equal(rng.integers(100,size=8),clone.integers(100,size=8))
            found,step=discover(1,d/'restore',roots=(d/'input',))
            self.assertEqual(found,root)
            with self.assertRaises(ValueError): inspect_pair(root,2)
            (d/'zips').mkdir()
            with zipfile.ZipFile(d/'zips'/'full.zip','w') as z:
                for f in root.iterdir(): z.write(f,'nested/output/'+f.name)
            found,_=discover(1,d/'restore',roots=(d/'zips',))
            self.assertEqual(inspect_pair(found,1)[0],5000)
            with (root/'control_checkpoint.pkl').open('ab') as f: f.write(b'broken')
            with self.assertRaises(ValueError): inspect_pair(root,1)

    def test_lr_scaling_matches_adamw_without_reset(self):
        p={'w':jnp.array([1.,2.])}; g={'w':jnp.array([.2,-.1])}
        old=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(6e-4,weight_decay=1e-4))
        new=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(3e-4,weight_decay=1e-4))
        s=old.init(p)
        for _ in range(4):
            u,s=old.update(g,s,p); p=optax.apply_updates(p,u)
        u,a=old.update(g,s,p); v,b=new.update(g,s,p)
        np.testing.assert_allclose(u['w']*.5,v['w'],atol=1e-10,rtol=1e-6)
        import jax
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)): np.testing.assert_array_equal(x,y)

if __name__=='__main__': unittest.main()
