"""Full objective TPU integration gate; synthetic bytes, not a BPC benchmark."""
import argparse
import json
import pickle
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import models
from segment_scale_trainer import segment_scale_memory_feedback_stateful
from two_stage import chunked
from tpu_lm_train import loss_fn
from run_probe import errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    jax.config.update('jax_default_matmul_precision', 'highest')
    assert jax.default_backend() == 'tpu' and jax.device_count() == 8
    mesh = Mesh(np.array(jax.devices()), ('data',))
    replicated = NamedSharding(mesh, P())
    batches = NamedSharding(mesh, P('data', None))
    cfg = models.ModelConfig(vocab_size=256, embed_dim=512, hidden_dim=1536,
        ax_res=512, mamba_state_dim=512, n_layers=12, seq_len=512,
        router_hidden=32, vector_router=False)
    params, fwd = models.make_model('Modus_X_MemoryFeedbackArchive_DeepSupervision',
        jax.random.key(1), cfg, auxiliary_layers=(6,), future_target_count=1,
        dropout_rate=0.0)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(6e-4, weight_decay=1e-4))
    params = jax.device_put(params, replicated)
    states = {name: (params, tx.init(params)) for name in ('canonical', 'two_stage')}
    report = {'scope': 'Synthetic byte integration, not language quality; batch8 seq512, all eight TPU devices',
        'params': models.count_params(params), 'jax': jax.__version__, 'optax': optax.__version__,
        'precision': 'highest', 'devices': list(map(str,jax.devices())), 'rows': [],
        'thresholds': {'loss': 1e-3, 'parameter_relative_l2': 1e-3, 'optimizer_relative_l2': 1e-3},
        'pass': False}
    path = args.outdir / 'integration_result.json'
    def batch(step):
        raw = np.random.default_rng(17000+step).integers(0,256,(8,513),dtype=np.int32)
        return jax.device_put(raw[:,:-1], batches), jax.device_put(raw[:,1:], batches)
    x,y = batch(0)
    compiled = {}
    def candidate_layer(p,x,s=None):
        if s is None:
            r=p['m_wk'].shape[0]
            s=(jnp.zeros((r,r)),jnp.zeros((r,r)),jnp.zeros(p['s_wu'].shape[0]))
        return chunked(p,x,s,32,512.)
    original = models.modus_x_memory_feedback_archive_layer_fwd_stateful
    try:
        for name in states:
            models.modus_x_memory_feedback_archive_layer_fwd_stateful = (
                segment_scale_memory_feedback_stateful if name == 'canonical' else
                candidate_layer)
            # Trace each executable while its own layer implementation is installed.
            def update(p,s,x,y):
                value,g = jax.value_and_grad(lambda p: loss_fn(p,fwd,x,y,.05,512,(2,),.5))(p)
                updates,s = tx.update(g,s,p)
                return optax.apply_updates(p,updates),s,value
            compiled[name] = jax.jit(update, in_shardings=(replicated,replicated,batches,batches),
                out_shardings=(replicated,replicated,replicated)).lower(*states[name],x,y).compile()
            jax.block_until_ready(compiled[name](*states[name],x,y))
            print('INTEGRATION_COMPILED',name,report['params'],flush=True)
        for step in range(1,21):
            x,y = batch(step)
            row = {'step': step}
            for name in (list(states) if step%2 else list(reversed(states))):
                start = time.perf_counter()
                p,s,value = compiled[name](*states[name],x,y)
                jax.block_until_ready((p,s,value))
                states[name] = (p,s)
                row[name] = {'loss':float(value), 'seconds':time.perf_counter()-start}
            row['loss_delta'] = abs(row['canonical']['loss']-row['two_stage']['loss'])
            report['rows'].append(row)
            print('INTEGRATION_UPDATE',json.dumps(row),flush=True)
            assert row['loss_delta'] < 1e-3, row
            if step == 10:
                report['resume'] = {}
                nx,ny = batch(step+1)
                for name in states:
                    checkpoint = args.outdir / (name+'_integration_checkpoint.pkl')
                    with checkpoint.open('wb') as f:
                        pickle.dump({'step':step, 'data_cursor':step+1,
                            'state':jax.device_get(states[name])},f,protocol=pickle.HIGHEST_PROTOCOL)
                    with checkpoint.open('rb') as f:
                        restored = pickle.load(f)
                    assert restored['step']==10 and restored['data_cursor']==11
                    restored_state = jax.device_put(restored['state'], replicated)
                    expected = compiled[name](*states[name],nx,ny)
                    actual = compiled[name](*restored_state,nx,ny)
                    result = errors(expected,actual)
                    assert result['max_abs']==0.0, result
                    report['resume'][name] = result
                    states[name] = restored_state
                    del expected, actual, restored, restored_state
        report['parameter_error'] = errors(states['canonical'][0],states['two_stage'][0])
        report['optimizer_error'] = errors(states['canonical'][1],states['two_stage'][1])
        assert report['parameter_error']['relative_l2']<1e-3
        assert report['optimizer_error']['relative_l2']<1e-3
        report['pass'] = True
    finally:
        models.modus_x_memory_feedback_archive_layer_fwd_stateful = original
        path.write_text(json.dumps(report,indent=2))
        print('INTEGRATION_REPORT',path,flush=True)
    print('INTEGRATION_PASS',json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)


if __name__ == '__main__':
    main()
