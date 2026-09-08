"""Same-state diagnostics after the failed integration trajectory gate."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import models
from segment_scale_trainer import segment_scale_memory_feedback_stateful as canonical
from two_stage import chunked
from tpu_lm_train import loss_fn
from run_probe import errors


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def discover(name):
    filename=name+'_integration_checkpoint.pkl'
    preferred=Path('/kaggle/working/mx_two_stage_/463b53e47df1/output')/filename
    if preferred.is_file(): return preferred
    found=[p for root in ('/kaggle/input','/kaggle/working')
           for p in Path(root).rglob(filename) if p.is_file()]
    unique={}
    for p in found: unique.setdefault(digest(p),p)
    if len(unique)!=1:
        raise RuntimeError(f'Need one distinct {filename}; found {found}. Attach original notebook outputs, not the compact ZIP (it omits checkpoints).')
    return next(iter(unique.values()))


def candidate(p,x,s=None):
    if s is None:
        r=p['m_wk'].shape[0]
        s=(jnp.zeros((r,r)),jnp.zeros((r,r)),jnp.zeros(p['s_wu'].shape[0]))
    return chunked(p,x,s,32,512.)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--outdir',type=Path,required=True)
    args=parser.parse_args()
    args.outdir.mkdir(parents=True,exist_ok=True)
    # Discover before compiling. These are the user's own local experiment pickles.
    sources={n:discover(n) for n in ('canonical','two_stage')}
    jax.config.update('jax_default_matmul_precision','highest')
    assert jax.default_backend()=='tpu' and jax.device_count()==8
    mesh=Mesh(np.asarray(jax.devices()),('data',))
    rep=NamedSharding(mesh,P())
    bs=NamedSharding(mesh,P('data',None))
    cfg=models.ModelConfig(vocab_size=256,embed_dim=512,hidden_dim=1536,ax_res=512,
        mamba_state_dim=512,n_layers=12,seq_len=512,router_hidden=32,vector_router=False)
    template,fwd=models.make_model('Modus_X_MemoryFeedbackArchive_DeepSupervision',
        jax.random.key(1),cfg,auxiliary_layers=(6,),future_target_count=1,dropout_rate=0.)
    tx=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(6e-4,weight_decay=1e-4))
    expected=jax.tree.structure((template,tx.init(template)))
    shapes=[v.shape for v in jax.tree.leaves((template,tx.init(template)))]
    del template
    states={}
    report={'scope':'Same-state full-objective diagnostic; not BPC or promotion',
        'prior_trajectory_gate':'FAILED at step14; unchanged', 'jax':jax.__version__,
        'optax':optax.__version__, 'precision':'highest','sources':{},'rows':[]}
    for name,path in sources.items():
        with path.open('rb') as f: saved=pickle.load(f)
        assert saved['step']==10 and saved['data_cursor']==11
        assert jax.tree.structure(saved['state'])==expected
        assert [v.shape for v in jax.tree.leaves(saved['state'])]==shapes
        states[name]=jax.device_put(saved['state'],rep)
        report['sources'][name]={'path':str(path),'sha256':digest(path)}
        del saved
    report['existing_parameter_drift']=errors(states['canonical'][0],states['two_stage'][0])
    report['existing_optimizer_drift']=errors(states['canonical'][1],states['two_stage'][1])
    def batch(step):
        raw=np.random.default_rng(17000+step).integers(0,256,(8,513),dtype=np.int32)
        return jax.device_put(raw[:,:-1],bs),jax.device_put(raw[:,1:],bs)
    x,y=batch(11)
    compiled={}
    original=models.modus_x_memory_feedback_archive_layer_fwd_stateful
    try:
        for name,impl in (('canonical',canonical),('two_stage',candidate)):
            models.modus_x_memory_feedback_archive_layer_fwd_stateful=impl
            def probe(p,s,x,y):
                value,g=jax.value_and_grad(lambda p:loss_fn(p,fwd,x,y,.05,512,(2,),.5))(p)
                norm=optax.global_norm(g)
                updates,new_s=tx.update(g,s,p)
                return value,g,norm,updates,new_s,optax.apply_updates(p,updates)
            compiled[name]=jax.jit(probe,in_shardings=(rep,rep,bs,bs),
                out_shardings=(rep,rep,rep,rep,rep,rep)).lower(*states['canonical'],x,y).compile()
            print('DRIFT_PROBE_COMPILED',name,flush=True)
        for origin,state in states.items():
            for step in (11,14,20):
                x,y=batch(step)
                a=compiled['canonical'](*state,x,y)
                b=compiled['two_stage'](*state,x,y)
                jax.block_until_ready((a,b))
                row={'checkpoint_origin':origin,'batch_index':step,
                    'canonical_loss':float(a[0]),'candidate_loss':float(b[0]),
                    'loss_delta':abs(float(a[0])-float(b[0])),
                    'raw_gradient_error':errors(a[1],b[1]),
                    'gradient_norms':[float(a[2]),float(b[2])],
                    'clip_factors':[min(1.,1./max(float(z[2]),1e-30)) for z in (a,b)],
                    'actual_update_error':errors(a[3],b[3]),
                    'next_optimizer_error':errors(a[4],b[4]),
                    'next_parameter_error':errors(a[5],b[5])}
                report['rows'].append(row)
                print('SAME_STATE_DIAGNOSTIC',json.dumps(row),flush=True)
                del a,b
        report['complete']=True
    finally:
        models.modus_x_memory_feedback_archive_layer_fwd_stateful=original
        (args.outdir/'drift_diagnostic.json').write_text(json.dumps(report,indent=2))
    print('DRIFT_DIAGNOSTIC_COMPLETE',str(args.outdir/'drift_diagnostic.json'),flush=True)


if __name__=='__main__': main()
