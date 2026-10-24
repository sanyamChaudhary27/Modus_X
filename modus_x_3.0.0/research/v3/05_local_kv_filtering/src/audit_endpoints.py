"""Read-only trained-weight validation and streaming execution audit."""
import argparse
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import models
import two_stage
import local_kv
from endpoint_recovery import discover, inspect_pair, load_pair
from enwik8_data import ensure


def zeros(p, candidate):
    layers, r, _ = p['layers']['m_wk'].shape
    state = (jnp.zeros((layers,r,r)), jnp.zeros((layers,r,r)),
             jnp.zeros((layers,p['layers']['s_wu'].shape[1])))
    return state + (jnp.zeros((layers,2,r)),jnp.zeros((layers,2,r))) if candidate else state


def forward(p, ids, state, candidate, chunked):
    module = local_kv if candidate else two_stage
    def layer_step(x, values):
        layer, previous = values
        new, y = (module.chunked(layer,x,previous,32,512.) if chunked
                  else module.forward(layer,x,previous,False,512.))
        return x+y, new
    x, state = jax.lax.scan(layer_step,p['embed'][ids],(p['layers'],state))
    return state, models.lm_head_fwd(p['head'],x)


def error(a,b):
    pairs = [(np.asarray(x),np.asarray(y)) for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b))]
    return {'max_abs': max(float(np.max(np.abs(x-y))) for x,y in pairs),
            'pass': all(np.allclose(x,y,atol=2e-4,rtol=2e-4) for x,y in pairs)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--outdir',type=Path,required=True)
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--working-only',action='store_true',help='Audit outputs produced in this notebook, never attached inputs.')
    a=ap.parse_args(); a.outdir.mkdir(parents=True,exist_ok=True)
    jax.config.update('jax_default_matmul_precision','highest')
    if jax.default_backend()!='tpu' or jax.device_count()!=8:
        raise RuntimeError('Fresh TPU v5e-8 required.')
    roots=('/kaggle/working',) if a.working_only else ('/kaggle/input',)
    source,step=discover(a.seed,a.outdir/'restore',roots=roots)
    if step!=25000: raise ValueError('Attach full25000 endpoint, not the screen or compact evidence ZIP.')
    _,hashes=inspect_pair(source,a.seed)
    pair=load_pair(source)
    expected_scores={1:{'control':1.4827387940010779,'local_kv':1.3954478092701448},
                     2:{'control':1.4424832014749431,'local_kv':1.4322875142646825},
                     3:{'control':1.4279850863336112,'local_kv':1.4067724810584166}}
    if a.seed not in expected_scores: raise ValueError('Only the reported seed1/2/3 endpoints are supported.')
    raw=np.memmap(ensure(),dtype=np.uint8,mode='r')
    valid=raw[90000000:95000000]
    report={'seed':a.seed,'source':str(source),'checkpoint_sha256':dict(zip(('control','local_kv'),hashes)),
            'step':step,'test_evaluated':False,'training_performed':False,
            'precision':'highest','jax':jax.__version__, 'cases':{},
            'reference_boundary':'Serial and chunk32 share layer equations; this is not an independent mathematical implementation.',
            'dense_protocol':'validation90M:95M reset512 offsets0,256 stride512, complete windows only',
            'carry_protocol':'first65536 validation bytes, one lane, contiguous512-token segments; diagnostic, not dense reset comparison'}
    try:
        for case in ('control','local_kv'):
            item=pair.pop(case)
            if item['step']!=25000 or item['seed']!=a.seed: raise ValueError('Checkpoint metadata mismatch')
            for key,value in {'embed_dim':512,'hidden_dim':1536,'ax_res':512,'n_layers':12,'mamba_state_dim':512,'seq_len':512,'router_hidden':32}.items():
                if item['config'].get(key)!=value: raise ValueError('Checkpoint configuration mismatch: '+key)
            candidate=case=='local_kv'; p=jax.tree.map(jnp.asarray,item['params']); del item
            if models.count_params(p)!=(47462344 if candidate else 47437768): raise ValueError('Wrong parameter count')
            if not all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(p)): raise ValueError('Nonfinite parameters')
            state=zeros(p,candidate)
            fast=jax.jit(lambda p,x,s:forward(p,x,s,candidate,True))
            serial=jax.jit(lambda p,x,s:forward(p,x,s,candidate,False))
            ids=jnp.asarray(valid[:512],dtype=jnp.int32)
            s,y=fast(p,ids,state)
            ref=serial(p,ids,state)
            s1,left=fast(p,ids[:256],state); s2,right=fast(p,ids[256:],s1)
            _,changed=fast(p,ids.at[256:].set((ids[256:]+71)%256),state)
            checks={'serial_vs_chunk32':error((s,y),ref),
                    'streaming_handoff':error((s,y),(s2,jnp.concatenate((left,right)))),
                    'causality':error(y[:256],changed[:256])}
            row={'checks':checks}; report['cases'][case]=row
            print('ENDPOINT_AUDIT_CHECKS',case,json.dumps(checks),flush=True)
            if not all(v['pass'] for v in checks.values()): raise ValueError('Trained-weight numerical audit failed')
            def bits(p,x,t):
                _,logits=forward(p,x,zeros(p,candidate),candidate,True)
                return -jnp.take_along_axis(jax.nn.log_softmax(logits),t[:,None],axis=-1).sum()/jnp.log(2.)
            # Eight independent windows distribute across all eight TPU devices.
            evaluate=jax.pmap(jax.vmap(bits,in_axes=(None,0,0)),in_axes=(None,0,0))
            total=0.; count=0
            for offset in (0,256):
                starts=np.arange(offset,len(valid)-512,512)
                for i in range(0,len(starts),8):
                    real=starts[i:i+8]; padded=np.pad(real,(0,8-len(real)),mode='edge')
                    batch=np.stack([valid[t:t+513] for t in padded]).astype(np.int32)
                    values=np.asarray(evaluate(p,batch[:,:-1].reshape(8,1,512),batch[:,1:].reshape(8,1,512))).reshape(8)
                    total+=float(values[:len(real)].sum()); count+=len(real)*512
                    if i%2048==0: print('ENDPOINT_AUDIT_PROGRESS',case,offset,i,len(starts),flush=True)
            row['dense_validation_bpc']=total/count
            row['reported_validation_bpc']=expected_scores[a.seed][case]
            row['reproduction_delta']=row['dense_validation_bpc']-row['reported_validation_bpc']
            row['reproduction_pass']=abs(row['reproduction_delta'])<=1e-4
            row['carry_probe']={}
            for carry in (False,True):
                st=state; total=0.
                for start in range(0,65536,512):
                    ids=jnp.asarray(valid[start:start+512],jnp.int32)
                    st,logits=fast(p,ids,st if carry else state)
                    targets=jnp.asarray(valid[start+1:start+513],jnp.int32)
                    total+=float(-jnp.take_along_axis(jax.nn.log_softmax(logits),targets[:,None],axis=-1).sum()/jnp.log(2.))
                row['carry_probe']['carry' if carry else 'reset']=total/65536
            print('ENDPOINT_AUDIT_CASE',case,json.dumps(row),flush=True)
            del p,state,fast,serial,evaluate,ref,s,y,s1,s2,left,right,changed,st,logits
            jax.clear_caches()
        report['dense_validation_gain']=report['cases']['control']['dense_validation_bpc']-report['cases']['local_kv']['dense_validation_bpc']
        report['audit_pass']=all(row['reproduction_pass'] for row in report['cases'].values())
        print('ENDPOINT_AUDIT_COMPLETE',json.dumps(report),flush=True)
    finally:
        (a.outdir/'endpoint_audit.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__': main()
