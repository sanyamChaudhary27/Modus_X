"""Paired from-scratch K/V local-filter screen; no test evaluation."""
import argparse
import json
import pickle
import statistics
import time
from pathlib import Path
import hashlib
import numpy as np
import jax
import jax.numpy as jnp
import optax
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
import models
import two_stage
import local_kv
from tpu_lm_train import loss_fn,batch_at
from enwik8_data import ensure, SHA256


def baseline(layer,x,state=None):
    if state is None:
        r=layer['m_wk'].shape[0]
        state=(jnp.zeros((r,r)),jnp.zeros((r,r)),jnp.zeros(layer['s_wu'].shape[0]))
    return two_stage.chunked(layer,x,state,32,512.)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--outdir',type=Path,required=True)
    a=ap.parse_args(); a.outdir.mkdir(parents=True,exist_ok=True)
    jax.config.update('jax_default_matmul_precision','highest')
    assert jax.default_backend()=='tpu' and jax.device_count()==8
    data=ensure(); raw=np.memmap(data,dtype=np.uint8,mode='r')
    train=raw[:90000000]; valid=raw[90000000:95000000]
    mesh=Mesh(np.asarray(jax.devices()),('data',)); rep=NamedSharding(mesh,P()); bs=NamedSharding(mesh,P('data',None))
    cfg=models.ModelConfig(vocab_size=256,embed_dim=512,hidden_dim=1536,ax_res=512,mamba_state_dim=512,
        n_layers=12,seq_len=512,router_hidden=32,vector_router=False)
    p,fwd=models.make_model('Modus_X_MemoryFeedbackArchive_DeepSupervision',jax.random.key(1),cfg,
        auxiliary_layers=(6,),future_target_count=1,dropout_rate=0.)
    q=dict(p); q['layers']=dict(p['layers'])
    q['layers'].update(kv_key_lags=jnp.zeros((12,2,512)),kv_value_lags=jnp.zeros((12,2,512)))
    assert models.count_params(p)==47437768
    assert models.count_params(q)-models.count_params(p)==24576
    tx=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(6e-4,weight_decay=1e-4))
    states={n:jax.device_put((v,tx.init(v)),rep) for n,v in (('control',p),('local_kv',q))}
    report={'seed':1,'steps':5000,'characters':20480000,'test_evaluated':False,
        'params':{'control':models.count_params(p),'local_kv':models.count_params(q)},
        'extra_inference_state_fp32_bytes_per_sequence':98304,
        'lr':.0006,'weight_decay':.0001,'future_target':2,'future_weight':.5,'auxiliary_weight':.05,
        'precision':'highest','jax':jax.__version__,'optax':optax.__version__,
        'dataset_sha256':SHA256,'data_path':str(data),'config':vars(cfg),
        'evaluation_protocol':'90M:95M; reset512; dense offsets 0,256; stride512; primary head',
        'training_protocol':'from scratch; identical random batches; batch8; 5000 updates; archive retention scale512',
        'devices':list(map(str,jax.devices())),
        'thresholds':{'minimum_dense_validation_gain':.005,'maximum_update_runtime_ratio':1.1},'rows':[]}
    rng=np.random.default_rng(1001)
    raw_batch=np.random.default_rng(19).integers(0,256,(8,513),dtype=np.int32)
    dx=jax.device_put(raw_batch[:,:-1],bs); dy=jax.device_put(raw_batch[:,1:],bs)
    compiled={}; evals={}; initial_loss={}; original=models.modus_x_memory_feedback_archive_layer_fwd_stateful
    try:
        for name,impl in (('control',baseline),('local_kv',lambda p,x,s=None:local_kv.chunked(p,x,s,32,512.))):
            models.modus_x_memory_feedback_archive_layer_fwd_stateful=impl
            def update(p,s,x,y):
                val,g=jax.value_and_grad(lambda p:loss_fn(p,fwd,x,y,.05,512,(2,),.5))(p)
                u,s=tx.update(g,s,p)
                return optax.apply_updates(p,u),s,val
            compiled[name]=jax.jit(update,in_shardings=(rep,rep,bs,bs),out_shardings=(rep,rep,rep)).lower(*states[name],dx,dy).compile()
            warm=compiled[name](*states[name],dx,dy); jax.block_until_ready(warm)
            initial_loss[name]=float(warm[2]); del warm
            def evaluate(p,x,y):
                logits=jax.vmap(lambda z:fwd(p,z)[0])(x)
                return -jnp.take_along_axis(jax.nn.log_softmax(logits),y[...,None],axis=-1).sum()/jnp.log(2.)
            evals[name]=jax.jit(evaluate,in_shardings=(rep,bs,bs)).lower(states[name][0],dx,dy).compile()
            print('LOCAL_KV_COMPILED',name,report['params'][name],flush=True)
        report['initial_full_objective_losses']=initial_loss
        assert abs(initial_loss['control']-initial_loss['local_kv'])<1e-5,'Identity initialization failed on TPU'
        for step in range(1,5001):
            starts=rng.integers(0,len(train)-513,size=8)
            x,y=batch_at(train,starts,512); x=jax.device_put(x,bs); y=jax.device_put(y,bs)
            row={'step':step}
            for name in (list(states) if step%2 else list(reversed(states))):
                t=time.perf_counter(); p,s,loss=compiled[name](*states[name],x,y); jax.block_until_ready((p,s,loss))
                row[name]={'loss':float(loss),'seconds':time.perf_counter()-t}
                assert np.isfinite(float(loss)); states[name]=(p,s)
            report['rows'].append(row)
            if step%100==0: print('LOCAL_KV_PROGRESS',json.dumps(row),flush=True)
            if step%1000==0:
                for name,(p,s) in states.items():
                    target=a.outdir/(name+'_checkpoint.pkl'); partial=target.with_suffix('.partial')
                    with partial.open('wb') as f: pickle.dump({'params':jax.device_get(p),'opt_state':jax.device_get(s),
                        'rng_state':rng.bit_generator.state,'step':step,'experiment':'local_kv_screen',
                        'case':name,'config':vars(cfg)},f,protocol=pickle.HIGHEST_PROTOCOL)
                    partial.replace(target)
                    digest=hashlib.sha256()
                    with target.open('rb') as f:
                        for block in iter(lambda:f.read(1024*1024),b''): digest.update(block)
                    (a.outdir/(name+'_recovery.json')).write_text(json.dumps({'step':step,
                        'file':target.name,'bytes':target.stat().st_size,'sha256':digest.hexdigest()},indent=2))
                (a.outdir/'screen_result.json').write_text(json.dumps(report,indent=2))
        dense={}
        for name,(p,_) in states.items():
            bits=0.; tokens=0
            for offset in (0,256):
                starts=np.arange(offset,len(valid)-512,512)
                for i in range(0,len(starts),8):
                    chunk=starts[i:i+8]
                    groups=[(chunk,1.)] if len(chunk)==8 else [(np.full(8,s),.125) for s in chunk]
                    for ss,weight in groups:
                        x,y=batch_at(valid,ss,512)
                        bits+=weight*float(evals[name](p,jax.device_put(x,bs),jax.device_put(y,bs))); tokens+=4096*weight
            dense[name]=bits/tokens
            print('LOCAL_KV_DENSE_VALIDATION',name,dense[name],flush=True)
        report['dense_validation']=dense
        med={n:statistics.median(r[n]['seconds'] for r in report['rows'][10:]) for n in states}
        report['median_update_seconds']=med
        report['gain']=dense['control']-dense['local_kv']
        report['runtime_ratio']=med['local_kv']/med['control']
        report['screen_pass']=report['gain']>=.005 and report['runtime_ratio']<=1.1
        report['next']='replicate unchanged if pass; otherwise freeze without tuning'
        print('LOCAL_KV_DECISION',json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
    finally:
        models.modus_x_memory_feedback_archive_layer_fwd_stateful=original
        (a.outdir/'screen_result.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__': main()
