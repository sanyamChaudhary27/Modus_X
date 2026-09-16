"""Paired screen/continuation; test is report-only at the frozen 25k endpoint."""
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
from endpoint_recovery import discover, load_pair, save_pair


def baseline(layer,x,state=None):
    if state is None:
        r=layer['m_wk'].shape[0]
        state=(jnp.zeros((r,r)),jnp.zeros((r,r)),jnp.zeros(layer['s_wu'].shape[0]))
    return two_stage.chunked(layer,x,state,32,512.)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--outdir',type=Path,required=True)
    ap.add_argument('--seed',type=int,choices=(1,2),required=True)
    ap.add_argument('--resume',action='store_true')
    a=ap.parse_args(); a.outdir.mkdir(parents=True,exist_ok=True)
    began=time.perf_counter()
    jax.config.update('jax_default_matmul_precision','highest')
    assert jax.default_backend()=='tpu' and jax.device_count()==8
    data=ensure(); raw=np.memmap(data,dtype=np.uint8,mode='r')
    train=raw[:90000000]; valid=raw[90000000:95000000]
    mesh=Mesh(np.asarray(jax.devices()),('data',)); rep=NamedSharding(mesh,P()); bs=NamedSharding(mesh,P('data',None))
    cfg=models.ModelConfig(vocab_size=256,embed_dim=512,hidden_dim=1536,ax_res=512,mamba_state_dim=512,
        n_layers=12,seq_len=512,router_hidden=32,vector_router=False)
    p,fwd=models.make_model('Modus_X_MemoryFeedbackArchive_DeepSupervision',jax.random.key(a.seed),cfg,
        auxiliary_layers=(6,),future_target_count=1,dropout_rate=0.)
    q=dict(p); q['layers']=dict(p['layers'])
    q['layers'].update(kv_key_lags=jnp.zeros((12,2,512)),kv_value_lags=jnp.zeros((12,2,512)))
    assert models.count_params(p)==47437768
    assert models.count_params(q)-models.count_params(p)==24576
    tx=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(6e-4,weight_decay=1e-4))
    states={n:jax.device_put((v,tx.init(v)),rep) for n,v in (('control',p),('local_kv',q))}
    report={'seed':a.seed,'steps':25000,'characters':102400000,'test_evaluated':False,
        'params':{'control':models.count_params(p),'local_kv':models.count_params(q)},
        'extra_inference_state_fp32_bytes_per_sequence':98304,
        'lr':.0006,'weight_decay':.0001,'future_target':2,'future_weight':.5,'auxiliary_weight':.05,
        'precision':'highest','jax':jax.__version__,'optax':optax.__version__,
        'dataset_sha256':SHA256,'data_path':str(data),'config':vars(cfg),
        'evaluation_protocol':'90M:95M; reset512; dense offsets 0,256; stride512; primary head',
        'training_protocol':'paired identical random batches; batch8; archive retention scale512',
        'schedule':[{'through_step':20000,'lr':.0006},{'through_step':25000,'lr':.0003}],
        'devices':list(map(str,jax.devices())),
        'thresholds':{'minimum_dense_validation_gain':.005,'maximum_update_runtime_ratio':1.1},'rows':[]}
    rng=np.random.default_rng(1000+a.seed); start=0
    if a.resume:
        source,start=discover(a.seed,a.outdir/'restored')
        if start==25000: raise RuntimeError('This pair is already complete at 25k; do not retrain it.')
        restored=load_pair(source)
        for name,expected in states.items():
            item=restored[name]
            if item['step']!=start: raise ValueError('Checkpoint and recovery step differ')
            if item['config']!=vars(cfg): raise ValueError('Checkpoint config differs')
            actual=(item['params'],item['opt_state'])
            if jax.tree.structure(actual)!=jax.tree.structure(expected): raise ValueError('Parameter/optimizer structure differs')
            for got,want in zip(jax.tree.leaves(actual),jax.tree.leaves(expected)):
                if np.shape(got)!=np.shape(want) or np.asarray(got).dtype!=want.dtype: raise ValueError('Checkpoint leaf shape/dtype mismatch')
                if not np.isfinite(got).all(): raise ValueError('Non-finite checkpoint')
            states[name]=jax.device_put(actual,rep)
            counts=[v for v in jax.tree.leaves(item['opt_state']) if np.shape(v)==() and np.issubdtype(np.asarray(v).dtype,np.integer)]
            if not counts or any(int(v)!=start for v in counts): raise ValueError('AdamW count does not match checkpoint step')
        rng.bit_generator.state=restored['control']['rng_state']
        report['source_checkpoint_directory']=str(source)
        del restored
    report['start_step']=start
    raw_batch=np.random.default_rng(19).integers(0,256,(8,513),dtype=np.int32)
    dx=jax.device_put(raw_batch[:,:-1],bs); dy=jax.device_put(raw_batch[:,1:],bs)
    compiled={}; evals={}; initial_loss={}; original=models.modus_x_memory_feedback_archive_layer_fwd_stateful
    try:
        for name,impl in (('control',baseline),('local_kv',lambda p,x,s=None:local_kv.chunked(p,x,s,32,512.))):
            models.modus_x_memory_feedback_archive_layer_fwd_stateful=impl
            def update(p,s,x,y,scale):
                val,g=jax.value_and_grad(lambda p:loss_fn(p,fwd,x,y,.05,512,(2,),.5))(p)
                u,s=tx.update(g,s,p)
                u=jax.tree.map(lambda v:v*scale,u)
                return optax.apply_updates(p,u),s,val
            compiled[name]=jax.jit(update,in_shardings=(rep,rep,bs,bs,rep),out_shardings=(rep,rep,rep)).lower(*states[name],dx,dy,jnp.array(1.,jnp.float32)).compile()
            warm=compiled[name](*states[name],dx,dy,jnp.array(1.,jnp.float32)); jax.block_until_ready(warm)
            initial_loss[name]=float(warm[2]); del warm
            def evaluate(p,x,y):
                logits=jax.vmap(lambda z:fwd(p,z)[0])(x)
                return -jnp.take_along_axis(jax.nn.log_softmax(logits),y[...,None],axis=-1).sum()/jnp.log(2.)
            evals[name]=jax.jit(evaluate,in_shardings=(rep,bs,bs)).lower(states[name][0],dx,dy).compile()
            print('LOCAL_KV_COMPILED',name,report['params'][name],flush=True)
        report['initial_full_objective_losses']=initial_loss
        if not start: assert abs(initial_loss['control']-initial_loss['local_kv'])<1e-5,'Identity initialization failed on TPU'
        if start:
            # Round-trip both arms, then compare the exact next batch and optimizer update.
            root=save_pair(a.outdir,a.seed,start,states,rng,cfg)
            reloaded=load_pair(root)
            clone=np.random.default_rng(); clone.bit_generator.state=reloaded['control']['rng_state']
            probe=np.random.default_rng(); probe.bit_generator.state=rng.bit_generator.state
            starts=probe.integers(0,len(train)-513,size=8)
            assert np.array_equal(starts,clone.integers(0,len(train)-513,size=8))
            xx,yy=batch_at(train,starts,512); xx=jax.device_put(xx,bs); yy=jax.device_put(yy,bs)
            scale=jnp.array(.5 if start>=20000 else 1.,jnp.float32)
            for name in states:
                lhs=compiled[name](*states[name],xx,yy,scale)
                rhs=compiled[name](*jax.device_put((reloaded[name]['params'],reloaded[name]['opt_state']),rep),xx,yy,scale)
                for l,r in zip(jax.tree.leaves(lhs),jax.tree.leaves(rhs)):
                    if not bool(jnp.array_equal(l,r)): raise RuntimeError('Resume update parity failed')
            del reloaded,lhs,rhs
            print('CHECKPOINT_RESTORE_NEXT_UPDATE_PASS',start,flush=True)
        def dense_eval(dataset,name):
            bits=0.; tokens=0
            for offset in (0,256):
                starts=np.arange(offset,len(dataset)-512,512)
                for i in range(0,len(starts),8):
                    chunk=starts[i:i+8]
                    groups=[(chunk,1.)] if len(chunk)==8 else [(np.full(8,s),.125) for s in chunk]
                    for ss,weight in groups:
                        x,y=batch_at(dataset,ss,512)
                        bits+=weight*float(evals[name](states[name][0],jax.device_put(x,bs),jax.device_put(y,bs))); tokens+=4096*weight
            return bits/tokens
        for step in range(start+1,25001):
            starts=rng.integers(0,len(train)-513,size=8)
            x,y=batch_at(train,starts,512); x=jax.device_put(x,bs); y=jax.device_put(y,bs)
            row={'step':step}
            for name in (list(states) if step%2 else list(reversed(states))):
                scale=jnp.array(.5 if step>20000 else 1.,jnp.float32)
                t=time.perf_counter(); p,s,loss=compiled[name](*states[name],x,y,scale); jax.block_until_ready((p,s,loss))
                row[name]={'loss':float(loss),'seconds':time.perf_counter()-t}
                assert np.isfinite(float(loss)); states[name]=(p,s)
            report['rows'].append(row)
            if step%100==0: print('LOCAL_KV_PROGRESS',json.dumps(row),flush=True)
            if step%1000==0:
                save_pair(a.outdir,a.seed,step,states,rng,cfg)
                for name in states:
                    total=0.
                    starts=np.linspace(0,len(valid)-513,128,dtype=np.int64)
                    for i in range(0,128,8):
                        vx,vy=batch_at(valid,starts[i:i+8],512)
                        total+=float(evals[name](states[name][0],jax.device_put(vx,bs),jax.device_put(vy,bs)))
                    print('CHECKPOINT',json.dumps({'case':name,'seed':a.seed,'step':step,
                        'processed_characters':step*4096,'loss':row[name]['loss'],'val_bpc':total/(128*512),
                        'evaluation':'sparse_progress_only','lr':.0003 if step>20000 else .0006,
                        'elapsed_s':time.perf_counter()-began,'elapsed_scope':'this paired session'}),flush=True)
                (a.outdir/'screen_result.json').write_text(json.dumps(report,indent=2))
            if step==5000:
                values={n:dense_eval(valid,n) for n in states}
                med={n:statistics.median(r[n]['seconds'] for r in report['rows'][10:]) for n in states}
                screen={'seed':a.seed,'dense_validation':values,'gain':values['control']-values['local_kv'],
                    'runtime_ratio':med['local_kv']/med['control'],'test_evaluated':False}
                screen['screen_pass']=screen['gain']>=.005 and screen['runtime_ratio']<=1.1
                report['replication_screen']=screen
                print('LOCAL_KV_REPLICATION_SCREEN_DECISION',json.dumps(screen),flush=True)
                if not screen['screen_pass']:
                    report['stopped_at_step']=step
                    print('LOCAL_KV_STOPPED_AT_SCREEN; checkpoints preserved; no test evaluation',flush=True)
                    return
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
        report['endpoint_pass']=report['gain']>=.005 and report['runtime_ratio']<=1.1
        report['next']='aggregate independent seed results; no tuning on test'
        (a.outdir/'endpoint_validation_decision.json').write_text(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
        print('LOCAL_KV_ENDPOINT_VALIDATION_DECISION',json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
        report['dense_test_report_only']={}
        report['test_evaluated']=True
        for name in states:
            value=dense_eval(raw[95000000:100000000],name)
            report['dense_test_report_only'][name]=value
            print('FINAL_DENSE_TEST_BPC',json.dumps({'case':name,'seed':a.seed,'bpc':value,'role':'report_only'}),flush=True)
        report['test_evaluated']=True
        print('LOCAL_KV_ENDPOINT_COMPLETE',json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
    finally:
        models.modus_x_memory_feedback_archive_layer_fwd_stateful=original
        (a.outdir/'screen_result.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__': main()
