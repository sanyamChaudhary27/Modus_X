"""Paired 500-update v3 continuation. Validation only, frozen execution screen."""
import argparse
import hashlib
import json
import pickle
import statistics
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import models
from segment_scale_trainer import segment_scale_memory_feedback_stateful as canonical
from diagnose_drift import candidate, digest
from tpu_lm_train import loss_fn, batch_at
from enwik8_data import ensure


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--outdir',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path)
    a=ap.parse_args()
    a.outdir.mkdir(parents=True,exist_ok=True)
    choices={}
    for path in ([a.checkpoint] if a.checkpoint else Path('/kaggle/input').rglob('checkpoint.pkl')):
        config=path.with_name('config.json')
        if not config.exists(): continue
        doc=json.loads(config.read_text()); args=doc.get('args',doc.get('config',{}).get('args',{}))
        if args.get('model')!='Modus_X_MemoryFeedbackSegmentRetention_DeepSupervision': continue
        if doc.get('params')!=47437768: continue
        choices.setdefault(digest(path),(path,doc,args))
    if len(choices)!=1:
        raise RuntimeError('Attach ONE trained 47M v3 SegmentRetention notebook output with checkpoint.pkl and config.json. Synthetic integration checkpoints and v2 checkpoints are not compatible. Candidates: '+str([str(v[0]) for v in choices.values()]))
    sha,(path,config,args)=next(iter(choices.items()))
    # Only the frozen constant schedule is supported; do not silently alter another recipe.
    assert args.get('schedule')=='constant' and args.get('optimizer','adamw')=='adamw'
    assert args.get('future_targets')=='2' and str(args.get('auxiliary_layers'))=='6'
    for key in ('dropout','input_corruption_rate','label_smoothing','auxiliary_decay_chars','future_target_decay_chars'):
        assert not args.get(key,0), (key,args.get(key))
    assert not args.get('paper_auxiliary_schedule',False) and not args.get('aux_future_targets',False)
    assert args.get('precision','float32')=='float32'
    assert args.get('batch')==8 and args.get('input_seq_len')==512
    data=ensure()
    print('ENWIK8_VERIFIED',str(data),data.stat().st_size,flush=True)
    raw=np.memmap(data,dtype=np.uint8,mode='r')
    train=raw[:90000000]; valid=raw[90000000:95000000]
    with path.open('rb') as f: saved=pickle.load(f)
    assert saved['step']>=5000 and 'rng_state' in saved
    jax.config.update('jax_default_matmul_precision','highest')
    assert jax.default_backend()=='tpu' and jax.device_count()==8
    mesh=Mesh(np.asarray(jax.devices()),('data',)); rep=NamedSharding(mesh,P()); bs=NamedSharding(mesh,P('data',None))
    cfg=models.ModelConfig(vocab_size=256,embed_dim=512,hidden_dim=1536,ax_res=512,
        mamba_state_dim=512,n_layers=12,seq_len=512,router_hidden=32,vector_router=False)
    template,fwd=models.make_model('Modus_X_MemoryFeedbackArchive_DeepSupervision',jax.random.key(1),cfg,
        auxiliary_layers=(6,),future_target_count=1,dropout_rate=0.)
    assert jax.tree.structure(template)==jax.tree.structure(saved['params'])
    assert [v.shape for v in jax.tree.leaves(template)]==[v.shape for v in jax.tree.leaves(saved['params'])]
    tx=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(args['lr'],weight_decay=args['weight_decay']))
    assert jax.tree.structure(tx.init(template))==jax.tree.structure(saved['opt_state'])
    initial=jax.device_put((saved['params'],saved['opt_state']),rep)
    rng=np.random.default_rng(); rng.bit_generator.state=saved['rng_state']
    start_step=int(saved['step']); del template,saved
    states={n:initial for n in ('canonical','two_stage')}
    report={'source_checkpoint':str(path),'source_sha256':sha,'source_config':config,
        'start_step':start_step,'updates':500,'added_characters':2048000,'test_evaluated':False,
        'validation_protocol':'90M:95M; reset512; dense offsets0,256; primary head; canonical evaluator for both endpoints',
        'precision':'highest','jax':jax.__version__,'optax':optax.__version__,
        'thresholds':{'candidate_minus_control_dense_validation_max':.002,'minimum_speedup':1.2},'rows':[]}
    compiled={}; original=models.modus_x_memory_feedback_archive_layer_fwd_stateful
    dummy=jax.device_put(np.zeros((8,512),np.int32),bs)
    try:
        for name,impl in (('canonical',canonical),('two_stage',candidate)):
            models.modus_x_memory_feedback_archive_layer_fwd_stateful=impl
            def update(p,s,x,y):
                val,g=jax.value_and_grad(lambda p:loss_fn(p,fwd,x,y,args['auxiliary_weight'],512,(2,),args['future_target_weight']))(p)
                u,s=tx.update(g,s,p)
                return optax.apply_updates(p,u),s,val
            compiled[name]=jax.jit(update,in_shardings=(rep,rep,bs,bs),out_shardings=(rep,rep,rep)).lower(*initial,dummy,dummy).compile()
            jax.block_until_ready(compiled[name](*initial,dummy,dummy))
            print('REAL_GATE_COMPILED',name,flush=True)
        models.modus_x_memory_feedback_archive_layer_fwd_stateful=canonical
        def ev(p,x,y):
            def seq(x):
                outputs=fwd(p,x)
                return outputs[0] if isinstance(outputs,tuple) else outputs
            logits=jax.vmap(seq)(x)
            return -jnp.take_along_axis(jax.nn.log_softmax(logits),y[...,None],axis=-1).sum()/jnp.log(2.)
        evaluator=jax.jit(ev,in_shardings=(rep,bs,bs)).lower(initial[0],dummy,dummy).compile()
        def dense(p):
            bits=0.; tokens=0
            for offset in (0,256):
                starts=np.arange(offset,len(valid)-512,512)
                for i in range(0,len(starts),8):
                    chunk=starts[i:i+8]
                    # Repeat final valid window to retain batch8, then evaluate its contribution separately.
                    if len(chunk)<8:
                        for s in chunk:
                            x,y=batch_at(valid,np.full(8,s),512)
                            bits+=float(evaluator(p,jax.device_put(x,bs),jax.device_put(y,bs)))/8; tokens+=512
                    else:
                        x,y=batch_at(valid,chunk,512)
                        bits+=float(evaluator(p,jax.device_put(x,bs),jax.device_put(y,bs))); tokens+=4096
            return bits/tokens
        report['initial_dense_validation']=dense(initial[0])
        print('REAL_GATE_INITIAL_VALIDATION',report['initial_dense_validation'],flush=True)
        for step in range(1,501):
            starts=rng.integers(0,len(train)-513,size=8)
            x,y=batch_at(train,starts,512); x=jax.device_put(x,bs); y=jax.device_put(y,bs)
            row={'update':step}
            for name in (list(states) if step%2 else list(reversed(states))):
                t=time.perf_counter(); p,s,loss=compiled[name](*states[name],x,y)
                jax.block_until_ready((p,s,loss)); row[name]={'seconds':time.perf_counter()-t,'loss':float(loss)}
                assert np.isfinite(float(loss))
                states[name]=(p,s)
            report['rows'].append(row)
            if step%50==0: print('REAL_PAIRED_UPDATE',json.dumps(row),flush=True)
        report['dense_validation']={name:dense(s[0]) for name,s in states.items()}
        med={n:statistics.median(r[n]['seconds'] for r in report['rows'][10:]) for n in states}
        report['median_update_seconds']=med; report['speedup']=med['canonical']/med['two_stage']
        report['validation_delta']=report['dense_validation']['two_stage']-report['dense_validation']['canonical']
        report['screen_pass']=report['validation_delta']<=.002 and report['speedup']>=1.2
        print('REAL_GATE_DECISION',json.dumps({k:v for k,v in report.items() if k not in ('rows','source_config')}),flush=True)
        for name,(p,s) in states.items():
            with (a.outdir/(name+'_paired_checkpoint.pkl')).open('wb') as f:
                pickle.dump({'params':jax.device_get(p),'opt_state':jax.device_get(s),
                    'step':start_step+500,'rng_state':rng.bit_generator.state,'source_config':config},f,protocol=pickle.HIGHEST_PROTOCOL)
    finally:
        models.modus_x_memory_feedback_archive_layer_fwd_stateful=original
        (a.outdir/'real_data_result.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__': main()
