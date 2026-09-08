"""Interleaved CPU/TPU timings and explicit gradient checks for v3 execution."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import jax
import jax.numpy as jnp
import numpy as np
from models import ModelConfig, init_modus_x_memory_feedback_archive_lm, lm_head_fwd, count_params
from segment_scale_trainer import segment_scale_memory_feedback_stateful as reference
from two_stage import forward, chunked
from run_probe import errors


def run(width,depth,length,repeats):
    cfg=ModelConfig(vocab_size=256,embed_dim=width,ax_res=width,hidden_dim=width*3,
                    mamba_state_dim=width,n_layers=depth,router_hidden=32,vector_router=True)
    p=init_modus_x_memory_feedback_archive_lm(jax.random.key(31),cfg)
    x=jax.random.normal(jax.random.key(9),(length,width))
    state=(jnp.ones((depth,width,width))*.001,jnp.ones((depth,width,width))*.002,
           jnp.ones((depth,width))*.01)
    targets=jax.random.randint(jax.random.key(10),(length,),0,256)
    @jax.checkpoint
    def canonical_chunk32(p,x,s):
        def step(carry,block):
            return reference(p,block,carry)
        s,y=jax.lax.scan(step,s,x.reshape((-1,32,width)))
        return s,y.reshape(x.shape)
    impls={'canonical_v3':reference,'two_stage_serial':lambda p,x,s:forward(p,x,s,False,512.),
           'two_stage_chunk32':lambda p,x,s:chunked(p,x,s,32,512.),
           'canonical_chunk32':canonical_chunk32}
    compiled={}
    outputs={}
    grads={}
    rows={}
    functions={}
    for name,impl in impls.items():
        def model(p,x,s,impl=impl):
            def step(h,pack):
                layer,h0,a0,s0=pack
                final,out=impl(layer,h,(h0,a0,s0))
                return h+out,final
            h,s=jax.lax.scan(step,x,(p['layers'],*s))
            return lm_head_fwd(p['head'],h),s
        def loss(p,x,s,model=model):
            logits,final=model(p,x,s)
            return -jnp.take_along_axis(jax.nn.log_softmax(logits),targets[:,None],axis=-1).mean()+1e-3*sum(jnp.mean(a*a) for a in final)
        grad=jax.value_and_grad(loss,argnums=(0,1,2))
        functions[name]=model
        t=time.perf_counter()
        cf=jax.jit(model).lower(p,x,state).compile()
        cg=jax.jit(grad).lower(p,x,state).compile()
        compiled[name]=(cf,cg)
        outputs[name]=cf(p,x,state)
        grads[name]=cg(p,x,state)
        jax.block_until_ready((outputs[name],grads[name]))
        rows[name]={'compile_and_first_execution_s':time.perf_counter()-t,
                    'forward_temp_bytes':cf.memory_analysis().temp_size_in_bytes,
                    'gradient_temp_bytes':cg.memory_analysis().temp_size_in_bytes,
                    'output_error':errors(outputs['canonical_v3'],outputs[name]),
                    'gradient_error':errors(grads['canonical_v3'][1],grads[name][1]),
                    'loss_error':abs(float(grads['canonical_v3'][0])-float(grads[name][0])),
                    'forward_times':[],'gradient_times':[]}
        print('STRESS_COMPILED',width,depth,name,json.dumps(rows[name]),flush=True)
    # Alternate order after compiling all cases to limit order/compile effects.
    for rep in range(repeats+1):
        names=list(impls) if rep%2 else list(reversed(impls))
        for name in names:
            for label,fn in zip(('forward_times','gradient_times'),compiled[name]):
                t=time.perf_counter()
                jax.block_until_ready(fn(p,x,state))
                elapsed=time.perf_counter()-t
                if rep:rows[name][label].append(elapsed)
    for name,row in rows.items():
        f=jax.jit(functions[name])
        left,mid=f(p,x[:length//2],state)
        right,end=f(p,x[length//2:],mid)
        row['chunk_handoff_error']=errors(outputs[name],(jnp.concatenate((left,right)),end))
        row['forward_median_s']=statistics.median(row['forward_times'])
        row['gradient_median_s']=statistics.median(row['gradient_times'])
        row['numerical_pass']=(row['output_error']['max_abs']<1e-5 and
            row['gradient_error']['relative_l2']<1e-4 and row['loss_error']<1e-5 and
            row['chunk_handoff_error']['max_abs']<1e-5)
    return dict(width=width,depth=depth,length=length,params=count_params(p),rows=rows)


if __name__=='__main__':
    jax.config.update('jax_default_matmul_precision','highest')
    parser=argparse.ArgumentParser()
    parser.add_argument('--outdir',type=Path,default=Path(__file__).parent)
    parser.add_argument('--full',action='store_true')
    parser.add_argument('--compact',action='store_true')
    args=parser.parse_args()
    args.outdir.mkdir(parents=True,exist_ok=True)
    results={'backend':jax.default_backend(),'jax':jax.__version__,'matmul_precision':jax.config.jax_default_matmul_precision,'devices':[str(x) for x in jax.devices()],
             'scope':'Random-initialization v3 numerical/system probe; batch one; no LM quality measurement',
             'hashes':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                       for name in ('models.py','segment_scale_trainer.py','two_stage.py','stress_probe.py')},'cases':[]}
    shapes=[(128,12,128),(512,1,128)]
    if args.compact:shapes=[(64,6,128)]
    if args.full:shapes.append((512,12,512))
    for shape in shapes:
        case=run(*shape,repeats=5)
        results['cases'].append(case)
        (args.outdir/'stress_result.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
        print('STRESS_CASE_DONE',json.dumps(case),flush=True)
        if not all(row['numerical_pass'] for row in case['rows'].values()):
            print('NUMERICAL_GATE_FAILED; no larger shape will run',flush=True)
            break
        jax.clear_caches()
    print('STRESS_COMPLETE',str(args.outdir/'stress_result.json'),flush=True)
