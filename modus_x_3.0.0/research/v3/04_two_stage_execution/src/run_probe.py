import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from models import ModelConfig, init_modus_x_memory_feedback_archive_lm, lm_head_fwd
from models import modus_x_memory_feedback_archive_layer_fwd_stateful as canonical
from two_stage import forward


def block(tree):
    jax.block_until_ready(tree)


def errors(a, b):
    aa, bb = jax.tree.leaves(a), jax.tree.leaves(b)
    assert jax.tree.structure(a) == jax.tree.structure(b)
    diff2, ref2, maximum = 0., 0., 0.
    for x, y in zip(aa, bb):
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        diff2 += np.square(x-y).sum()
        ref2 += np.square(x).sum()
        maximum = max(maximum, float(np.max(np.abs(x-y))))
    return dict(max_abs=maximum, relative_l2=float(np.sqrt(diff2 / max(ref2, 1e-30))))


def bench(fn, args, repeats=7):
    t = time.perf_counter()
    compiled = jax.jit(fn).lower(*args).compile()
    compile_s = time.perf_counter() - t
    for _ in range(2):
        block(compiled(*args))
    timings = []
    for _ in range(repeats):
        t = time.perf_counter()
        block(compiled(*args))
        timings.append(time.perf_counter() - t)
    memory = compiled.memory_analysis()
    return compiled, dict(compile_s=compile_s, median_s=statistics.median(timings),
                         timings_s=timings,
                         temporary_bytes=int(memory.temp_size_in_bytes) if memory else None)


def run(width, depth, length, seed):
    cfg = ModelConfig(vocab_size=256, embed_dim=width, ax_res=width // 4 * 3,
                      hidden_dim=width*3, mamba_state_dim=width+8, n_layers=depth,
                      router_hidden=8, vector_router=True)
    p = init_modus_x_memory_feedback_archive_lm(jax.random.key(seed), cfg)
    x = jax.random.normal(jax.random.key(seed+99), (length, width))
    initial = (jax.random.normal(jax.random.key(9), (depth, cfg.ax_res, cfg.ax_res))*.01,
               jax.random.normal(jax.random.key(10), (depth, cfg.ax_res, cfg.ax_res))*.01,
               jax.random.normal(jax.random.key(11), (depth, cfg.mamba_state_dim))*.01)
    # Exercise the full residual depth and LM head, including state cotangents.
    def model(params, inputs, state, impl):
        def step(h, packed):
            layer, h0, a0, s0 = packed
            final, out = impl(layer, h, (h0, a0, s0))
            return h + out, final
        h, final = jax.lax.scan(step, inputs, (params['layers'], *state))
        return lm_head_fwd(params['head'], h), final
    targets = jax.random.randint(jax.random.key(7), (length,), 0, 256)
    def objective(params, inputs, state, impl):
        logits, final = model(params, inputs, state, impl)
        nll = -jnp.take_along_axis(jax.nn.log_softmax(logits), targets[:, None], axis=-1).mean()
        return nll + 1e-3 * sum(jnp.mean(z*z) for z in final)
    funcs = {'canonical': canonical,
             'two_stage_serial': lambda p,x,s: forward(p,x,s,False),
             'two_stage_associative': lambda p,x,s: forward(p,x,s,True)}
    rows = {}
    outputs, gradients = {}, {}
    for name, impl in funcs.items():
        fn = lambda p,x,s: model(p,x,s,impl)
        grad = jax.value_and_grad(lambda p,x,s: objective(p,x,s,impl), argnums=(0,1,2))
        compiled, ft = bench(fn, (p,x,initial))
        compiled_grad, gt = bench(grad, (p,x,initial))
        outputs[name] = compiled(p,x,initial)
        gradients[name] = compiled_grad(p,x,initial)
        # Streaming handoff uses exactly the same state layout.
        left, mid = jax.jit(fn)(p,x[:length//2],initial)
        right, end = jax.jit(fn)(p,x[length//2:],mid)
        chunk = (jnp.concatenate((left,right)), end)
        rows[name] = dict(forward=ft, backward=gt,
                         chunk_handoff_error=errors(outputs[name],chunk),
                         output_error=errors(outputs['canonical'], outputs[name]),
                         loss_and_gradient_error=errors(gradients['canonical'], gradients[name]))
        print('CASE', json.dumps(dict(width=width, depth=depth, seed=seed, name=name, **rows[name])), flush=True)
    return dict(width=width, depth=depth, length=length, seed=seed, rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    report = dict(backend=jax.default_backend(), devices=[str(x) for x in jax.devices()],
                  jax=jax.__version__, python=platform.python_version(), cases=[],
                  source_sha256=hashlib.sha256(Path(__file__).with_name('models.py').read_bytes()).hexdigest())
    for width, depth, length, seed in ((32,12,64,1),(32,12,64,2),(128,1,256,1)):
        report['cases'].append(run(width,depth,length,seed))
        (args.outdir/'result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('PROBE_COMPLETE', str(args.outdir/'result.json'), flush=True)
