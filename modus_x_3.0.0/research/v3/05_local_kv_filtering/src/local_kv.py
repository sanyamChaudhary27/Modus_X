"""Dependency-preserving execution of the canonical MemoryFeedback layer.

The matrix recurrence runs first. Its reads determine vector coefficients,
which can then be computed over the sequence in parallel. No gates are added.
"""
import jax
from functools import partial
import jax.numpy as jnp
from jax import lax
from models import layer_norm, normalize
from local_context import filter_sequence


@partial(jax.checkpoint, static_argnums=(3,))
def forward(layer, x, initial_state=None, associative=False, archive_clock=1.0):
    r = layer['m_wk'].shape[0]
    if initial_state is None:
        initial_state = (jnp.zeros((r, r)), jnp.zeros((r, r)),
                         jnp.zeros(layer['s_wu'].shape[0]),jnp.zeros((2,r)),jnp.zeros((2,r)))
    current, archive, vector, key_history, value_history = initial_state
    # All of these depend on the layer input, never on its vector state.
    e = layer_norm(x, layer['pre_g'], layer['pre_b'])
    project = lambda name: jax.vmap(lambda token: layer[name] @ token)(e)
    key_history, key_raw = filter_sequence(project('m_wk'),layer['kv_key_lags'],key_history)
    value_history, value_raw = filter_sequence(project('m_wv'),layer['kv_value_lags'],value_history)
    key = jax.vmap(normalize)(key_raw)
    query = jax.vmap(normalize)(project('m_wq'))
    value = jnp.tanh(value_raw)
    gate = lambda name: jax.nn.sigmoid(project('m_w_' + name) + layer['m_b_' + name])
    eta, write, retain = (gate(name)[:, 0] for name in ('eta', 'write', 'ret'))
    archive_write = gate('archive_write')[:, 0]
    archive_logit = (project('m_w_archive_ret') + layer['m_b_archive_ret'])[:, 0]
    archive_retain = jax.nn.sigmoid(archive_logit + jnp.log(archive_clock))

    def matrix_step(state, inputs):
        h, a = state
        k, q, v, rate, w, ret, aw, ar = inputs
        old = h @ k
        h = ret * h + (rate * w) * jnp.outer(v - old, k)
        old_a = a @ k
        a = ar * a + (rate * w * aw) * jnp.outer(v - old_a, k)
        return (h, a), (h @ q, a @ q)

    (current, archive), (reads, archive_reads) = lax.scan(
        matrix_step, (current, archive),
        (key, query, value, eta, write, retain, archive_write, archive_retain))
    reads = layer_norm(reads, layer['m_ln_g'], layer['m_ln_b'])
    archive_reads = layer_norm(archive_reads, layer['m_ln_g'], layer['m_ln_b'])
    mix = gate('archive_mix')
    context = gate('read') * (mix * reads + (1 - mix) * archive_reads)
    proposal = jax.vmap(lambda token: layer['m_proj_w'] @ token)(
        jnp.concatenate((x, context), axis=-1)) + layer['m_proj_b']
    matrix_out = gate('out') * proposal
    feedback_gate = jax.nn.sigmoid(project('s_w_memory_feedback') + layer['s_b_memory_feedback'])
    feedback = jax.vmap(lambda c: layer['s_memory_up'] @ jnp.tanh(layer['s_memory_down'] @ c))(context)
    ev = layer_norm(x + feedback_gate * feedback, layer['pre_g'], layer['pre_b'])
    pv = lambda name: jax.vmap(lambda token: layer[name] @ token)(ev)
    u = jnp.tanh(pv('s_wu'))
    delta = jax.nn.sigmoid(pv('s_w_delta') + layer['s_b_delta'])
    ret = jax.nn.sigmoid(pv('s_w_ret') + layer['s_b_ret'])
    if associative:
        def compose(left, right):
            a, b = left
            c, d = right
            return c * a, c * b + d
        prefix_a, prefix_b = lax.associative_scan(compose, (ret, delta * u))
        states = prefix_a * vector + prefix_b
        vector = states[-1]
    else:
        def vector_step(s, inputs):
            a, b, c = inputs
            s = a * s + b * c
            return s, s
        vector, states = lax.scan(vector_step, vector, (ret, delta, u))
    select = jax.nn.sigmoid(pv('s_w_c'))
    out_gate = jax.nn.sigmoid(pv('s_w_gate') + layer['s_b_gate'])
    vector_out = out_gate * (jax.vmap(lambda y: layer['s_proj_w'] @ y)(select * states) + layer['s_proj_b'])
    router_hidden = jax.nn.gelu(project('r_w') + layer['r_b'])
    router_logits = jax.vmap(lambda y: layer['r_proj'] @ y)(router_hidden) + layer['r_proj_b']
    router = jax.nn.sigmoid(router_logits)
    return (current, archive, vector,key_history,value_history), router * matrix_out + (1 - router) * vector_out


@partial(jax.checkpoint, static_argnums=(3,))
def chunked(layer, x, state, chunk_size, archive_clock=1.0):
    if state is None:
        r=layer['m_wk'].shape[0]
        state=(jnp.zeros((r,r)),jnp.zeros((r,r)),jnp.zeros(layer['s_wu'].shape[0]),jnp.zeros((2,r)),jnp.zeros((2,r)))
    if x.shape[0] % chunk_size:
        raise ValueError('Sequence length must be divisible by chunk size')
    blocks = x.reshape((-1, chunk_size, x.shape[-1]))
    def step(carry, block):
        return forward(layer, block, carry, False, archive_clock)
    state, out = lax.scan(step, state, blocks)
    return state, out.reshape(x.shape)
