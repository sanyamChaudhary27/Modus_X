"""Opt-in v3 execution; never mutates learned parameters or checkpoint state."""
from contextlib import contextmanager
import jax
import jax.numpy as jnp
import models
from segment_scale_trainer import segment_scale_memory_feedback_stateful
from two_stage import chunked


def two_stage_chunk32(layer, tokens, state=None):
    if state is None:
        r=layer['m_wk'].shape[0]
        state=(jnp.zeros((r,r)),jnp.zeros((r,r)),jnp.zeros(layer['s_wu'].shape[0]))
    return chunked(layer,tokens,state,32,512.)


@contextmanager
def execution_backend(name='canonical'):
    """Trace/compile within this context. Not thread-safe; not a live JIT switch."""
    choices={'canonical':segment_scale_memory_feedback_stateful,
             'two_stage_chunk32':two_stage_chunk32}
    if name not in choices: raise ValueError(f'Unknown execution backend: {name}')
    original=models.modus_x_memory_feedback_archive_layer_fwd_stateful
    try:
        with jax.default_matmul_precision('highest'):
            models.modus_x_memory_feedback_archive_layer_fwd_stateful=choices[name]
            yield
    finally:
        models.modus_x_memory_feedback_archive_layer_fwd_stateful=original
