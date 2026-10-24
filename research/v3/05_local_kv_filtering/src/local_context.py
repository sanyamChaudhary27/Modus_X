"""Identity-initialized causal depthwise residual filter prototype, not integrated MX."""
import jax.numpy as jnp


def filter_sequence(x,lag_weights,history=None):
    """x: T,D; lag_weights: L,D; history stores last L projected tokens oldest first."""
    length,width=x.shape
    lags=lag_weights.shape[0]
    if history is None: history=jnp.zeros((lags,width),dtype=x.dtype)
    if history.shape!=(lags,width): raise ValueError('History shape mismatch')
    joined=jnp.concatenate((history,x),axis=0)
    output=x
    for lag in range(1,lags+1):
        output=output+lag_weights[lag-1]*joined[lags-lag:lags-lag+length]
    return joined[-lags:] if lags else joined[:0],output
