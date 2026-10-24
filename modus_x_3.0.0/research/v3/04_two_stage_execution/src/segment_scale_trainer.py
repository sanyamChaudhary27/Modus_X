from __future__ import annotations

import math
import runpy
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import lax

import models


@jax.checkpoint
def segment_scale_memory_feedback_stateful(layer, x_seq, initial_state=None):
    """Canonical MemoryFeedbackArchive with archive retention on a 512-token clock."""
    r = layer["m_wk"].shape[0]

    def step(carry, e_raw):
        H_current, H_archive, s = carry
        e = models.layer_norm(e_raw, layer["pre_g"], layer["pre_b"])

        k = models.normalize(layer["m_wk"] @ e)
        q = models.normalize(layer["m_wq"] @ e)
        val = jnp.tanh(layer["m_wv"] @ e)
        eta = jax.nn.sigmoid((layer["m_w_eta"] @ e + layer["m_b_eta"])[0])
        write = jax.nn.sigmoid((layer["m_w_write"] @ e + layer["m_b_write"])[0])
        retain = jax.nn.sigmoid((layer["m_w_ret"] @ e + layer["m_b_ret"])[0])
        old_current = H_current @ k
        H_current = retain * H_current + (eta * write) * jnp.outer(val - old_current, k)

        archive_write = jax.nn.sigmoid(
            (layer["m_w_archive_write"] @ e + layer["m_b_archive_write"])[0]
        )
        archive_logit = (layer["m_w_archive_ret"] @ e + layer["m_b_archive_ret"])[0]
        archive_retain = jax.nn.sigmoid(archive_logit + math.log(512.0))
        old_archive = H_archive @ k
        H_archive = archive_retain * H_archive + (
            eta * write * archive_write
        ) * jnp.outer(val - old_archive, k)

        read_gate = jax.nn.sigmoid(layer["m_w_read"] @ e + layer["m_b_read"])
        current_context = models.layer_norm(H_current @ q, layer["m_ln_g"], layer["m_ln_b"])
        archive_context = models.layer_norm(H_archive @ q, layer["m_ln_g"], layer["m_ln_b"])
        archive_mix = jax.nn.sigmoid(layer["m_w_archive_mix"] @ e + layer["m_b_archive_mix"])
        context = read_gate * (
            archive_mix * current_context + (1.0 - archive_mix) * archive_context
        )
        proposal = layer["m_proj_w"] @ jnp.concatenate([e_raw, context]) + layer["m_proj_b"]
        out_gate = jax.nn.sigmoid(layer["m_w_out"] @ e + layer["m_b_out"])
        modus_out = out_gate * proposal

        feedback_gate = jax.nn.sigmoid(
            (layer["s_w_memory_feedback"] @ e + layer["s_b_memory_feedback"])[0]
        )
        compressed_memory = jnp.tanh(layer["s_memory_down"] @ context)
        memory_feedback = layer["s_memory_up"] @ compressed_memory
        e_vector = models.layer_norm(
            e_raw + feedback_gate * memory_feedback,
            layer["pre_g"],
            layer["pre_b"],
        )

        u = jnp.tanh(layer["s_wu"] @ e_vector)
        delta = jax.nn.sigmoid(layer["s_w_delta"] @ e_vector + layer["s_b_delta"])
        ret_s = jax.nn.sigmoid(layer["s_w_ret"] @ e_vector + layer["s_b_ret"])
        s = ret_s * s + delta * u
        c = jax.nn.sigmoid(layer["s_w_c"] @ e_vector)
        y_s = c * s
        gate_s = jax.nn.sigmoid(layer["s_w_gate"] @ e_vector + layer["s_b_gate"])
        mamba_out = gate_s * (layer["s_proj_w"] @ y_s + layer["s_proj_b"])

        r_hidden = jax.nn.gelu(layer["r_w"] @ e + layer["r_b"])
        r_logits = layer["r_proj"] @ r_hidden + layer["r_proj_b"]
        router = jax.nn.sigmoid(
            r_logits[0] if layer["r_proj"].shape[0] == 1 else r_logits
        )
        out = router * modus_out + (1.0 - router) * mamba_out
        return (H_current, H_archive, s), out

    if initial_state is None:
        initial_state = (
            jnp.zeros((r, r)),
            jnp.zeros((r, r)),
            jnp.zeros(layer["s_wu"].shape[0]),
        )
    return lax.scan(step, initial_state, x_seq)


def main():
    models.modus_x_memory_feedback_archive_layer_fwd_stateful = (
        segment_scale_memory_feedback_stateful
    )
    original_make_model = models.make_model

    def candidate_make_model(name, *args, **kwargs):
        if name == "Modus_X_MemoryFeedbackSegmentRetention_DeepSupervision":
            name = "Modus_X_MemoryFeedbackArchive_DeepSupervision"
        return original_make_model(name, *args, **kwargs)

    models.make_model = candidate_make_model
    trainer = Path(__file__).with_name("tpu_lm_train.py")
    runpy.run_path(str(trainer), run_name="__main__")


if __name__ == "__main__":
    main()
