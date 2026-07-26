from __future__ import annotations

import jax
import jax.numpy as jnp

from models import (
    ModelConfig,
    count_params,
    make_model,
    modus_x_memory_feedback_archive_diagnostics,
)


def main() -> None:
    cfg = ModelConfig(
        vocab_size=256,
        embed_dim=32,
        hidden_dim=96,
        ax_res=16,
        n_layers=2,
        n_heads_attn=4,
        seq_len=16,
        mamba_state_dim=16,
        vector_router=False,
        router_hidden=8,
    )
    key = jax.random.key(1)
    params, forward = make_model(
        "Modus_X_MemoryFeedbackArchive_DeepSupervision",
        key,
        cfg,
        auxiliary_layers=(1,),
        future_target_count=1,
    )
    control, _ = make_model(
        "Modus_X_CurrentArchive_DeepSupervision",
        key,
        cfg,
        auxiliary_layers=(1,),
        future_target_count=1,
    )
    tokens = jnp.arange(cfg.seq_len, dtype=jnp.int32)
    outputs = forward(params, tokens)
    assert len(outputs) == 4
    assert outputs[0].shape == (cfg.seq_len, cfg.vocab_size)
    loss_fn = lambda tree: jnp.mean(jnp.square(forward(tree, tokens)[0]))
    loss, gradients = jax.value_and_grad(loss_fn)(params)
    assert bool(jnp.isfinite(loss))
    nonfinite_paths = [
        jax.tree_util.keystr(path)
        for path, leaf in jax.tree_util.tree_leaves_with_path(gradients)
        if not bool(jnp.all(jnp.isfinite(leaf)))
    ]
    assert not nonfinite_paths, f"Non-finite gradient leaves: {nonfinite_paths}"

    changed = tokens.at[9:].set((tokens[9:] + 37) % cfg.vocab_size)
    assert bool(
        jnp.allclose(
            forward(params, tokens)[0][:9],
            forward(params, changed)[0][:9],
            atol=1e-6,
            rtol=1e-6,
        )
    )
    rank = min(32, cfg.ax_res, cfg.mamba_state_dim)
    expected_extra = cfg.n_layers * (
        rank * cfg.ax_res
        + cfg.embed_dim * rank
        + cfg.embed_dim
        + 1
    )
    extra = count_params(params) - count_params(control)
    assert extra == expected_extra, (extra, expected_extra)
    diagnostics = modus_x_memory_feedback_archive_diagnostics(params, tokens, cfg)
    assert diagnostics["feedback_gate"].shape == (cfg.n_layers, cfg.seq_len)
    assert all(bool(jnp.all(jnp.isfinite(value))) for value in diagnostics.values())
    gate = jax.nn.sigmoid(params["layers"]["s_b_memory_feedback"])
    assert bool(jnp.all((gate > 0.1) & (gate < 0.13)))
    print(
        {
            "status": "PASS",
            "params": count_params(params),
            "control_params": count_params(control),
            "extra_params": extra,
            "loss": float(loss),
            "feedback_gate_prior": float(jnp.mean(gate)),
        }
    )


if __name__ == "__main__":
    main()
