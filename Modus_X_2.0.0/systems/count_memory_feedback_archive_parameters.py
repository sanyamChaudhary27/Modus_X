"""Catalog the parameter and recurrent-state cost of a 1B MFA candidate."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Config:
    vocab_size: int = 50_257
    embed_dim: int = 1_536
    hidden_dim: int = 4_608
    matrix_state_dim: int = 896
    vector_state_dim: int = 960
    n_layers: int = 32
    router_hidden: int = 128
    vector_router: bool = True
    feedback_rank: int = 32


FROZEN_V1_PARAMS = 1_058_963_121


def head_params(cfg: Config) -> int:
    return (
        cfg.hidden_dim * cfg.embed_dim
        + cfg.hidden_dim
        + cfg.vocab_size * cfg.hidden_dim
        + cfg.vocab_size
    )


def base_layer_params(cfg: Config) -> int:
    d = cfg.embed_dim
    r = cfg.matrix_state_dim
    n = cfg.vector_state_dim
    rh = cfg.router_hidden
    router_out = d if cfg.vector_router else 1
    pre_norm = 2 * d
    matrix_stream = (
        3 * r * d
        + 3 * (d + 1)
        + r * d
        + r
        + d * d
        + d
        + d * (d + r)
        + d
        + 2 * r
    )
    vector_stream = 4 * n * d + 2 * n + d * d + d + d * n + d
    router = rh * d + rh + router_out * rh + router_out
    return pre_norm + matrix_stream + vector_stream + router


def archive_controller_params(cfg: Config) -> int:
    d, r = cfg.embed_dim, cfg.matrix_state_dim
    return (d + 1) + (d + 1) + (r * d + r)


def feedback_bridge_params(cfg: Config) -> int:
    d, r, rank = cfg.embed_dim, cfg.matrix_state_dim, cfg.feedback_rank
    return rank * r + d * rank + d + 1


def total_params(cfg: Config) -> int:
    embedding = cfg.vocab_size * cfg.embed_dim
    layer = (
        base_layer_params(cfg)
        + archive_controller_params(cfg)
        + feedback_bridge_params(cfg)
    )
    return embedding + cfg.n_layers * layer + head_params(cfg)


def recurrent_state_bytes(cfg: Config, bytes_per_value: int = 2) -> int:
    per_layer_values = 2 * cfg.matrix_state_dim**2 + cfg.vector_state_dim
    return cfg.n_layers * per_layer_values * bytes_per_value


def report(cfg: Config = Config()) -> dict[str, object]:
    params = total_params(cfg)
    frozen_state_bytes = 32 * (1_024**2 + 1_024) * 2
    candidate_state_bytes = recurrent_state_bytes(cfg)
    return {
        "architecture": "Modus_X_MemoryFeedbackArchive",
        "config": asdict(cfg),
        "params": params,
        "parameter_delta_vs_frozen": params - FROZEN_V1_PARAMS,
        "parameter_delta_fraction": (params - FROZEN_V1_PARAMS) / FROZEN_V1_PARAMS,
        "bf16_recurrent_state_bytes_per_sequence": candidate_state_bytes,
        "frozen_v1_bf16_recurrent_state_bytes_per_sequence": frozen_state_bytes,
        "recurrent_state_ratio_vs_frozen": candidate_state_bytes / frozen_state_bytes,
        "base_layer_params": base_layer_params(cfg),
        "archive_controller_params_per_layer": archive_controller_params(cfg),
        "feedback_bridge_params_per_layer": feedback_bridge_params(cfg),
    }


if __name__ == "__main__":
    result = report()
    assert result["params"] == 1_058_467_601, result
    assert abs(result["parameter_delta_fraction"]) < 0.0005, result
    print(json.dumps(result, indent=2))
