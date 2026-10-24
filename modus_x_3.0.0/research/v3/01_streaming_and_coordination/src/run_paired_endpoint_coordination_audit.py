from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from models import layer_norm, lm_head_fwd, normalize


LN2 = math.log(2.0)
VALIDATION_START = 90_000_000
VALIDATION_END = 95_000_000
EXPECTED_PARAMS = 47_437_768
EXPECTED_STEP = 25_000
EXPECTED_CHARACTERS = 102_400_000
LOCAL_SAMPLES = 128
MODE_NAMES = (
    "baseline",
    "feedback_off",
    "read_off",
    "write_off",
    "router_neutral",
    "matrix_only",
    "vector_only",
)
PERSISTENT_MODES = ("feedback_off", "router_neutral")
DIAGNOSTIC_NAMES = (
    "write_strength",
    "archive_write_strength",
    "read_gate",
    "feedback_gate",
    "feedback_to_input_ratio",
    "router",
    "router_confidence",
)
ACTIVATION_FOR_MODE = {
    "feedback_off": "feedback_gate",
    "read_off": "read_gate",
    "write_off": "write_strength",
    "router_neutral": "router_confidence",
    "matrix_only": "router",
    "vector_only": "router",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--canonical-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def checkpoint_characters(saved) -> int:
    if "processed_characters" in saved:
        return int(saved["processed_characters"])
    return int(saved.get("step", -1)) * 4096


def load_params(path, base):
    with path.open("rb") as handle:
        saved = pickle.load(handle)
    step = int(saved.get("step", -1))
    characters = checkpoint_characters(saved)
    params = saved["params"]
    count = int(base.count_params(params))
    if step != EXPECTED_STEP or characters != EXPECTED_CHARACTERS or count != EXPECTED_PARAMS:
        raise RuntimeError(
            f"Checkpoint mismatch for {path}: step={step}, characters={characters}, params={count}"
        )
    return params


def intervention_layer_forward(layer, x_seq, initial_state, mode, archive_clock_offset):
    def step(carry, e_raw):
        current, archive, vector = carry
        e = layer_norm(e_raw, layer["pre_g"], layer["pre_b"])
        key = normalize(layer["m_wk"] @ e)
        query = normalize(layer["m_wq"] @ e)
        value = jnp.tanh(layer["m_wv"] @ e)
        eta = jax.nn.sigmoid((layer["m_w_eta"] @ e + layer["m_b_eta"])[0])
        write = jax.nn.sigmoid((layer["m_w_write"] @ e + layer["m_b_write"])[0])
        retain = jax.nn.sigmoid((layer["m_w_ret"] @ e + layer["m_b_ret"])[0])
        old_current = current @ key
        updated_current = retain * current + (eta * write) * jnp.outer(value - old_current, key)

        archive_write = jax.nn.sigmoid(
            (layer["m_w_archive_write"] @ e + layer["m_b_archive_write"])[0]
        )
        archive_logit = (layer["m_w_archive_ret"] @ e + layer["m_b_archive_ret"])[0]
        archive_retain = jax.nn.sigmoid(archive_logit + archive_clock_offset)
        old_archive = archive @ key
        updated_archive = archive_retain * archive + (
            eta * write * archive_write
        ) * jnp.outer(value - old_archive, key)
        write_off = mode == 3
        current = jnp.where(write_off, retain * current, updated_current)
        archive = jnp.where(write_off, archive_retain * archive, updated_archive)

        read_gate = jax.nn.sigmoid(layer["m_w_read"] @ e + layer["m_b_read"])
        current_context = layer_norm(current @ query, layer["m_ln_g"], layer["m_ln_b"])
        archive_context = layer_norm(archive @ query, layer["m_ln_g"], layer["m_ln_b"])
        archive_mix = jax.nn.sigmoid(layer["m_w_archive_mix"] @ e + layer["m_b_archive_mix"])
        context = read_gate * (
            archive_mix * current_context + (1.0 - archive_mix) * archive_context
        )
        context = jnp.where(mode == 2, jnp.zeros_like(context), context)

        proposal = layer["m_proj_w"] @ jnp.concatenate([e_raw, context]) + layer["m_proj_b"]
        out_gate = jax.nn.sigmoid(layer["m_w_out"] @ e + layer["m_b_out"])
        matrix_output = out_gate * proposal

        feedback_gate = jax.nn.sigmoid(
            (layer["s_w_memory_feedback"] @ e + layer["s_b_memory_feedback"])[0]
        )
        memory_feedback = layer["s_memory_up"] @ jnp.tanh(layer["s_memory_down"] @ context)
        effective_feedback = jnp.where(
            mode == 1, jnp.zeros_like(memory_feedback), feedback_gate * memory_feedback
        )
        e_vector = layer_norm(e_raw + effective_feedback, layer["pre_g"], layer["pre_b"])
        proposal_vector = jnp.tanh(layer["s_wu"] @ e_vector)
        delta = jax.nn.sigmoid(layer["s_w_delta"] @ e_vector + layer["s_b_delta"])
        vector_retain = jax.nn.sigmoid(layer["s_w_ret"] @ e_vector + layer["s_b_ret"])
        vector = vector_retain * vector + delta * proposal_vector
        selection = jax.nn.sigmoid(layer["s_w_c"] @ e_vector)
        vector_output_gate = jax.nn.sigmoid(
            layer["s_w_gate"] @ e_vector + layer["s_b_gate"]
        )
        vector_output = vector_output_gate * (
            layer["s_proj_w"] @ (selection * vector) + layer["s_proj_b"]
        )

        router_hidden = jax.nn.gelu(layer["r_w"] @ e + layer["r_b"])
        router_logits = layer["r_proj"] @ router_hidden + layer["r_proj_b"]
        router = jax.nn.sigmoid(
            router_logits[0] if layer["r_proj"].shape[0] == 1 else router_logits
        )
        effective_router = jnp.where(mode == 4, 0.5, router)
        effective_router = jnp.where(mode == 5, 1.0, effective_router)
        effective_router = jnp.where(mode == 6, 0.0, effective_router)
        output = effective_router * matrix_output + (1.0 - effective_router) * vector_output

        diagnostics = jnp.stack(
            [
                eta * write,
                eta * write * archive_write,
                jnp.mean(read_gate),
                feedback_gate,
                jnp.linalg.norm(feedback_gate * memory_feedback)
                / (jnp.linalg.norm(e_raw) + 1e-6),
                jnp.mean(router),
                jnp.mean(jnp.abs(router - 0.5) * 2.0),
            ]
        )
        return (current, archive, vector), (output, diagnostics)

    return lax.scan(step, initial_state, x_seq)


def stateful_forward(params, token_ids, cfg, initial_state, mode, archive_clock_offset):
    del cfg
    hidden = params["embed"][token_ids]

    def scan_layer(hidden_in, inputs):
        layer, current, archive, vector = inputs
        next_state, (layer_output, diagnostics) = intervention_layer_forward(
            layer,
            hidden_in,
            (current, archive, vector),
            mode,
            archive_clock_offset,
        )
        return hidden_in + layer_output, (next_state, diagnostics)

    hidden, (final_state, diagnostics) = lax.scan(
        scan_layer, hidden, (params["layers"], *initial_state)
    )
    return lm_head_fwd(params["head"], hidden), final_state, diagnostics


def batch_forward(params, states, tokens, cfg, mode, archive_clock_offset):
    return jax.vmap(
        lambda sequence, state: stateful_forward(
            params, sequence, cfg, state, mode, archive_clock_offset
        ),
        in_axes=(0, 0),
    )(tokens, states)


def rankdata(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def correlation(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def calibration(activation, regret):
    activation = np.asarray(activation, dtype=np.float64)
    regret = np.asarray(regret, dtype=np.float64)
    order = np.argsort(activation)
    quartile = max(1, len(order) // 4)
    bottom = float(np.mean(regret[order[:quartile]]))
    top = float(np.mean(regret[order[-quartile:]]))
    return {
        "pearson": correlation(activation, regret),
        "spearman": correlation(rankdata(activation), rankdata(regret)),
        "bottom_quartile_regret_bpc": bottom,
        "top_quartile_regret_bpc": top,
        "top_minus_bottom_quartile_regret_bpc": top - bottom,
        "activation_mean": float(np.mean(activation)),
        "activation_std": float(np.std(activation)),
    }


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    if isinstance(value, (int, float, np.number)):
        return bool(np.isfinite(value))
    return True


def main():
    args = parse_args()
    if jax.default_backend() != "tpu" or jax.device_count() != 8:
        raise RuntimeError(
            f"Expected TPU v5e-8, found {jax.default_backend()} with {jax.device_count()} devices"
        )
    if not args.data_path.is_file() or args.data_path.stat().st_size != 100_000_000:
        raise RuntimeError("Expected canonical 100,000,000-byte enwik8")
    sys.path.insert(0, str(args.code_root))
    import run_contiguous_training_screen as base
    from segment_scale_trainer import segment_scale_memory_feedback_stateful

    control_host = load_params(args.canonical_checkpoint, base)
    candidate_host = load_params(args.candidate_checkpoint, base)
    cfg = base.model_config()
    mesh = Mesh(np.asarray(jax.devices(), dtype=object), ("data",))
    replicated = NamedSharding(mesh, P())
    batch_sharding = NamedSharding(mesh, P("data", None))
    state_shardings = (
        NamedSharding(mesh, P("data", None, None, None)),
        NamedSharding(mesh, P("data", None, None, None)),
        NamedSharding(mesh, P("data", None, None)),
    )
    control = jax.device_put(control_host, replicated)
    candidate = jax.device_put(candidate_host, replicated)
    del control_host, candidate_host

    def initialize_state_unsharded():
        single = base.zero_state_single(cfg, jnp.float32)
        return tuple(jnp.broadcast_to(value, (8,) + value.shape) for value in single)

    initialize_state = jax.jit(initialize_state_unsharded, out_shardings=state_shardings)

    @jax.jit
    def evaluate_step(model_params, states, tokens, targets, mode, archive_clock_offset):
        logits, next_state, diagnostics = batch_forward(
            model_params, states, tokens, cfg, mode, archive_clock_offset
        )
        logp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
        nll = -jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
        return (
            jax.tree_util.tree_map(lax.stop_gradient, next_state),
            jnp.sum(nll),
            jnp.mean(diagnostics, axis=(0, 1, 2)),
        )

    data = np.memmap(args.data_path, dtype=np.uint8, mode="r")
    validation = data[VALIDATION_START:VALIDATION_END]
    x_host, y_host = base.validation_lanes(validation)

    parity_tokens = jnp.asarray(np.asarray(validation[:64], dtype=np.int32))
    parity_state = base.zero_state_single(cfg, jnp.float32)
    original_stateful = base.modus_x_memory_feedback_archive_layer_fwd_stateful

    def parity(model_params, candidate_mode):
        base.modus_x_memory_feedback_archive_layer_fwd_stateful = (
            segment_scale_memory_feedback_stateful if candidate_mode else original_stateful
        )
        reference = jax.jit(
            lambda p, t, s: base.stateful_deep_forward(p, t, cfg, s)
        )
        custom = jax.jit(
            lambda p, t, s: stateful_forward(
                p,
                t,
                cfg,
                s,
                jnp.asarray(0),
                jnp.asarray(math.log(512.0) if candidate_mode else 0.0, jnp.float32),
            )
        )
        reference_outputs, reference_state = reference(model_params, parity_tokens, parity_state)
        custom_logits, custom_state, _ = custom(model_params, parity_tokens, parity_state)
        errors = [float(jnp.max(jnp.abs(reference_outputs[0] - custom_logits)))]
        errors.extend(
            float(jnp.max(jnp.abs(left - right)))
            for left, right in zip(reference_state, custom_state)
        )
        return {"max_error": max(errors), "errors": errors}

    parity_results = {
        "canonical": parity(control, False),
        "candidate": parity(candidate, True),
    }
    base.modus_x_memory_feedback_archive_layer_fwd_stateful = original_stateful
    print("PAIRED_COORDINATION_PARITY", json.dumps(parity_results), flush=True)
    if max(row["max_error"] for row in parity_results.values()) > 1e-5:
        raise RuntimeError(f"Paired coordination parity failed: {parity_results}")

    sample_indices = np.unique(
        np.linspace(0, x_host.shape[0] - 1, LOCAL_SAMPLES, dtype=np.int32)
    )
    sample_set = set(int(value) for value in sample_indices)
    tokens_per_segment = int(x_host.shape[1] * x_host.shape[2])

    def audit_one(label, model_params, offset):
        baseline_state = initialize_state()
        baseline_total_nll = 0.0
        local_nll = {name: 0.0 for name in MODE_NAMES}
        segment_regret = {name: [] for name in MODE_NAMES[1:]}
        activations = {name: [] for name in DIAGNOSTIC_NAMES}
        started = time.perf_counter()
        for segment in range(x_host.shape[0]):
            tokens = jax.device_put(x_host[segment], batch_sharding)
            targets = jax.device_put(y_host[segment], batch_sharding)
            next_state, baseline_nll, diagnostic = evaluate_step(
                model_params,
                baseline_state,
                tokens,
                targets,
                jnp.asarray(0),
                jnp.asarray(offset, jnp.float32),
            )
            baseline_value = float(baseline_nll)
            baseline_total_nll += baseline_value
            if segment in sample_set:
                local_nll["baseline"] += baseline_value
                host_diagnostic = np.asarray(jax.device_get(diagnostic), dtype=np.float64)
                for name, value in zip(DIAGNOSTIC_NAMES, host_diagnostic):
                    activations[name].append(float(value))
                for mode, name in enumerate(MODE_NAMES[1:], start=1):
                    _, counterfactual_nll, _ = evaluate_step(
                        model_params,
                        baseline_state,
                        tokens,
                        targets,
                        jnp.asarray(mode),
                        jnp.asarray(offset, jnp.float32),
                    )
                    value = float(counterfactual_nll)
                    local_nll[name] += value
                    segment_regret[name].append(
                        (value - baseline_value) / (tokens_per_segment * LN2)
                    )
            baseline_state = next_state
            if segment == 0 or (segment + 1) % 256 == 0 or segment + 1 == x_host.shape[0]:
                print("PAIRED_COORDINATION_BASELINE_PROGRESS", label, segment + 1, x_host.shape[0], flush=True)

        baseline_bpc = baseline_total_nll / (x_host.size * LN2)
        sampled_tokens = len(sample_indices) * tokens_per_segment
        local_bpc = {
            name: value / (sampled_tokens * LN2) for name, value in local_nll.items()
        }
        local_regret = {
            name: value - local_bpc["baseline"]
            for name, value in local_bpc.items()
            if name != "baseline"
        }
        persistent_bpc = {"baseline": baseline_bpc}
        persistent_state = {"baseline": base.state_summary(baseline_state)}
        for mode_name in PERSISTENT_MODES:
            mode = MODE_NAMES.index(mode_name)
            state = initialize_state()
            total_nll = 0.0
            for segment in range(x_host.shape[0]):
                state, nll, _ = evaluate_step(
                    model_params,
                    state,
                    jax.device_put(x_host[segment], batch_sharding),
                    jax.device_put(y_host[segment], batch_sharding),
                    jnp.asarray(mode),
                    jnp.asarray(offset, jnp.float32),
                )
                total_nll += float(nll)
            persistent_bpc[mode_name] = total_nll / (x_host.size * LN2)
            persistent_state[mode_name] = base.state_summary(state)
            print("PAIRED_COORDINATION_PERSISTENT_COMPLETE", label, mode_name, persistent_bpc[mode_name], flush=True)
        persistent_regret = {
            name: value - baseline_bpc
            for name, value in persistent_bpc.items()
            if name != "baseline"
        }
        result = {
            "checkpoint_role": label,
            "baseline_full_validation_bpc": baseline_bpc,
            "matched_local_segments": sample_indices.tolist(),
            "matched_local_bpc": local_bpc,
            "matched_local_regret_bpc": local_regret,
            "persistent_bpc": persistent_bpc,
            "persistent_regret_bpc": persistent_regret,
            "calibration": {
                mode: calibration(activations[ACTIVATION_FOR_MODE[mode]], segment_regret[mode])
                for mode in MODE_NAMES[1:]
            },
            "baseline_diagnostics": {
                name: {"mean": float(np.mean(values)), "std": float(np.std(values))}
                for name, values in activations.items()
            },
            "persistent_final_state": persistent_state,
            "elapsed_s": time.perf_counter() - started,
        }
        print("PAIRED_COORDINATION_MODEL_COMPLETE", json.dumps(result), flush=True)
        return result

    canonical = audit_one("canonical", control, 0.0)
    segment_retention = audit_one("segment_retention", candidate, math.log(512.0))
    local_delta = {
        name: segment_retention["matched_local_regret_bpc"][name]
        - canonical["matched_local_regret_bpc"][name]
        for name in MODE_NAMES[1:]
    }
    persistent_delta = {
        name: segment_retention["persistent_regret_bpc"][name]
        - canonical["persistent_regret_bpc"][name]
        for name in PERSISTENT_MODES
    }
    candidate_local = segment_retention["matched_local_regret_bpc"]
    candidate_persistent = segment_retention["persistent_regret_bpc"]
    checks = {
        "exact_forward_parity": max(row["max_error"] for row in parity_results.values()) <= 1e-5,
        "candidate_language_not_worse_by_0p005": (
            segment_retention["baseline_full_validation_bpc"]
            <= canonical["baseline_full_validation_bpc"] + 0.005
        ),
        "feedback_locally_useful": candidate_local["feedback_off"] >= 0.0,
        "read_locally_useful": candidate_local["read_off"] >= 0.002,
        "write_locally_useful": candidate_local["write_off"] >= 0.002,
        "router_locally_useful": candidate_local["router_neutral"] >= 0.002,
        "router_persistently_useful": candidate_persistent["router_neutral"] >= 0.002,
        "matrix_branch_not_sufficient": candidate_local["matrix_only"] >= 0.002,
        "vector_branch_not_sufficient": candidate_local["vector_only"] >= 0.002,
    }
    report = {
        "seed": args.seed,
        "stage": "matched_102p4m_endpoint_coordination",
        "parameters_each": EXPECTED_PARAMS,
        "processed_characters_each": EXPECTED_CHARACTERS,
        "parity": parity_results,
        "canonical": canonical,
        "segment_retention": segment_retention,
        "candidate_minus_canonical_baseline_bpc": (
            segment_retention["baseline_full_validation_bpc"]
            - canonical["baseline_full_validation_bpc"]
        ),
        "candidate_minus_canonical_local_regret_bpc": local_delta,
        "candidate_minus_canonical_persistent_regret_bpc": persistent_delta,
        "promotion_checks": checks,
        "coordination_closure_pass": all(checks.values()),
        "all_values_finite": True,
        "test_data_read": False,
        "claim_boundary": "Frozen endpoint operation attribution; activation persistence is not semantic recall.",
    }
    report["all_values_finite"] = finite_tree(report)
    if not report["all_values_finite"]:
        report["coordination_closure_pass"] = False
    report["next"] = (
        "close seed-1 coordination gate and integrate the bounded result"
        if report["coordination_closure_pass"]
        else "inspect the failed causal check before any architecture change"
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.outdir / "paired_endpoint_coordination_audit.json", report)
    print("PAIRED_ENDPOINT_COORDINATION_DECISION", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
