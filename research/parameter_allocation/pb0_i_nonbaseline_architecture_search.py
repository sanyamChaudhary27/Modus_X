from __future__ import annotations

# =============================================================================
# MODUS_X PB0-I
# EXACT NON-BASELINE ARCHITECTURE VERIFICATION
#
# Purpose
# -------
# PB0-G searched the calibrated parameter space using the exact PB0-F formula.
# PB0-I is the final pre-training gate:
#
#   1. Load PB0-G candidates.
#   2. Import the actual Modus_X language/models.py implementation.
#   3. Rebuild each selected candidate through the real public make_model API.
#   4. Count the actual instantiated parameter tree.
#   5. Compare actual count against:
#        - PB0-G predicted count
#        - canonical B0 parameter budget
#   6. Record exact verification results.
#
# IMPORTANT
# ---------
# This script does NOT:
#   - modify language/models.py
#   - modify checkpoints
#   - train models
#   - load enwik8
#   - invent a new parameter formula
#   - independently expose feedback rank
#
# The actual architecture implementation is the source of truth.
#
# PB0-F established that the current initializer derives:
#
#     feedback_rank = min(32, ax_res, mamba_state_dim)
#
# Therefore PB0-I treats feedback_rank_f as an observed consequence of r/n,
# not as an independently configurable architecture parameter.
#
# The previous PB0-I failure was an API mismatch:
#
#     make_model(..., config=cfg)
#
# The actual public API uses:
#
#     make_model(..., cfg=cfg)
#
# This implementation uses the exact public signature and validates it before
# running candidate verification.
# =============================================================================

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import csv
import hashlib
import importlib
import inspect
import json
import sys
import traceback

import jax


# =============================================================================
# PROJECT PATHS
# =============================================================================

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

RESEARCH_DIR = PROJECT_ROOT / "research" / "parameter_allocation"
OUTPUT_DIR = RESEARCH_DIR / "outputs"

B0_JSON_PATH = OUTPUT_DIR / "b0_parameter_census.json"
PB0_G_JSON_PATH = OUTPUT_DIR / "PB0_G_CALIBRATED_CANDIDATE_SEARCH.json"
PB0_G_CSV_PATH = OUTPUT_DIR / "PB0_G_CALIBRATED_CANDIDATES.csv"

OUTPUT_JSON_PATH = OUTPUT_DIR / "PB0_I_NONBASELINE_ARCHITECTURE_SEARCH.json"
OUTPUT_MD_PATH = OUTPUT_DIR / "PB0_I_NONBASELINE_ARCHITECTURE_SEARCH.md"
OUTPUT_CSV_PATH = OUTPUT_DIR / "PB0_I_NONBASELINE_CANDIDATES.csv"
OUTPUT_VERIFY_CSV_PATH = OUTPUT_DIR / "PB0_I_NONBASELINE_VERIFICATION.csv"


# =============================================================================
# CANONICAL ARCHITECTURE SETTINGS
# =============================================================================

MODEL_NAME = "Modus_X_MemoryFeedbackArchive"

CANONICAL_VOCAB_SIZE = 256
CANONICAL_EMBED_DIM = 512
CANONICAL_HIDDEN_DIM = 1536
CANONICAL_AX_RES = 512
CANONICAL_N_LAYERS = 12
CANONICAL_N_HEADS_ATTN = 8
CANONICAL_SEQ_LEN = 512
CANONICAL_MAMBA_STATE_DIM = 512

CANONICAL_VECTOR_ROUTER = True
CANONICAL_ROUTER_HIDDEN = 32

# B0/PB0-F established that the canonical census includes one future target
# head attached to the final auxiliary layer.
CANONICAL_AUXILIARY_LAYERS = (11,)
CANONICAL_FUTURE_TARGET_COUNT = 1

FEEDBACK_CAP = 32

# PB0-I is a verification gate, not a broad new search.
#
# PB0-G stores up to 500 candidates, sorted by exact-match status, parameter
# error, allocation score, and allocation distance. We verify the strongest
# non-baseline candidates rather than blindly instantiating every stored row.
MAX_CANDIDATES_TO_VERIFY = 25

# A candidate must remain inside this absolute budget window to be considered
# a fixed-budget candidate.
MAX_ABSOLUTE_PARAMETER_ERROR = 50_000


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class Candidate:
    rank_r: int
    vector_dimension_n: int
    router_hidden_h: int
    feedback_rank_f: int
    parameter_count: int
    parameter_error: int
    absolute_parameter_error: int
    exact_match: bool
    matrix_parameter_change: int
    vector_parameter_change: int
    router_parameter_change: int
    feedback_parameter_change: int
    rank_ratio: float
    vector_ratio: float
    router_ratio: float
    allocation_distance: float
    score: float


@dataclass(frozen=True)
class VerificationResult:
    candidate_index: int
    rank_r: int
    vector_dimension_n: int
    router_hidden_h: int
    expected_feedback_rank: int
    pb0_g_parameter_count: int
    actual_parameter_count: int
    parameter_count_delta: int
    actual_vs_pb0_g_exact: bool
    actual_vs_canonical_exact: bool
    actual_within_budget_window: bool
    candidate_is_nonbaseline: bool
    tree_fingerprint: str
    status: str
    error_type: str
    error_message: str


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def print_banner(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} does not exist:\n{path}"
        )


def load_json(path: Path) -> dict[str, Any]:
    require_file(path, f"Required JSON file: {path.name}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid JSON in {path}:\n{exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected top-level JSON object in {path}, "
            f"got {type(payload).__name__}"
        )

    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# B0 TARGET
# =============================================================================

def load_b0_parameter_count() -> int:
    payload = load_json(B0_JSON_PATH)

    actual = payload.get("actual_parameter_count")
    expected = payload.get("expected_parameter_count")

    values: list[tuple[str, int]] = []

    for field_name, value in (
        ("actual_parameter_count", actual),
        ("expected_parameter_count", expected),
    ):
        if value is None:
            continue

        if isinstance(value, bool):
            raise TypeError(
                f"B0 field {field_name!r} cannot be boolean."
            )

        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"B0 field {field_name!r} is not an integer: {value!r}"
            ) from exc

        if parsed <= 0:
            raise ValueError(
                f"B0 field {field_name!r} must be positive: {parsed}"
            )

        values.append((field_name, parsed))

    if not values:
        raise KeyError(
            "B0 does not contain actual_parameter_count or "
            "expected_parameter_count."
        )

    if len(values) == 2 and values[0][1] != values[1][1]:
        raise RuntimeError(
            "B0 canonical parameter-count fields disagree:\n"
            f"actual={values[0][1]:,}\n"
            f"expected={values[1][1]:,}"
        )

    return values[0][1]


# =============================================================================
# PB0-G INPUT
# =============================================================================

def candidate_from_dict(row: dict[str, Any]) -> Candidate:
    required = (
        "rank_r",
        "vector_dimension_n",
        "router_hidden_h",
        "feedback_rank_f",
        "parameter_count",
        "parameter_error",
        "exact_match",
        "matrix_parameter_change",
        "vector_parameter_change",
        "router_parameter_change",
        "feedback_parameter_change",
        "rank_ratio",
        "vector_ratio",
        "router_ratio",
        "allocation_distance",
        "score",
    )

    missing = [field for field in required if field not in row]
    if missing:
        raise KeyError(
            "PB0-G candidate is missing required fields:\n"
            + "\n".join(f"  - {field}" for field in missing)
        )

    return Candidate(
        rank_r=int(row["rank_r"]),
        vector_dimension_n=int(row["vector_dimension_n"]),
        router_hidden_h=int(row["router_hidden_h"]),
        feedback_rank_f=int(row["feedback_rank_f"]),
        parameter_count=int(row["parameter_count"]),
        parameter_error=int(row["parameter_error"]),
        absolute_parameter_error=abs(int(row["parameter_error"])),
        exact_match=bool(row["exact_match"]),
        matrix_parameter_change=int(row["matrix_parameter_change"]),
        vector_parameter_change=int(row["vector_parameter_change"]),
        router_parameter_change=int(row["router_parameter_change"]),
        feedback_parameter_change=int(row["feedback_parameter_change"]),
        rank_ratio=float(row["rank_ratio"]),
        vector_ratio=float(row["vector_ratio"]),
        router_ratio=float(row["router_ratio"]),
        allocation_distance=float(row["allocation_distance"]),
        score=float(row["score"]),
    )


def load_pb0_g_candidates() -> list[Candidate]:
    payload = load_json(PB0_G_JSON_PATH)

    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list):
        raise KeyError(
            "PB0-G JSON does not contain a valid 'candidates' list."
        )

    candidates = [
        candidate_from_dict(row)
        for row in raw_candidates
        if isinstance(row, dict)
    ]

    if not candidates:
        raise RuntimeError("PB0-G contains no usable candidates.")

    return candidates


# =============================================================================
# MODEL IMPORT
# =============================================================================

def import_project_models() -> Any:
    language_dir = PROJECT_ROOT / "language"

    if not language_dir.exists():
        raise FileNotFoundError(
            f"Canonical language directory does not exist:\n{language_dir}"
        )

    root_string = str(PROJECT_ROOT)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)

    try:
        models = importlib.import_module("language.models")
    except Exception as exc:
        raise RuntimeError(
            "Failed to import the actual project language.models module."
        ) from exc

    required_symbols = (
        "ModelConfig",
        "make_model",
        "count_params",
    )

    missing = [
        symbol
        for symbol in required_symbols
        if not hasattr(models, symbol)
    ]

    if missing:
        raise AttributeError(
            "language.models is missing required symbols:\n"
            + "\n".join(f"  - {symbol}" for symbol in missing)
        )

    return models


def validate_make_model_signature(models: Any) -> None:
    signature = inspect.signature(models.make_model)

    parameters = signature.parameters

    required = ("name", "key", "cfg")
    missing = [name for name in required if name not in parameters]

    if missing:
        raise RuntimeError(
            "The imported make_model API does not match the expected "
            "2.1.x public signature.\n"
            f"Missing parameters: {missing}\n"
            f"Actual signature: {signature}"
        )

    if "config" in parameters and "cfg" not in parameters:
        raise RuntimeError(
            "The imported make_model API exposes 'config' instead of "
            "the expected 'cfg'.\n"
            f"Actual signature: {signature}"
        )

    print(f"make_model signature: {signature}")


# =============================================================================
# CONFIGURATION
# =============================================================================

def build_canonical_config(models: Any) -> Any:
    return models.ModelConfig(
        vocab_size=CANONICAL_VOCAB_SIZE,
        embed_dim=CANONICAL_EMBED_DIM,
        hidden_dim=CANONICAL_HIDDEN_DIM,
        ax_res=CANONICAL_AX_RES,
        n_layers=CANONICAL_N_LAYERS,
        n_heads_attn=CANONICAL_N_HEADS_ATTN,
        seq_len=CANONICAL_SEQ_LEN,
        mamba_state_dim=CANONICAL_MAMBA_STATE_DIM,
        vector_router=CANONICAL_VECTOR_ROUTER,
        router_hidden=CANONICAL_ROUTER_HIDDEN,
    )


def build_variant_config(
    canonical_cfg: Any,
    matrix_rank: int,
    vector_dim: int,
    router_hidden: int,
) -> Any:
    if matrix_rank <= 0:
        raise ValueError(f"matrix_rank must be positive: {matrix_rank}")

    if vector_dim <= 0:
        raise ValueError(f"vector_dim must be positive: {vector_dim}")

    if router_hidden <= 0:
        raise ValueError(f"router_hidden must be positive: {router_hidden}")

    # These are the ONLY three architecture dimensions PB0-G searched.
    # Feedback rank is deliberately not inserted into ModelConfig because the
    # current initializer derives it internally.
    from dataclasses import replace

    return replace(
        canonical_cfg,
        ax_res=int(matrix_rank),
        mamba_state_dim=int(vector_dim),
        router_hidden=int(router_hidden),
        vector_router=True,
    )


def expected_feedback_rank(matrix_rank: int, vector_dim: int) -> int:
    return min(
        FEEDBACK_CAP,
        int(matrix_rank),
        int(vector_dim),
    )


# =============================================================================
# ACTUAL MODEL INSTANTIATION
# =============================================================================

def instantiate_actual_model(
    models: Any,
    cfg: Any,
) -> dict[str, Any]:
    key = jax.random.PRNGKey(0)

    # IMPORTANT:
    # The public API parameter is named `cfg`, not `config`.
    params, _forward = models.make_model(
        MODEL_NAME,
        key,
        cfg,
        auxiliary_layers=CANONICAL_AUXILIARY_LAYERS,
        future_target_count=CANONICAL_FUTURE_TARGET_COUNT,
    )

    return params


def count_actual_parameters(
    models: Any,
    cfg: Any,
) -> tuple[int, dict[str, Any]]:
    params = instantiate_actual_model(
        models=models,
        cfg=cfg,
    )

    count = int(models.count_params(params))

    if count <= 0:
        raise RuntimeError(
            f"Actual model parameter count is invalid: {count}"
        )

    return count, params


# =============================================================================
# PARAMETER-TREE FINGERPRINT
# =============================================================================

def _flatten_tree_items(
    value: Any,
    prefix: str = "",
) -> list[tuple[str, tuple[int, ...], str]]:
    items: list[tuple[str, tuple[int, ...], str]] = []

    if isinstance(value, dict):
        for key in sorted(value):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            items.extend(
                _flatten_tree_items(
                    value[key],
                    child_prefix,
                )
            )
        return items

    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            items.extend(
                _flatten_tree_items(
                    child,
                    child_prefix,
                )
            )
        return items

    shape = tuple(int(x) for x in getattr(value, "shape", ()))
    dtype = str(getattr(value, "dtype", type(value).__name__))

    items.append(
        (
            prefix,
            shape,
            dtype,
        )
    )

    return items


def parameter_tree_fingerprint(params: dict[str, Any]) -> str:
    leaves = _flatten_tree_items(params)

    payload = "\n".join(
        f"{path}|shape={shape}|dtype={dtype}"
        for path, shape, dtype in leaves
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


# =============================================================================
# CANONICAL VERIFICATION
# =============================================================================

def verify_canonical_baseline(
    models: Any,
    b0_count: int,
    canonical_cfg: Any,
) -> tuple[int, str]:
    print_banner("VERIFYING CANONICAL BASELINE")

    actual_count, params = count_actual_parameters(
        models=models,
        cfg=canonical_cfg,
    )

    fingerprint = parameter_tree_fingerprint(params)

    print(f"B0 parameter count       : {b0_count:,}")
    print(f"Actual canonical count   : {actual_count:,}")
    print(f"Difference                : {actual_count - b0_count:+,}")
    print(f"Tree fingerprint          : {fingerprint}")

    if actual_count != b0_count:
        raise RuntimeError(
            "CANONICAL BASELINE FAILED.\n"
            f"B0={b0_count:,}\n"
            f"actual={actual_count:,}\n"
            f"difference={actual_count - b0_count:+,}\n\n"
            "PB0-I refuses to verify non-baseline candidates against a "
            "canonical architecture that does not reproduce B0 exactly."
        )

    return actual_count, fingerprint


# =============================================================================
# CANDIDATE SELECTION
# =============================================================================

def select_nonbaseline_candidates(
    candidates: list[Candidate],
) -> list[Candidate]:
    nonbaseline = [
        candidate
        for candidate in candidates
        if not (
            candidate.rank_r == CANONICAL_AX_RES
            and candidate.vector_dimension_n == CANONICAL_MAMBA_STATE_DIM
            and candidate.router_hidden_h == CANONICAL_ROUTER_HIDDEN
        )
        and candidate.absolute_parameter_error <= MAX_ABSOLUTE_PARAMETER_ERROR
    ]

    nonbaseline.sort(
        key=lambda candidate: (
            0 if candidate.exact_match else 1,
            candidate.absolute_parameter_error,
            candidate.score,
            candidate.allocation_distance,
        )
    )

    return nonbaseline[:MAX_CANDIDATES_TO_VERIFY]


# =============================================================================
# CANDIDATE VERIFICATION
# =============================================================================

def verify_candidate(
    index: int,
    candidate: Candidate,
    models: Any,
    canonical_cfg: Any,
    target_parameter_count: int,
) -> VerificationResult:
    feedback_rank = expected_feedback_rank(
        matrix_rank=candidate.rank_r,
        vector_dim=candidate.vector_dimension_n,
    )

    try:
        cfg = build_variant_config(
            canonical_cfg=canonical_cfg,
            matrix_rank=candidate.rank_r,
            vector_dim=candidate.vector_dimension_n,
            router_hidden=candidate.router_hidden_h,
        )

        actual_count, params = count_actual_parameters(
            models=models,
            cfg=cfg,
        )

        fingerprint = parameter_tree_fingerprint(params)
        delta = actual_count - candidate.parameter_count

        exact_pb0_g = actual_count == candidate.parameter_count
        exact_canonical = actual_count == target_parameter_count
        within_budget = (
            abs(actual_count - target_parameter_count)
            <= MAX_ABSOLUTE_PARAMETER_ERROR
        )

        status = (
            "PASS_EXACT_PB0_G"
            if exact_pb0_g
            else "FAIL_FORMULA_OR_IMPLEMENTATION_MISMATCH"
        )

        return VerificationResult(
            candidate_index=index,
            rank_r=candidate.rank_r,
            vector_dimension_n=candidate.vector_dimension_n,
            router_hidden_h=candidate.router_hidden_h,
            expected_feedback_rank=feedback_rank,
            pb0_g_parameter_count=candidate.parameter_count,
            actual_parameter_count=actual_count,
            parameter_count_delta=delta,
            actual_vs_pb0_g_exact=exact_pb0_g,
            actual_vs_canonical_exact=exact_canonical,
            actual_within_budget_window=within_budget,
            candidate_is_nonbaseline=True,
            tree_fingerprint=fingerprint,
            status=status,
            error_type="",
            error_message="",
        )

    except Exception as exc:
        return VerificationResult(
            candidate_index=index,
            rank_r=candidate.rank_r,
            vector_dimension_n=candidate.vector_dimension_n,
            router_hidden_h=candidate.router_hidden_h,
            expected_feedback_rank=feedback_rank,
            pb0_g_parameter_count=candidate.parameter_count,
            actual_parameter_count=-1,
            parameter_count_delta=0,
            actual_vs_pb0_g_exact=False,
            actual_vs_canonical_exact=False,
            actual_within_budget_window=False,
            candidate_is_nonbaseline=True,
            tree_fingerprint="",
            status="FAIL_INSTANTIATION",
            error_type=type(exc).__name__,
            error_message=str(exc),
        )


# =============================================================================
# REPORTING
# =============================================================================

def write_reports(
    b0_count: int,
    canonical_count: int,
    canonical_fingerprint: str,
    all_pb0_g_candidates: list[Candidate],
    selected_candidates: list[Candidate],
    verification_results: list[VerificationResult],
) -> None:
    exact_verified = [
        result
        for result in verification_results
        if result.actual_vs_pb0_g_exact
    ]

    failed_verified = [
        result
        for result in verification_results
        if not result.actual_vs_pb0_g_exact
    ]

    passing_budget_candidates = [
        result
        for result in exact_verified
        if result.actual_within_budget_window
    ]

    # ------------------------------
    # CSV: selected candidates
    # ------------------------------
    candidate_rows = [
        asdict(candidate)
        for candidate in selected_candidates
    ]

    write_csv(
        OUTPUT_CSV_PATH,
        candidate_rows,
        fieldnames=[
            "rank_r",
            "vector_dimension_n",
            "router_hidden_h",
            "feedback_rank_f",
            "parameter_count",
            "parameter_error",
            "absolute_parameter_error",
            "exact_match",
            "matrix_parameter_change",
            "vector_parameter_change",
            "router_parameter_change",
            "feedback_parameter_change",
            "rank_ratio",
            "vector_ratio",
            "router_ratio",
            "allocation_distance",
            "score",
        ],
    )

    # ------------------------------
    # CSV: actual verification
    # ------------------------------
    verification_rows = [
        asdict(result)
        for result in verification_results
    ]

    write_csv(
        OUTPUT_VERIFY_CSV_PATH,
        verification_rows,
        fieldnames=[
            "candidate_index",
            "rank_r",
            "vector_dimension_n",
            "router_hidden_h",
            "expected_feedback_rank",
            "pb0_g_parameter_count",
            "actual_parameter_count",
            "parameter_count_delta",
            "actual_vs_pb0_g_exact",
            "actual_vs_canonical_exact",
            "actual_within_budget_window",
            "candidate_is_nonbaseline",
            "tree_fingerprint",
            "status",
            "error_type",
            "error_message",
        ],
    )

    # ------------------------------
    # JSON
    # ------------------------------
    payload = {
        "report": "PB0-I Exact Non-Baseline Architecture Verification",
        "generated_at_utc": utc_now(),
        "project_root": str(PROJECT_ROOT),
        "model_name": MODEL_NAME,
        "canonical_configuration": {
            "vocab_size": CANONICAL_VOCAB_SIZE,
            "embed_dim": CANONICAL_EMBED_DIM,
            "hidden_dim": CANONICAL_HIDDEN_DIM,
            "ax_res": CANONICAL_AX_RES,
            "n_layers": CANONICAL_N_LAYERS,
            "n_heads_attn": CANONICAL_N_HEADS_ATTN,
            "seq_len": CANONICAL_SEQ_LEN,
            "mamba_state_dim": CANONICAL_MAMBA_STATE_DIM,
            "vector_router": CANONICAL_VECTOR_ROUTER,
            "router_hidden": CANONICAL_ROUTER_HIDDEN,
            "auxiliary_layers": list(CANONICAL_AUXILIARY_LAYERS),
            "future_target_count": CANONICAL_FUTURE_TARGET_COUNT,
        },
        "feedback_rank_rule": "min(32, ax_res, mamba_state_dim)",
        "b0_parameter_count": b0_count,
        "actual_canonical_parameter_count": canonical_count,
        "canonical_exact": canonical_count == b0_count,
        "canonical_tree_fingerprint": canonical_fingerprint,
        "pb0_g_candidate_count_loaded": len(all_pb0_g_candidates),
        "pb0_i_candidate_count_verified": len(selected_candidates),
        "pb0_i_exact_verified_count": len(exact_verified),
        "pb0_i_failed_count": len(failed_verified),
        "pb0_i_budget_valid_exact_count": len(passing_budget_candidates),
        "selected_candidates": [
            asdict(candidate)
            for candidate in selected_candidates
        ],
        "verification_results": [
            asdict(result)
            for result in verification_results
        ],
        "gate": {
            "canonical_baseline_exact": canonical_count == b0_count,
            "all_selected_candidates_exact_against_pb0_g": (
                len(selected_candidates) > 0
                and len(exact_verified) == len(selected_candidates)
            ),
            "at_least_one_nonbaseline_exact_budget_candidate": (
                len(passing_budget_candidates) > 0
            ),
            "training_allowed": (
                canonical_count == b0_count
                and len(passing_budget_candidates) > 0
            ),
        },
        "next_step": (
            "Only candidates with exact actual parameter-tree agreement may "
            "advance to a training screen. If no non-baseline candidate "
            "survives exact verification, PB0-I has not demonstrated a "
            "realizable fixed-budget alternative."
        ),
    }

    write_json(
        OUTPUT_JSON_PATH,
        payload,
    )

    # ------------------------------
    # Markdown
    # ------------------------------
    lines: list[str] = []

    lines.append("# PB0-I Exact Non-Baseline Architecture Verification")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "PB0-I is the final parameter-accounting gate before any candidate "
        "training. PB0-G generated calibrated fixed-budget candidates; PB0-I "
        "instantiates the actual architecture and verifies the real parameter "
        "tree."
    )
    lines.append("")
    lines.append("## Architecture Source of Truth")
    lines.append("")
    lines.append(f"- Model: `{MODEL_NAME}`")
    lines.append(
        "- Public factory call: "
        "`make_model(name, key, cfg, auxiliary_layers=..., "
        "future_target_count=...)`"
    )
    lines.append(
        "- Feedback rank rule: "
        "`min(32, ax_res, mamba_state_dim)`"
    )
    lines.append("")
    lines.append("## Canonical Baseline Gate")
    lines.append("")
    lines.append(f"- B0 count: **{b0_count:,}**")
    lines.append(f"- Actual instantiated count: **{canonical_count:,}**")
    lines.append(
        f"- Difference: **{canonical_count - b0_count:+,}**"
    )
    lines.append(
        f"- Exact: **{canonical_count == b0_count}**"
    )
    lines.append(
        f"- Tree fingerprint: `{canonical_fingerprint}`"
    )
    lines.append("")

    lines.append("## PB0-G Candidate Pool")
    lines.append("")
    lines.append(
        f"- Candidates loaded from PB0-G: **{len(all_pb0_g_candidates):,}**"
    )
    lines.append(
        f"- Non-baseline candidates verified here: "
        f"**{len(selected_candidates):,}**"
    )
    lines.append("")

    lines.append("## Exact Verification Results")
    lines.append("")
    lines.append(
        "| # | r | n | h | f | PB0-G params | Actual params | "
        "Delta | Exact | Status |"
    )
    lines.append(
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"
    )

    for result in verification_results:
        lines.append(
            f"| {result.candidate_index} "
            f"| {result.rank_r} "
            f"| {result.vector_dimension_n} "
            f"| {result.router_hidden_h} "
            f"| {result.expected_feedback_rank} "
            f"| {result.pb0_g_parameter_count:,} "
            f"| {result.actual_parameter_count:,} "
            f"| {result.parameter_count_delta:+,} "
            f"| {result.actual_vs_pb0_g_exact} "
            f"| `{result.status}` |"
        )

    lines.append("")
    lines.append("## Gate Decision")
    lines.append("")

    if not (canonical_count == b0_count):
        lines.append(
            "### FAILED — canonical baseline mismatch"
        )
        lines.append("")
        lines.append(
            "PB0-I must stop. The actual architecture does not reproduce "
            "the B0 canonical parameter count."
        )
    elif failed_verified:
        lines.append(
            "### FAILED — candidate accounting mismatch"
        )
        lines.append("")
        lines.append(
            "At least one selected PB0-G candidate does not reproduce its "
            "calibrated parameter count when instantiated through the actual "
            "architecture. Those candidates must not be trained."
        )
    elif not passing_budget_candidates:
        lines.append(
            "### FAILED — no non-baseline fixed-budget candidate"
        )
        lines.append("")
        lines.append(
            "The selected non-baseline candidates instantiate correctly, "
            "but none satisfies the fixed-budget window."
        )
    else:
        lines.append(
            "### PASSED — realizable fixed-budget candidates exist"
        )
        lines.append("")
        lines.append(
            f"**{len(passing_budget_candidates)}** verified non-baseline "
            "candidate(s) reproduce the PB0-G parameter count exactly and "
            "remain within the fixed canonical budget window."
        )

    lines.append("")
    lines.append("## Important Interpretation")
    lines.append("")
    lines.append(
        "PB0-I does not establish that any candidate improves BPC. It only "
        "establishes that the candidate is a real, reproducible architecture "
        "under the current implementation and parameter accounting."
    )
    lines.append("")
    lines.append(
        "A candidate must still pass an empirical training/validation screen "
        "before any architecture change is accepted."
    )
    lines.append("")
    lines.append("## Next Gate")
    lines.append("")
    lines.append(
        "If PB0-I passes, train only the strongest verified candidate(s) under "
        "the same data, optimizer, schedule, seed policy, and evaluation "
        "protocol used for the canonical baseline."
    )
    lines.append("")
    lines.append(
        "Do not modify `language/models.py` merely because PB0-G found a "
        "different allocation. Let the empirical BPC result determine whether "
        "a parameter-allocation change is justified."
    )
    lines.append("")

    OUTPUT_MD_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print_banner(
        "MODUS_X PB0-I EXACT NON-BASELINE ARCHITECTURE VERIFICATION"
    )

    print(f"Project root : {PROJECT_ROOT}")
    print(f"B0 input     : {B0_JSON_PATH}")
    print(f"PB0-G input  : {PB0_G_JSON_PATH}")
    print(f"Output dir   : {OUTPUT_DIR}")

    require_file(
        B0_JSON_PATH,
        "B0 parameter census",
    )
    require_file(
        PB0_G_JSON_PATH,
        "PB0-G calibrated candidate search",
    )

    print_banner("LOADING CANONICAL BUDGET")
    b0_count = load_b0_parameter_count()
    print(f"Canonical B0 parameter count: {b0_count:,}")

    print_banner("LOADING PB0-G CANDIDATES")
    all_candidates = load_pb0_g_candidates()
    print(f"PB0-G candidates loaded: {len(all_candidates):,}")

    selected_candidates = select_nonbaseline_candidates(
        all_candidates
    )

    if not selected_candidates:
        raise RuntimeError(
            "PB0-G produced no non-baseline candidates inside the configured "
            "parameter window."
        )

    print(
        f"Non-baseline candidates selected for verification: "
        f"{len(selected_candidates)}"
    )

    for index, candidate in enumerate(
        selected_candidates,
        start=1,
    ):
        print(
            f"{index:02d} "
            f"r={candidate.rank_r:3d} "
            f"n={candidate.vector_dimension_n:3d} "
            f"h={candidate.router_hidden_h:3d} "
            f"f={candidate.feedback_rank_f:2d} "
            f"params={candidate.parameter_count:,} "
            f"error={candidate.parameter_error:+,}"
        )

    print_banner("IMPORTING ACTUAL ARCHITECTURE")
    models = import_project_models()
    print(f"Imported module: {models.__file__}")

    validate_make_model_signature(models)

    print_banner("BUILDING CANONICAL CONFIGURATION")
    canonical_cfg = build_canonical_config(models)
    print(canonical_cfg)

    canonical_count, canonical_fingerprint = verify_canonical_baseline(
        models=models,
        b0_count=b0_count,
        canonical_cfg=canonical_cfg,
    )

    print_banner("VERIFYING NON-BASELINE CANDIDATES")

    verification_results: list[VerificationResult] = []

    for index, candidate in enumerate(
        selected_candidates,
        start=1,
    ):
        print()
        print(
            f"[{index}/{len(selected_candidates)}] "
            f"r={candidate.rank_r}, "
            f"n={candidate.vector_dimension_n}, "
            f"h={candidate.router_hidden_h}, "
            f"f={candidate.feedback_rank_f}"
        )

        result = verify_candidate(
            index=index,
            candidate=candidate,
            models=models,
            canonical_cfg=canonical_cfg,
            target_parameter_count=b0_count,
        )

        verification_results.append(result)

        print(
            f"  PB0-G params : {result.pb0_g_parameter_count:,}"
        )
        print(
            f"  Actual params: {result.actual_parameter_count:,}"
        )
        print(
            f"  Delta        : {result.parameter_count_delta:+,}"
        )
        print(
            f"  Exact        : {result.actual_vs_pb0_g_exact}"
        )
        print(
            f"  Status       : {result.status}"
        )

        if result.error_message:
            print(
                f"  Error        : "
                f"{result.error_type}: {result.error_message}"
            )

    print_banner("WRITING PB0-I OUTPUTS")

    write_reports(
        b0_count=b0_count,
        canonical_count=canonical_count,
        canonical_fingerprint=canonical_fingerprint,
        all_pb0_g_candidates=all_candidates,
        selected_candidates=selected_candidates,
        verification_results=verification_results,
    )

    exact_count = sum(
        result.actual_vs_pb0_g_exact
        for result in verification_results
    )

    budget_exact_count = sum(
        result.actual_vs_pb0_g_exact
        and result.actual_within_budget_window
        for result in verification_results
    )

    print_banner("PB0-I COMPLETE")

    print(
        f"Canonical exact         : "
        f"{canonical_count == b0_count}"
    )
    print(
        f"Candidates verified     : "
        f"{len(verification_results)}"
    )
    print(
        f"Exact PB0-G matches     : "
        f"{exact_count}"
    )
    print(
        f"Exact + budget-valid    : "
        f"{budget_exact_count}"
    )

    print()
    print("Output files:")
    print(f"  {OUTPUT_JSON_PATH}")
    print(f"  {OUTPUT_MD_PATH}")
    print(f"  {OUTPUT_CSV_PATH}")
    print(f"  {OUTPUT_VERIFY_CSV_PATH}")

    if canonical_count != b0_count:
        raise RuntimeError(
            "PB0-I FAILED: canonical baseline did not reproduce B0."
        )

    if exact_count != len(verification_results):
        raise RuntimeError(
            "PB0-I FAILED: one or more selected candidates did not reproduce "
            "the PB0-G parameter count exactly."
        )

    if budget_exact_count == 0:
        raise RuntimeError(
            "PB0-I FAILED: no verified non-baseline candidate remains inside "
            "the fixed-budget window."
        )

    print()
    print(
        "PB0-I PASS: at least one non-baseline candidate is "
        "architecturally realizable and fixed-budget valid."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 80)
        print("PB0-I FAILED")
        print("=" * 80)
        print(f"{type(exc).__name__}: {exc}")
        print()
        traceback.print_exc()
        raise
