from __future__ import annotations
from pathlib import Path
import csv
import json
import sys
import traceback
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import jax
import numpy as np


# =============================================================================
# PB0-F
# ACTUAL ARCHITECTURE PARAMETER FORMULA CALIBRATION
#
# Purpose
# -------
# PB0-E exposed a large discrepancy between the analytical parameter-allocation
# model and the actual architecture tensor tree.
#
# This script resolves that discrepancy by:
#
#   1. Importing the actual canonical language/models.py implementation.
#   2. Instantiating the actual Modus_X_MemoryFeedbackArchive architecture.
#   3. Measuring exact parameter-tree counts for controlled dimension changes.
#   4. Separating parameter changes caused by:
#        - matrix rank / ax_res
#        - vector state dimension / mamba_state_dim
#        - router hidden width
#        - actual feedback bottleneck behaviour
#   5. Deriving an exact empirical parameter formula for the current
#      implementation.
#   6. Validating that formula against held-out architecture configurations.
#
# IMPORTANT
# ---------
# This script does not modify language/models.py.
# This script does not train.
# This script does not load datasets.
# This script does not modify checkpoints.
#
# The architecture is treated as the source of truth.
# =============================================================================


# =============================================================================
# PATH RESOLUTION
# =============================================================================

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
RESEARCH_DIR = PROJECT_ROOT / "research" / "parameter_allocation"
OUTPUT_DIR = RESEARCH_DIR / "outputs"
B0_CENSUS_PATH = (
    PROJECT_ROOT
    / "research"
    / "parameter_allocation"
    / "outputs"
    / "b0_parameter_census.json"
)
MODELS_PATH = PROJECT_ROOT / "language" / "models.py"
B0_JSON_PATH = OUTPUT_DIR / "b0_parameter_census.json"

OUTPUT_JSON_PATH = OUTPUT_DIR / "PB0_F_PARAMETER_FORMULA_CALIBRATION.json"
OUTPUT_MD_PATH = OUTPUT_DIR / "PB0_F_PARAMETER_FORMULA_CALIBRATION.md"
OUTPUT_CSV_PATH = OUTPUT_DIR / "PB0_F_PARAMETER_CALIBRATION_SAMPLES.csv"


# =============================================================================
# CANONICAL EXPERIMENT SETTINGS
#
# These values represent the canonical architecture used by the B0 census.
#
# PB0-F validates this instantiated parameter tree against the measured B0
# parameter count. If it does not match, the script fails instead of silently
# fitting a formula to the wrong architecture configuration.
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

# The actual make_model implementation forces vector_router=True for
# Modus_X_MemoryFeedbackArchive. The B0 census showed the canonical router
# parameter budget corresponding to the actual architecture configuration.
CANONICAL_VECTOR_ROUTER = True
CANONICAL_ROUTER_HIDDEN = 32

# B0 reported:
#
#   output_head              1,181,440
#   future_prediction_head   1,181,440
#
# This indicates that the canonical census includes one auxiliary future head.
CANONICAL_DEEP_SUPERVISION = True
CANONICAL_AUXILIARY_LAYERS = (11,)
CANONICAL_FUTURE_TARGET_COUNT = 1


# =============================================================================
# CALIBRATION DIMENSION GRID
#
# We intentionally include values both below and above the current
# implementation's feedback-rank cap.
#
# In the current canonical implementation:
#
#     feedback_rank = min(32, ax_res, mamba_state_dim)
#
# Therefore feedback rank is NOT independently configurable through ModelConfig.
#
# This is exactly one of the architectural facts PB0-F must expose.
# =============================================================================

CALIBRATION_MATRIX_RANKS = (
    96,
    128,
    168,
    192,
    256,
    384,
    512,
)

CALIBRATION_VECTOR_DIMS = (
    96,
    128,
    192,
    256,
    384,
    512,
)

CALIBRATION_ROUTER_WIDTHS = (
    8,
    16,
    20,
    32,
    48,
    64,
)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass(frozen=True)
class ArchitectureSample:
    sample_id: str
    matrix_rank: int
    vector_dim: int
    router_hidden: int
    feedback_rank_actual: int
    parameter_count: int
    parameter_delta_from_baseline: int
    source: str


@dataclass(frozen=True)
class FormulaCoefficients:
    intercept: int
    matrix_rank: int
    vector_dim: int
    router_hidden: int
    feedback_rank_times_matrix_rank: int
    feedback_rank: int


@dataclass(frozen=True)
class ValidationResult:
    sample_id: str
    matrix_rank: int
    vector_dim: int
    router_hidden: int
    actual_feedback_rank: int
    actual_parameters: int
    predicted_parameters: int
    prediction_error: int
    exact: bool


# =============================================================================
# GENERAL HELPERS
# =============================================================================


def print_banner(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} does not exist:\n{path}\n\n"
            "PB0-F cannot continue because the required source of truth is "
            "missing."
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_json_compatible(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, tuple):
        return [to_json_compatible(item) for item in value]

    if isinstance(value, list):
        return [to_json_compatible(item) for item in value]

    if isinstance(value, dict):
        return {
            str(key): to_json_compatible(item)
            for key, item in value.items()
        }

    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            to_json_compatible(payload),
            handle,
            indent=2,
            sort_keys=False,
        )
        handle.write("\n")


def write_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field: row.get(field)
                    for field in fieldnames
                }
            )


# =============================================================================
# PROJECT IMPORTS
# =============================================================================


def import_project_models() -> Any:
    language_dir = PROJECT_ROOT / "language"

    if not language_dir.exists():
        raise FileNotFoundError(
            f"language directory does not exist:\n{language_dir}"
        )

    project_root_string = str(PROJECT_ROOT)

    if project_root_string not in sys.path:
        sys.path.insert(0, project_root_string)

    try:
        import language.models as models
    except Exception as exc:
        raise RuntimeError(
            "Failed to import language.models. "
            "PB0-F requires the actual canonical architecture implementation."
        ) from exc

    required_symbols = (
        "ModelConfig",
        "make_model",
        "count_params",
    )

    missing_symbols = [
        symbol
        for symbol in required_symbols
        if not hasattr(models, symbol)
    ]

    if missing_symbols:
        raise AttributeError(
            "language.models is missing required symbols: "
            + ", ".join(missing_symbols)
        )

    return models


# =============================================================================
# B0 BASELINE LOADING
# =============================================================================
def load_b0_expected_parameter_count() -> int:
    """
    Load the canonical parameter count from the B0 census.

    The canonical B0 census stores the authoritative model count at the
    top level. Component-level values must never be selected as a fallback
    before these canonical fields are checked.

    Expected authoritative fields:

        actual_parameter_count
        expected_parameter_count

    Both are present in the canonical B0 census and should resolve to the
    same value for a valid baseline.
    """

    if not B0_JSON_PATH.exists():
        raise FileNotFoundError(
            "B0 parameter census does not exist:\n"
            f"{B0_JSON_PATH}"
        )

    try:
        with B0_JSON_PATH.open(
            "r",
            encoding="utf-8",
        ) as handle:
            payload = json.load(handle)

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "B0 parameter census is not valid JSON:\n"
            f"{B0_JSON_PATH}"
        ) from exc

    if not isinstance(payload, dict):
        raise TypeError(
            "B0 parameter census root must be a JSON object.\n"
            f"Found: {type(payload).__name__}"
        )

    authoritative_fields = (
        "actual_parameter_count",
        "expected_parameter_count",
    )

    resolved_values: dict[str, int] = {}

    for field_name in authoritative_fields:
        value = payload.get(field_name)

        if value is None:
            continue

        if isinstance(value, bool):
            raise TypeError(
                f"B0 field '{field_name}' must be an integer parameter "
                f"count, not a boolean."
            )

        try:
            parameter_count = int(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"B0 field '{field_name}' is not a valid integer parameter "
                f"count: {value!r}"
            ) from exc

        if parameter_count <= 0:
            raise ValueError(
                f"B0 field '{field_name}' must be positive. "
                f"Found: {parameter_count}"
            )

        resolved_values[field_name] = parameter_count

    if not resolved_values:
        raise KeyError(
            "Could not find an authoritative canonical parameter count in "
            f"{B0_JSON_PATH}.\n\n"
            "Expected one of:\n"
            "  - actual_parameter_count\n"
            "  - expected_parameter_count\n"
        )

    actual_value = resolved_values.get(
        "actual_parameter_count"
    )

    expected_value = resolved_values.get(
        "expected_parameter_count"
    )

    if (
        actual_value is not None
        and expected_value is not None
        and actual_value != expected_value
    ):
        raise RuntimeError(
            "B0 canonical parameter-count fields disagree.\n\n"
            f"actual_parameter_count   : {actual_value:,}\n"
            f"expected_parameter_count : {expected_value:,}\n\n"
            "PB0-F refuses to calibrate against an internally inconsistent "
            "canonical baseline."
        )

    if actual_value is not None:
        resolved_field = "actual_parameter_count"
        resolved_count = actual_value
    else:
        resolved_field = "expected_parameter_count"
        resolved_count = expected_value

    if resolved_count is None:
        raise RuntimeError(
            "Internal error while resolving the B0 canonical parameter count."
        )

    print(
        "Resolved canonical B0 parameter field: "
        f"{resolved_field}"
    )

    return int(resolved_count)
# =============================================================================
# CANONICAL CONFIGURATION
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
        raise ValueError(
            f"matrix_rank must be positive, got {matrix_rank}"
        )

    if vector_dim <= 0:
        raise ValueError(
            f"vector_dim must be positive, got {vector_dim}"
        )

    if router_hidden <= 0:
        raise ValueError(
            f"router_hidden must be positive, got {router_hidden}"
        )

    return replace(
        canonical_cfg,
        ax_res=int(matrix_rank),
        mamba_state_dim=int(vector_dim),
        router_hidden=int(router_hidden),
        vector_router=True,
    )


# =============================================================================
# ACTUAL ARCHITECTURE INSTANTIATION
# =============================================================================


def instantiate_actual_model(
    models: Any,
    cfg: Any,
) -> dict[str, Any]:
    """
    Instantiate the actual architecture through the same public model factory.

    We intentionally do not recreate parameter formulas manually here.
    """

    key = jax.random.PRNGKey(0)

    try:
        if CANONICAL_DEEP_SUPERVISION:
            params, _forward = models.make_model(
                MODEL_NAME,
                key,
                cfg,
                auxiliary_layers=CANONICAL_AUXILIARY_LAYERS,
                future_target_count=CANONICAL_FUTURE_TARGET_COUNT,
            )
        else:
            params, _forward = models.make_model(
                MODEL_NAME,
                key,
                cfg,
            )
    except TypeError as exc:
        raise RuntimeError(
            "The make_model call signature did not accept the PB0-F "
            "canonical arguments. This indicates that language/models.py "
            "differs from the architecture version assumed by the B0 census."
        ) from exc

    return params


def count_actual_parameters(
    models: Any,
    cfg: Any,
) -> int:
    params = instantiate_actual_model(
        models=models,
        cfg=cfg,
    )

    count = int(models.count_params(params))

    if count <= 0:
        raise RuntimeError(
            "Actual model instantiation produced a non-positive parameter count."
        )

    return count


def actual_feedback_rank(
    matrix_rank: int,
    vector_dim: int,
) -> int:
    """
    This mirrors the actual current initializer implementation:

        feedback_rank = min(32, cfg.ax_res, cfg.mamba_state_dim)

    PB0-F treats this as an observed architecture constraint, not a desired
    research choice.
    """

    return min(
        32,
        int(matrix_rank),
        int(vector_dim),
    )


# =============================================================================
# SAMPLE COLLECTION
# =============================================================================


def collect_sample(
    models: Any,
    canonical_cfg: Any,
    baseline_count: int,
    sample_id: str,
    matrix_rank: int,
    vector_dim: int,
    router_hidden: int,
    source: str,
) -> ArchitectureSample:
    cfg = build_variant_config(
        canonical_cfg=canonical_cfg,
        matrix_rank=matrix_rank,
        vector_dim=vector_dim,
        router_hidden=router_hidden,
    )

    parameter_count = count_actual_parameters(
        models=models,
        cfg=cfg,
    )

    feedback_rank = actual_feedback_rank(
        matrix_rank=matrix_rank,
        vector_dim=vector_dim,
    )

    return ArchitectureSample(
        sample_id=sample_id,
        matrix_rank=int(matrix_rank),
        vector_dim=int(vector_dim),
        router_hidden=int(router_hidden),
        feedback_rank_actual=int(feedback_rank),
        parameter_count=int(parameter_count),
        parameter_delta_from_baseline=int(
            parameter_count - baseline_count
        ),
        source=source,
    )


def build_calibration_samples(
    models: Any,
    canonical_cfg: Any,
    baseline_count: int,
) -> list[ArchitectureSample]:
    """
    Build a minimal but structurally informative calibration design.

    The formula basis is:

        total
        = intercept
        + a * r
        + b * n
        + c * h
        + d * (f * r)
        + e * f

    where:

        r = matrix rank
        n = vector state dimension
        h = router hidden width
        f = min(32, r, n)

    We vary one dimension at a time and include low-rank cases that expose the
    feedback-rank cap.
    """

    samples: list[ArchitectureSample] = []

    sample_counter = 1

    baseline_dimensions = (
        CANONICAL_AX_RES,
        CANONICAL_MAMBA_STATE_DIM,
        CANONICAL_ROUTER_HIDDEN,
    )

    def add(
        matrix_rank: int,
        vector_dim: int,
        router_hidden: int,
        source: str,
    ) -> None:
        nonlocal sample_counter

        sample = collect_sample(
            models=models,
            canonical_cfg=canonical_cfg,
            baseline_count=baseline_count,
            sample_id=f"S{sample_counter:03d}",
            matrix_rank=matrix_rank,
            vector_dim=vector_dim,
            router_hidden=router_hidden,
            source=source,
        )

        samples.append(sample)
        sample_counter += 1

        print(
            f"{sample.sample_id} "
            f"r={sample.matrix_rank:>3} "
            f"n={sample.vector_dim:>3} "
            f"h={sample.router_hidden:>3} "
            f"f={sample.feedback_rank_actual:>2} "
            f"params={sample.parameter_count:,}"
        )

    # Baseline sample.
    add(
        matrix_rank=baseline_dimensions[0],
        vector_dim=baseline_dimensions[1],
        router_hidden=baseline_dimensions[2],
        source="canonical_baseline",
    )

    # Matrix-rank sweep.
    for rank in CALIBRATION_MATRIX_RANKS:
        add(
            matrix_rank=rank,
            vector_dim=CANONICAL_MAMBA_STATE_DIM,
            router_hidden=CANONICAL_ROUTER_HIDDEN,
            source="matrix_rank_sweep",
        )

    # Vector-state sweep.
    for vector_dim in CALIBRATION_VECTOR_DIMS:
        add(
            matrix_rank=CANONICAL_AX_RES,
            vector_dim=vector_dim,
            router_hidden=CANONICAL_ROUTER_HIDDEN,
            source="vector_dimension_sweep",
        )

    # Router-width sweep.
    for router_hidden in CALIBRATION_ROUTER_WIDTHS:
        add(
            matrix_rank=CANONICAL_AX_RES,
            vector_dim=CANONICAL_MAMBA_STATE_DIM,
            router_hidden=router_hidden,
            source="router_width_sweep",
        )

    # Feedback-cap interaction probes.
    interaction_cases = (
        (16, 512, CANONICAL_ROUTER_HIDDEN),
        (24, 512, CANONICAL_ROUTER_HIDDEN),
        (32, 512, CANONICAL_ROUTER_HIDDEN),
        (48, 512, CANONICAL_ROUTER_HIDDEN),
        (512, 16, CANONICAL_ROUTER_HIDDEN),
        (512, 24, CANONICAL_ROUTER_HIDDEN),
        (512, 32, CANONICAL_ROUTER_HIDDEN),
        (512, 48, CANONICAL_ROUTER_HIDDEN),
        (16, 16, CANONICAL_ROUTER_HIDDEN),
        (24, 24, CANONICAL_ROUTER_HIDDEN),
        (32, 32, CANONICAL_ROUTER_HIDDEN),
        (64, 64, CANONICAL_ROUTER_HIDDEN),
    )

    for matrix_rank, vector_dim, router_hidden in interaction_cases:
        add(
            matrix_rank=matrix_rank,
            vector_dim=vector_dim,
            router_hidden=router_hidden,
            source="feedback_rank_interaction",
        )

    return samples


# =============================================================================
# FORMULA FITTING
# =============================================================================


def feature_row(
    matrix_rank: int,
    vector_dim: int,
    router_hidden: int,
) -> list[int]:
    feedback_rank = actual_feedback_rank(
        matrix_rank=matrix_rank,
        vector_dim=vector_dim,
    )

    return [
        1,
        int(matrix_rank),
        int(vector_dim),
        int(router_hidden),
        int(feedback_rank * matrix_rank),
        int(feedback_rank),
    ]


def build_design_matrix(
    samples: list[ArchitectureSample],
) -> tuple[np.ndarray, np.ndarray]:
    x_rows = [
        feature_row(
            sample.matrix_rank,
            sample.vector_dim,
            sample.router_hidden,
        )
        for sample in samples
    ]

    y_values = [
        sample.parameter_count
        for sample in samples
    ]

    return (
        np.asarray(x_rows, dtype=np.float64),
        np.asarray(y_values, dtype=np.float64),
    )


def solve_integer_formula(
    samples: list[ArchitectureSample],
) -> FormulaCoefficients:
    """
    Solve using least squares, then require integer coefficients and exact
    reconstruction of all calibration samples.

    Parameter counts are integer arithmetic. A non-integer or inexact formula
    is rejected rather than rounded silently.
    """

    x, y = build_design_matrix(samples)

    rank = np.linalg.matrix_rank(x)

    if rank < x.shape[1]:
        raise RuntimeError(
            "Calibration design matrix is rank deficient.\n"
            f"Matrix rank: {rank}\n"
            f"Required rank: {x.shape[1]}\n\n"
            "PB0-F cannot derive an identifiable parameter formula from the "
            "current calibration samples."
        )

    coefficients_float, residuals, solved_rank, singular_values = (
        np.linalg.lstsq(
            x,
            y,
            rcond=None,
        )
    )

    rounded = np.rint(coefficients_float).astype(np.int64)

    max_float_distance = float(
        np.max(
            np.abs(
                coefficients_float - rounded.astype(np.float64)
            )
        )
    )

    if max_float_distance > 1e-6:
        raise RuntimeError(
            "Derived parameter formula coefficients are not numerically "
            "integer-valued.\n"
            f"Maximum coefficient distance from nearest integer: "
            f"{max_float_distance}\n\n"
            "This means the selected formula basis is insufficient or the "
            "architecture contains additional interactions."
        )

    predictions = x.astype(np.int64) @ rounded
    actual = y.astype(np.int64)

    errors = predictions - actual

    if np.any(errors != 0):
        max_error = int(np.max(np.abs(errors)))

        raise RuntimeError(
            "The proposed parameter formula does not exactly reproduce the "
            "actual calibration samples.\n"
            f"Maximum absolute reconstruction error: {max_error}\n\n"
            "PB0-F intentionally refuses approximate parameter formulas."
        )

    return FormulaCoefficients(
        intercept=int(rounded[0]),
        matrix_rank=int(rounded[1]),
        vector_dim=int(rounded[2]),
        router_hidden=int(rounded[3]),
        feedback_rank_times_matrix_rank=int(rounded[4]),
        feedback_rank=int(rounded[5]),
    )


def predict_parameters(
    coefficients: FormulaCoefficients,
    matrix_rank: int,
    vector_dim: int,
    router_hidden: int,
) -> int:
    features = feature_row(
        matrix_rank=matrix_rank,
        vector_dim=vector_dim,
        router_hidden=router_hidden,
    )

    coefficient_vector = (
        coefficients.intercept,
        coefficients.matrix_rank,
        coefficients.vector_dim,
        coefficients.router_hidden,
        coefficients.feedback_rank_times_matrix_rank,
        coefficients.feedback_rank,
    )

    return int(
        sum(
            feature * coefficient
            for feature, coefficient in zip(
                features,
                coefficient_vector,
                strict=True,
            )
        )
    )


# =============================================================================
# HELD-OUT VALIDATION
# =============================================================================


def build_validation_cases() -> tuple[tuple[int, int, int], ...]:
    """
    These configurations are intentionally not the main sweep points used for
    the core formula fit.
    """

    return (
        (80, 144, 12),
        (120, 224, 28),
        (168, 384, 20),
        (200, 300, 24),
        (288, 160, 40),
        (448, 448, 56),
        (512, 320, 18),
    )


def validate_formula(
    models: Any,
    canonical_cfg: Any,
    coefficients: FormulaCoefficients,
) -> list[ValidationResult]:
    results: list[ValidationResult] = []

    for index, (
        matrix_rank,
        vector_dim,
        router_hidden,
    ) in enumerate(
        build_validation_cases(),
        start=1,
    ):
        cfg = build_variant_config(
            canonical_cfg=canonical_cfg,
            matrix_rank=matrix_rank,
            vector_dim=vector_dim,
            router_hidden=router_hidden,
        )

        actual = count_actual_parameters(
            models=models,
            cfg=cfg,
        )

        predicted = predict_parameters(
            coefficients=coefficients,
            matrix_rank=matrix_rank,
            vector_dim=vector_dim,
            router_hidden=router_hidden,
        )

        result = ValidationResult(
            sample_id=f"V{index:03d}",
            matrix_rank=matrix_rank,
            vector_dim=vector_dim,
            router_hidden=router_hidden,
            actual_feedback_rank=actual_feedback_rank(
                matrix_rank,
                vector_dim,
            ),
            actual_parameters=actual,
            predicted_parameters=predicted,
            prediction_error=predicted - actual,
            exact=actual == predicted,
        )

        results.append(result)

        print(
            f"{result.sample_id} "
            f"r={result.matrix_rank:>3} "
            f"n={result.vector_dim:>3} "
            f"h={result.router_hidden:>3} "
            f"actual={result.actual_parameters:,} "
            f"predicted={result.predicted_parameters:,} "
            f"error={result.prediction_error:+,}"
        )

    return results


# =============================================================================
# MARKDOWN REPORT
# =============================================================================


def formula_as_text(
    coefficients: FormulaCoefficients,
) -> str:
    return (
        "P = "
        f"{coefficients.intercept}"
        f" + ({coefficients.matrix_rank} * r)"
        f" + ({coefficients.vector_dim} * n)"
        f" + ({coefficients.router_hidden} * h)"
        f" + ({coefficients.feedback_rank_times_matrix_rank} * f * r)"
        f" + ({coefficients.feedback_rank} * f)"
        "\n\n"
        "where:\n\n"
        "- `r` = matrix rank / `ax_res`\n"
        "- `n` = vector recurrent state dimension / `mamba_state_dim`\n"
        "- `h` = router hidden width\n"
        "- `f = min(32, r, n)` in the current canonical "
        "`MemoryFeedbackArchive` initializer\n"
    )


def write_markdown_report(
    expected_b0_count: int,
    actual_canonical_count: int,
    coefficients: FormulaCoefficients,
    calibration_samples: list[ArchitectureSample],
    validation_results: list[ValidationResult],
) -> None:
    baseline_exact = (
        expected_b0_count == actual_canonical_count
    )

    all_validation_exact = all(
        result.exact
        for result in validation_results
    )

    lines: list[str] = []

    lines.append("# PB0-F Actual Architecture Parameter Formula Calibration")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "PB0-F calibrates parameter accounting against the actual "
        "`language/models.py` implementation."
    )
    lines.append("")
    lines.append(
        "No canonical architecture source file, checkpoint, dataset, "
        "or training configuration was modified."
    )
    lines.append("")
    lines.append("## Canonical Architecture Verification")
    lines.append("")
    lines.append(
        f"- B0 measured parameter count: **{expected_b0_count:,}**"
    )
    lines.append(
        f"- PB0-F actual instantiated count: "
        f"**{actual_canonical_count:,}**"
    )
    lines.append(
        f"- Difference: "
        f"**{actual_canonical_count - expected_b0_count:+,}**"
    )
    lines.append(
        f"- Exact canonical match: **{baseline_exact}**"
    )
    lines.append("")
    lines.append("## Calibrated Formula")
    lines.append("")
    lines.append("```text")
    lines.append(formula_as_text(coefficients).rstrip())
    lines.append("```")
    lines.append("")
    lines.append("## Critical Architectural Finding")
    lines.append("")
    lines.append(
        "The current `Modus_X_MemoryFeedbackArchive` initializer does not "
        "expose feedback rank as an independent `ModelConfig` dimension."
    )
    lines.append("")
    lines.append(
        "The actual initializer derives feedback rank as:"
    )
    lines.append("")
    lines.append("```text")
    lines.append("feedback_rank = min(32, ax_res, mamba_state_dim)")
    lines.append("```")
    lines.append("")
    lines.append(
        "Therefore the PB0-E proposed value `feedback_rank = 7` cannot be "
        "realized through the current canonical configuration alone. "
        "Implementing that value would require an explicit architecture "
        "change to the initializer."
    )
    lines.append("")
    lines.append("## Calibration Samples")
    lines.append("")
    lines.append(
        "| Sample | r | n | h | Actual feedback rank | Parameters | Source |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---|"
    )

    for sample in calibration_samples:
        lines.append(
            f"| `{sample.sample_id}` "
            f"| {sample.matrix_rank:,} "
            f"| {sample.vector_dim:,} "
            f"| {sample.router_hidden:,} "
            f"| {sample.feedback_rank_actual:,} "
            f"| {sample.parameter_count:,} "
            f"| `{sample.source}` |"
        )

    lines.append("")
    lines.append("## Held-Out Formula Validation")
    lines.append("")
    lines.append(
        "| Sample | r | n | h | f | Actual | Predicted | Error | Exact |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---|"
    )

    for result in validation_results:
        lines.append(
            f"| `{result.sample_id}` "
            f"| {result.matrix_rank:,} "
            f"| {result.vector_dim:,} "
            f"| {result.router_hidden:,} "
            f"| {result.actual_feedback_rank:,} "
            f"| {result.actual_parameters:,} "
            f"| {result.predicted_parameters:,} "
            f"| {result.prediction_error:+,} "
            f"| {result.exact} |"
        )

    lines.append("")
    lines.append("## Calibration Gate")
    lines.append("")

    if baseline_exact and all_validation_exact:
        lines.append(
            "PB0-F **PASSED**. The calibrated formula reproduces both the "
            "canonical architecture and all held-out validation "
            "configurations exactly."
        )
    else:
        lines.append(
            "PB0-F **FAILED**. The current analytical formula must not be "
            "used for Candidate A dimension search."
        )

    lines.append("")
    lines.append("## Next Gate")
    lines.append("")
    lines.append(
        "Do not modify `language/models.py` and do not train Candidate A "
        "until the next search uses the PB0-F calibrated formula."
    )
    lines.append("")
    lines.append(
        "The next step is **PB0-G: calibrated Candidate A dimension search**, "
        "with two separate modes:"
    )
    lines.append("")
    lines.append(
        "1. Search dimensions that are realizable without modifying the "
        "canonical architecture."
    )
    lines.append(
        "2. Separately identify which PB0-D targets require explicit "
        "architecture changes, including independently configurable "
        "feedback rank."
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
        "MODUS_X PB0-F ACTUAL ARCHITECTURE PARAMETER FORMULA CALIBRATION"
    )

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Models source: {MODELS_PATH}")
    print(f"B0 input     : {B0_JSON_PATH}")
    print(f"Output dir   : {OUTPUT_DIR}")

    require_file(
        MODELS_PATH,
        "Canonical language/models.py",
    )

    require_file(
        B0_JSON_PATH,
        "B0 parameter census",
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print_banner(
        "IMPORTING ACTUAL CANONICAL ARCHITECTURE"
    )

    models = import_project_models()

    print(
        f"Imported module: {models.__file__}"
    )

    print_banner(
        "LOADING B0 CANONICAL BASELINE"
    )

    expected_b0_count = (
        load_b0_expected_parameter_count()
    )

    print(
        f"B0 measured parameter count: "
        f"{expected_b0_count:,}"
    )

    print_banner(
        "INSTANTIATING CANONICAL CONFIGURATION"
    )

    canonical_cfg = build_canonical_config(
        models
    )

    actual_canonical_count = (
        count_actual_parameters(
            models=models,
            cfg=canonical_cfg,
        )
    )

    print(
        f"PB0-F instantiated count   : "
        f"{actual_canonical_count:,}"
    )

    print(
        f"Difference from B0         : "
        f"{actual_canonical_count - expected_b0_count:+,}"
    )

    if actual_canonical_count != expected_b0_count:
        raise RuntimeError(
            "\n"
            "PB0-F canonical verification failed.\n\n"
            f"B0 expected: {expected_b0_count:,}\n"
            f"Actual init: {actual_canonical_count:,}\n"
            f"Difference : "
            f"{actual_canonical_count - expected_b0_count:+,}\n\n"
            "Do not calibrate Candidate A from a mismatched baseline.\n"
            "The canonical ModelConfig or make_model options used by B0 must "
            "be reconciled first."
        )

    print_banner(
        "COLLECTING ACTUAL ARCHITECTURE CALIBRATION SAMPLES"
    )

    calibration_samples = (
        build_calibration_samples(
            models=models,
            canonical_cfg=canonical_cfg,
            baseline_count=actual_canonical_count,
        )
    )

    print_banner(
        "DERIVING EXACT PARAMETER FORMULA"
    )

    coefficients = solve_integer_formula(
        calibration_samples
    )

    print("Calibrated formula:")
    print()
    print(
        formula_as_text(
            coefficients
        )
    )

    print_banner(
        "VALIDATING FORMULA ON HELD-OUT ARCHITECTURES"
    )

    validation_results = validate_formula(
        models=models,
        canonical_cfg=canonical_cfg,
        coefficients=coefficients,
    )

    validation_exact = all(
        result.exact
        for result in validation_results
    )

    if not validation_exact:
        failures = [
            result
            for result in validation_results
            if not result.exact
        ]

        raise RuntimeError(
            "PB0-F formula validation failed on held-out architectures.\n\n"
            + "\n".join(
                (
                    f"{result.sample_id}: "
                    f"actual={result.actual_parameters:,}, "
                    f"predicted={result.predicted_parameters:,}, "
                    f"error={result.prediction_error:+,}"
                )
                for result in failures
            )
        )

    print_banner(
        "WRITING PB0-F OUTPUTS"
    )

    calibration_rows = [
        {
            "sample_id": sample.sample_id,
            "matrix_rank": sample.matrix_rank,
            "vector_dim": sample.vector_dim,
            "router_hidden": sample.router_hidden,
            "feedback_rank_actual": sample.feedback_rank_actual,
            "parameter_count": sample.parameter_count,
            "parameter_delta_from_baseline": (
                sample.parameter_delta_from_baseline
            ),
            "source": sample.source,
        }
        for sample in calibration_samples
    ]

    write_csv(
        path=OUTPUT_CSV_PATH,
        rows=calibration_rows,
        fieldnames=[
            "sample_id",
            "matrix_rank",
            "vector_dim",
            "router_hidden",
            "feedback_rank_actual",
            "parameter_count",
            "parameter_delta_from_baseline",
            "source",
        ],
    )

    payload = {
        "generated_at_utc": utc_now(),
        "project_root": str(PROJECT_ROOT),
        "models_path": str(MODELS_PATH),
        "model_name": MODEL_NAME,
        "scope": (
            "Actual architecture parameter formula calibration. "
            "No model source, checkpoint, dataset, or training run modified."
        ),
        "canonical_configuration": asdict(
            canonical_cfg
        ),
        "b0_expected_parameter_count": (
            expected_b0_count
        ),
        "pb0_f_actual_canonical_parameter_count": (
            actual_canonical_count
        ),
        "canonical_parameter_difference": (
            actual_canonical_count
            - expected_b0_count
        ),
        "canonical_exact_match": (
            actual_canonical_count
            == expected_b0_count
        ),
        "feedback_rank_rule": (
            "min(32, ax_res, mamba_state_dim)"
        ),
        "feedback_rank_independently_configurable": False,
        "formula_coefficients": asdict(
            coefficients
        ),
        "formula": (
            "P = intercept"
            " + matrix_rank_coefficient * r"
            " + vector_dim_coefficient * n"
            " + router_hidden_coefficient * h"
            " + feedback_rank_times_matrix_rank_coefficient * f * r"
            " + feedback_rank_coefficient * f"
        ),
        "formula_variables": {
            "r": "matrix rank / ax_res",
            "n": "vector recurrent state dimension / mamba_state_dim",
            "h": "router hidden width",
            "f": "min(32, r, n)",
        },
        "calibration_samples": [
            asdict(sample)
            for sample in calibration_samples
        ],
        "held_out_validation": [
            asdict(result)
            for result in validation_results
        ],
        "held_out_validation_exact": (
            validation_exact
        ),
        "status": (
            "passed"
            if validation_exact
            else "failed"
        ),
        "next_gate": (
            "PB0-G calibrated Candidate A dimension search. "
            "Do not modify language/models.py or train Candidate A before "
            "using the PB0-F calibrated formula."
        ),
    }

    write_json(
        OUTPUT_JSON_PATH,
        payload,
    )

    write_markdown_report(
        expected_b0_count=expected_b0_count,
        actual_canonical_count=actual_canonical_count,
        coefficients=coefficients,
        calibration_samples=calibration_samples,
        validation_results=validation_results,
    )

    print_banner(
        "PB0-F CALIBRATION COMPLETE"
    )

    print(
        f"Canonical parameter count : "
        f"{actual_canonical_count:,}"
    )

    print(
        f"Calibration samples       : "
        f"{len(calibration_samples)}"
    )

    print(
        f"Held-out validations      : "
        f"{len(validation_results)}"
    )

    print(
        f"All validation exact      : "
        f"{validation_exact}"
    )

    print()
    print("Formula:")
    print(
        formula_as_text(
            coefficients
        )
    )

    print("Output files:")
    print(f"  {OUTPUT_JSON_PATH}")
    print(f"  {OUTPUT_MD_PATH}")
    print(f"  {OUTPUT_CSV_PATH}")

    print()
    print(
        "NEXT GATE: inspect PB0_F_PARAMETER_FORMULA_CALIBRATION.md."
    )
    print(
        "Do not modify language/models.py and do not train Candidate A yet."
    )
    print(
        "If PB0-F passes, the next action is PB0-G calibrated Candidate A "
        "dimension search."
    )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print()
        print("=" * 80)
        print("PB0-F INTERRUPTED")
        print("=" * 80)
        print(
            "No architecture source file was modified."
        )
        sys.exit(130)

    except Exception as exc:
        print()
        print("=" * 80)
        print("PB0-F CALIBRATION FAILED")
        print("=" * 80)
        print()
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print()
        traceback.print_exc()
        sys.exit(1)