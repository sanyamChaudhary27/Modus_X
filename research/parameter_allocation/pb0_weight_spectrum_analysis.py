from __future__ import annotations

import csv
import json
import math
import pickle
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
OUTPUT_DIR = SCRIPT_PATH.parent / "outputs"

SEED_DIRECTORIES: dict[str, Path] = {
    "seed1": Path(r"E:\seed1"),
    "seed2": Path(r"E:\seed2"),
}

CHECKPOINT_FILENAME = "checkpoint.pkl"

TARGET_FAMILIES: tuple[str, ...] = (
    "m_wq",
    "m_wk",
    "m_wv",
    "m_w_read",
    "m_w_out",
)

CANDIDATE_RANKS: tuple[int, ...] = (
    384,
    320,
    256,
    192,
    160,
    128,
    96,
    64,
)

OUTPUT_JSON = OUTPUT_DIR / "PB0_WEIGHT_SPECTRUM_ANALYSIS.json"
OUTPUT_MARKDOWN = OUTPUT_DIR / "PB0_WEIGHT_SPECTRUM_ANALYSIS.md"
OUTPUT_LAYER_CSV = OUTPUT_DIR / "PB0_WEIGHT_SPECTRUM_LAYERS.csv"
OUTPUT_SUMMARY_CSV = OUTPUT_DIR / "PB0_WEIGHT_SPECTRUM_SUMMARY.csv"
OUTPUT_SEED_COMPARISON_CSV = OUTPUT_DIR / "PB0_WEIGHT_SPECTRUM_SEED_COMPARISON.csv"


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass(frozen=True)
class RankMetrics:
    rank: int
    retained_energy_fraction: float
    retained_energy_percent: float
    relative_frobenius_error: float


@dataclass(frozen=True)
class MatrixAnalysis:
    seed: str
    family: str
    layer: int
    parameter_path: str
    shape: tuple[int, int]
    dtype: str
    elements: int

    numerical_rank: int
    stable_rank: float

    largest_singular_value: float
    smallest_singular_value: float
    spectral_norm: float
    frobenius_norm: float

    condition_number: float | None
    effective_rank_entropy: float

    rank_metrics: tuple[RankMetrics, ...]


@dataclass(frozen=True)
class FamilySeedSummary:
    seed: str
    family: str
    matrices: int

    mean_stable_rank: float
    median_stable_rank: float
    min_stable_rank: float
    max_stable_rank: float

    mean_effective_rank_entropy: float
    median_effective_rank_entropy: float

    mean_condition_number: float | None
    median_condition_number: float | None

    rank_energy_mean: dict[str, float]
    rank_energy_min: dict[str, float]
    rank_error_mean: dict[str, float]
    rank_error_max: dict[str, float]


@dataclass(frozen=True)
class CrossSeedComparison:
    family: str
    matrices_per_seed: int

    stable_rank_seed1_mean: float
    stable_rank_seed2_mean: float
    stable_rank_absolute_difference: float
    stable_rank_relative_difference_percent: float | None

    effective_rank_seed1_mean: float
    effective_rank_seed2_mean: float
    effective_rank_absolute_difference: float

    rank_energy_absolute_differences: dict[str, float]

    consistency_label: str


# =============================================================================
# CONSOLE HELPERS
# =============================================================================


def banner(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def fail(message: str) -> None:
    raise RuntimeError(message)


def format_int(value: int) -> str:
    return f"{value:,}"


def format_float(value: float, digits: int = 6) -> str:
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Inf"
    return f"{value:.{digits}f}"


def format_percent(value: float, digits: int = 4) -> str:
    return f"{value * 100:.{digits}f}%"


# =============================================================================
# CHECKPOINT LOADING
# =============================================================================


def validate_checkpoint_path(seed_name: str, seed_directory: Path) -> Path:
    if not seed_directory.exists():
        fail(
            f"Seed directory for {seed_name} does not exist:\n"
            f"{seed_directory}"
        )

    checkpoint_path = seed_directory / CHECKPOINT_FILENAME

    if not checkpoint_path.exists():
        fail(
            f"Checkpoint for {seed_name} does not exist:\n"
            f"{checkpoint_path}"
        )

    if not checkpoint_path.is_file():
        fail(
            f"Checkpoint path for {seed_name} is not a regular file:\n"
            f"{checkpoint_path}"
        )

    return checkpoint_path


def load_pickle_checkpoint(checkpoint_path: Path) -> Any:
    try:
        with checkpoint_path.open("rb") as checkpoint_file:
            return pickle.load(checkpoint_file)
    except Exception as exc:
        fail(
            f"Unable to load checkpoint:\n"
            f"{checkpoint_path}\n\n"
            f"Original error:\n"
            f"{type(exc).__name__}: {exc}"
        )


def get_checkpoint_params(checkpoint_object: Any, checkpoint_path: Path) -> Any:
    if not isinstance(checkpoint_object, Mapping):
        fail(
            f"Checkpoint root is not a mapping:\n"
            f"{checkpoint_path}\n"
            f"Actual type: {type(checkpoint_object).__name__}"
        )

    if "params" not in checkpoint_object:
        fail(
            f"Checkpoint does not contain a top-level 'params' key:\n"
            f"{checkpoint_path}\n"
            f"Available keys: {sorted(str(key) for key in checkpoint_object.keys())}"
        )

    return checkpoint_object["params"]


# =============================================================================
# PARAMETER TREE ACCESS
# =============================================================================


def try_direct_path_lookup(
    tree: Any,
    path: Sequence[str],
) -> Any | None:
    current = tree

    for key in path:
        if not isinstance(current, Mapping):
            return None

        if key not in current:
            return None

        current = current[key]

    return current


def recursively_find_key(
    tree: Any,
    target_key: str,
) -> list[tuple[str, Any]]:
    results: list[tuple[str, Any]] = []

    def walk(value: Any, prefix: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_text = str(key)
                child_path = (
                    key_text
                    if prefix == ""
                    else f"{prefix}.{key_text}"
                )

                if key_text == target_key:
                    results.append((child_path, child))

                walk(child, child_path)

    walk(tree, "")
    return results


def normalize_parameter_array(value: Any, parameter_path: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except Exception as exc:
        fail(
            f"Unable to convert parameter to NumPy array:\n"
            f"{parameter_path}\n"
            f"Type: {type(value).__name__}\n"
            f"Error: {type(exc).__name__}: {exc}"
        )

    if array.ndim != 3:
        fail(
            f"Expected a rank-3 tensor [layers, rows, columns] for:\n"
            f"{parameter_path}\n"
            f"Actual shape: {tuple(array.shape)}"
        )

    if array.shape[0] != 12:
        fail(
            f"Expected 12 layers for:\n"
            f"{parameter_path}\n"
            f"Actual shape: {tuple(array.shape)}"
        )

    if array.shape[1:] != (512, 512):
        fail(
            f"Expected [12, 512, 512] for:\n"
            f"{parameter_path}\n"
            f"Actual shape: {tuple(array.shape)}"
        )

    if not np.issubdtype(array.dtype, np.number):
        fail(
            f"Parameter is not numeric:\n"
            f"{parameter_path}\n"
            f"Dtype: {array.dtype}"
        )

    if not np.all(np.isfinite(array)):
        non_finite_count = int(np.size(array) - np.count_nonzero(np.isfinite(array)))

        fail(
            f"Parameter contains NaN or Inf values:\n"
            f"{parameter_path}\n"
            f"Non-finite values: {format_int(non_finite_count)}"
        )

    return np.asarray(array, dtype=np.float64)


def resolve_family_tensor(
    params: Any,
    family: str,
) -> tuple[str, np.ndarray]:
    direct_candidates: tuple[tuple[str, ...], ...] = (
        ("layers", family),
        (f"layers.{family}",),
        (family,),
    )

    for candidate in direct_candidates:
        value = try_direct_path_lookup(params, candidate)

        if value is not None:
            if len(candidate) == 2:
                path = ".".join(candidate)
            else:
                path = candidate[0]

            return path, normalize_parameter_array(value, path)

    matches = recursively_find_key(params, family)

    if len(matches) == 0:
        fail(
            f"Could not locate parameter family '{family}' in checkpoint params."
        )

    if len(matches) > 1:
        candidate_paths = [path for path, _ in matches]

        fail(
            f"Found multiple possible parameter matches for '{family}'.\n"
            f"Candidates:\n"
            + "\n".join(f"  - {path}" for path in candidate_paths)
        )

    parameter_path, value = matches[0]

    return (
        parameter_path,
        normalize_parameter_array(value, parameter_path),
    )


# =============================================================================
# SPECTRAL METRICS
# =============================================================================


def calculate_numerical_rank(
    singular_values: np.ndarray,
    rows: int,
    columns: int,
) -> int:
    if singular_values.size == 0:
        return 0

    maximum_singular_value = float(singular_values[0])

    if maximum_singular_value == 0.0:
        return 0

    machine_epsilon = np.finfo(np.float64).eps
    tolerance = (
        max(rows, columns)
        * machine_epsilon
        * maximum_singular_value
    )

    return int(np.count_nonzero(singular_values > tolerance))


def calculate_stable_rank(
    singular_values: np.ndarray,
) -> float:
    if singular_values.size == 0:
        return 0.0

    spectral_norm_squared = float(singular_values[0] ** 2)
    frobenius_norm_squared = float(np.sum(singular_values ** 2))

    if spectral_norm_squared <= 0.0:
        return 0.0

    return frobenius_norm_squared / spectral_norm_squared


def calculate_entropy_effective_rank(
    singular_values: np.ndarray,
) -> float:
    squared_singular_values = singular_values ** 2
    total_energy = float(np.sum(squared_singular_values))

    if total_energy <= 0.0:
        return 0.0

    probabilities = squared_singular_values / total_energy

    positive_probabilities = probabilities[probabilities > 0.0]

    if positive_probabilities.size == 0:
        return 0.0

    entropy = float(
        -np.sum(
            positive_probabilities
            * np.log(positive_probabilities)
        )
    )

    return float(np.exp(entropy))


def calculate_condition_number(
    singular_values: np.ndarray,
) -> float | None:
    if singular_values.size == 0:
        return None

    largest = float(singular_values[0])
    smallest = float(singular_values[-1])

    if smallest <= 0.0:
        return None

    return largest / smallest


def calculate_rank_metrics(
    singular_values: np.ndarray,
    candidate_ranks: Iterable[int],
) -> tuple[RankMetrics, ...]:
    squared_singular_values = singular_values ** 2
    total_energy = float(np.sum(squared_singular_values))

    if total_energy <= 0.0:
        return tuple(
            RankMetrics(
                rank=rank,
                retained_energy_fraction=0.0,
                retained_energy_percent=0.0,
                relative_frobenius_error=1.0,
            )
            for rank in candidate_ranks
        )

    cumulative_energy = np.cumsum(squared_singular_values)

    metrics: list[RankMetrics] = []

    maximum_rank = singular_values.size

    for requested_rank in candidate_ranks:
        actual_rank = min(requested_rank, maximum_rank)

        retained_energy = float(
            cumulative_energy[actual_rank - 1] / total_energy
        )

        retained_energy = min(
            max(retained_energy, 0.0),
            1.0,
        )

        relative_frobenius_error = float(
            math.sqrt(
                max(
                    0.0,
                    1.0 - retained_energy,
                )
            )
        )

        metrics.append(
            RankMetrics(
                rank=requested_rank,
                retained_energy_fraction=retained_energy,
                retained_energy_percent=retained_energy * 100.0,
                relative_frobenius_error=relative_frobenius_error,
            )
        )

    return tuple(metrics)


def analyze_single_matrix(
    *,
    seed: str,
    family: str,
    layer: int,
    parameter_path: str,
    matrix: np.ndarray,
) -> MatrixAnalysis:
    if matrix.shape != (512, 512):
        fail(
            f"Expected matrix shape [512, 512] but received:\n"
            f"{parameter_path}[{layer}]\n"
            f"Shape: {tuple(matrix.shape)}"
        )

    try:
        singular_values = np.linalg.svd(
            matrix,
            compute_uv=False,
            full_matrices=False,
        )
    except np.linalg.LinAlgError as exc:
        fail(
            f"SVD failed for:\n"
            f"{parameter_path}[{layer}]\n\n"
            f"{type(exc).__name__}: {exc}"
        )

    if singular_values.size != 512:
        fail(
            f"Unexpected singular value count for:\n"
            f"{parameter_path}[{layer}]\n"
            f"Expected: 512\n"
            f"Actual: {singular_values.size}"
        )

    numerical_rank = calculate_numerical_rank(
        singular_values,
        rows=matrix.shape[0],
        columns=matrix.shape[1],
    )

    stable_rank = calculate_stable_rank(
        singular_values
    )

    effective_rank_entropy = calculate_entropy_effective_rank(
        singular_values
    )

    largest_singular_value = float(singular_values[0])
    smallest_singular_value = float(singular_values[-1])

    spectral_norm = largest_singular_value

    frobenius_norm = float(
        math.sqrt(
            float(
                np.sum(
                    singular_values ** 2
                )
            )
        )
    )

    condition_number = calculate_condition_number(
        singular_values
    )

    rank_metrics = calculate_rank_metrics(
        singular_values,
        CANDIDATE_RANKS,
    )

    return MatrixAnalysis(
        seed=seed,
        family=family,
        layer=layer,
        parameter_path=f"{parameter_path}[{layer}]",
        shape=(512, 512),
        dtype=str(matrix.dtype),
        elements=int(matrix.size),
        numerical_rank=numerical_rank,
        stable_rank=stable_rank,
        largest_singular_value=largest_singular_value,
        smallest_singular_value=smallest_singular_value,
        spectral_norm=spectral_norm,
        frobenius_norm=frobenius_norm,
        condition_number=condition_number,
        effective_rank_entropy=effective_rank_entropy,
        rank_metrics=rank_metrics,
    )


# =============================================================================
# AGGREGATION
# =============================================================================


def values_for_family(
    analyses: Sequence[MatrixAnalysis],
    seed: str,
    family: str,
) -> list[MatrixAnalysis]:
    selected = [
        analysis
        for analysis in analyses
        if analysis.seed == seed
        and analysis.family == family
    ]

    if not selected:
        fail(
            f"No matrix analyses found for seed={seed}, family={family}"
        )

    return selected


def rank_metric_map(
    analysis: MatrixAnalysis,
) -> dict[int, RankMetrics]:
    return {
        metric.rank: metric
        for metric in analysis.rank_metrics
    }


def optional_mean(
    values: Sequence[float | None],
) -> float | None:
    finite_values = [
        value
        for value in values
        if value is not None
        and math.isfinite(value)
    ]

    if not finite_values:
        return None

    return float(np.mean(finite_values))


def optional_median(
    values: Sequence[float | None],
) -> float | None:
    finite_values = [
        value
        for value in values
        if value is not None
        and math.isfinite(value)
    ]

    if not finite_values:
        return None

    return float(np.median(finite_values))


def summarize_family_seed(
    analyses: Sequence[MatrixAnalysis],
    seed: str,
    family: str,
) -> FamilySeedSummary:
    selected = values_for_family(
        analyses,
        seed,
        family,
    )

    stable_ranks = np.asarray(
        [analysis.stable_rank for analysis in selected],
        dtype=np.float64,
    )

    effective_ranks = np.asarray(
        [
            analysis.effective_rank_entropy
            for analysis in selected
        ],
        dtype=np.float64,
    )

    rank_energy_mean: dict[str, float] = {}
    rank_energy_min: dict[str, float] = {}
    rank_error_mean: dict[str, float] = {}
    rank_error_max: dict[str, float] = {}

    for rank in CANDIDATE_RANKS:
        rank_metrics = [
            rank_metric_map(analysis)[rank]
            for analysis in selected
        ]

        energies = np.asarray(
            [
                metric.retained_energy_fraction
                for metric in rank_metrics
            ],
            dtype=np.float64,
        )

        errors = np.asarray(
            [
                metric.relative_frobenius_error
                for metric in rank_metrics
            ],
            dtype=np.float64,
        )

        rank_key = str(rank)

        rank_energy_mean[rank_key] = float(
            np.mean(energies)
        )

        rank_energy_min[rank_key] = float(
            np.min(energies)
        )

        rank_error_mean[rank_key] = float(
            np.mean(errors)
        )

        rank_error_max[rank_key] = float(
            np.max(errors)
        )

    return FamilySeedSummary(
        seed=seed,
        family=family,
        matrices=len(selected),
        mean_stable_rank=float(np.mean(stable_ranks)),
        median_stable_rank=float(np.median(stable_ranks)),
        min_stable_rank=float(np.min(stable_ranks)),
        max_stable_rank=float(np.max(stable_ranks)),
        mean_effective_rank_entropy=float(
            np.mean(effective_ranks)
        ),
        median_effective_rank_entropy=float(
            np.median(effective_ranks)
        ),
        mean_condition_number=optional_mean(
            [
                analysis.condition_number
                for analysis in selected
            ]
        ),
        median_condition_number=optional_median(
            [
                analysis.condition_number
                for analysis in selected
            ]
        ),
        rank_energy_mean=rank_energy_mean,
        rank_energy_min=rank_energy_min,
        rank_error_mean=rank_error_mean,
        rank_error_max=rank_error_max,
    )


def classify_consistency(
    stable_rank_relative_difference_percent: float | None,
    energy_differences: Sequence[float],
) -> str:
    if stable_rank_relative_difference_percent is None:
        return "unknown"

    max_energy_difference = (
        max(energy_differences)
        if energy_differences
        else float("inf")
    )

    if (
        stable_rank_relative_difference_percent <= 5.0
        and max_energy_difference <= 0.005
    ):
        return "high"

    if (
        stable_rank_relative_difference_percent <= 15.0
        and max_energy_difference <= 0.02
    ):
        return "moderate"

    return "low"

def compare_family_between_seeds(
    summaries: Sequence[FamilySeedSummary],
    family: str,
) -> CrossSeedComparison:
    seed1: FamilySeedSummary | None = next(
        (
            summary
            for summary in summaries
            if summary.seed == "seed1"
            and summary.family == family
        ),
        None,
    )

    seed2: FamilySeedSummary | None = next(
        (
            summary
            for summary in summaries
            if summary.seed == "seed2"
            and summary.family == family
        ),
        None,
    )

    if seed1 is None:
        fail(
            f"Cannot compare family '{family}' because the Seed1 "
            "summary is missing."
        )

    if seed2 is None:
        fail(
            f"Cannot compare family '{family}' because the Seed2 "
            "summary is missing."
        )

    if seed1.matrices != seed2.matrices:
        fail(
            f"Cannot compare family '{family}' because the seed matrix "
            f"counts do not match. "
            f"Seed1={seed1.matrices}, "
            f"Seed2={seed2.matrices}"
        )

    stable_difference = abs(
        seed1.mean_stable_rank
        - seed2.mean_stable_rank
    )

    stable_baseline = (
        abs(seed1.mean_stable_rank)
        + abs(seed2.mean_stable_rank)
    ) / 2.0

    stable_rank_relative_difference_percent: float | None

    if stable_baseline <= 0.0:
        stable_rank_relative_difference_percent = None
    else:
        stable_rank_relative_difference_percent = (
            stable_difference
            / stable_baseline
            * 100.0
        )

    effective_rank_difference = abs(
        seed1.mean_effective_rank_entropy
        - seed2.mean_effective_rank_entropy
    )

    rank_energy_absolute_differences: dict[str, float] = {}

    for rank in CANDIDATE_RANKS:
        rank_key = str(rank)

        seed1_energy = seed1.rank_energy_mean.get(
            rank_key
        )

        seed2_energy = seed2.rank_energy_mean.get(
            rank_key
        )

        if seed1_energy is None:
            fail(
                f"Missing rank-energy summary for family '{family}', "
                f"rank {rank} in Seed1."
            )

        if seed2_energy is None:
            fail(
                f"Missing rank-energy summary for family '{family}', "
                f"rank {rank} in Seed2."
            )

        rank_energy_absolute_differences[rank_key] = abs(
            seed1_energy
            - seed2_energy
        )

    consistency_label = classify_consistency(
        stable_rank_relative_difference_percent,
        list(
            rank_energy_absolute_differences.values()
        ),
    )

    return CrossSeedComparison(
        family=family,
        matrices_per_seed=seed1.matrices,
        stable_rank_seed1_mean=seed1.mean_stable_rank,
        stable_rank_seed2_mean=seed2.mean_stable_rank,
        stable_rank_absolute_difference=stable_difference,
        stable_rank_relative_difference_percent=(
            stable_rank_relative_difference_percent
        ),
        effective_rank_seed1_mean=(
            seed1.mean_effective_rank_entropy
        ),
        effective_rank_seed2_mean=(
            seed2.mean_effective_rank_entropy
        ),
        effective_rank_absolute_difference=(
            effective_rank_difference
        ),
        rank_energy_absolute_differences=(
            rank_energy_absolute_differences
        ),
        consistency_label=consistency_label,
    )

def make_json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, tuple):
        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(value, list):
        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None

        return value

    return value


def write_json_report(
    *,
    analyses: Sequence[MatrixAnalysis],
    summaries: Sequence[FamilySeedSummary],
    comparisons: Sequence[CrossSeedComparison],
    checkpoint_metadata: Mapping[str, Any],
) -> None:
    payload = {
        "analysis": {
            "name": "PB0 Weight Spectrum Analysis",
            "purpose": (
                "Frozen-checkpoint spectral analysis of the five "
                "matrix-memory projection families before any architecture "
                "modification."
            ),
            "candidate_ranks": list(CANDIDATE_RANKS),
            "target_families": list(TARGET_FAMILIES),
        },
        "project_root": str(PROJECT_ROOT),
        "checkpoint_metadata": dict(checkpoint_metadata),
        "family_seed_summaries": [
            asdict(summary)
            for summary in summaries
        ],
        "cross_seed_comparisons": [
            asdict(comparison)
            for comparison in comparisons
        ],
        "layer_analyses": [
            asdict(analysis)
            for analysis in analyses
        ],
    }

    OUTPUT_JSON.write_text(
        json.dumps(
            make_json_safe(payload),
            indent=2,
        ),
        encoding="utf-8",
    )


def layer_analysis_to_row(
    analysis: MatrixAnalysis,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "seed": analysis.seed,
        "family": analysis.family,
        "layer": analysis.layer,
        "parameter_path": analysis.parameter_path,
        "shape": "x".join(
            str(dimension)
            for dimension in analysis.shape
        ),
        "dtype": analysis.dtype,
        "elements": analysis.elements,
        "numerical_rank": analysis.numerical_rank,
        "stable_rank": analysis.stable_rank,
        "effective_rank_entropy": (
            analysis.effective_rank_entropy
        ),
        "largest_singular_value": (
            analysis.largest_singular_value
        ),
        "smallest_singular_value": (
            analysis.smallest_singular_value
        ),
        "spectral_norm": analysis.spectral_norm,
        "frobenius_norm": analysis.frobenius_norm,
        "condition_number": analysis.condition_number,
    }

    for metric in analysis.rank_metrics:
        prefix = f"rank_{metric.rank}"

        row[f"{prefix}_energy_fraction"] = (
            metric.retained_energy_fraction
        )
        row[f"{prefix}_energy_percent"] = (
            metric.retained_energy_percent
        )
        row[f"{prefix}_relative_frobenius_error"] = (
            metric.relative_frobenius_error
        )

    return row


def write_layer_csv(
    analyses: Sequence[MatrixAnalysis],
) -> None:
    rows = [
        layer_analysis_to_row(analysis)
        for analysis in analyses
    ]

    if not rows:
        fail("No layer rows available for CSV output.")

    fieldnames = list(rows[0].keys())

    with OUTPUT_LAYER_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
            extrasaction="raise",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def summary_to_row(
    summary: FamilySeedSummary,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "seed": summary.seed,
        "family": summary.family,
        "matrices": summary.matrices,
        "mean_stable_rank": (
            summary.mean_stable_rank
        ),
        "median_stable_rank": (
            summary.median_stable_rank
        ),
        "min_stable_rank": (
            summary.min_stable_rank
        ),
        "max_stable_rank": (
            summary.max_stable_rank
        ),
        "mean_effective_rank_entropy": (
            summary.mean_effective_rank_entropy
        ),
        "median_effective_rank_entropy": (
            summary.median_effective_rank_entropy
        ),
        "mean_condition_number": (
            summary.mean_condition_number
        ),
        "median_condition_number": (
            summary.median_condition_number
        ),
    }

    for rank in CANDIDATE_RANKS:
        rank_key = str(rank)

        row[f"rank_{rank}_energy_mean"] = (
            summary.rank_energy_mean[rank_key]
        )
        row[f"rank_{rank}_energy_min"] = (
            summary.rank_energy_min[rank_key]
        )
        row[f"rank_{rank}_error_mean"] = (
            summary.rank_error_mean[rank_key]
        )
        row[f"rank_{rank}_error_max"] = (
            summary.rank_error_max[rank_key]
        )

    return row


def write_summary_csv(
    summaries: Sequence[FamilySeedSummary],
) -> None:
    rows = [
        summary_to_row(summary)
        for summary in summaries
    ]

    if not rows:
        fail("No family summary rows available for CSV output.")

    fieldnames = list(rows[0].keys())

    with OUTPUT_SUMMARY_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
            extrasaction="raise",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def comparison_to_row(
    comparison: CrossSeedComparison,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "family": comparison.family,
        "matrices_per_seed": (
            comparison.matrices_per_seed
        ),
        "stable_rank_seed1_mean": (
            comparison.stable_rank_seed1_mean
        ),
        "stable_rank_seed2_mean": (
            comparison.stable_rank_seed2_mean
        ),
        "stable_rank_absolute_difference": (
            comparison.stable_rank_absolute_difference
        ),
        "stable_rank_relative_difference_percent": (
            comparison.stable_rank_relative_difference_percent
        ),
        "effective_rank_seed1_mean": (
            comparison.effective_rank_seed1_mean
        ),
        "effective_rank_seed2_mean": (
            comparison.effective_rank_seed2_mean
        ),
        "effective_rank_absolute_difference": (
            comparison.effective_rank_absolute_difference
        ),
        "consistency_label": (
            comparison.consistency_label
        ),
    }

    for rank in CANDIDATE_RANKS:
        rank_key = str(rank)

        row[
            f"rank_{rank}_energy_absolute_difference"
        ] = (
            comparison.rank_energy_absolute_differences[
                rank_key
            ]
        )

    return row


def write_seed_comparison_csv(
    comparisons: Sequence[CrossSeedComparison],
) -> None:
    rows = [
        comparison_to_row(comparison)
        for comparison in comparisons
    ]

    if not rows:
        fail(
            "No cross-seed comparison rows available for CSV output."
        )

    fieldnames = list(rows[0].keys())

    with OUTPUT_SEED_COMPARISON_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
            extrasaction="raise",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


# =============================================================================
# MARKDOWN REPORT
# =============================================================================


def markdown_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> str:
    header_line = (
        "| "
        + " | ".join(headers)
        + " |"
    )

    separator_line = (
        "|"
        + "|".join("---" for _ in headers)
        + "|"
    )

    body_lines = [
        "| "
        + " | ".join(row)
        + " |"
        for row in rows
    ]

    return "\n".join(
        [
            header_line,
            separator_line,
            *body_lines,
        ]
    )


def write_markdown_report(
    *,
    analyses: Sequence[MatrixAnalysis],
    summaries: Sequence[FamilySeedSummary],
    comparisons: Sequence[CrossSeedComparison],
    checkpoint_metadata: Mapping[str, Any],
) -> None:
    lines: list[str] = []

    lines.append("# PB0 Weight Spectrum Analysis")
    lines.append("")

    lines.append("## Scope")
    lines.append("")
    lines.append(
        "This analysis inspects frozen trained checkpoints and measures the "
        "spectral structure of the five repeated matrix-memory projection "
        "families before any architecture modification."
    )
    lines.append("")
    lines.append(
        "The analysis does not modify `language/models.py`, checkpoint files, "
        "parameter counts, recurrent-state dimensions, or compiled kernels."
    )
    lines.append("")

    lines.append("## Checkpoints")
    lines.append("")

    checkpoint_rows: list[list[str]] = []

    for seed_name in sorted(checkpoint_metadata.keys()):
        metadata = checkpoint_metadata[seed_name]

        checkpoint_rows.append(
            [
                seed_name,
                str(metadata["path"]),
                format_int(
                    int(metadata["parameter_count"])
                ),
                str(metadata["step"]),
            ]
        )

    lines.append(
        markdown_table(
            [
                "Seed",
                "Checkpoint",
                "Parameter count",
                "Step",
            ],
            checkpoint_rows,
        )
    )
    lines.append("")

    lines.append("## Target Families")
    lines.append("")

    for family in TARGET_FAMILIES:
        lines.append(f"- `{family}`")

    lines.append("")

    lines.append("## Candidate Ranks")
    lines.append("")
    lines.append(
        ", ".join(
            str(rank)
            for rank in CANDIDATE_RANKS
        )
    )
    lines.append("")

    lines.append("## Per-Family Seed Summary")
    lines.append("")

    summary_rows: list[list[str]] = []

    for summary in summaries:
        summary_rows.append(
            [
                summary.seed,
                summary.family,
                str(summary.matrices),
                format_float(
                    summary.mean_stable_rank,
                    3,
                ),
                format_float(
                    summary.median_stable_rank,
                    3,
                ),
                format_float(
                    summary.mean_effective_rank_entropy,
                    3,
                ),
            ]
        )

    lines.append(
        markdown_table(
            [
                "Seed",
                "Family",
                "Matrices",
                "Mean stable rank",
                "Median stable rank",
                "Mean entropy effective rank",
            ],
            summary_rows,
        )
    )
    lines.append("")

    lines.append("## Mean Retained Spectral Energy")
    lines.append("")

    energy_headers = [
        "Seed",
        "Family",
        *[
            f"Rank {rank}"
            for rank in CANDIDATE_RANKS
        ],
    ]

    energy_rows: list[list[str]] = []

    for summary in summaries:
        row = [
            summary.seed,
            summary.family,
        ]

        for rank in CANDIDATE_RANKS:
            row.append(
                format_percent(
                    summary.rank_energy_mean[
                        str(rank)
                    ],
                    4,
                )
            )

        energy_rows.append(row)

    lines.append(
        markdown_table(
            energy_headers,
            energy_rows,
        )
    )
    lines.append("")

    lines.append("## Worst-Layer Retained Spectral Energy")
    lines.append("")
    lines.append(
        "For each family and seed, this table shows the minimum retained "
        "energy across the 12 layers. This is important because a good average "
        "can hide a compression-sensitive layer."
    )
    lines.append("")

    minimum_energy_rows: list[list[str]] = []

    for summary in summaries:
        row = [
            summary.seed,
            summary.family,
        ]

        for rank in CANDIDATE_RANKS:
            row.append(
                format_percent(
                    summary.rank_energy_min[
                        str(rank)
                    ],
                    4,
                )
            )

        minimum_energy_rows.append(row)

    lines.append(
        markdown_table(
            energy_headers,
            minimum_energy_rows,
        )
    )
    lines.append("")

    lines.append("## Cross-Seed Consistency")
    lines.append("")

    comparison_rows: list[list[str]] = []

    for comparison in comparisons:
        stable_relative = (
            "N/A"
            if comparison.stable_rank_relative_difference_percent
            is None
            else (
                f"{comparison.stable_rank_relative_difference_percent:.3f}%"
            )
        )

        comparison_rows.append(
            [
                comparison.family,
                format_float(
                    comparison.stable_rank_seed1_mean,
                    3,
                ),
                format_float(
                    comparison.stable_rank_seed2_mean,
                    3,
                ),
                stable_relative,
                format_float(
                    comparison.effective_rank_absolute_difference,
                    3,
                ),
                comparison.consistency_label,
            ]
        )

    lines.append(
        markdown_table(
            [
                "Family",
                "Seed1 stable rank",
                "Seed2 stable rank",
                "Stable-rank difference",
                "Entropy-effective-rank difference",
                "Consistency",
            ],
            comparison_rows,
        )
    )
    lines.append("")

    lines.append("## Layer-Level Variation")
    lines.append("")
    lines.append(
        "Layer-level metrics are written to "
        "`PB0_WEIGHT_SPECTRUM_LAYERS.csv`."
    )
    lines.append("")

    for family in TARGET_FAMILIES:
        family_analyses = [
            analysis
            for analysis in analyses
            if analysis.family == family
        ]

        stable_ranks = [
            analysis.stable_rank
            for analysis in family_analyses
        ]

        lines.append(f"### `{family}`")
        lines.append("")
        lines.append(
            f"- Matrices analyzed: **{len(family_analyses)}**"
        )
        lines.append(
            "- Stable rank range across both seeds: "
            f"**{min(stable_ranks):.3f} → {max(stable_ranks):.3f}**"
        )
        lines.append("")

    lines.append("## Interpretation Rules")
    lines.append("")
    lines.append(
        "This analysis does not itself prove that a projection can be "
        "compressed without language-quality loss."
    )
    lines.append("")
    lines.append(
        "A candidate is stronger when all of the following are observed:"
    )
    lines.append("")
    lines.append(
        "1. High retained energy at the candidate rank."
    )
    lines.append(
        "2. Low reconstruction error."
    )
    lines.append(
        "3. No severely sensitive individual layer."
    )
    lines.append(
        "4. Similar results across Seed1 and Seed2."
    )
    lines.append("")
    lines.append(
        "The next experimental gate is PB0-B: replace one projection family "
        "at a time with dense truncated-SVD approximations and evaluate the "
        "frozen checkpoint without changing architecture topology."
    )
    lines.append("")

    OUTPUT_MARKDOWN.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# =============================================================================
# MAIN ANALYSIS
# =============================================================================


def count_parameter_elements(
    params: Any,
) -> int:
    total = 0

    def walk(value: Any) -> None:
        nonlocal total

        if isinstance(value, Mapping):
            for child in value.values():
                walk(child)
            return

        try:
            array = np.asarray(value)
        except Exception:
            return

        if array.ndim == 0:
            return

        if np.issubdtype(array.dtype, np.number):
            total += int(array.size)

    walk(params)

    return total


def analyze_seed(
    *,
    seed_name: str,
    checkpoint_path: Path,
) -> tuple[list[MatrixAnalysis], dict[str, Any]]:
    banner(
        f"LOADING {seed_name.upper()}"
    )

    print(f"Checkpoint : {checkpoint_path}")
    print(
        "Size       : "
        f"{checkpoint_path.stat().st_size / (1024 * 1024):.2f} MiB"
    )

    checkpoint_object = load_pickle_checkpoint(
        checkpoint_path
    )

    params = get_checkpoint_params(
        checkpoint_object,
        checkpoint_path,
    )

    parameter_count = count_parameter_elements(
        params
    )

    checkpoint_step: Any = None

    if isinstance(checkpoint_object, Mapping):
        checkpoint_step = checkpoint_object.get(
            "step",
            "unknown",
        )

    print()
    print("Checkpoint loaded successfully.")
    print(
        f"Parameter count : {format_int(parameter_count)}"
    )
    print(
        f"Checkpoint step : {checkpoint_step}"
    )

    analyses: list[MatrixAnalysis] = []

    for family in TARGET_FAMILIES:
        print()
        print(
            f"Analyzing family: {family}"
        )

        parameter_path, tensor = resolve_family_tensor(
            params,
            family,
        )

        print(
            f"  Path  : {parameter_path}"
        )
        print(
            f"  Shape : {tuple(tensor.shape)}"
        )

        for layer in range(tensor.shape[0]):
            matrix = tensor[layer]

            analysis = analyze_single_matrix(
                seed=seed_name,
                family=family,
                layer=layer,
                parameter_path=parameter_path,
                matrix=matrix,
            )

            analyses.append(analysis)

            print(
                f"  Layer {layer:02d} | "
                f"stable rank {analysis.stable_rank:8.3f} | "
                f"entropy rank "
                f"{analysis.effective_rank_entropy:8.3f}"
            )

    metadata = {
        "path": str(checkpoint_path),
        "size_bytes": int(
            checkpoint_path.stat().st_size
        ),
        "parameter_count": parameter_count,
        "step": checkpoint_step,
    }

    return analyses, metadata


def main() -> None:
    banner(
        "MODUS_X PB0-A WEIGHT SPECTRUM ANALYSIS"
    )

    print(
        f"Project root : {PROJECT_ROOT}"
    )
    print(
        f"Output dir   : {OUTPUT_DIR}"
    )
    print()

    print("Target families:")
    for family in TARGET_FAMILIES:
        print(f"  - {family}")

    print()
    print(
        "Candidate ranks: "
        + ", ".join(
            str(rank)
            for rank in CANDIDATE_RANKS
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_analyses: list[MatrixAnalysis] = []
    checkpoint_metadata: dict[str, Any] = {}

    for seed_name, seed_directory in SEED_DIRECTORIES.items():
        checkpoint_path = validate_checkpoint_path(
            seed_name,
            seed_directory,
        )

        seed_analyses, seed_metadata = analyze_seed(
            seed_name=seed_name,
            checkpoint_path=checkpoint_path,
        )

        all_analyses.extend(
            seed_analyses
        )

        checkpoint_metadata[seed_name] = (
            seed_metadata
        )

    banner(
        "BUILDING FAMILY SUMMARIES"
    )

    summaries: list[FamilySeedSummary] = []

    for seed_name in SEED_DIRECTORIES:
        for family in TARGET_FAMILIES:
            summary = summarize_family_seed(
                all_analyses,
                seed_name,
                family,
            )

            summaries.append(summary)

            print(
                f"{seed_name.upper():5s} "
                f"{family:10s} "
                f"stable rank mean "
                f"{summary.mean_stable_rank:8.3f}"
            )

    comparisons: list[CrossSeedComparison] = []

    banner(
        "CROSS-SEED COMPARISON"
    )

    for family in TARGET_FAMILIES:
        comparison = compare_family_between_seeds(
            summaries,
            family,
        )

        comparisons.append(comparison)

        print(
            f"{family:10s} | "
            f"Seed1 stable rank "
            f"{comparison.stable_rank_seed1_mean:8.3f} | "
            f"Seed2 stable rank "
            f"{comparison.stable_rank_seed2_mean:8.3f} | "
            f"Consistency: "
            f"{comparison.consistency_label}"
        )

    write_json_report(
        analyses=all_analyses,
        summaries=summaries,
        comparisons=comparisons,
        checkpoint_metadata=checkpoint_metadata,
    )

    write_layer_csv(
        all_analyses
    )

    write_summary_csv(
        summaries
    )

    write_seed_comparison_csv(
        comparisons
    )

    write_markdown_report(
        analyses=all_analyses,
        summaries=summaries,
        comparisons=comparisons,
        checkpoint_metadata=checkpoint_metadata,
    )

    banner(
        "PB0-A ANALYSIS COMPLETE"
    )

    print()
    print(
        f"Matrices analyzed : {format_int(len(all_analyses))}"
    )
    print(
        f"Families analyzed : {len(TARGET_FAMILIES)}"
    )
    print(
        f"Seeds analyzed    : {len(SEED_DIRECTORIES)}"
    )
    print()

    print("Output files:")
    print(f"  {OUTPUT_JSON}")
    print(f"  {OUTPUT_MARKDOWN}")
    print(f"  {OUTPUT_LAYER_CSV}")
    print(f"  {OUTPUT_SUMMARY_CSV}")
    print(f"  {OUTPUT_SEED_COMPARISON_CSV}")

    print()
    print(
        "NEXT GATE: inspect PB0_WEIGHT_SPECTRUM_ANALYSIS.md and identify "
        "projection families and ranks that are consistently compressible "
        "across both trained seeds before implementing PB0-B dense truncated-"
        "SVD sensitivity testing."
    )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        banner(
            "PB0-A ANALYSIS INTERRUPTED"
        )
        print(
            "The analysis was interrupted by the user."
        )
        sys.exit(130)

    except Exception as exc:
        banner(
            "PB0-A ANALYSIS FAILED"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )
        print()
        traceback.print_exc()

        sys.exit(1)