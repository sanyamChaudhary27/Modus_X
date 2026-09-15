from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np


# =============================================================================
# PROJECT PATHS
# =============================================================================


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
LANGUAGE_DIR = PROJECT_ROOT / "language"
OUTPUT_DIR = SCRIPT_PATH.parent / "outputs"

if not LANGUAGE_DIR.exists():
    raise RuntimeError(
        "Unable to locate the canonical language directory:\n"
        f"{LANGUAGE_DIR}"
    )

if str(LANGUAGE_DIR) not in sys.path:
    sys.path.insert(0, str(LANGUAGE_DIR))


try:
    from models import ModelConfig, count_params, make_model
except Exception as exc:
    raise RuntimeError(
        "Unable to import canonical Modus_X model definitions from:\n"
        f"{LANGUAGE_DIR}\n\n"
        f"{type(exc).__name__}: {exc}"
    ) from exc


# =============================================================================
# BASELINE CONFIGURATION
# =============================================================================


BASELINE_MODEL_NAME = (
    "Modus_X_MemoryFeedbackArchive_DeepSupervision"
)

BASELINE_PARAMETER_COUNT = 47_437_768

BASELINE_VOCAB_SIZE = 256
BASELINE_EMBED_DIM = 512
BASELINE_HIDDEN_DIM = 1536
BASELINE_STATE_DIM = 512
BASELINE_N_LAYERS = 12
BASELINE_ROUTER_HIDDEN = 32

BASELINE_INPUT_SEQ_LEN = 512
BASELINE_LOSS_TAIL = 512

BASELINE_AUXILIARY_LAYERS: tuple[int, ...] = (
    6,
)

BASELINE_FUTURE_TARGET_COUNT = 1
BASELINE_DROPOUT_RATE = 0.0

CHECKPOINT_FILENAME = "checkpoint.pkl"

DEFAULT_SEED_DIRECTORIES: dict[str, Path] = {
    "seed1": Path(r"E:\seed1"),
    "seed2": Path(r"E:\seed2"),
}

PHASE_ONE_FAMILIES: tuple[str, ...] = (
    "m_wq",
    "m_wk",
    "m_w_out",
)

PHASE_ONE_RANKS: tuple[int, ...] = (
    256,
    128,
    64,
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "PB0_DENSE_SVD_SENSITIVITY.json"
)

OUTPUT_MARKDOWN = (
    OUTPUT_DIR
    / "PB0_DENSE_SVD_SENSITIVITY.md"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "PB0_DENSE_SVD_SENSITIVITY.csv"
)

OUTPUT_SUMMARY_CSV = (
    OUTPUT_DIR
    / "PB0_DENSE_SVD_SENSITIVITY_SUMMARY.csv"
)

OUTPUT_PARAMETER_INVENTORY_CSV = (
    OUTPUT_DIR
    / "PB0_PARAMETER_INVENTORY.csv"
)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass(frozen=True)
class ParameterLeaf:
    path: str
    shape: tuple[int, ...]
    dtype: str
    elements: int
    bytes: int


@dataclass(frozen=True)
class ParameterInventoryRow:
    category: str
    path: str
    shape: tuple[int, ...]
    dtype: str
    elements: int
    bytes: int
    percent_of_total: float


@dataclass(frozen=True)
class ExperimentResult:
    seed: str
    family: str
    rank: int

    checkpoint_path: str
    checkpoint_step: int | str

    baseline_bpc: float
    compressed_bpc: float
    bpc_delta: float

    retained_energy_mean: float
    retained_energy_min: float
    relative_frobenius_error_mean: float
    relative_frobenius_error_max: float

    elapsed_seconds: float


@dataclass(frozen=True)
class ExperimentSummary:
    family: str
    rank: int

    experiments: int

    mean_baseline_bpc: float
    mean_compressed_bpc: float
    mean_bpc_delta: float

    max_bpc_delta: float
    min_bpc_delta: float

    mean_retained_energy: float
    min_retained_energy: float

    mean_relative_frobenius_error: float
    max_relative_frobenius_error: float

    recommendation: str


@dataclass(frozen=True)
class SeedCheckpoint:
    seed: str
    path: Path
    step: int | str
    params: Mapping[str, Any]


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


def format_float(
    value: float,
    digits: int = 6,
) -> str:
    if math.isnan(value):
        return "NaN"

    if math.isinf(value):
        return "Inf"

    return f"{value:.{digits}f}"


def format_percent(
    value: float,
    digits: int = 4,
) -> str:
    return f"{value * 100.0:.{digits}f}%"


# =============================================================================
# ARGUMENT PARSING
# =============================================================================


def parse_csv_strings(
    value: str,
) -> tuple[str, ...]:
    parsed = tuple(
        item.strip()
        for item in value.split(",")
        if item.strip()
    )

    if not parsed:
        fail(
            "Expected at least one comma-separated value."
        )

    return parsed


def parse_csv_integers(
    value: str,
) -> tuple[int, ...]:
    parsed_values: list[int] = []

    for item in parse_csv_strings(value):
        try:
            parsed_value = int(item)
        except ValueError as exc:
            fail(
                f"Invalid integer value: {item}"
            )
            raise exc

        if parsed_value <= 0:
            fail(
                f"Rank values must be positive: {parsed_value}"
            )

        parsed_values.append(parsed_value)

    return tuple(parsed_values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "PB0-B frozen dense truncated-SVD sensitivity analysis for "
            "Modus_X parameter allocation research."
        )
    )

    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help=(
            "Path to the canonical enwik8 byte dataset used by the "
            "Modus_X training/evaluation path."
        ),
    )

    parser.add_argument(
        "--seed1-dir",
        type=Path,
        default=DEFAULT_SEED_DIRECTORIES["seed1"],
    )

    parser.add_argument(
        "--seed2-dir",
        type=Path,
        default=DEFAULT_SEED_DIRECTORIES["seed2"],
    )

    parser.add_argument(
        "--families",
        default=",".join(PHASE_ONE_FAMILIES),
        help=(
            "Comma-separated projection families. "
            f"Default: {','.join(PHASE_ONE_FAMILIES)}"
        ),
    )

    parser.add_argument(
        "--ranks",
        default=",".join(
            str(rank)
            for rank in PHASE_ONE_RANKS
        ),
        help=(
            "Comma-separated truncated-SVD ranks. "
            f"Default: {','.join(str(rank) for rank in PHASE_ONE_RANKS)}"
        ),
    )

    parser.add_argument(
        "--eval-chunks",
        type=int,
        default=128,
        help=(
            "Canonical evaluation chunk count."
        ),
    )

    parser.add_argument(
        "--eval-batch",
        type=int,
        default=8,
        help=(
            "Evaluation batch size."
        ),
    )

    parser.add_argument(
        "--run-test",
        action="store_true",
        help=(
            "Evaluate the canonical test split in addition to validation. "
            "The default PB0-B gate evaluates validation only."
        ),
    )

    return parser.parse_args()


# =============================================================================
# CHECKPOINT LOADING
# =============================================================================


def validate_regular_file(
    path: Path,
    description: str,
) -> None:
    if not path.exists():
        fail(
            f"{description} does not exist:\n"
            f"{path}"
        )

    if not path.is_file():
        fail(
            f"{description} is not a regular file:\n"
            f"{path}"
        )


def validate_directory(
    path: Path,
    description: str,
) -> None:
    if not path.exists():
        fail(
            f"{description} does not exist:\n"
            f"{path}"
        )

    if not path.is_dir():
        fail(
            f"{description} is not a directory:\n"
            f"{path}"
        )


def load_checkpoint(
    seed: str,
    seed_directory: Path,
) -> SeedCheckpoint:
    validate_directory(
        seed_directory,
        f"Seed directory for {seed}",
    )

    checkpoint_path = (
        seed_directory
        / CHECKPOINT_FILENAME
    )

    validate_regular_file(
        checkpoint_path,
        f"Checkpoint for {seed}",
    )

    try:
        with checkpoint_path.open("rb") as file:
            checkpoint = pickle.load(file)
    except Exception as exc:
        fail(
            f"Unable to load checkpoint:\n"
            f"{checkpoint_path}\n\n"
            f"{type(exc).__name__}: {exc}"
        )

    if not isinstance(checkpoint, Mapping):
        fail(
            f"Checkpoint root is not a mapping:\n"
            f"{checkpoint_path}\n"
            f"Actual type: {type(checkpoint).__name__}"
        )

    if "params" not in checkpoint:
        fail(
            f"Checkpoint is missing the required 'params' field:\n"
            f"{checkpoint_path}"
        )

    params = checkpoint["params"]

    if not isinstance(params, Mapping):
        fail(
            f"Checkpoint params are not a mapping:\n"
            f"{checkpoint_path}\n"
            f"Actual type: {type(params).__name__}"
        )

    step = checkpoint.get(
        "step",
        "unknown",
    )

    if isinstance(step, np.generic):
        step = step.item()

    return SeedCheckpoint(
        seed=seed,
        path=checkpoint_path,
        step=step,
        params=params,
    )


# =============================================================================
# PARAMETER TREE HELPERS
# =============================================================================


def recursively_collect_parameter_leaves(
    tree: Any,
) -> list[ParameterLeaf]:
    leaves: list[ParameterLeaf] = []

    def walk(
        value: Any,
        prefix: str,
    ) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_text = str(key)

                child_path = (
                    key_text
                    if prefix == ""
                    else (
                        f"{prefix}.{key_text}"
                    )
                )

                walk(
                    child,
                    child_path,
                )

            return

        try:
            array = np.asarray(value)
        except Exception:
            return

        if array.ndim == 0:
            return

        if not np.issubdtype(
            array.dtype,
            np.number,
        ):
            return

        leaves.append(
            ParameterLeaf(
                path=prefix,
                shape=tuple(
                    int(dimension)
                    for dimension in array.shape
                ),
                dtype=str(array.dtype),
                elements=int(array.size),
                bytes=int(array.nbytes),
            )
        )

    walk(
        tree,
        "",
    )

    return leaves


def count_parameter_elements(
    params: Mapping[str, Any],
) -> int:
    leaves = recursively_collect_parameter_leaves(
        params
    )

    return int(
        sum(
            leaf.elements
            for leaf in leaves
        )
    )


def classify_parameter_path(
    path: str,
) -> str:
    path_lower = path.lower()

    if (
        path_lower == "embed"
        or path_lower.startswith(
            "embed."
        )
    ):
        return "embeddings"

    if (
        "m_wq" in path_lower
        or "m_wk" in path_lower
        or "m_wv" in path_lower
        or "m_w_read" in path_lower
        or "m_w_out" in path_lower
        or "m_b_" in path_lower
        or "matrix" in path_lower
        or "archive" in path_lower
    ):
        return "matrix_memory"

    if (
        "s_" in path_lower
        or "mamba" in path_lower
        or "vector" in path_lower
        or "state" in path_lower
    ):
        return "vector_mamba"

    if (
        "router" in path_lower
        or "mix" in path_lower
    ):
        return "router"

    if (
        "feedback" in path_lower
        or "fb_" in path_lower
        or "bridge" in path_lower
    ):
        return "feedback_bridge"

    if (
        "head" in path_lower
        or "out" in path_lower
        or "logit" in path_lower
    ):
        return "output_head"

    return "other"


def build_parameter_inventory(
    params: Mapping[str, Any],
    total_parameter_count: int,
) -> list[ParameterInventoryRow]:
    if total_parameter_count <= 0:
        fail(
            "Total parameter count must be positive."
        )

    leaves = recursively_collect_parameter_leaves(
        params
    )

    inventory: list[
        ParameterInventoryRow
    ] = []

    for leaf in leaves:
        category = classify_parameter_path(
            leaf.path
        )

        inventory.append(
            ParameterInventoryRow(
                category=category,
                path=leaf.path,
                shape=leaf.shape,
                dtype=leaf.dtype,
                elements=leaf.elements,
                bytes=leaf.bytes,
                percent_of_total=(
                    leaf.elements
                    / total_parameter_count
                ),
            )
        )

    return inventory


# =============================================================================
# CANONICAL MODEL CONSTRUCTION
# =============================================================================


def build_canonical_model() -> Any:
    cfg = ModelConfig(
        vocab_size=BASELINE_VOCAB_SIZE,
        embed_dim=BASELINE_EMBED_DIM,
        hidden_dim=BASELINE_HIDDEN_DIM,
        ax_res=BASELINE_STATE_DIM,
        n_layers=BASELINE_N_LAYERS,
        n_heads_attn=8,
        seq_len=BASELINE_INPUT_SEQ_LEN,
        mamba_state_dim=BASELINE_STATE_DIM,
        vector_router=False,
        router_hidden=BASELINE_ROUTER_HIDDEN,
    )

    try:
        initialized_params, fwd_fn = make_model(
            BASELINE_MODEL_NAME,
            jax.random.key(1),
            cfg,
            auxiliary_layers=(
                BASELINE_AUXILIARY_LAYERS
            ),
            future_target_count=(
                BASELINE_FUTURE_TARGET_COUNT
            ),
            dropout_rate=(
                BASELINE_DROPOUT_RATE
            ),
        )
    except Exception as exc:
        fail(
            "Unable to construct the canonical Modus_X model:\n\n"
            f"{type(exc).__name__}: {exc}"
        )

    initialized_count = int(
        count_params(
            initialized_params
        )
    )

    if (
        initialized_count
        != BASELINE_PARAMETER_COUNT
    ):
        fail(
            "Canonical model construction produced an unexpected parameter "
            "count.\n\n"
            f"Expected: {format_int(BASELINE_PARAMETER_COUNT)}\n"
            f"Actual  : {format_int(initialized_count)}\n\n"
            "PB0-B must not evaluate a differently shaped architecture."
        )

    return fwd_fn


# =============================================================================
# CANONICAL DATA LOADING
# =============================================================================


def load_canonical_splits(
    data_path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    validate_regular_file(
        data_path,
        "Canonical dataset",
    )

    raw: np.memmap | None = None

    try:
        raw = np.memmap(
            data_path,
            dtype=np.uint8,
            mode="r",
        )
    except Exception as exc:
        fail(
            f"Unable to memory-map dataset:\n"
            f"{data_path}\n\n"
            f"{type(exc).__name__}: {exc}"
        )

    required_length = 100_000_000

    if len(raw) < required_length:
        fail(
            "Dataset is shorter than the canonical enwik8 split length.\n\n"
            f"Required: {format_int(required_length)} bytes\n"
            f"Actual  : {format_int(len(raw))} bytes"
        )

    train = raw[
        :90_000_000
    ]

    valid = raw[
        90_000_000:95_000_000
    ]

    test = raw[
        95_000_000:100_000_000
    ]

    return (
        train,
        valid,
        test,
    )


def batch_at(
    data: np.ndarray,
    starts: np.ndarray,
    seq_len: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    if seq_len <= 0:
        fail(
            "Sequence length must be positive."
        )

    if starts.ndim != 1:
        fail(
            "Batch starts must be rank-1."
        )

    offsets = np.arange(
        seq_len + 1
    )

    chunks = data[
        starts[:, None]
        + offsets[None, :]
    ]

    x = chunks[
        :,
        :-1,
    ].astype(
        np.int32
    )

    y = chunks[
        :,
        1:,
    ].astype(
        np.int32
    )

    return (
        x,
        y,
    )


# =============================================================================
# CANONICAL EVALUATION LOSS
# =============================================================================


def extract_primary_logits(
    outputs: Any,
) -> jax.Array:
    if isinstance(
        outputs,
        tuple,
    ):
        if len(outputs) == 0:
            fail(
                "Model forward function returned an empty tuple."
            )

        logits = next(iter(outputs))
    else:
        logits = outputs

    return logits


def evaluation_loss(
    params: Mapping[str, Any],
    fwd_fn: Any,
    x: np.ndarray,
    y: np.ndarray,
    loss_tail: int,
) -> jax.Array:
    
    if loss_tail <= 0:
        fail(
            "Loss tail must be positive."
        )

    outputs = [
        fwd_fn(
            params,
            sequence,
        )
        for sequence in x
    ]

    logits = jnp.stack(
        [
            extract_primary_logits(output)
            for output in outputs
        ],
        axis=0,
    )


    if (
        loss_tail
        > logits.shape[1]
    ):
        fail(
            "Loss tail exceeds evaluated sequence length."
        )

    logits = logits[
        :,
        -loss_tail:,
    ]

    targets = y[
        :,
        -loss_tail:,
    ]

    log_probabilities = (
        jax.nn.log_softmax(
            logits,
            axis=-1,
        )
    )

    negative_log_likelihood = (
        -jnp.take_along_axis(
            log_probabilities,
            targets[..., None],
            axis=-1,
        )[..., 0]
    )

    return jnp.mean(
        negative_log_likelihood
    )


def evaluate_bpc(
    params: Mapping[str, Any],
    fwd_fn: Any,
    data: np.ndarray,
    *,
    seq_len: int,
    chunks: int,
    batch_size: int,
    loss_tail: int,
) -> float:
    if chunks <= 0:
        fail(
            "Evaluation chunk count must be positive."
        )

    if batch_size <= 0:
        fail(
            "Evaluation batch size must be positive."
        )

    max_start = (
        len(data)
        - seq_len
        - 1
    )

    if max_start <= 0:
        fail(
            "Evaluation data is shorter than one sequence."
        )

    starts = np.linspace(
        0,
        max_start,
        chunks,
        dtype=np.int64,
    )

    # IMPORTANT:
    # Do not JIT this function.
    #
    # The canonical Modus_X forward path performs a NumPy-backed
    # embedding lookup. JAX-tracing batch_x would therefore produce
    # TracerArrayConversionError.
    #
    # Keep evaluation in ordinary Python/NumPy and let the model
    # forward function execute one sequence at a time.
    def eval_batch(
        batch_x: np.ndarray,
        batch_y: np.ndarray,
    ) -> jax.Array:
        return evaluation_loss(
            params,
            fwd_fn,
            batch_x,
            batch_y,
            loss_tail,
        )

    losses: list[float] = []

    for offset in range(
        0,
        chunks,
        batch_size,
    ):
        selected = starts[
            offset:
            offset + batch_size
        ]

        if (
            len(selected)
            < batch_size
        ):
            selected = np.pad(
                selected,
                (
                    0,
                    batch_size
                    - len(selected),
                ),
                mode="edge",
            )

        x, y = batch_at(
            data,
            selected,
            seq_len,
        )

        # Keep x/y as NumPy arrays.
        # Do NOT call jnp.asarray() here.
        value = eval_batch(
            x,
            y,
        )

        losses.append(
            float(value)
        )
        print(
            f"  Evaluation progress: "
            f"{min(offset + batch_size, chunks)}/{chunks} windows",
            flush=True,
        )

    if not losses:
        fail(
            "Canonical evaluation produced no batches."
        )

    mean_loss = float(
        np.mean(losses)
    )

    if not math.isfinite(
        mean_loss
    ):
        fail(
            f"Evaluation produced non-finite mean loss: {mean_loss}"
        )

    return float(
        mean_loss
        / math.log(2.0)
    )


# =============================================================================
# FAMILY VALIDATION
# =============================================================================


def get_layers_mapping(
    params: Mapping[str, Any],
) -> Mapping[str, Any]:
    if "layers" not in params:
        fail(
            "Checkpoint parameter tree is missing top-level 'layers'."
        )

    layers = params["layers"]

    if not isinstance(
        layers,
        Mapping,
    ):
        fail(
            "Checkpoint 'layers' value is not a mapping."
        )

    return layers


def get_family_tensor(
    params: Mapping[str, Any],
    family: str,
) -> np.ndarray:
    layers = get_layers_mapping(
        params
    )

    if family not in layers:
        fail(
            f"Projection family '{family}' was not found in params['layers'].\n"
            f"Available keys:\n"
            + "\n".join(
                f"  - {key}"
                for key in sorted(
                    str(key)
                    for key in layers.keys()
                )
            )
        )

    tensor = np.asarray(
        layers[family]
    )

    expected_shape = (
        BASELINE_N_LAYERS,
        BASELINE_EMBED_DIM,
        BASELINE_EMBED_DIM,
    )

    if (
        tuple(tensor.shape)
        != expected_shape
    ):
        fail(
            f"Unexpected tensor shape for family '{family}'.\n"
            f"Expected: {expected_shape}\n"
            f"Actual  : {tuple(tensor.shape)}"
        )

    if not np.issubdtype(
        tensor.dtype,
        np.number,
    ):
        fail(
            f"Family '{family}' is not numeric:\n"
            f"Dtype: {tensor.dtype}"
        )

    if not np.all(
        np.isfinite(tensor)
    ):
        fail(
            f"Family '{family}' contains NaN or Inf."
        )

    return tensor


# =============================================================================
# DENSE TRUNCATED SVD
# =============================================================================


def reconstruct_matrix_with_rank(
    matrix: np.ndarray,
    rank: int,
) -> tuple[
    np.ndarray,
    float,
    float,
]:
    if matrix.ndim != 2:
        fail(
            f"Expected rank-2 matrix, received shape {tuple(matrix.shape)}"
        )

    rows, columns = matrix.shape

    maximum_rank = min(
        rows,
        columns,
    )

    if rank <= 0:
        fail(
            f"Rank must be positive: {rank}"
        )

    if rank > maximum_rank:
        fail(
            f"Requested rank {rank} exceeds matrix maximum rank "
            f"{maximum_rank}."
        )

    matrix64 = np.asarray(
        matrix,
        dtype=np.float64,
    )

    try:
        u, singular_values, vh = (
            np.linalg.svd(
                matrix64,
                full_matrices=False,
            )
        )
    except np.linalg.LinAlgError as exc:
        fail(
            f"SVD failed for matrix shape {tuple(matrix.shape)}:\n"
            f"{type(exc).__name__}: {exc}"
        )

    retained_singular_values = (
        singular_values[:rank]
    )

    reconstructed = (
        u[:, :rank]
        * retained_singular_values[None, :]
    ) @ vh[:rank, :]

    total_energy = float(
        np.sum(
            singular_values ** 2
        )
    )

    retained_energy_value = float(
        np.sum(
            retained_singular_values ** 2
        )
    )

    if total_energy <= 0.0:
        retained_energy_fraction = 0.0
    else:
        retained_energy_fraction = (
            retained_energy_value
            / total_energy
        )

    retained_energy_fraction = min(
        max(
            retained_energy_fraction,
            0.0,
        ),
        1.0,
    )

    relative_frobenius_error = float(
        math.sqrt(
            max(
                0.0,
                1.0
                - retained_energy_fraction,
            )
        )
    )

    return (
        reconstructed,
        retained_energy_fraction,
        relative_frobenius_error,
    )


def reconstruct_family_with_rank(
    tensor: np.ndarray,
    rank: int,
) -> tuple[
    np.ndarray,
    float,
    float,
    float,
    float,
]:
    if tensor.ndim != 3:
        fail(
            f"Expected rank-3 family tensor, received {tuple(tensor.shape)}"
        )

    reconstructed_layers: list[
        np.ndarray
    ] = []

    retained_energies: list[
        float
    ] = []

    reconstruction_errors: list[
        float
    ] = []

    for layer_index in range(
        tensor.shape[0]
    ):
        (
            reconstructed,
            retained_energy,
            reconstruction_error,
        ) = reconstruct_matrix_with_rank(
            tensor[layer_index],
            rank,
        )

        reconstructed_layers.append(
            reconstructed.astype(
                tensor.dtype,
                copy=False,
            )
        )

        retained_energies.append(
            retained_energy
        )

        reconstruction_errors.append(
            reconstruction_error
        )

    reconstructed_tensor = np.stack(
        reconstructed_layers,
        axis=0,
    )

    return (
        reconstructed_tensor,
        float(
            np.mean(
                retained_energies
            )
        ),
        float(
            np.min(
                retained_energies
            )
        ),
        float(
            np.mean(
                reconstruction_errors
            )
        ),
        float(
            np.max(
                reconstruction_errors
            )
        ),
    )


def replace_family_tensor(
    params: Mapping[str, Any],
    family: str,
    replacement: np.ndarray,
) -> Mapping[str, Any]:
    layers = get_layers_mapping(
        params
    )

    if (
        family
        not in layers
    ):
        fail(
            f"Cannot replace missing family '{family}'."
        )

    original = np.asarray(
        layers[family]
    )

    if (
        tuple(replacement.shape)
        != tuple(original.shape)
    ):
        fail(
            f"Replacement shape mismatch for '{family}'.\n"
            f"Original   : {tuple(original.shape)}\n"
            f"Replacement: {tuple(replacement.shape)}"
        )

    replacement_array = jnp.asarray(
        replacement,
        dtype=original.dtype,
    )

    new_layers = dict(
        layers
    )

    new_layers[family] = (
        replacement_array
    )

    new_params = dict(
        params
    )

    new_params["layers"] = (
        new_layers
    )

    return new_params


# =============================================================================
# RESULT INTERPRETATION
# =============================================================================


def classify_recommendation(
    mean_bpc_delta: float,
    max_bpc_delta: float,
) -> str:
    if (
        max_bpc_delta
        <= 0.001
    ):
        return (
            "strong_candidate"
        )

    if (
        mean_bpc_delta
        <= 0.003
        and max_bpc_delta
        <= 0.005
    ):
        return (
            "candidate"
        )

    if (
        mean_bpc_delta
        <= 0.010
        and max_bpc_delta
        <= 0.015
    ):
        return (
            "sensitive"
        )

    return (
        "reject_at_this_rank"
    )


def summarize_results(
    results: Sequence[
        ExperimentResult
    ],
) -> list[
    ExperimentSummary
]:
    grouped: dict[
        tuple[str, int],
        list[ExperimentResult],
    ] = {}

    for result in results:
        key = (
            result.family,
            result.rank,
        )

        grouped.setdefault(
            key,
            [],
        ).append(
            result
        )

    summaries: list[
        ExperimentSummary
    ] = []

    for (
        family,
        rank,
    ), group in sorted(
        grouped.items()
    ):
        baseline_values = np.asarray(
            [
                result.baseline_bpc
                for result in group
            ],
            dtype=np.float64,
        )

        compressed_values = np.asarray(
            [
                result.compressed_bpc
                for result in group
            ],
            dtype=np.float64,
        )

        deltas = np.asarray(
            [
                result.bpc_delta
                for result in group
            ],
            dtype=np.float64,
        )

        retained_energies = np.asarray(
            [
                result.retained_energy_mean
                for result in group
            ],
            dtype=np.float64,
        )

        minimum_energies = np.asarray(
            [
                result.retained_energy_min
                for result in group
            ],
            dtype=np.float64,
        )

        reconstruction_errors = np.asarray(
            [
                result.relative_frobenius_error_mean
                for result in group
            ],
            dtype=np.float64,
        )

        maximum_errors = np.asarray(
            [
                result.relative_frobenius_error_max
                for result in group
            ],
            dtype=np.float64,
        )

        mean_delta = float(
            np.mean(
                deltas
            )
        )

        max_delta = float(
            np.max(
                deltas
            )
        )

        summaries.append(
            ExperimentSummary(
                family=family,
                rank=rank,
                experiments=len(group),
                mean_baseline_bpc=float(
                    np.mean(
                        baseline_values
                    )
                ),
                mean_compressed_bpc=float(
                    np.mean(
                        compressed_values
                    )
                ),
                mean_bpc_delta=mean_delta,
                max_bpc_delta=max_delta,
                min_bpc_delta=float(
                    np.min(
                        deltas
                    )
                ),
                mean_retained_energy=float(
                    np.mean(
                        retained_energies
                    )
                ),
                min_retained_energy=float(
                    np.min(
                        minimum_energies
                    )
                ),
                mean_relative_frobenius_error=float(
                    np.mean(
                        reconstruction_errors
                    )
                ),
                max_relative_frobenius_error=float(
                    np.max(
                        maximum_errors
                    )
                ),
                recommendation=(
                    classify_recommendation(
                        mean_delta,
                        max_delta,
                    )
                ),
            )
        )

    return summaries


# =============================================================================
# CSV SERIALIZATION
# =============================================================================


def write_parameter_inventory_csv(
    inventory: Sequence[
        ParameterInventoryRow
    ],
) -> None:
    if not inventory:
        fail(
            "Parameter inventory is empty."
        )

    fieldnames = [
        "category",
        "path",
        "shape",
        "dtype",
        "elements",
        "bytes",
        "percent_of_total",
    ]

    with OUTPUT_PARAMETER_INVENTORY_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="raise",
        )

        writer.writeheader()

        for row in inventory:
            writer.writerow(
                {
                    "category": row.category,
                    "path": row.path,
                    "shape": "x".join(
                        str(dimension)
                        for dimension in row.shape
                    ),
                    "dtype": row.dtype,
                    "elements": row.elements,
                    "bytes": row.bytes,
                    "percent_of_total": (
                        row.percent_of_total
                    ),
                }
            )


def write_results_csv(
    results: Sequence[
        ExperimentResult
    ],
) -> None:
    if not results:
        fail(
            "No PB0-B results available."
        )

    rows = [
        asdict(result)
        for result in results
    ]

    fieldnames = list(
        rows[0].keys()
    )

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="raise",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                row
            )


def write_summary_csv(
    summaries: Sequence[
        ExperimentSummary
    ],
) -> None:
    if not summaries:
        fail(
            "No PB0-B summaries available."
        )

    rows = [
        asdict(summary)
        for summary in summaries
    ]

    fieldnames = list(
        rows[0].keys()
    )

    with OUTPUT_SUMMARY_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="raise",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                row
            )


# =============================================================================
# JSON SERIALIZATION
# =============================================================================


def make_json_safe(
    value: Any,
) -> Any:
    if isinstance(
        value,
        Path,
    ):
        return str(value)

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if isinstance(
        value,
        tuple,
    ):
        return [
            make_json_safe(
                item
            )
            for item in value
        ]

    if isinstance(
        value,
        list,
    ):
        return [
            make_json_safe(
                item
            )
            for item in value
        ]

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): make_json_safe(
                item
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        float,
    ):
        if (
            math.isnan(value)
            or math.isinf(value)
        ):
            return None

        return value

    return value


def write_json_report(
    *,
    args: argparse.Namespace,
    results: Sequence[
        ExperimentResult
    ],
    summaries: Sequence[
        ExperimentSummary
    ],
    inventory: Sequence[
        ParameterInventoryRow
    ],
    baseline_parameter_count: int,
) -> None:
    payload = {
        "analysis": {
            "name": (
                "PB0-B Dense Truncated-SVD Sensitivity"
            ),
            "purpose": (
                "Measure functional sensitivity of frozen "
                "Modus_X checkpoints to low-rank dense "
                "approximations before architecture-level "
                "parameter reallocation."
            ),
            "parameter_budget": (
                BASELINE_PARAMETER_COUNT
            ),
            "model": (
                BASELINE_MODEL_NAME
            ),
            "families": list(
                parse_csv_strings(
                    args.families
                )
            ),
            "ranks": list(
                parse_csv_integers(
                    args.ranks
                )
            ),
        },
        "baseline_parameter_count": (
            baseline_parameter_count
        ),
        "evaluation": {
            "data_path": str(
                args.data_path
            ),
            "sequence_length": (
                BASELINE_INPUT_SEQ_LEN
            ),
            "loss_tail": (
                BASELINE_LOSS_TAIL
            ),
            "eval_chunks": (
                args.eval_chunks
            ),
            "eval_batch": (
                args.eval_batch
            ),
        },
        "parameter_inventory": [
            asdict(row)
            for row in inventory
        ],
        "experiments": [
            asdict(result)
            for result in results
        ],
        "summaries": [
            asdict(summary)
            for summary in summaries
        ],
    }

    OUTPUT_JSON.write_text(
        json.dumps(
            make_json_safe(
                payload
            ),
            indent=2,
        ),
        encoding="utf-8",
    )


# =============================================================================
# MARKDOWN REPORT
# =============================================================================


def markdown_table(
    headers: Sequence[str],
    rows: Sequence[
        Sequence[str]
    ],
) -> str:
    header_line = (
        "| "
        + " | ".join(headers)
        + " |"
    )

    separator_line = (
        "|"
        + "|".join(
            "---"
            for _ in headers
        )
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
    results: Sequence[
        ExperimentResult
    ],
    summaries: Sequence[
        ExperimentSummary
    ],
    inventory: Sequence[
        ParameterInventoryRow
    ],
) -> None:
    lines: list[str] = []

    lines.append(
        "# PB0-B Dense Truncated-SVD Sensitivity"
    )

    lines.append("")

    lines.append(
        "## Purpose"
    )

    lines.append("")

    lines.append(
        "PB0-B measures whether the spectral redundancy observed in PB0-A "
        "is functionally removable. Each experiment replaces one complete "
        "projection family with rank-k truncated-SVD reconstructions while "
        "preserving the original dense tensor shape."
    )

    lines.append("")

    lines.append(
        "No architecture topology, tensor shape, recurrent state dimension, "
        "or checkpoint parameter count is changed."
    )

    lines.append("")

    lines.append(
        "## Parameter Inventory"
    )

    lines.append("")

    category_totals: dict[
        str,
        int,
    ] = {}

    for row in inventory:
        category_totals[
            row.category
        ] = (
            category_totals.get(
                row.category,
                0,
            )
            + row.elements
        )

    total_parameters = sum(
        category_totals.values()
    )

    inventory_rows: list[
        list[str]
    ] = []

    for category, elements in sorted(
        category_totals.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    ):
        fraction = (
            0.0
            if total_parameters == 0
            else elements
            / total_parameters
        )

        inventory_rows.append(
            [
                category,
                format_int(
                    elements
                ),
                format_percent(
                    fraction
                ),
            ]
        )

    lines.append(
        markdown_table(
            [
                "Category",
                "Parameters",
                "Share",
            ],
            inventory_rows,
        )
    )

    lines.append("")

    lines.append(
        "The category inventory is a bookkeeping input to the later "
        "fixed-budget allocation matrix. It does not itself determine "
        "where parameters should be moved."
    )

    lines.append("")

    lines.append(
        "## Experiment Results"
    )

    lines.append("")

    experiment_rows: list[
        list[str]
    ] = []

    for result in results:
        experiment_rows.append(
            [
                result.seed,
                result.family,
                str(
                    result.rank
                ),
                format_float(
                    result.baseline_bpc,
                    6,
                ),
                format_float(
                    result.compressed_bpc,
                    6,
                ),
                (
                    f"{result.bpc_delta:+.6f}"
                ),
                format_percent(
                    result.retained_energy_mean
                ),
                format_percent(
                    result.retained_energy_min
                ),
                format_float(
                    result.relative_frobenius_error_mean,
                    6,
                ),
            ]
        )

    lines.append(
        markdown_table(
            [
                "Seed",
                "Family",
                "Rank",
                "Baseline BPC",
                "Compressed BPC",
                "Δ BPC",
                "Mean energy",
                "Min energy",
                "Mean relative error",
            ],
            experiment_rows,
        )
    )

    lines.append("")

    lines.append(
        "## Cross-Seed Summary"
    )

    lines.append("")

    summary_rows: list[
        list[str]
    ] = []

    for summary in summaries:
        summary_rows.append(
            [
                summary.family,
                str(
                    summary.rank
                ),
                format_float(
                    summary.mean_bpc_delta,
                    6,
                ),
                format_float(
                    summary.max_bpc_delta,
                    6,
                ),
                format_percent(
                    summary.mean_retained_energy
                ),
                format_percent(
                    summary.min_retained_energy
                ),
                summary.recommendation,
            ]
        )

    lines.append(
        markdown_table(
            [
                "Family",
                "Rank",
                "Mean Δ BPC",
                "Worst Δ BPC",
                "Mean energy",
                "Minimum energy",
                "Recommendation",
            ],
            summary_rows,
        )
    )

    lines.append("")

    lines.append(
        "## Interpretation"
    )

    lines.append("")

    lines.append(
        "PB0-A measured spectral structure. PB0-B measures functional "
        "sensitivity. A family should not be considered a parameter "
        "reallocation donor merely because it has a low stable rank."
    )

    lines.append("")

    lines.append(
        "A family becomes a stronger donor candidate when low-rank "
        "reconstruction preserves validation BPC consistently across both "
        "trained seeds."
    )

    lines.append("")

    lines.append(
        "The next research stage is fixed-budget PB0-C parameter allocation. "
        "PB0-C converts empirically compressible capacity into an explicit "
        "47M allocation matrix spanning matrix memory, vector/Mamba capacity, "
        "router capacity, feedback bridge capacity, embeddings, and output "
        "capacity."
    )

    lines.append("")

    OUTPUT_MARKDOWN.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )


# =============================================================================
# MAIN PB0-B EXPERIMENT
# =============================================================================


def evaluate_baseline(
    *,
    checkpoint: SeedCheckpoint,
    fwd_fn: Any,
    validation_data: np.ndarray,
    eval_chunks: int,
    eval_batch: int,
) -> float:
    banner(
        f"{checkpoint.seed.upper()} BASELINE EVALUATION"
    )

    started = time.perf_counter()

    baseline_bpc = evaluate_bpc(
        checkpoint.params,
        fwd_fn,
        validation_data,
        seq_len=(
            BASELINE_INPUT_SEQ_LEN
        ),
        chunks=eval_chunks,
        batch_size=eval_batch,
        loss_tail=(
            BASELINE_LOSS_TAIL
        ),
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    print(
        f"Baseline BPC : {baseline_bpc:.8f}"
    )

    print(
        f"Elapsed      : {elapsed:.2f} seconds"
    )

    if not math.isfinite(
        baseline_bpc
    ):
        fail(
            f"Baseline evaluation produced non-finite BPC for {checkpoint.seed}."
        )

    return baseline_bpc


def run_single_experiment(
    *,
    checkpoint: SeedCheckpoint,
    family: str,
    rank: int,
    baseline_bpc: float,
    fwd_fn: Any,
    validation_data: np.ndarray,
    eval_chunks: int,
    eval_batch: int,
) -> ExperimentResult:
    banner(
        f"{checkpoint.seed.upper()} | "
        f"{family} | RANK {rank}"
    )

    original_tensor = (
        get_family_tensor(
            checkpoint.params,
            family,
        )
    )

    started = time.perf_counter()

    (
        reconstructed_tensor,
        retained_energy_mean,
        retained_energy_min,
        reconstruction_error_mean,
        reconstruction_error_max,
    ) = reconstruct_family_with_rank(
        original_tensor,
        rank,
    )

    compressed_params = (
        replace_family_tensor(
            checkpoint.params,
            family,
            reconstructed_tensor,
        )
    )

    compressed_parameter_count = (
        count_parameter_elements(
            compressed_params
        )
    )

    original_parameter_count = (
        count_parameter_elements(
            checkpoint.params
        )
    )

    if (
        compressed_parameter_count
        != original_parameter_count
    ):
        fail(
            "Dense PB0-B replacement changed parameter count.\n\n"
            f"Original   : {format_int(original_parameter_count)}\n"
            f"Compressed : {format_int(compressed_parameter_count)}"
        )

    compressed_bpc = (
        evaluate_bpc(
            compressed_params,
            fwd_fn,
            validation_data,
            seq_len=(
                BASELINE_INPUT_SEQ_LEN
            ),
            chunks=eval_chunks,
            batch_size=eval_batch,
            loss_tail=(
                BASELINE_LOSS_TAIL
            ),
        )
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    if not math.isfinite(
        compressed_bpc
    ):
        fail(
            "Compressed evaluation produced non-finite BPC.\n"
            f"Seed   : {checkpoint.seed}\n"
            f"Family : {family}\n"
            f"Rank   : {rank}"
        )

    bpc_delta = (
        compressed_bpc
        - baseline_bpc
    )

    print(
        f"Baseline BPC           : {baseline_bpc:.8f}"
    )

    print(
        f"Compressed BPC         : {compressed_bpc:.8f}"
    )

    print(
        f"Delta BPC              : {bpc_delta:+.8f}"
    )

    print(
        "Mean retained energy   : "
        f"{retained_energy_mean:.8f}"
    )

    print(
        "Minimum retained energy: "
        f"{retained_energy_min:.8f}"
    )

    print(
        f"Elapsed                : {elapsed:.2f} seconds"
    )

    return ExperimentResult(
        seed=checkpoint.seed,
        family=family,
        rank=rank,
        checkpoint_path=str(
            checkpoint.path
        ),
        checkpoint_step=(
            checkpoint.step
        ),
        baseline_bpc=baseline_bpc,
        compressed_bpc=compressed_bpc,
        bpc_delta=bpc_delta,
        retained_energy_mean=(
            retained_energy_mean
        ),
        retained_energy_min=(
            retained_energy_min
        ),
        relative_frobenius_error_mean=(
            reconstruction_error_mean
        ),
        relative_frobenius_error_max=(
            reconstruction_error_max
        ),
        elapsed_seconds=elapsed,
    )


def main() -> None:
    args = parse_args()

    families = (
        parse_csv_strings(
            args.families
        )
    )

    ranks = (
        parse_csv_integers(
            args.ranks
        )
    )

    if args.eval_chunks <= 0:
        fail(
            "--eval-chunks must be positive."
        )

    if args.eval_batch <= 0:
        fail(
            "--eval-batch must be positive."
        )

    if (
        args.eval_batch
        > args.eval_chunks
    ):
        fail(
            "--eval-batch cannot exceed --eval-chunks."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    banner(
        "MODUS_X PB0-B DENSE TRUNCATED-SVD SENSITIVITY"
    )

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"Language dir : {LANGUAGE_DIR}"
    )

    print(
        f"Output dir   : {OUTPUT_DIR}"
    )

    print(
        f"JAX backend  : {jax.default_backend()}"
    )

    print()

    print(
        "Target families:"
    )

    for family in families:
        print(
            f"  - {family}"
        )

    print()

    print(
        "Candidate ranks:"
    )

    for rank in ranks:
        print(
            f"  - {rank}"
        )

    banner(
        "VALIDATING CANONICAL MODEL"
    )

    fwd_fn = (
        build_canonical_model()
    )

    print(
        "Canonical model construction: PASS"
    )

    print(
        "Expected parameter count    : "
        f"{format_int(BASELINE_PARAMETER_COUNT)}"
    )

    banner(
        "LOADING CANONICAL DATA"
    )

    (
        _train_data,
        validation_data,
        test_data,
    ) = load_canonical_splits(
        args.data_path
    )

    print(
        f"Dataset    : {args.data_path}"
    )

    print(
        "Validation : "
        f"{format_int(len(validation_data))} bytes"
    )

    print(
        "Test       : "
        f"{format_int(len(test_data))} bytes"
    )

    seed_directories = {
        "seed1": args.seed1_dir,
        "seed2": args.seed2_dir,
    }

    checkpoints: list[
        SeedCheckpoint
    ] = []

    for (
        seed,
        seed_directory,
    ) in seed_directories.items():
        banner(
            f"LOADING {seed.upper()}"
        )

        checkpoint = (
            load_checkpoint(
                seed,
                seed_directory,
            )
        )

        parameter_count = (
            count_parameter_elements(
                checkpoint.params
            )
        )

        if (
            parameter_count
            != BASELINE_PARAMETER_COUNT
        ):
            fail(
                f"Checkpoint parameter count mismatch for {seed}.\n"
                f"Expected: {format_int(BASELINE_PARAMETER_COUNT)}\n"
                f"Actual  : {format_int(parameter_count)}"
            )

        print(
            f"Checkpoint : {checkpoint.path}"
        )

        print(
            f"Step       : {checkpoint.step}"
        )

        print(
            "Parameters : "
            f"{format_int(parameter_count)}"
        )

        checkpoints.append(
            checkpoint
        )

    banner(
        "BUILDING PARAMETER INVENTORY"
    )

    inventory = (
        build_parameter_inventory(
            checkpoints[0].params,
            BASELINE_PARAMETER_COUNT,
        )
    )

    write_parameter_inventory_csv(
        inventory
    )

    category_totals: dict[
        str,
        int,
    ] = {}

    for row in inventory:
        category_totals[
            row.category
        ] = (
            category_totals.get(
                row.category,
                0,
            )
            + row.elements
        )

    for (
        category,
        elements,
    ) in sorted(
        category_totals.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    ):
        print(
            f"{category:20s} "
            f"{format_int(elements):>15s} "
            f"({elements / BASELINE_PARAMETER_COUNT * 100.0:6.2f}%)"
        )

    banner(
        "BASELINE EVALUATION"
    )

    baseline_results: dict[
        str,
        float,
    ] = {}

    for checkpoint in checkpoints:
        baseline_results[
            checkpoint.seed
        ] = evaluate_baseline(
            checkpoint=checkpoint,
            fwd_fn=fwd_fn,
            validation_data=validation_data,
            eval_chunks=args.eval_chunks,
            eval_batch=args.eval_batch,
        )

    results: list[
        ExperimentResult
    ] = []

    banner(
        "PB0-B SENSITIVITY EXPERIMENTS"
    )

    total_experiments = (
        len(checkpoints)
        * len(families)
        * len(ranks)
    )

    experiment_index = 0

    for checkpoint in checkpoints:
        for family in families:
            for rank in ranks:
                experiment_index += 1

                print()
                print(
                    f"Experiment {experiment_index}/"
                    f"{total_experiments}"
                )

                result = (
                    run_single_experiment(
                        checkpoint=checkpoint,
                        family=family,
                        rank=rank,
                        baseline_bpc=(
                            baseline_results[
                                checkpoint.seed
                            ]
                        ),
                        fwd_fn=fwd_fn,
                        validation_data=validation_data,
                        eval_chunks=(
                            args.eval_chunks
                        ),
                        eval_batch=(
                            args.eval_batch
                        ),
                    )
                )

                results.append(
                    result
                )

                write_results_csv(
                    results
                )

    banner(
        "BUILDING PB0-B SUMMARY"
    )

    summaries = (
        summarize_results(
            results
        )
    )

    for summary in summaries:
        print(
            f"{summary.family:10s} "
            f"rank={summary.rank:3d} "
            f"mean_delta={summary.mean_bpc_delta:+.6f} "
            f"worst_delta={summary.max_bpc_delta:+.6f} "
            f"{summary.recommendation}"
        )

    write_results_csv(
        results
    )

    write_summary_csv(
        summaries
    )

    write_json_report(
        args=args,
        results=results,
        summaries=summaries,
        inventory=inventory,
        baseline_parameter_count=(
            BASELINE_PARAMETER_COUNT
        ),
    )

    write_markdown_report(
        results=results,
        summaries=summaries,
        inventory=inventory,
    )

    if args.run_test:
        banner(
            "CANONICAL TEST EVALUATION"
        )

        print(
            "PB0-B test evaluation is requested, but the current "
            "sensitivity result table is validation-gated. Test evaluation "
            "should only be interpreted after selecting validation survivors."
        )

        for checkpoint in checkpoints:
            test_bpc = (
                evaluate_bpc(
                    checkpoint.params,
                    fwd_fn,
                    test_data,
                    seq_len=(
                        BASELINE_INPUT_SEQ_LEN
                    ),
                    chunks=(
                        args.eval_chunks
                    ),
                    batch_size=(
                        args.eval_batch
                    ),
                    loss_tail=(
                        BASELINE_LOSS_TAIL
                    ),
                )
            )

            print(
                f"{checkpoint.seed.upper()} "
                f"baseline test BPC: "
                f"{test_bpc:.8f}"
            )

    banner(
        "PB0-B COMPLETE"
    )

    print()

    print(
        f"Experiments completed : {len(results)}"
    )

    print(
        f"Seeds                 : {len(checkpoints)}"
    )

    print(
        f"Families              : {len(families)}"
    )

    print(
        f"Ranks                 : {len(ranks)}"
    )

    print()

    print(
        "Output files:"
    )

    print(
        f"  {OUTPUT_JSON}"
    )

    print(
        f"  {OUTPUT_MARKDOWN}"
    )

    print(
        f"  {OUTPUT_CSV}"
    )

    print(
        f"  {OUTPUT_SUMMARY_CSV}"
    )

    print(
        f"  {OUTPUT_PARAMETER_INVENTORY_CSV}"
    )

    print()

    print(
        "NEXT GATE: use PB0-B functional sensitivity together with the "
        "parameter inventory to construct PB0-C, a fixed ~47M parameter "
        "allocation matrix. Only empirically compressible matrix capacity "
        "should become a candidate donor for vector/Mamba, feedback, router, "
        "embedding, or output-head capacity."
    )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        banner(
            "PB0-B INTERRUPTED"
        )

        print(
            "The analysis was interrupted by the user."
        )

        sys.exit(130)

    except Exception as exc:
        banner(
            "PB0-B FAILED"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        print()

        traceback.print_exc()

        sys.exit(1)