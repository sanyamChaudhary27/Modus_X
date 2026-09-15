from __future__ import annotations

import csv
import hashlib
import json
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import jax
import jax.numpy as jnp


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "research" / "parameter_allocation"
OUTPUT_DIR = RESEARCH_DIR / "outputs"

MODELS_PATH = PROJECT_ROOT / "language" / "models.py"

PB0_G_PATH = (
    OUTPUT_DIR
    / "PB0_G_CALIBRATED_CANDIDATE_SEARCH.json"
)

B0_CENSUS_PATH = (
    OUTPUT_DIR
    / "b0_parameter_census.json"
)

OUTPUT_JSON_PATH = (
    OUTPUT_DIR
    / "PB0_H_EXACT_PARAMETER_TREE_RECOUNT.json"
)

OUTPUT_MARKDOWN_PATH = (
    OUTPUT_DIR
    / "PB0_H_EXACT_PARAMETER_TREE_RECOUNT.md"
)

OUTPUT_CSV_PATH = (
    OUTPUT_DIR
    / "PB0_H_PARAMETER_LEAVES.csv"
)

MODEL_NAME = "Modus_X_MemoryFeedbackArchive"

CANONICAL_VOCAB_SIZE = 256
CANONICAL_EMBED_DIM = 512
CANONICAL_HIDDEN_DIM = 1536
CANONICAL_N_LAYERS = 12
CANONICAL_N_HEADS_ATTN = 8
CANONICAL_SEQ_LEN = 512

CANONICAL_VECTOR_ROUTER = True
CANONICAL_AUXILIARY_LAYERS = (11,)
CANONICAL_FUTURE_TARGET_COUNT = 1


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass(frozen=True)
class SelectedCandidate:
    rank_r: int
    vector_dimension_n: int
    router_hidden_h: int
    feedback_rank_f: int
    predicted_parameter_count: int
    parameter_error: int
    exact_match: bool
    source_index: int


@dataclass(frozen=True)
class ParameterLeaf:
    path: str
    shape: tuple[int, ...]
    dtype: str
    parameter_count: int
    component: str


# =============================================================================
# GENERIC HELPERS
# =============================================================================


def print_banner(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    print()


def require_file(
    path: Path,
    description: str,
) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} does not exist:\n{path}"
        )

    if not path.is_file():
        raise FileNotFoundError(
            f"{description} is not a file:\n{path}"
        )


def load_json(
    path: Path,
) -> dict[str, Any]:
    require_file(
        path,
        f"Required JSON file '{path.name}'",
    )

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            payload = json.load(handle)

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid JSON in:\n{path}"
        ) from exc

    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected JSON object at root of:\n{path}"
        )

    return payload


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def normalize_path_component(
    value: Any,
) -> str:
    if isinstance(value, str):
        return value

    if isinstance(value, int):
        return str(value)

    return str(value)


def flatten_parameter_tree(
    tree: Any,
) -> list[tuple[str, Any]]:
    leaves_with_paths = jax.tree_util.tree_flatten_with_path(
        tree
    )

    flattened: list[tuple[str, Any]] = []

    for key_path, leaf in leaves_with_paths[0]:
        parts: list[str] = []

        for entry in key_path:
            if hasattr(
                entry,
                "key",
            ):
                parts.append(
                    normalize_path_component(
                        entry.key
                    )
                )

            elif hasattr(
                entry,
                "idx",
            ):
                parts.append(
                    normalize_path_component(
                        entry.idx
                    )
                )

            elif hasattr(
                entry,
                "name",
            ):
                parts.append(
                    normalize_path_component(
                        entry.name
                    )
                )

            else:
                parts.append(
                    str(entry)
                )

        path = ".".join(parts)

        flattened.append(
            (
                path,
                leaf,
            )
        )

    return flattened


def leaf_parameter_count(
    value: Any,
) -> int:
    array = jnp.asarray(
        value
    )

    return int(
        array.size
    )


def leaf_shape(
    value: Any,
) -> tuple[int, ...]:
    array = jnp.asarray(
        value
    )

    return tuple(
        int(dimension)
        for dimension in array.shape
    )


def leaf_dtype(
    value: Any,
) -> str:
    array = jnp.asarray(
        value
    )

    return str(
        array.dtype
    )


def infer_component(
    parameter_path: str,
) -> str:
    normalized = (
        parameter_path
        .lower()
        .replace(
            "/",
            ".",
        )
    )

    if (
        normalized.startswith(
            "embed"
        )
        or ".embed" in normalized
    ):
        return "embeddings"

    if (
        normalized.startswith(
            "head"
        )
        or ".head." in normalized
    ):
        return "output_head"

    if (
        normalized.startswith(
            "future_heads"
        )
        or ".future_heads" in normalized
    ):
        return "auxiliary"

    if (
        ".r_" in normalized
        or normalized.startswith(
            "layers.r"
        )
        or ".router" in normalized
    ):
        return "router"

    if (
        ".s_" in normalized
        or normalized.startswith(
            "layers.s"
        )
    ):
        if (
            "memory_feedback" in normalized
            or "memory_down" in normalized
            or "memory_up" in normalized
        ):
            return "feedback_bridge"

        return "vector_mamba_pathway"

    if (
        ".m_" in normalized
        or normalized.startswith(
            "layers.m"
        )
    ):
        if "archive" in normalized:
            return "archive_memory_control"

        return "matrix_memory"

    if (
        "norm" in normalized
        or ".ln_" in normalized
        or ".pre_g" in normalized
        or ".pre_b" in normalized
    ):
        return "normalization"

    return "other"


# =============================================================================
# MODEL IMPORT
# =============================================================================


def import_project_models() -> Any:
    if not MODELS_PATH.exists():
        raise FileNotFoundError(
            "Canonical model source does not exist:\n"
            f"{MODELS_PATH}"
        )

    project_root_string = str(
        PROJECT_ROOT
    )

    if (
        project_root_string
        not in sys.path
    ):
        sys.path.insert(
            0,
            project_root_string,
        )

    try:
        import language.models as models

    except Exception as exc:
        raise RuntimeError(
            "Failed to import language.models. "
            "PB0-H requires the actual canonical architecture."
        ) from exc

    required_symbols = (
        "ModelConfig",
        "make_model",
        "count_params",
    )

    missing = [
        symbol
        for symbol in required_symbols
        if not hasattr(
            models,
            symbol,
        )
    ]

    if missing:
        raise AttributeError(
            "language.models is missing required symbols: "
            + ", ".join(
                missing
            )
        )

    return models


# =============================================================================
# PB0-G CANDIDATE LOADING
# =============================================================================


def candidate_value(
    candidate: dict[str, Any],
    names: tuple[str, ...],
    description: str,
) -> Any:
    for name in names:
        if name in candidate:
            return candidate[name]

    raise KeyError(
        f"Could not find {description} in selected PB0-G candidate.\n"
        f"Expected one of: {list(names)}\n"
        f"Available keys: {sorted(candidate.keys())}"
    )


def load_selected_candidate() -> SelectedCandidate:
    payload = load_json(
        PB0_G_PATH
    )

    raw_candidates = (
        payload.get(
            "candidates"
        )
    )

    if not isinstance(
        raw_candidates,
        list,
    ):
        raise KeyError(
            "PB0-G output does not contain a valid 'candidates' list."
        )

    if not raw_candidates:
        raise RuntimeError(
            "PB0-G candidate list is empty."
        )

    exact_candidates: list[
        tuple[
            int,
            dict[str, Any],
        ]
    ] = []

    for index, candidate in enumerate(
        raw_candidates
    ):
        if not isinstance(
            candidate,
            dict,
        ):
            continue

        exact_value = candidate.get(
            "exact_match"
        )

        if exact_value is True:
            exact_candidates.append(
                (
                    index,
                    candidate,
                )
            )

    if exact_candidates:
        selected_index, selected = (
            exact_candidates[0]
        )

    else:
        selected_index = 0

        first_candidate = raw_candidates[0]

        if not isinstance(
            first_candidate,
            dict,
        ):
            raise TypeError(
                "First PB0-G candidate is not a JSON object."
            )

        selected = first_candidate

    rank_r = int(
        candidate_value(
            selected,
            (
                "rank_r",
                "matrix_rank",
                "r",
            ),
            "matrix rank",
        )
    )

    vector_dimension_n = int(
        candidate_value(
            selected,
            (
                "vector_dimension_n",
                "vector_dim",
                "n",
            ),
            "vector dimension",
        )
    )

    router_hidden_h = int(
        candidate_value(
            selected,
            (
                "router_hidden_h",
                "router_hidden",
                "h",
            ),
            "router hidden width",
        )
    )

    feedback_rank_f = int(
        candidate_value(
            selected,
            (
                "feedback_rank_f",
                "feedback_rank",
                "f",
            ),
            "feedback rank",
        )
    )

    predicted_parameter_count = int(
        candidate_value(
            selected,
            (
                "parameter_count",
                "predicted_parameter_count",
                "parameters",
            ),
            "predicted parameter count",
        )
    )

    parameter_error = int(
        selected.get(
            "parameter_error",
            selected.get(
                "error",
                0,
            ),
        )
    )

    exact_match = bool(
        selected.get(
            "exact_match",
            parameter_error == 0,
        )
    )

    if rank_r <= 0:
        raise ValueError(
            f"Selected matrix rank must be positive, got {rank_r}"
        )

    if vector_dimension_n <= 0:
        raise ValueError(
            "Selected vector dimension must be positive, "
            f"got {vector_dimension_n}"
        )

    if router_hidden_h <= 0:
        raise ValueError(
            "Selected router hidden width must be positive, "
            f"got {router_hidden_h}"
        )

    if feedback_rank_f <= 0:
        raise ValueError(
            "Selected feedback rank must be positive, "
            f"got {feedback_rank_f}"
        )

    if predicted_parameter_count <= 0:
        raise ValueError(
            "Selected predicted parameter count must be positive, "
            f"got {predicted_parameter_count}"
        )

    return SelectedCandidate(
        rank_r=rank_r,
        vector_dimension_n=vector_dimension_n,
        router_hidden_h=router_hidden_h,
        feedback_rank_f=feedback_rank_f,
        predicted_parameter_count=(
            predicted_parameter_count
        ),
        parameter_error=parameter_error,
        exact_match=exact_match,
        source_index=selected_index,
    )


# =============================================================================
# B0 BASELINE LOADING
# =============================================================================


def load_b0_parameter_count() -> int:
    payload = load_json(
        B0_CENSUS_PATH
    )

    preferred_keys = (
        "actual_parameter_count",
        "expected_parameter_count",
        "baseline_parameter_count",
        "parameter_count",
        "total_parameters",
    )

    for key in preferred_keys:
        value = payload.get(
            key
        )

        if (
            isinstance(
                value,
                int,
            )
            and not isinstance(
                value,
                bool,
            )
            and value > 0
        ):
            return int(
                value
            )

    raise KeyError(
        "Could not resolve the canonical parameter count "
        "from the B0 census.\n"
        f"Available root keys: {sorted(payload.keys())}"
    )


# =============================================================================
# CONFIGURATION
# =============================================================================


def build_selected_config(
    models: Any,
    candidate: SelectedCandidate,
) -> Any:
    return models.ModelConfig(
        vocab_size=CANONICAL_VOCAB_SIZE,
        embed_dim=CANONICAL_EMBED_DIM,
        hidden_dim=CANONICAL_HIDDEN_DIM,
        ax_res=candidate.rank_r,
        n_layers=CANONICAL_N_LAYERS,
        n_heads_attn=CANONICAL_N_HEADS_ATTN,
        seq_len=CANONICAL_SEQ_LEN,
        mamba_state_dim=(
            candidate.vector_dimension_n
        ),
        vector_router=CANONICAL_VECTOR_ROUTER,
        router_hidden=(
            candidate.router_hidden_h
        ),
    )


def actual_feedback_rank(
    candidate: SelectedCandidate,
) -> int:
    return min(
        32,
        candidate.rank_r,
        candidate.vector_dimension_n,
    )


# =============================================================================
# REAL MODEL INSTANTIATION
# =============================================================================


def instantiate_selected_candidate(
    models: Any,
    cfg: Any,
) -> Any:
    key = jax.random.PRNGKey(
        0
    )

    try:
        params, _forward = (
            models.make_model(
                MODEL_NAME,
                key,
                cfg,
                auxiliary_layers=(
                    CANONICAL_AUXILIARY_LAYERS
                ),
                future_target_count=(
                    CANONICAL_FUTURE_TARGET_COUNT
                ),
            )
        )

    except TypeError as exc:
        raise RuntimeError(
            "The actual make_model factory rejected the PB0-H "
            "canonical deep-supervision arguments. "
            "This indicates a mismatch between PB0-F/PB0-G "
            "and the current language/models.py implementation."
        ) from exc

    return params


# =============================================================================
# EXACT PARAMETER TREE CENSUS
# =============================================================================


def collect_parameter_leaves(
    params: Any,
) -> list[ParameterLeaf]:
    flattened = (
        flatten_parameter_tree(
            params
        )
    )

    if not flattened:
        raise RuntimeError(
            "The instantiated parameter tree contains no leaves."
        )

    leaves: list[
        ParameterLeaf
    ] = []

    seen_paths: set[
        str
    ] = set()

    for path, value in flattened:
        if not path:
            raise RuntimeError(
                "Encountered an empty parameter path."
            )

        if path in seen_paths:
            raise RuntimeError(
                "Duplicate parameter path encountered: "
                f"{path}"
            )

        seen_paths.add(
            path
        )

        count = (
            leaf_parameter_count(
                value
            )
        )

        if count <= 0:
            raise RuntimeError(
                f"Parameter leaf has non-positive size: {path}"
            )

        shape = (
            leaf_shape(
                value
            )
        )

        dtype = (
            leaf_dtype(
                value
            )
        )

        component = (
            infer_component(
                path
            )
        )

        leaves.append(
            ParameterLeaf(
                path=path,
                shape=shape,
                dtype=dtype,
                parameter_count=count,
                component=component,
            )
        )

    leaves.sort(
        key=lambda leaf: leaf.path
    )

    return leaves


def count_tree_parameters(
    leaves: Iterable[
        ParameterLeaf
    ],
) -> int:
    return sum(
        leaf.parameter_count
        for leaf in leaves
    )


def component_summary(
    leaves: Iterable[
        ParameterLeaf
    ],
) -> dict[
    str,
    dict[str, int],
]:
    summary: dict[
        str,
        dict[str, int],
    ] = defaultdict(
        lambda: {
            "parameters": 0,
            "leaf_count": 0,
        }
    )

    for leaf in leaves:
        summary[
            leaf.component
        ][
            "parameters"
        ] += (
            leaf.parameter_count
        )

        summary[
            leaf.component
        ][
            "leaf_count"
        ] += 1

    return dict(
        sorted(
            summary.items(),
            key=lambda item: (
                -item[1][
                    "parameters"
                ],
                item[0],
            ),
        )
    )


def build_tree_fingerprint(
    leaves: Iterable[
        ParameterLeaf
    ],
) -> str:
    digest = hashlib.sha256()

    for leaf in leaves:
        record = (
            f"{leaf.path}|"
            f"{leaf.shape}|"
            f"{leaf.dtype}|"
            f"{leaf.parameter_count}\n"
        )

        digest.update(
            record.encode(
                "utf-8"
            )
        )

    return (
        digest.hexdigest()
    )


# =============================================================================
# OUTPUT WRITING
# =============================================================================


def write_csv(
    leaves: list[
        ParameterLeaf
    ],
) -> None:
    fieldnames = [
        "path",
        "component",
        "shape",
        "dtype",
        "parameter_count",
    ]

    with OUTPUT_CSV_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for leaf in leaves:
            writer.writerow(
                {
                    "path": leaf.path,
                    "component": leaf.component,
                    "shape": (
                        "x".join(
                            str(
                                dimension
                            )
                            for dimension
                            in leaf.shape
                        )
                    ),
                    "dtype": leaf.dtype,
                    "parameter_count": (
                        leaf.parameter_count
                    ),
                }
            )


def write_json_output(
    candidate: SelectedCandidate,
    b0_parameter_count: int,
    instantiated_parameter_count: int,
    count_params_result: int,
    leaves: list[
        ParameterLeaf
    ],
    components: dict[
        str,
        dict[str, int],
    ],
    tree_fingerprint: str,
) -> None:
    actual_f = (
        actual_feedback_rank(
            candidate
        )
    )

    predicted_difference = (
        instantiated_parameter_count
        - candidate.predicted_parameter_count
    )

    b0_difference = (
        instantiated_parameter_count
        - b0_parameter_count
    )

    independent_count_difference = (
        count_params_result
        - instantiated_parameter_count
    )

    payload = {
        "generated_at_utc": (
            utc_now()
        ),
        "project_root": str(
            PROJECT_ROOT
        ),
        "models_path": str(
            MODELS_PATH
        ),
        "model_name": MODEL_NAME,
        "scope": (
            "Exact parameter-tree recount through the actual "
            "Modus_X initializer. No model source, checkpoint, "
            "dataset, or training run modified."
        ),
        "selected_pb0_g_candidate": {
            "source_index": (
                candidate.source_index
            ),
            "matrix_rank_r": (
                candidate.rank_r
            ),
            "vector_dimension_n": (
                candidate.vector_dimension_n
            ),
            "router_hidden_h": (
                candidate.router_hidden_h
            ),
            "pb0_g_feedback_rank_f": (
                candidate.feedback_rank_f
            ),
            "actual_initializer_feedback_rank": (
                actual_f
            ),
            "pb0_g_predicted_parameter_count": (
                candidate.predicted_parameter_count
            ),
            "pb0_g_parameter_error": (
                candidate.parameter_error
            ),
            "pb0_g_exact_match": (
                candidate.exact_match
            ),
        },
        "architecture_configuration": {
            "vocab_size": (
                CANONICAL_VOCAB_SIZE
            ),
            "embed_dim": (
                CANONICAL_EMBED_DIM
            ),
            "hidden_dim": (
                CANONICAL_HIDDEN_DIM
            ),
            "ax_res": (
                candidate.rank_r
            ),
            "n_layers": (
                CANONICAL_N_LAYERS
            ),
            "n_heads_attn": (
                CANONICAL_N_HEADS_ATTN
            ),
            "seq_len": (
                CANONICAL_SEQ_LEN
            ),
            "mamba_state_dim": (
                candidate.vector_dimension_n
            ),
            "vector_router": (
                CANONICAL_VECTOR_ROUTER
            ),
            "router_hidden": (
                candidate.router_hidden_h
            ),
            "auxiliary_layers": list(
                CANONICAL_AUXILIARY_LAYERS
            ),
            "future_target_count": (
                CANONICAL_FUTURE_TARGET_COUNT
            ),
        },
        "parameter_counts": {
            "b0_frozen_parameter_count": (
                b0_parameter_count
            ),
            "pb0_g_predicted_parameter_count": (
                candidate.predicted_parameter_count
            ),
            "pb0_h_tree_recount_parameter_count": (
                instantiated_parameter_count
            ),
            "models_count_params_parameter_count": (
                count_params_result
            ),
            "tree_vs_pb0_g_difference": (
                predicted_difference
            ),
            "tree_vs_b0_difference": (
                b0_difference
            ),
            "count_params_vs_tree_difference": (
                independent_count_difference
            ),
        },
        "verification": {
            "tree_matches_pb0_g_prediction": (
                predicted_difference == 0
            ),
            "tree_matches_b0_budget": (
                b0_difference == 0
            ),
            "models_count_params_matches_tree": (
                independent_count_difference == 0
            ),
            "feedback_rank_matches_pb0_g": (
                actual_f
                == candidate.feedback_rank_f
            ),
            "all_exact": (
                predicted_difference == 0
                and b0_difference == 0
                and independent_count_difference == 0
            ),
        },
        "tree": {
            "leaf_count": len(
                leaves
            ),
            "fingerprint_sha256": (
                tree_fingerprint
            ),
            "components": (
                components
            ),
            "parameter_leaves": [
                {
                    "path": leaf.path,
                    "component": (
                        leaf.component
                    ),
                    "shape": list(
                        leaf.shape
                    ),
                    "dtype": (
                        leaf.dtype
                    ),
                    "parameter_count": (
                        leaf.parameter_count
                    ),
                }
                for leaf in leaves
            ],
        },
        "status": (
            "passed"
            if (
                predicted_difference == 0
                and b0_difference == 0
                and independent_count_difference == 0
            )
            else "failed"
        ),
        "next_gate": (
            "PB0-H exact tree recount completed. "
            "Do not train yet until the research objective "
            "distinguishes whether an exact parameter-preserving "
            "architecture change is still required. "
            "If PB0-H passes and the selected candidate is "
            "identical to baseline dimensions, the next gate must "
            "search for a non-baseline architecture under explicit "
            "architectural constraints."
        ),
    }

    with OUTPUT_JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
        )
        handle.write(
            "\n"
        )


def write_markdown_output(
    candidate: SelectedCandidate,
    b0_parameter_count: int,
    instantiated_parameter_count: int,
    count_params_result: int,
    leaves: list[
        ParameterLeaf
    ],
    components: dict[
        str,
        dict[str, int],
    ],
    tree_fingerprint: str,
) -> None:
    actual_f = (
        actual_feedback_rank(
            candidate
        )
    )

    predicted_difference = (
        instantiated_parameter_count
        - candidate.predicted_parameter_count
    )

    b0_difference = (
        instantiated_parameter_count
        - b0_parameter_count
    )

    count_params_difference = (
        count_params_result
        - instantiated_parameter_count
    )

    all_exact = (
        predicted_difference == 0
        and b0_difference == 0
        and count_params_difference == 0
    )

    lines: list[
        str
    ] = []

    lines.append(
        "# PB0-H Exact Parameter-Tree Recount"
    )
    lines.append("")

    lines.append(
        "## Scope"
    )
    lines.append("")

    lines.append(
        "PB0-H instantiates the selected PB0-G architecture "
        "through the actual `language.models.make_model` "
        "factory and performs an exact recursive recount of "
        "every parameter leaf."
    )
    lines.append("")

    lines.append(
        "No model source, checkpoint, dataset, or training "
        "run was modified."
    )
    lines.append("")

    lines.append(
        "## Selected PB0-G Candidate"
    )
    lines.append("")

    lines.append(
        f"- Matrix rank `r`: **{candidate.rank_r:,}**"
    )

    lines.append(
        f"- Vector dimension `n`: "
        f"**{candidate.vector_dimension_n:,}**"
    )

    lines.append(
        f"- Router hidden width `h`: "
        f"**{candidate.router_hidden_h:,}**"
    )

    lines.append(
        f"- PB0-G feedback rank `f`: "
        f"**{candidate.feedback_rank_f:,}**"
    )

    lines.append(
        f"- Actual initializer feedback rank: "
        f"**{actual_f:,}**"
    )

    lines.append(
        f"- PB0-G predicted parameters: "
        f"**{candidate.predicted_parameter_count:,}**"
    )

    lines.append("")

    lines.append(
        "## Exact Count Verification"
    )
    lines.append("")

    lines.append(
        "| Measurement | Parameters | Difference |"
    )

    lines.append(
        "|---|---:|---:|"
    )

    lines.append(
        f"| Frozen B0 budget "
        f"| {b0_parameter_count:,} "
        f"| {instantiated_parameter_count - b0_parameter_count:+,} |"
    )

    lines.append(
        f"| PB0-G prediction "
        f"| {candidate.predicted_parameter_count:,} "
        f"| {predicted_difference:+,} |"
    )

    lines.append(
        f"| PB0-H recursive tree recount "
        f"| {instantiated_parameter_count:,} "
        f"| {0:+,} |"
    )

    lines.append(
        f"| `models.count_params` "
        f"| {count_params_result:,} "
        f"| {count_params_difference:+,} |"
    )

    lines.append("")

    lines.append(
        f"Overall exact verification: **{all_exact}**"
    )

    lines.append("")

    lines.append(
        "## Parameter Tree"
    )
    lines.append("")

    lines.append(
        f"- Parameter leaves: **{len(leaves):,}**"
    )

    lines.append(
        f"- SHA-256 structural fingerprint: "
        f"`{tree_fingerprint}`"
    )

    lines.append("")

    lines.append(
        "## Component Summary"
    )
    lines.append("")

    lines.append(
        "| Component | Leaves | Parameters |"
    )

    lines.append(
        "|---|---:|---:|"
    )

    for component, summary in components.items():
        lines.append(
            f"| `{component}` "
            f"| {summary['leaf_count']:,} "
            f"| {summary['parameters']:,} |"
        )

    lines.append("")

    lines.append(
        "## Gate Result"
    )
    lines.append("")

    if all_exact:
        lines.append(
            "PB0-H **PASSED**. The selected PB0-G candidate "
            "was instantiated through the actual Modus_X "
            "initializer and the recursive parameter-tree "
            "recount exactly matches both the PB0-G prediction "
            "and the frozen B0 parameter budget."
        )

    else:
        lines.append(
            "PB0-H **FAILED**. At least one independent "
            "parameter count disagrees. Do not train this "
            "candidate until the mismatch is reconciled."
        )

    lines.append("")

    lines.append(
        "## Important Interpretation"
    )
    lines.append("")

    if (
        candidate.rank_r == 512
        and candidate.vector_dimension_n == 512
        and candidate.router_hidden_h == 32
    ):
        lines.append(
            "The exact selected candidate uses the same "
            "search dimensions as the canonical baseline. "
            "Therefore an exact parameter-budget match alone "
            "does not yet demonstrate a new architecture."
        )

        lines.append("")

        lines.append(
            "Any subsequent architecture change must therefore "
            "be evaluated under explicit research constraints "
            "rather than assuming that a different configuration "
            "exists within the current three-dimensional "
            "configuration space."
        )

    lines.append("")

    lines.append(
        "## Next Gate"
    )
    lines.append("")

    lines.append(
        "Do not train yet. The next step is to decide whether "
        "the research objective requires a non-baseline "
        "architecture while preserving the fixed parameter "
        "budget. If so, the next analysis must explicitly "
        "search or design an architectural change that is not "
        "expressible solely through `ax_res`, "
        "`mamba_state_dim`, and `router_hidden`."
    )

    lines.append("")

    with OUTPUT_MARKDOWN_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "\n".join(
                lines
            )
        )


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    print_banner(
        "MODUS_X PB0-H EXACT PARAMETER TREE RECOUNT"
    )

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"Models source: {MODELS_PATH}"
    )

    print(
        f"PB0-G input  : {PB0_G_PATH}"
    )

    print(
        f"B0 input     : {B0_CENSUS_PATH}"
    )

    print(
        f"Output dir   : {OUTPUT_DIR}"
    )

    require_file(
        MODELS_PATH,
        "Canonical language/models.py",
    )

    require_file(
        PB0_G_PATH,
        "PB0-G calibrated candidate search output",
    )

    require_file(
        B0_CENSUS_PATH,
        "B0 parameter census",
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print_banner(
        "IMPORTING ACTUAL CANONICAL ARCHITECTURE"
    )

    models = (
        import_project_models()
    )

    print(
        f"Imported module: {models.__file__}"
    )

    print_banner(
        "LOADING SELECTED PB0-G CANDIDATE"
    )

    candidate = (
        load_selected_candidate()
    )

    print(
        f"PB0-G candidate index : "
        f"{candidate.source_index}"
    )

    print(
        f"Matrix rank r         : "
        f"{candidate.rank_r:,}"
    )

    print(
        f"Vector dimension n    : "
        f"{candidate.vector_dimension_n:,}"
    )

    print(
        f"Router hidden h       : "
        f"{candidate.router_hidden_h:,}"
    )

    print(
        f"PB0-G feedback f      : "
        f"{candidate.feedback_rank_f:,}"
    )

    print(
        f"Predicted parameters  : "
        f"{candidate.predicted_parameter_count:,}"
    )

    print(
        f"PB0-G exact match     : "
        f"{candidate.exact_match}"
    )

    actual_f = (
        actual_feedback_rank(
            candidate
        )
    )

    print(
        f"Actual feedback rank  : "
        f"{actual_f:,}"
    )

    print_banner(
        "LOADING FROZEN B0 PARAMETER BUDGET"
    )

    b0_parameter_count = (
        load_b0_parameter_count()
    )

    print(
        f"Frozen B0 parameters: "
        f"{b0_parameter_count:,}"
    )

    print_banner(
        "BUILDING SELECTED ARCHITECTURE CONFIGURATION"
    )

    cfg = (
        build_selected_config(
            models=models,
            candidate=candidate,
        )
    )

    print(
        f"Model name             : "
        f"{MODEL_NAME}"
    )

    print(
        f"Vocabulary size        : "
        f"{CANONICAL_VOCAB_SIZE:,}"
    )

    print(
        f"Embedding dimension    : "
        f"{CANONICAL_EMBED_DIM:,}"
    )

    print(
        f"Hidden dimension       : "
        f"{CANONICAL_HIDDEN_DIM:,}"
    )

    print(
        f"Matrix rank / ax_res   : "
        f"{candidate.rank_r:,}"
    )

    print(
        f"Vector state dimension : "
        f"{candidate.vector_dimension_n:,}"
    )

    print(
        f"Router hidden          : "
        f"{candidate.router_hidden_h:,}"
    )

    print(
        f"Layers                 : "
        f"{CANONICAL_N_LAYERS:,}"
    )

    print_banner(
        "INSTANTIATING THROUGH REAL MODUS_X INITIALIZER"
    )

    params = (
        instantiate_selected_candidate(
            models=models,
            cfg=cfg,
        )
    )

    print(
        "Real parameter tree instantiated successfully."
    )

    print_banner(
        "PERFORMING EXACT RECURSIVE PARAMETER TREE RECOUNT"
    )

    leaves = (
        collect_parameter_leaves(
            params
        )
    )

    tree_parameter_count = (
        count_tree_parameters(
            leaves
        )
    )

    count_params_result = int(
        models.count_params(
            params
        )
    )

    if tree_parameter_count <= 0:
        raise RuntimeError(
            "Recursive parameter-tree recount produced "
            "a non-positive total."
        )

    if count_params_result <= 0:
        raise RuntimeError(
            "models.count_params produced a non-positive total."
        )

    print(
        f"Parameter leaves         : "
        f"{len(leaves):,}"
    )

    print(
        f"Recursive tree recount   : "
        f"{tree_parameter_count:,}"
    )

    print(
        f"models.count_params      : "
        f"{count_params_result:,}"
    )

    print(
        f"PB0-G prediction         : "
        f"{candidate.predicted_parameter_count:,}"
    )

    print(
        f"Frozen B0 budget         : "
        f"{b0_parameter_count:,}"
    )

    print(
        f"Tree vs PB0-G difference : "
        f"{tree_parameter_count - candidate.predicted_parameter_count:+,}"
    )

    print(
        f"Tree vs B0 difference    : "
        f"{tree_parameter_count - b0_parameter_count:+,}"
    )

    print(
        f"count_params vs tree     : "
        f"{count_params_result - tree_parameter_count:+,}"
    )

    components = (
        component_summary(
            leaves
        )
    )

    tree_fingerprint = (
        build_tree_fingerprint(
            leaves
        )
    )

    print_banner(
        "COMPONENT PARAMETER SUMMARY"
    )

    for component, summary in components.items():
        print(
            f"{component:<28} "
            f"leaves={summary['leaf_count']:>4,} "
            f"parameters={summary['parameters']:>12,}"
        )

    print_banner(
        "WRITING PB0-H OUTPUTS"
    )

    write_csv(
        leaves
    )

    write_json_output(
        candidate=candidate,
        b0_parameter_count=(
            b0_parameter_count
        ),
        instantiated_parameter_count=(
            tree_parameter_count
        ),
        count_params_result=(
            count_params_result
        ),
        leaves=leaves,
        components=components,
        tree_fingerprint=(
            tree_fingerprint
        ),
    )

    write_markdown_output(
        candidate=candidate,
        b0_parameter_count=(
            b0_parameter_count
        ),
        instantiated_parameter_count=(
            tree_parameter_count
        ),
        count_params_result=(
            count_params_result
        ),
        leaves=leaves,
        components=components,
        tree_fingerprint=(
            tree_fingerprint
        ),
    )

    tree_matches_prediction = (
        tree_parameter_count
        == candidate.predicted_parameter_count
    )

    tree_matches_b0 = (
        tree_parameter_count
        == b0_parameter_count
    )

    count_params_matches_tree = (
        count_params_result
        == tree_parameter_count
    )

    all_exact = (
        tree_matches_prediction
        and tree_matches_b0
        and count_params_matches_tree
    )

    print_banner(
        "PB0-H EXACT PARAMETER TREE RECOUNT COMPLETE"
    )

    print(
        f"PB0-H tree count       : "
        f"{tree_parameter_count:,}"
    )

    print(
        f"PB0-G predicted count  : "
        f"{candidate.predicted_parameter_count:,}"
    )

    print(
        f"B0 frozen count        : "
        f"{b0_parameter_count:,}"
    )

    print(
        f"models.count_params    : "
        f"{count_params_result:,}"
    )

    print()

    print(
        f"Tree matches PB0-G     : "
        f"{tree_matches_prediction}"
    )

    print(
        f"Tree matches B0        : "
        f"{tree_matches_b0}"
    )

    print(
        f"Independent count match: "
        f"{count_params_matches_tree}"
    )

    print(
        f"All exact              : "
        f"{all_exact}"
    )

    print()

    print(
        f"Tree fingerprint       : "
        f"{tree_fingerprint}"
    )

    print()

    print(
        "Output files:"
    )

    print(
        f"  {OUTPUT_JSON_PATH}"
    )

    print(
        f"  {OUTPUT_MARKDOWN_PATH}"
    )

    print(
        f"  {OUTPUT_CSV_PATH}"
    )

    print()

    if all_exact:
        print(
            "NEXT GATE:"
        )

        print(
            "PB0-H PASSED. The actual JAX parameter tree "
            "exactly matches PB0-G and B0."
        )

        print(
            "Do not train yet. Determine whether the next "
            "research objective requires a genuinely non-baseline "
            "architecture under the fixed parameter budget."
        )

    else:
        raise RuntimeError(
            "PB0-H verification failed.\n\n"
            f"PB0-G predicted: "
            f"{candidate.predicted_parameter_count:,}\n"
            f"Tree recount   : "
            f"{tree_parameter_count:,}\n"
            f"B0 frozen      : "
            f"{b0_parameter_count:,}\n"
            f"count_params   : "
            f"{count_params_result:,}\n\n"
            "Do not train until the parameter-count mismatch "
            "has been reconciled."
        )


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print()
        print("=" * 80)
        print(
            "PB0-H EXACT PARAMETER TREE RECOUNT FAILED"
        )
        print("=" * 80)
        print()

        print(
            f"{type(exc).__name__}: {exc}"
        )

        print()
        traceback.print_exc()

        raise