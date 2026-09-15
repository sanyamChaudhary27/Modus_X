from __future__ import annotations

import inspect
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LANGUAGE_DIRECTORY = REPOSITORY_ROOT / "language"


def print_separator() -> None:
    print("=" * 100)


def print_function_signature(module: object, function_name: str) -> None:
    print_separator()
    print(f"FUNCTION: {function_name}")
    print_separator()

    if not hasattr(module, function_name):
        print("STATUS: NOT FOUND")
        print()
        return

    function = getattr(module, function_name)

    print(f"OBJECT TYPE: {type(function)}")

    try:
        signature = inspect.signature(function)
        print(f"SIGNATURE: {function_name}{signature}")
    except (TypeError, ValueError) as exc:
        print(f"SIGNATURE: UNAVAILABLE ({type(exc).__name__}: {exc})")

    try:
        source_file = inspect.getsourcefile(function)
        print(f"SOURCE FILE: {source_file}")
    except (TypeError, OSError) as exc:
        print(f"SOURCE FILE: UNAVAILABLE ({type(exc).__name__}: {exc})")

    try:
        source_lines, starting_line = inspect.getsourcelines(function)

        print(f"STARTING LINE: {starting_line}")
        print()
        print("SOURCE:")
        print("-" * 100)

        for offset, source_line in enumerate(source_lines):
            line_number = starting_line + offset
            print(f"{line_number:6d} | {source_line.rstrip()}")

    except (TypeError, OSError) as exc:
        print(f"SOURCE: UNAVAILABLE ({type(exc).__name__}: {exc})")

    print()


def main() -> None:
    print_separator()
    print("MODUS_X CANONICAL RECURRENT-STATE API INSPECTION")
    print_separator()

    print(f"Repository root: {REPOSITORY_ROOT}")
    print(f"Language directory: {LANGUAGE_DIRECTORY}")
    print()

    if not LANGUAGE_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Expected language directory does not exist: {LANGUAGE_DIRECTORY}"
        )

    language_directory_string = str(LANGUAGE_DIRECTORY)

    if language_directory_string not in sys.path:
        sys.path.insert(0, language_directory_string)

    try:
        import models
    except ImportError as exc:
        raise RuntimeError(
            "Could not import language/models.py. "
            "Ensure the repository virtual environment is activated and all "
            "dependencies from requirements.txt are installed."
        ) from exc

    print(f"Imported module: {models.__file__}")
    print()

    print_separator()
    print("MODEL REGISTRY ENTRY")
    print_separator()

    registry_key = "Modus_X_MemoryFeedbackArchive"

    if registry_key not in models.MODEL_REGISTRY:
        available_models = sorted(models.MODEL_REGISTRY.keys())

        raise KeyError(
            f"Model registry key {registry_key!r} was not found. "
            f"Available keys: {available_models}"
        )

    initializer, forward = models.MODEL_REGISTRY[registry_key]

    print(f"Registry key: {registry_key}")
    print(f"Initializer: {initializer.__name__}")
    print(f"Forward:     {forward.__name__}")
    print()

    functions_to_inspect = [
        "init_modus_x_state",
        "init_modus_x_memory_feedback_archive_lm",
        "modus_x_memory_feedback_archive_lm_fwd",
        "modus_x_memory_feedback_archive_lm_fwd_stateful",
        "modus_x_memory_feedback_archive_layer_fwd",
        "modus_x_memory_feedback_archive_layer_fwd_stateful",
        "modus_x_memory_feedback_archive_diagnostics",
    ]

    for function_name in functions_to_inspect:
        print_function_signature(models, function_name)

    print_separator()
    print("REGISTRY INITIALIZER SIGNATURE")
    print_separator()

    try:
        print(f"{initializer.__name__}{inspect.signature(initializer)}")
    except (TypeError, ValueError) as exc:
        print(
            f"Could not inspect initializer signature: "
            f"{type(exc).__name__}: {exc}"
        )

    print()

    print_separator()
    print("REGISTRY FORWARD SIGNATURE")
    print_separator()

    try:
        print(f"{forward.__name__}{inspect.signature(forward)}")
    except (TypeError, ValueError) as exc:
        print(
            f"Could not inspect forward signature: "
            f"{type(exc).__name__}: {exc}"
        )

    print()

    print_separator()
    print("INSPECTION COMPLETE")
    print_separator()


if __name__ == "__main__":
    main()
    