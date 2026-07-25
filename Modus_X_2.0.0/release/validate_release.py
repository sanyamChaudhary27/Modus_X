from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "README.md",
    "LICENSE",
    "CITATION.cff",
    ".zenodo.json",
    "paper/whitepaper.md",
    "paper/build_pdf.py",
    "docs/architecture.md",
    "docs/BENCHMARK_PROTOCOL.md",
    "docs/CLAIMS_AND_EVIDENCE.md",
    "docs/LIMITATIONS.md",
    "docs/MODEL_CARD.md",
    "docs/PROVENANCE.md",
    "docs/REPRODUCIBILITY.md",
    "evidence/EVIDENCE_INDEX.md",
    "evidence/v1_1_1_context/RESULTS_LEDGER.md",
    "evidence/language/scaling/MEMORY_FEEDBACK_SCALING_RESULT_2026-07-24.md",
    "evidence/language/scaling/memory_feedback_81m_scaling_point.json",
    "evidence/language/scaling/memory_feedback_99m_scaling_point.json",
    "evidence/language/scaling/memory_feedback_scaling.png",
    "release/CHANGELOG.md",
    "release/MANIFEST.md",
    "release/RELEASE_GATES.md",
    "release/ZENODO_DESCRIPTION.md",
    "release/ZENODO_RELEASE_CHECKLIST.md",
    "release/build_release.py",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--final",
        action="store_true",
        help="Fail if DOI_PENDING remains or a release gate is unchecked.",
    )
    args = parser.parse_args()

    missing = [path for path in REQUIRED if not (ROOT / path).exists()]
    if missing:
        raise SystemExit("Missing required release files:\n- " + "\n- ".join(missing))

    forbidden = sorted(ROOT.rglob("__pycache__")) + sorted(ROOT.rglob("*.pyc"))
    if forbidden:
        print(
            "Warning: temporary source-tree files will be excluded from the archive:\n- "
            + "\n- ".join(map(str, forbidden))
        )

    metadata = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    if metadata.get("version") != "2.0.0":
        raise SystemExit(".zenodo.json must declare version 2.0.0")
    if (
        metadata.get("upload_type") != "publication"
        or metadata.get("publication_type") != "preprint"
    ):
        raise SystemExit(
            ".zenodo.json must use publication/preprint for this versioned paper record"
        )

    scaling = [
        json.loads(
            (ROOT / "evidence/language/scaling" / name).read_text(encoding="utf-8")
        )
        for name in (
            "memory_feedback_81m_scaling_point.json",
            "memory_feedback_99m_scaling_point.json",
        )
    ]
    expected = {
        "memory_feedback_81m": (81_486_728, 1.4331383109092712, 1.4438726902008057),
        "memory_feedback_99m": (99_438_920, 1.4342829585075378, 1.4420339465141296),
    }
    for row in scaling:
        label = row["label"]
        params, dense_val, dense_test = expected[label]
        observed = (
            row["params"],
            row["dense_validation_bpc"],
            row["dense_test_bpc"],
        )
        if observed != (params, dense_val, dense_test):
            raise SystemExit(f"Scaling evidence mismatch for {label}: {observed}")

    compile_failures = []
    for path in sorted(ROOT.rglob("*.py")):
        try:
            compile(
                path.read_text(encoding="utf-8"),
                str(path.relative_to(ROOT)),
                "exec",
            )
        except (SyntaxError, UnicodeDecodeError) as exc:
            compile_failures.append(f"{path.relative_to(ROOT)}: {exc}")
    if compile_failures:
        raise SystemExit(
            "Python compilation failures:\n- " + "\n- ".join(compile_failures)
        )

    if args.final:
        pending = []
        for path in (ROOT / "CITATION.cff", ROOT / "paper/build_pdf.py"):
            if "DOI_PENDING" in path.read_text(encoding="utf-8"):
                pending.append(str(path.relative_to(ROOT)))
        gates = (ROOT / "release/RELEASE_GATES.md").read_text(encoding="utf-8")
        if "- [ ]" in gates:
            pending.append("release/RELEASE_GATES.md")
        if pending:
            raise SystemExit(
                "Final-release conditions remain open:\n- " + "\n- ".join(pending)
            )

    print(f"Structure OK: {len(REQUIRED)} required files present.")
    print(f"Python compile OK: {len(list(ROOT.rglob('*.py')))} files.")
    print("Scaling evidence constants verified.")
    print("Use --final only after DOI reservation and all release gates close.")


if __name__ == "__main__":
    main()
