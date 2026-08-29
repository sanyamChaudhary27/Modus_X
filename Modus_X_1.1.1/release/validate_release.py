from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "README.md",
    "LICENSE",
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
    "evidence/RESULTS_LEDGER.md",
    "release/CHANGELOG.md",
    "release/MANIFEST.md",
    "release/RELEASE_GATES.md",
]


def main() -> None:
    missing = [path for path in REQUIRED if not (ROOT / path).exists()]
    if missing:
        raise SystemExit("Missing required release files:\n- " + "\n- ".join(missing))

    forbidden = list(ROOT.rglob("__pycache__")) + list(ROOT.rglob("*.pyc"))
    if forbidden:
        raise SystemExit("Temporary files found:\n- " + "\n- ".join(map(str, forbidden)))

    print(f"Structure OK: {len(REQUIRED)} required files present.")
    print("Evidence and paper-content gates remain tracked in release/RELEASE_GATES.md.")


if __name__ == "__main__":
    main()

