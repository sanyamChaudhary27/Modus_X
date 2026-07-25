from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1]
PROJECT = SOURCE.parent
OUTPUT = PROJECT / "output"
STAGE = OUTPUT / "Modus_X_2.0.0"
ARCHIVE = OUTPUT / "Modus_X_2.0.0_release.zip"

EXCLUDED_PARTS = {"__pycache__", ".git", ".pytest_cache", ".mypy_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".tmp"}
EXCLUDED_NAMES = {"whitepaper.html", "MANIFEST.sha256"}
FIXED_ZIP_TIME = (2026, 7, 24, 0, 0, 0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def included(path: Path) -> bool:
    relative = path.relative_to(SOURCE)
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    if path.name in EXCLUDED_NAMES:
        return False
    return True


def build_manifest() -> None:
    rows = []
    for path in sorted(STAGE.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.sha256":
            rows.append(f"{sha256(path)}  {path.relative_to(STAGE).as_posix()}")
    (STAGE / "MANIFEST.sha256").write_text("\n".join(rows) + "\n", encoding="ascii")


def build_zip() -> None:
    if ARCHIVE.exists():
        ARCHIVE.unlink()
    with zipfile.ZipFile(
        ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(STAGE.rglob("*")):
            if not path.is_file():
                continue
            relative = Path("Modus_X_2.0.0") / path.relative_to(STAGE)
            info = zipfile.ZipInfo(relative.as_posix(), FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--final",
        action="store_true",
        help="Require DOI replacement and all release gates to be checked.",
    )
    parser.add_argument(
        "--skip-pdf",
        action="store_true",
        help="Package without rebuilding the whitepaper PDF.",
    )
    args = parser.parse_args()

    validator = SOURCE / "release/validate_release.py"
    command = [sys.executable, str(validator)]
    if args.final:
        command.append("--final")
    subprocess.run(command, check=True, cwd=SOURCE)

    if not args.skip_pdf:
        subprocess.run(
            [sys.executable, str(SOURCE / "paper/build_pdf.py")],
            check=True,
            cwd=SOURCE / "paper",
        )

    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    for source in sorted(SOURCE.rglob("*")):
        if source.is_file() and included(source):
            destination = STAGE / source.relative_to(SOURCE)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    build_manifest()
    build_zip()
    archive_hash = sha256(ARCHIVE)
    (OUTPUT / "Modus_X_2.0.0_release.zip.sha256").write_text(
        f"{archive_hash}  {ARCHIVE.name}\n", encoding="ascii"
    )

    print(f"STAGED {STAGE}")
    print(f"ARCHIVE {ARCHIVE}")
    print(f"SHA256 {archive_hash}")


if __name__ == "__main__":
    main()

