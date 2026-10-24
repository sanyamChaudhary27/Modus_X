from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


ENWIK8_BYTES = 100_000_000
ENWIK8_SHA256 = "2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8"
ENWIK8_URL = "https://mattmahoney.net/dc/enwik8.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_data(workdir: Path) -> Path:
    data = workdir / "enwik8"
    archive_path = workdir / "enwik8.zip"
    if not data.exists():
        print("ONBOARDING_DATA_DOWNLOAD", ENWIK8_URL, flush=True)
        urllib.request.urlretrieve(ENWIK8_URL, archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            member = archive.getinfo("enwik8")
            with archive.open(member) as source, data.open("wb") as target:
                shutil.copyfileobj(source, target)

    size = data.stat().st_size
    digest = sha256(data)
    if size != ENWIK8_BYTES or digest != ENWIK8_SHA256:
        raise RuntimeError(
            "enwik8 verification failed: "
            f"bytes={size}, sha256={digest}; expected "
            f"bytes={ENWIK8_BYTES}, sha256={ENWIK8_SHA256}"
        )
    print(
        "ONBOARDING_DATA_READY",
        str(data),
        size,
        digest,
        flush=True,
    )
    return data


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    trainer = repo / "language" / "tpu_lm_train.py"
    if not trainer.exists():
        raise RuntimeError(f"Missing trainer: {trainer}")

    workdir = Path("/kaggle/working/modus_x_onboarding_data")
    outdir = Path("/kaggle/working/modus_x_onboarding_smoke")
    workdir.mkdir(parents=True, exist_ok=True)
    if outdir.exists():
        raise RuntimeError(
            f"Output already exists at {outdir}. Remove it only if you intend "
            "to restart the onboarding smoke from scratch."
        )
    data = prepare_data(workdir)

    command = [
        sys.executable,
        "-u",
        str(trainer),
        "--data-path",
        str(data),
        "--outdir",
        str(outdir),
        "--model",
        "Modus_X_MemoryFeedbackArchive_DeepSupervision",
        "--batch",
        "8",
        "--target-chars",
        "4096000",
        "--stop-chars",
        "409600",
        "--checkpoint-chars",
        "409600",
        "--eval-batch",
        "8",
        "--eval-chunks",
        "32",
        "--embed-dim",
        "512",
        "--hidden-dim",
        "1536",
        "--state-dim",
        "512",
        "--n-layers",
        "12",
        "--router-hidden",
        "32",
        "--lr",
        "6e-4",
        "--auxiliary-weight",
        "0.05",
        "--weight-decay",
        "1e-4",
        "--schedule",
        "constant",
        "--precision",
        "float32",
        "--input-seq-len",
        "512",
        "--loss-tail",
        "512",
        "--auxiliary-layers",
        "6",
        "--future-targets",
        "2",
        "--future-target-weight",
        "0.5",
        "--seed",
        "1",
    ]
    print("ONBOARDING_RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=trainer.parent)
    print("ONBOARDING_TPU_SMOKE_COMPLETE", outdir, flush=True)


if __name__ == "__main__":
    main()
