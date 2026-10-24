# Manifest and Build

`modus_x_3.0.0/MANIFEST.sha256` is the authoritative versioned-payload manifest.
The root MANIFEST.sha256 covers tracked repository files, excluding itself and
untracked local work. Frozen upstream source hashes additionally appear in the
versioned SOURCE_PROVENANCE.json.

Run `python release/build_release.py --final` to validate and rebuild the release
ZIP, paper PDF, flat arXiv source ZIP and artifact SHA-256 list under `dist/`.
The source/evidence ZIP uses a deterministic timestamp and one enclosing
`modus_x_3.0.0/` directory. The arXiv ZIP contains the manuscript and figure PDFs
only, with main.tex at archive root. It is not the code ZIP.

No checkpoints, Python caches, local review notes or private correspondence are
included. Save complete paired endpoint outputs separately for recovery.
