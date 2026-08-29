# Release Manifest

The final release is built deterministically by:

```bash
python release/build_release.py
```

After the Zenodo DOI is reserved and all gates close:

```bash
python release/build_release.py --final
```

The staged package contains `MANIFEST.sha256`, which hashes every packaged file
except itself. The output directory also contains
`Modus_X_2.1.0_release.zip.sha256` for the archive.

## Primary publication artifacts

| Artifact | Purpose | Status |
|---|---|---|
| `paper/Modus_X_2.1.0_whitepaper.pdf` | rendered research paper | generated during build |
| `Modus_X_2.1.0_release.zip` | source, evidence, docs, and paper | generated during build |
| `MANIFEST.sha256` | file-level integrity | generated inside archive |
| `.zenodo.json` | archival metadata | included |

Large training checkpoints are excluded from the compact archive unless
explicitly added and documented before final publication.
