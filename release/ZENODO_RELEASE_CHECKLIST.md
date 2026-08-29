# Zenodo Release Checklist

Zenodo creates a new version as a separate record with its own version DOI,
linked to the prior versions and concept DOI.

1. Open the published Modus_X record and choose **New version**.
2. Import prior files only when they remain valid; remove superseded archives
   and PDFs.
3. Upload:
   - `Modus_X_2.1.0_whitepaper.pdf`;
   - `Modus_X_2.1.0_release.zip`;
   - optional compact raw-run archives listed in `MANIFEST.md`.
4. Use version `2.1.0`, resource type `Publication / Preprint`, publication
   date matching the actual publish date, and CC BY 4.0 for the record. Code
   inside the archive remains MIT-licensed.
5. Add the prior v1.1.1 DOI as `isNewVersionOf` only if Zenodo has not already
   created the version relationship automatically.
6. Reserve the v2.1.0 DOI in the new-version draft, then embed it in the final
   paper and citation metadata before upload. Do not reuse the v2.0.0 DOI
   `10.5281/zenodo.21538210`.
7. Preview title, description, creators, license, related identifiers, files,
   and visibility.
8. Publish only after `validate_release.py`, manifest verification, and visual
   PDF inspection pass.

Zenodo currently allows up to 100 files and 50GB per upload. The compact v2
release intentionally uses a small number of preservation-friendly files.
