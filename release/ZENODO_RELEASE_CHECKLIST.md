# Zenodo and arXiv Checklist

The publication workflow uses the final files in `dist/`, not rc1 artifacts:

- Modus_X_3.0.0_release.zip
- Modus_X_3.0.0_paper.pdf
- Modus_X_3.0.0_arxiv_source.zip
- ARTIFACTS_SHA256.txt

Zenodo receives all four; arXiv receives only the flat LaTeX source ZIP. Check
author Sanyam Chaudhary, version3.0.0,2026-10-24,CC BY4.0 and publication/preprint.
arXiv primary is cs.LG; journal reference and journal DOI stay blank.

Check server compilation, PDF text/layout and all uploaded file checksums before
the final publish/submit action. Record actual success URLs/statuses in
PUBLICATION_STATUS.md. A reserved DOI or an upload is not proof of publication;
arXiv submission is not moderator acceptance or announcement.

The release branch stays separate from main for later review/merge. Do not
silently create a broad teammate merge or modify older version directories.
