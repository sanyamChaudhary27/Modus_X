# Modus_X

Modus_X is an open research program exploring how bounded associative matrix
memory can cooperate with recurrent computation. This repository preserves the
published v1.1.1 line and includes the v2 research series as separate,
versioned packages.

## Releases

### Modus_X 2.1.0

The current research release is browsable at this repository root and archived
in full in [`Modus_X_2.1.0/`](Modus_X_2.1.0/), which additionally holds the
raw evidence archive. It contains the restructured 32-page
whitepaper, source and protocols, controlled-memory evidence, dense language
audits, rejected interventions, systems-readiness appendices, and a
deterministic integrity manifest.

Its two evidence leads remain deliberately separate:

- **MemoryFeedbackArchive** is the language-modeling lead. The promoted
  81.49M-parameter run reaches `1.382445` dense test BPC at `163.84M`
  processed characters, narrowly improving the v1.1.1 endpoint with `1.54%`
  fewer parameters.
- **CurrentArchiveDelta** is the bounded storage and update lead. It
  outperforms the tested truncated Transformer KV baseline after that cache
  can no longer retain full context, while full-context KV remains superior.

Official Mamba remains stronger on the matched dense language endpoint. The
two v2 advantages have not yet been demonstrated in one trained checkpoint;
the `1.1` BPC target and trained 1B quality remain open.

The version DOI is
[`10.5281/zenodo.21590445`](https://doi.org/10.5281/zenodo.21590445).

### Modus_X 2.0.0

The preceding v2 package is preserved in
[`Modus_X_2.0.0/`](Modus_X_2.0.0/).
It introduces two distinct experimental leads:

- **MemoryFeedbackArchive**, where retrieved matrix context conditions the
  recurrent stream and improves the tested dense enwik8 result over the
  matched CurrentArchive control;
- **CurrentArchiveDelta**, which separates bounded current and historical
  storage and performs strongly on controlled mixed clean/update retrieval
  after a matched Transformer KV cache can no longer retain full context.

These strengths have not yet been demonstrated together in one model.
Official Mamba remains stronger on the tested matched generic-language
comparison, the `1.1` BPC target remains unmet, and the 1B artifacts establish
systems readiness rather than trained model quality.

The published v2.0.0 DOI is
[`10.5281/zenodo.21538210`](https://doi.org/10.5281/zenodo.21538210).

### Modus_X v1.1.1

The published v1.1.1 research release is preserved in
[`Modus_X_1.1.1/`](Modus_X_1.1.1/). Its Zenodo DOI is
[`10.5281/zenodo.20923248`](https://doi.org/10.5281/zenodo.20923248).
