# Modus_X 3

## Calibrated associative memory. Faster recurrent execution.

Modus_X couples current and archive matrix memories to a vector recurrence.
Retrieved memory changes the recurrence's input before its state update; a
router also preserves the direct matrix output. The model processes sequences
without a growing attention cache.

**Modus_X 3.0.0 is a source, paper and experimental-evidence release.**
Version DOI: [10.5281/zenodo.23621101](https://doi.org/10.5281/zenodo.23621101).
Zenodo publication is live. The paper is also submitted to arXiv cs.LG,
awaiting moderation/public announcement; no public arXiv identifier is assigned
yet. See [publication status](release/PUBLICATION_STATUS.md).
Its frozen experiment snapshot is based on upstream commit
`766379e89f1395d98b9fc0df3cf4fc7f5a73de6f`. Large trained checkpoints are not
bundled; this is not a pretrained assistant or a full checkpoint distribution.

### Measured Results

| Change | Result | Comparison |
|---|---|---|
| Two-stage execution | **2.923x median update speedup** | Two 47M TPU continuation experiments, same equations, highest precision |
| Two-lag K/V context | **3/3 dense-validation endpoint wins** | Paired 25k updates, 102.4M characters, 47.46M vs 47.44M parameters |
| Endpoint validation gain | **0.021213 median; 0.039566 mean BPC** | Seed gains 0.087291, 0.010196, 0.021213; not a uniform effect |
| Filter overhead | **24,576 parameters; 96 KiB FP32 state/sequence** | Approximately 0.115% mean update-time overhead across seeds |

The language control already uses archive calibration and two-stage execution.
These gains must not be added to older retention-only gains. The model's total
FP32 inference state is 25,288,704 bytes per sequence, excluding weights and
temporary activations. Its size is constant in sequence length, not in width.

### Start Here

- [Paper source](paper/manuscript/main.tex) and [compiled paper](paper/Modus_X_3.0.0_paper.pdf).
- [Claims and comparisons](docs/CLAIMS_AND_COMPARISONS.md).
- [Reproduction guide](REPRODUCIBILITY.md) and [environment record](ENVIRONMENTS.md).
- [Release review](docs/RELEASE_REVIEW.md), including remaining archival work.
- [Evidence map](docs/EVIDENCE_MAP.md) and [calculated statistics](docs/AGGREGATE.json).
- [Source provenance](SOURCE_PROVENANCE.json) and `MANIFEST.sha256`.

### Experiment Map

| Track | Question |
|---|---|
| [01_streaming_and_coordination](research/v3/01_streaming_and_coordination/README.md) | Does carried state affect prediction, and which paths matter? |
| [02_segment_scale_retention](research/v3/02_segment_scale_retention/README.md) | Does archive calibration preserve useful memory without destructive over-retention? |
| [03_language_endpoints](research/v3/03_language_endpoints/README.md) | Does retention calibration improve its development language endpoints? |
| [04_two_stage_execution](research/v3/04_two_stage_execution/README.md) | Can the same dependencies execute more efficiently? |
| [05_local_kv_filtering](research/v3/05_local_kv_filtering/README.md) | Does local write context improve frozen language endpoints? |
| [negative_results](research/v3/negative_results/README.md) | Which alternatives failed and must remain visible? |

Track files are preserved source snapshots. Some older track summaries retain
their historical pending status; this root and the evidence map identify the
later records rather than silently rewriting their chronology.

### Relationship to Earlier Releases

v1.1.1 established routing and associative-retrieval experiments. v2 added
current/archive storage and MemoryFeedback coordination. v3 calibrates archive
retention, optimizes execution, and tests causal local K/V context.

The historical official-Mamba run has better enwik8 compression than the v2
model at its recorded endpoint. Earlier associative-memory tests show a large
Modus_X advantage under their particular retrieval/update protocol. Those are
different model configurations and budgets. They are not evidence that the
new v3 language checkpoint beats every Mamba variant or Transformer.

Code is MIT licensed under [LICENSE](LICENSE); the manuscript and newly authored
release documentation are [CC BY 4.0](PAPER_LICENSE.md). Sole author:
Sanyam Chaudhary. This is a preprint research release, not a conference
acceptance or a peer-review certification. See [citation metadata](CITATION.cff).

For a reduced local check, run `python language/run_modus_x3.py smoke`. Full
endpoint reproduction uses the frozen Kaggle TPU cells described in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Versioned Releases

The current complete release snapshot is in [modus_x_3.0.0/](modus_x_3.0.0/).
Earlier releases remain unchanged in [Modus_X_2.1.0/](Modus_X_2.1.0/),
[Modus_X_2.0.0/](Modus_X_2.0.0/) and [Modus_X_1.1.1/](Modus_X_1.1.1/).
The predecessor DOI is [10.5281/zenodo.21590445](https://doi.org/10.5281/zenodo.21590445).
Historical root language/memory code remains for prior protocols; the v3 entry
point resolves the frozen source inside the versioned release.

`release/v3.0.0` stages this release for review and later merge into `main`.
The branch does not merge the reviewed teammate changes from other research refs.
