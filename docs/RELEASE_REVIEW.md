# v3 Release Review

Publication preparation update: the author authorized the v3.0.0 source/evidence
release on 2026-10-24, confirmed sole authorship, and explicitly approved CC BY
4.0 for the manuscript and MIT for code. Reserved version DOI:
10.5281/zenodo.23621101. The review below preserves the rc1 decision history.
Checkpoint payload distribution and direct cross-lane comparisons remain
outside the released artifact scope unless subsequently supplied and verified.

Prepared 2026-10-24 against official research commit
`766379e89f1395d98b9fc0df3cf4fc7f5a73de6f`.

## Decision

**Prepare v3.0.0, using this rc1 for review.** The revision has a defined
architecture, distinct mechanism and systems evidence, three paired local-K/V
endpoints and reported endpoint audits. Calling this only a v2 backend patch
would hide the model changes. Calling it state of the art would exceed the
comparisons. A professional research release can make its actual wins prominent
without pretending that every scientific question has been closed.

The manuscript has been rewritten around v3, rather than adding new paragraphs
to a v1/v2 whitepaper. Tables and plots are computed from the normalized result
JSON. Equations were checked against `models.py`, `local_kv.py`, `two_stage.py`,
the complete loss and the actual fixed-horizon trainer.

## Findings Resolved in This Candidate

1. **Release identity.** The manuscript, root README, citation metadata and
   release notes consistently identify v3.0.0-rc1. No v2 DOI is assigned to v3.
2. **Mechanism accuracy.** Matrix feedback enters the vector recurrence before
   its update. The direct matrix route remains. This is not official Mamba in
   the Modus_X core and must not be labeled as such.
3. **Calibration, not expressivity.** `log(512)` shifts archive-retention logits;
   a free learned bias can absorb it. The paper describes optimization and
   operating point, not increased theoretical memory capacity.
4. **Execution, not a new function.** Two-stage scheduling and chunk32 are
   separated from architecture changes. Both scans remain sequential. The
   2.923x number is synchronized median update time in two continuations.
5. **Unequal seed effects.** Every endpoint and the sample SD are visible.
   Median 0.021213 and mean 0.039566 BPC are both stated. No population
   significance or token-level pseudo-replication is claimed.
6. **Screen history.** Failed seed-2/3 5k screens remain visible. Seed 2 is a
   post-screen fixed-horizon follow-up; seed 3 prospectively fixes its endpoint
   after development. The series is not retroactively preregistered.
7. **Comparison hygiene.** Old Mamba/xLSTM/Modus_X results are a historical
   panel, not points on a new v3 scaling law. New K/V comparisons isolate K/V
   within the retention-calibrated recipe. Gains from different lanes are not
   added. Dense/reset and subset/carry results remain distinct.
8. **Current literature.** Related work includes DeltaNet, Gated DeltaNet, KDA,
   Gated DeltaNet-2, Mamba-3, TTT, Titans and recent local/global recurrent designs. Short
   filtering and delta rules are explicitly credited as established mechanisms.
9. **Bounded state accounting.** Total FP32 state is 25,288,704 bytes/sequence;
   the filter increment is 98,304 bytes. No claim of constant training memory,
   infinite retention or semantic version correctness is made.
10. **Historical source preservation.** Six-file experiment records and cells
    are copied from git objects, excluding untracked files and reviewed changes
    from other branches. A separate release index supersedes stale operational
    summaries without rewriting frozen history.

## Still Required Before Calling the Full Artifact Reproducible

**Durable checkpoints.** Publish or otherwise durably index the six 25k local-K/V
checkpoint payloads and their pair/recovery manifests. The source records six
hashes, but a hash is not a functioning download link. Independently download,
hash and run the read-only audit from those public artifacts. Capture exact
environment and notebook versions. The source/evidence release and a paper can
disclose missing weights; they must not claim full artifact reproducibility.

**Recipe reconciliation.** Earlier retention-only endpoints and later K/V
control endpoints differ, especially seed 1. Their source records are preserved
and the new within-pair K/V result is supported by the audits; the reason for
the cross-lane difference is not established. Do not attribute the difference
to RNG, precision or scheduling without tracing the actual checkpoint metadata.
For a direct full-v2-to-full-v3 statement, add an explicitly matched comparison
or release a metadata reconciliation demonstrating what changed. This is not a
reason to erase the current K/V result.

**Author sign-off and publication metadata.** User confirmed Sanyam Chaudhary
as sole author. Review the final PDF, chosen paper license, title/abstract and
checkpoint disclosure; then reserve the new version DOI and create the final
tag. arXiv cs.LG endorsement is eligibility, not acceptance or peer review.
No external person's name is used as an implied endorsement of the results.

## Research That Would Strengthen a Venue Submission

The highest-value new comparison is a modern recurrent baseline under the same
training/evaluation recipe, with state, parameters, precision and end-to-end
cost accounted for. A second corpus and a real memory-intensive developer task
would test whether the result transfers. These are research extensions, not
uncompleted checks secretly counted in this release. Avoid launching another
architecture sweep merely to make a stronger headline.

A rank/dimension-matched DeltaNet/Gated DeltaNet-2 comparison is more relevant
to the matrix-mechanism claim than another unmatched large official-Mamba run.
An equal-parameter headroom control can test whether the extra 24,576 parameters
alone explain the benefit; causal/filter interventions at trained endpoints
can test how learned lags are used. Neither experiment has been performed in
this release preparation. Developer quality/cost claims require that workload,
not a conversion of enwik8 BPC into an assistant-quality score.

## Validation Boundary

See `LOCAL_VALIDATION.json` for checks actually run during preparation. Frozen
TPU results and checkpoint audits were not rerun on this CPU. Package integrity,
source pinning, arithmetic, Python syntax, reduced model tests and PDF layout
can be verified locally without inventing another accelerator experiment.
