# Modus_X v2.1.0 Paper Restructure

Status: **required editorial correction for v2.1.0**

The v2.0.0 package contains the new architecture, implementation, and evidence,
but the whitepaper preserves too much of the v1.1.1 narrative structure. A
reader can therefore reach the end without a sufficiently sharp distinction
between inherited Modus_X evidence and the contributions introduced in v2.

This is a presentation and scientific-lineage problem, not permission to
reinterpret the measured results.

## Required changes

1. Put a one-page `v1.1.1 -> v2` contribution map immediately after the
   introduction.
2. Introduce MemoryFeedbackArchive and CurrentArchiveDelta as two explicit v2
   experimental branches before presenting inherited v1.1.1 results.
3. Add a side-by-side architecture figure:
   - v1.1.1: late fusion between vector and matrix streams;
   - v2 MemoryFeedbackArchive: retrieved matrix context conditions recurrent
     computation;
   - v2 CurrentArchiveDelta: bounded current and archive stores separate
     recent-value and historical retrieval behavior.
4. Give each v2 mechanism its own algorithm box, equations, parameter/state
   accounting, and implementation path.
5. Mark every result table row as `Inherited v1.1.1`, `Measured v2`, or
   `Modeled`.
6. Separate the evidence sections into:
   - generic language modeling;
   - controlled memory and update behavior;
   - systems readiness;
   - limitations and rejected branches.
7. State prominently that MemoryFeedbackArchive is the v2 language lead and
   CurrentArchiveDelta is the v2 bounded-memory lead; no experiment yet shows
   one model owning both advantages.
8. Move the v1.1.1 scientific history into a clearly labeled background and
   prior-results section instead of allowing it to carry the main v2 story.
9. Rewrite the abstract and conclusion around the actual v2 contribution:
   coordinated matrix-to-vector feedback plus bounded current/archive memory,
   with separate measured strengths and unresolved integration.
10. Retain the negative results and explicit boundaries: official Mamba wins
    the tested generic-language comparison, `1.1` BPC is unmet, and the 1B
    model has systems-readiness evidence rather than trained quality evidence.

## Acceptance test

A technically informed reader who sees only the abstract, contribution map,
architecture spread, and results table must be able to answer:

- what v1.1.1 already contained;
- what changed in v2;
- which v2 branch produced each result;
- which claims are measured, inherited, or modeled;
- what remains unproven.

The v2.1.0 paper should not ship until all five answers are unambiguous.
