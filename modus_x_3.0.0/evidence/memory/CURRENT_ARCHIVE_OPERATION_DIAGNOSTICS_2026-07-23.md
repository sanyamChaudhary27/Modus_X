# CurrentArchive operation diagnostics

Date: 2026-07-23

## Question

After latest-heavy supervision reduced but did not eliminate stale recall, where
does the remaining error arise?

The frozen alternatives were:

1. stale content in the current matrix;
2. stale correction from the vector path;
3. final fusion/router behavior;
4. archive readout dominance.

## Protocol

- Model: `CurrentArchiveDelta`
- Parameters: `166,791`
- Recurrent state: `16,512` bytes
- Curricula: balanced and latest-heavy
- Seeds: `17, 27, 37`
- Exact seed-derived balanced test reconstruction
- Six saved validation-selected checkpoints; no retraining
- Runner:
  `experiments/matrix_memory_capacity/run_current_archive_operation_diagnostics.py`
- Raw report:
  `experiments/matrix_memory_capacity/results/current_archive_operation_diagnostics_2026-07-23/current_archive_operation_diagnostics.json`
- Raw report SHA256:
  `FDEF84B74E40D90BC369F148E0AD41F4F16BD9C7DC571BD68C1B723E76235416`

The branch predictions reuse the trained final classifier. They are
shared-head diagnostic probes, not independently trained heads or causal
ablations.

## Key results

On latest-overwritten, stale-eligible queries:

| Curriculum | Final accuracy | Final stale | Current-probe accuracy | Current-probe stale | Vector-probe stale |
|---|---:|---:|---:|---:|---:|
| Balanced | 68.263% | 9.950% | 65.874% | 11.151% | 3.450% |
| Latest-heavy | 73.877% | 8.090% | 72.147% | 8.889% | 2.258% |

For examples where the final answer was stale:

- final/current-probe agreement was `82.73%` under balanced training and
  `83.88%` under latest-heavy training;
- final/vector-probe agreement was `0.00%` and `2.38%`;
- the current probe itself produced a stale value on `82.73%` and `83.88%`;
- the history probe was stale on more than `92%`, as expected for historical
  memory, but latest-role readout hard-selects current memory.

Across all six case-seed pairs, comparing stale failures with correct latest
answers:

- current/history disagreement was lower in `6/6` pairs, with mean difference
  `-0.622`;
- history-read norm was higher in `6/6` pairs, with mean difference `+0.325`;
- router direction split `3/3`;
- current-read norm direction split `4/2`.

## Interpretation

The remaining stale error is not primarily caused by the vector path or a
router that selects archive readout. For tagged latest queries, the
implementation already hard-selects current memory. The stale answer is
usually present in the current-memory/shared-head probe before final fusion.

The stable reduction in current/history disagreement and increase in
history-read norm indicate current/history representational convergence or
interference on failed examples. This supports a current-slot
refresh/separation hypothesis.

The implementation provides a specific mechanism to test. CurrentArchive uses
role-aware write keys, but unlike the latest-shadow variants it does not place
the first fact into a consistent latest/current slot. A later update can
therefore write a latest-role address without explicitly replacing a
same-address first value. This is a narrower hypothesis than generic
arbitration tuning.

## Decision

The diagnostic passes the pre-registered requirement for one correction:
current/history disagreement has the same failure direction in both curricula
and all three seeds.

Permit exactly one state-neutral correction:

`CurrentArchiveDelta + latest_shadow_write`

The first fact initializes the current/latest address; later updates refresh
that same address. The archive continues to preserve historical information.
No new router, loss sweep, attention path, or state growth is permitted in
this gate.

Compare the corrected model with an identically initialized
`CurrentArchiveDelta` control under the latest-heavy curriculum. Require:

- three seeds and validation-only checkpoint selection;
- equal parameters and recurrent-state bytes;
- latest-overwritten accuracy improves by at least `3` points **or** stale
  false recall improves by at least `2` points;
- stale false recall reaches at most `7%`;
- overall accuracy loses at most `1` point;
- previous and first overwritten accuracy each lose at most `3` points;
- runtime overhead is at most `5%`.

If it fails, freeze the synthetic architecture lane for v2 packaging.

## Claim boundary

This diagnostic identifies a stable association and a localized implementation
hypothesis. It does not yet prove that latest-shadow refresh causally fixes the
error. Only the paired correction gate can establish that.
