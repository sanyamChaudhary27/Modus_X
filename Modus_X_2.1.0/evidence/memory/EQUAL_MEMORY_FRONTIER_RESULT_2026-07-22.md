# Equal-memory retrieval frontier result (2026-07-22)

## Decision

The corrected three-seed run establishes a measured crossover, not universal
superiority. At full-context state (`64.25 KiB`), the tied-Q/K Transformer wins
decisively. At `36.19 KiB` and below, CurrentArchiveDelta wins overall versioned
overwrite retrieval because its fixed matrix state preserves older bindings
that have fallen outside the Transformer's bounded KV window.

| BF16 state / sequence | Transformer KV window | CurrentArchive accuracy | Transformer accuracy | CA - Transformer |
|---:|---:|---:|---:|---:|
| 64.25 KiB | 128 | 71.03 +/- 2.12 | 98.63 +/- 0.20 | -27.60 |
| 36.19 KiB | 72 | 71.78 +/- 0.54 | 59.34 +/- 2.12 | +12.43 |
| 16.13 KiB | 32 | 72.92 +/- 1.83 | 17.71 +/- 2.64 | +55.21 |
| 4.06 KiB | 8 | 76.20 +/- 2.69 | 4.65 +/- 0.37 | +71.55 |
| 1.03 KiB | 2 | 36.46 +/- 1.30 | 4.00 +/- 0.35 | +32.45 |

Values are mean +/- sample standard deviation over seeds `17,27,37`, with
`1,024` test queries per seed and point.

## Strongest controlled point

At `16.13 KiB`, parameter counts are also close: CurrentArchiveDelta has
`166,791` parameters and TransformerKV has `174,112`. CurrentArchive reaches
`72.92%` versus `17.71%`. The corresponding role accuracies are:

| Model | Latest | Previous | First |
|---|---:|---:|---:|
| CurrentArchiveDelta | 70.06 | 72.53 | 76.17 |
| TransformerKV | 33.40 | 7.52 | 11.05 |

At `36.19 KiB`, the Transformer remains stronger on the latest value
(`80.10%` versus `69.02%`) while CurrentArchive is stronger on previous and
first values (`72.44/73.95%` versus `47.72/48.85%`). This is evidence of the
intended recency-versus-durable-compression tradeoff.

Using the explicitly defined proxy
`state_bytes / (32 bindings * mean accuracy)`, the near-parameter-matched
`16.13 KiB` point uses approximately `708` bytes per correctly retained
binding for CurrentArchive and `2,914` for TransformerKV. This proxy is only
meaningful inside this protocol.

## Capacity boundary

CurrentArchive remains near `71-76%` from widths `128` through `32`, then
drops to `36.46%` at width `16`. The matrix therefore has a measured low-state
failure region; it is not an unlimited store. The non-monotonic peak at width
`32` also means this run should not be presented as a clean architectural
scaling law.

## Protocol and caveats

- This is a controlled synthetic versioned-key overwrite task, not language
  modeling, general reasoning, or an end-to-end Transformer comparison.
- All reported test queries in this run are overwritten queries
  (`clean_query_count=0`). A mixed clean/overwrite follow-up is required for a
  broader update claim.
- The Transformer uses tied query/key projections and an identity-preserving
  input initialization. This favorable retrieval inductive bias was frozen
  only after the independent-Q/K control proved unlearnable (`4.49%`) and the
  tied control proved learnable (`98.63%`). It is a specialized learned causal
  Transformer baseline, not a claim about every Transformer implementation.
- The primary match is BF16 inference-state bytes. Parameter counts differ at
  most points; width `64` is the cleanest near-parameter match.
- Transformer state is a bounded KV window and CurrentArchive state is fixed
  matrix/vector memory. Their information representations differ by design.
- The supplied transcript contains all `30` learned-model rows but only `1`
  of the expected `15` exact-key-oracle rows. Retrieve the Kaggle result JSON
  before publishing an oracle frontier.

## Artifacts

- `equal_memory_frontier_2026-07-22/equal_memory_frontier_parsed_rows.json`
- `equal_memory_frontier_2026-07-22/equal_memory_frontier_aggregate.json`
- `equal_memory_frontier_2026-07-22/equal_memory_frontier_aggregate.csv`
- `equal_memory_frontier_2026-07-22/equal_memory_frontier_curve.png`
- `analyze_equal_memory_frontier.py`

The professor-facing statement is: **full-context KV is superior when its
state budget retains the sequence; under a constrained equal state budget,
CurrentArchive preserves versioned historical bindings substantially better,
with a measured matrix-capacity collapse at the smallest state.**
