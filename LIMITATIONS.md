# Limitations and open questions

1. **Separate leads.** MemoryFeedbackArchive leads language modeling;
   CurrentArchiveDelta leads the controlled bounded-memory study. No single
   checkpoint yet demonstrates both advantages.
2. **Generic language quality.** Official Mamba remains better on the matched
   dense enwik8 comparison available to the project.
3. **Scale.** The exact 1B path has systems smokes, not a trained quality
   result.
4. **Stale recall.** CurrentArchive retrieves clean and overwritten history
   strongly but still produces stale values on some latest queries.
5. **Synthetic-to-language bridge.** Controlled version tags and query roles
   are not supplied by ordinary language data.
6. **Compute matching.** Parameters, processed data, optimizer updates,
   recurrent state, wall time, and hardware are different matching axes. No
   single experiment matches all of them.
7. **Efficiency.** Bounded recurrent state does not imply constant total
   serving cost. Model weights, activations, batching, kernels, and hardware
   utilization still matter.
8. **Kernel maturity.** The research implementation is not a fused production
   kernel. Runtime measurements include implementation quality.
9. **Benchmark breadth.** enwik8 and synthetic retrieval are insufficient to
   establish reasoning, instruction following, factuality, or tool use.
10. **The `1.1` target.** The external target remains unmet.
11. **Single-seed scaling.** The 47M/81M/99M MemoryFeedback points are one seed
    each, the 47M learning-rate schedule differs from the larger runs, and the
    matched v2 improvement over v1.1.1 is only `0.001735` BPC.
12. **Scale allocation.** The 99M point is not a self-similar enlargement:
    matrix/vector state grows less than backbone width and feedback rank stays
    fixed at 32.
13. **Single endpoint selection.** The 81M continuation was promoted after the
    pre-registered 99M efficiency gate and evaluated at one matched endpoint.
    Further enwik8 tuning is frozen to avoid selecting against final test data.

The next funded study should train matched multi-size language models, measure
long-context serving behavior, and test natural-language memory/update tasks
without weakening these claim boundaries.
