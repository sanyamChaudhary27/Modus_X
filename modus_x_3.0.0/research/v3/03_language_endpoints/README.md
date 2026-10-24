# Matched language endpoints

Memory mechanisms only matter to the main Modus_X claim if they improve the
language model. The segment-retention candidate was therefore continued under
the same frozen 47,437,768-parameter recipe as its exact MemoryFeedbackArchive
control. Both runs reached 102.4M processed characters and were selected by the
mean of two dense validation offsets.

Seed 1 improved dense validation by `0.026833 BPC`. Seed 2 improved it by
`0.006981 BPC`. Mean gain was `0.016907 BPC`, with effectively unchanged
runtime. Test values were read only after each endpoint was frozen and are
report-only.

This is a development confirmation, not the final v3 replication. Seed 3 is
reserved for paper closure, and a second corpus is still required.
