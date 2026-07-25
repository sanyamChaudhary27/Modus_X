# Limitations

See the root `LIMITATIONS.md` for the canonical list. The most important
boundaries are:

- language and controlled-memory leads belong to different promoted models;
- official Mamba remains better on the available matched dense enwik8 result;
- the 99M MemoryFeedback point saturates at the current data/schedule budget;
- language scaling evidence is single-seed;
- stale latest-value recall remains unresolved;
- the controlled-memory tasks are synthetic;
- the v2 1B model is not quality-trained;
- bounded recurrent state does not imply lower total cost on every workload;
- no production fused kernel is included;
- no broad downstream reasoning result is claimed.

