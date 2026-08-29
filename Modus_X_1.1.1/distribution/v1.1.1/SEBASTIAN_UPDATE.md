Subject: Modus_X v1.1.1: reproducible component attribution and updated comparison package

Hi Sebastian,

Thank you again for the xLSTM comparison advice. I have now released Modus_X
v1.1.1 with a more complete evidence package.

The new addition is a three-seed component ablation on the recovered balanced
key-value recall and same-key-overwrite protocol. MatrixOnly retains
`96.992 +/- 0.427%` at length 2048 without overwrite, while VectorOnly remains
near 32-way chance (`3.100 +/- 0.109%`). The raw archive contains all 24
seed-level outputs, commands, and runner code. I have framed this narrowly as
matrix-memory attribution on a controlled protocol.

The fair language-modeling story remains mixed and visible: official Mamba is
stronger on our matched 80M enwik8 dense test, while Modus_X exceeds the tested
official xLSTM configuration. I am now using the NXAI scaling-laws work as the
reference for the next fair scaling plan.

Release: https://doi.org/10.5281/zenodo.20923248
Code and evidence: https://github.com/sanyamChaudhary27/Modus_X

If you have one pointer on the minimum xLSTM-style scaling/evaluation protocol
that would make a 1B Modus_X comparison genuinely useful to your team, I would
be grateful.

Best,
Sanyam
