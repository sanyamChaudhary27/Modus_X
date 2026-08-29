# LinkedIn

Modus_X v1.1.1 is now published.

This release adds reproducible three-seed evidence for a question we needed to
answer clearly: where does Modus_X's controlled associative-recall capability
actually come from?

On a balanced key-value recall protocol at length 2048, MatrixOnly retains
`96.99%` recall while VectorOnly is near the `3.125%` chance level. Under 50%
same-key overwrite, matrix-based variants remain around `87-88%`. The released
raw archive contains all 24 seed-level outputs, plus the exact runners and
aggregate results.

The broader picture remains deliberately mixed. On matched 80M enwik8 dense
evaluation, official Mamba is still stronger (`1.34578` BPC versus Modus_X
`1.38418`), while Modus_X exceeds the tested official xLSTM configuration
(`1.41962`). The point is not a one-number victory. It is evidence for a
different capability profile: constant recurrent state and strong controlled
associative binding, with the matrix stream now directly attributed.

Zenodo: https://doi.org/10.5281/zenodo.20923248
Code and evidence: https://github.com/sanyamChaudhary27/Modus_X

#AIResearch #MachineLearning #LLM #StateSpaceModels #OpenScience

---

# X

Modus_X v1.1.1 is out: reproducible 3-seed matrix-memory attribution.

At L=2048 balanced KV recall: MatrixOnly 96.99%, VectorLean 96.76%,
VectorOnly 3.10% (32-way chance: 3.125%). Under 50% same-key overwrite,
matrix-based variants stay around 87-88%.

Honest counterpoint: official Mamba still wins matched 80M enwik8 BPC.
Zenodo: https://doi.org/10.5281/zenodo.20923248
Raw outputs + runners: https://github.com/sanyamChaudhary27/Modus_X
