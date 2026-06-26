# Title

Modus_X v1.1.1: three-seed matrix-memory attribution, raw results, and a
bounded comparison story

# Post

I am releasing Modus_X v1.1.1, an attention-free causal architecture that
combines a selective recurrent vector stream with a delta-rule associative
matrix-memory stream.

The core addition is a three-seed component ablation on a recovered balanced
key-value associative-recall protocol. We compared ScalarPM, VectorLeanPM,
MatrixOnly, and VectorOnly with the same task generator, seeds, data counts,
optimizer, epoch budget, and length sweep. Parameter counts are reported
explicitly rather than hidden.

At length 2048 with no overwrite:

- MatrixOnly: `96.992 +/- 0.427%`
- VectorLeanPM: `96.758 +/- 0.317%`
- ScalarPM: `96.383 +/- 0.496%`
- VectorOnly: `3.100 +/- 0.109%`

With 50% same-key overwrite:

- MatrixOnly: `87.625 +/- 0.745%`
- VectorLeanPM: `87.750 +/- 0.763%`
- ScalarPM: `88.108 +/- 0.447%`
- VectorOnly: `3.308 +/- 0.506%`

There are 32 possible values, so chance is `3.125%`. The claim is deliberately
narrow: the delta-rule matrix stream carries the observed controlled binding
and overwrite behavior. VectorOnly is not sufficient in this protocol. This
does not prove natural-language long-context recall, a general vector-router
advantage, or universal model superiority.

The language-modeling evidence is also mixed and reported as such. On the
matched 80M enwik8 dense test, official Mamba reaches `1.34578` BPC and
Modus_X reaches `1.38418`; Modus_X exceeds the tested official xLSTM
configuration at `1.41962`. I would value critique on the next falsifiable
language-level experiment that best tests whether this associative capability
scales usefully.

Repository, raw results, runner scripts, paper, and release archive:
https://github.com/sanyamChaudhary27/Modus_X
