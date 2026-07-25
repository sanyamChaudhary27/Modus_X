# Modus_X 2.0.0: Coordinating Bounded Associative Memory with Recurrent Computation

Modus_X 2.0.0 is an experimental research release studying whether bounded
associative matrix memory can complement recurrent vector computation without
a sequence-length-growing inference cache.

The release promotes two configurations for different evidence layers.
**MemoryFeedbackArchive** is the language-modeling lead: retrieved matrix
context is compressed and gated back into the vector computation. At a fixed
102.4M-character enwik8 budget, it improves substantially from 47.44M to
81.49M parameters, while the measured 81.49M-to-99.44M interval saturates.
**CurrentArchiveDelta** is the controlled-memory lead: it separates current and
archived matrix state and demonstrates useful clean/update retrieval under a
constrained recurrent-state budget.

The release also preserves counterevidence. Official Mamba remains better on
the available matched dense enwik8 result. A Transformer with enough KV state
to retain the complete synthetic context wins the equal-memory study; the
CurrentArchive advantage appears after that cache is truncated. Stale
latest-value recall remains unresolved, the 1.1 BPC target is unmet, and no
quality-trained 1B model is claimed.

Included artifacts cover source code, benchmark protocols, dense evaluation
records, controlled-memory studies, negative architecture results, exact 1B
systems-readiness appendices, a full whitepaper, model card, provenance,
reproducibility guide, and integrity manifest.

The architectural contribution is a specific coordination pattern rather than
a claim that fast weights or delta-rule memory are individually new.
CurrentArchiveDelta maintains shared-address current and archive matrices with
independent update timescales. MemoryFeedbackArchive compresses their mixed
retrieval through a low-rank projection and gates it into the vector-stream
input before recurrence. The package places this mechanism in the context of
DeltaNet, Gated DeltaNet, Kimi Linear, Titans, and associative recurrent
memory, and reports both positive and rejected interventions.
