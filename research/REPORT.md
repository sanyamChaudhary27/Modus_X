# Modus_X --- Fixed-Budget Parameter Reallocation Research Report

**Status:** Engineering freeze complete; final short-budget dense
control may be appended if/when available.\
**Final candidate:** C1 (`q_rank=128`, `k_rank=128`)\
**Canonical dense budget:** 47,437,768 parameters\
**C1:** 47,440,328 parameters

## 1. Executive Summary

Modus_X investigates whether neural-network capacity can be
**reallocated rather than merely removed**. The hypothesis is that
highly compressible Q/K projections contain enough redundancy to permit
low-rank factorization, allowing the recovered parameter budget to be
reinvested into hidden and recurrent capacity while keeping the total
model size approximately fixed.

The research question is:

> Can capacity be removed from highly compressible Q/K projections and
> placed elsewhere while keeping approximately the same parameter
> budget, and does that reallocation improve language-modeling BPC?

The experimental flow was:

``` text
Exact census
   ↓
Structural matrix analysis
   ↓
SVD compressibility
   ↓
Functional BPC sensitivity
   ↓
Fixed-budget allocation search
   ↓
Exact parameter recount
   ↓
Functional smoke tests
   ↓
4M screening / 10M TPU runs
   ↓
Candidate comparison
   ↓
Native C1 integration
   ↓
Engineering freeze
```

The experiments establish that Q/K can be strongly compressed and that
the recovered budget can be moved into hidden/recurrent capacity. Among
the three final candidates, C1 achieved the best held-out test BPC.

**C1 test BPC: 2.508946329389158**

However, this does **not** by itself prove that C1 beats the original
dense baseline, because the original dense baseline used a different and
much larger training protocol. A matched dense control is therefore
treated separately.

------------------------------------------------------------------------

## 2. Core Research Idea

The project is based on a fixed-budget allocation principle:

``` text
47.44M parameter budget
        │
        ▼
Find redundant capacity
        │
        ▼
Compress Q/K
        │
        ▼
Recover parameters
        │
        ▼
Increase hidden/recurrent capacity
        │
        ▼
Remain ≈47.44M parameters
        │
        ▼
Measure BPC
```

The objective is not model compression in the usual sense:

``` text
47.44M → smaller model
```

It is:

``` text
47.44M → different parameter distribution
       → potentially better use of capacity
```

------------------------------------------------------------------------

## 3. Original Architecture

The canonical architecture contained exactly:

``` text
47,437,768 parameters
```

The main architecture includes:

-   vocabulary size 256
-   embedding dimension 512
-   hidden dimension 1536
-   axial/residual dimension 512
-   12 layers
-   8 attention heads
-   Mamba/recurrent state dimension 512
-   memory/archive pathway
-   memory feedback
-   router
-   auxiliary/deep-supervision pathway
-   future prediction head

The major repeated 512×512 projection families analyzed were:

``` text
m_wq
m_wk
m_wv
m_w_read
m_w_out
```

`m_proj_w` is a larger `[12, 512, 1024]` projection family.

------------------------------------------------------------------------

## 4. Why Q/K Were Chosen

The project first measured matrix structure instead of arbitrarily
shrinking dimensions.

Approximate effective/stable rank measurements showed:

  Matrix     Effective/stable rank   Alternate measure
  -------- ----------------------- -------------------
  Q                          7.807              45.943
  K                          8.518              49.724
  V                         30.864             185.582
  Read                      14.989             170.335
  Wout                       7.031             155.162

At rank 128, approximate retained spectral energy was:

  Matrix     Rank-128 energy
  -------- -----------------
  Q                   96.46%
  K                   96.63%
  V                   80.94%
  Read                80.59%
  Wout                78.90%

The important result was the comparative gap: Q/K were much more
tolerant of low-rank approximation than V/read/output under the tested
criteria.

------------------------------------------------------------------------

## 5. Functional Compression Evidence

SVD energy alone is not sufficient, so model-level BPC sensitivity was
also measured.

Representative BPC changes:

  Intervention      BPC change
  --------------- ------------
  Q rank 128         +0.000060
  K rank 128         +0.000135
  Wout rank 128      +0.006460
  Q rank 64          +0.002300
  K rank 64          +0.000774
  Wout rank 64       +0.028459

This supported the decision to take capacity from Q/K rather than
aggressively compressing other projection families.

The methodology therefore became:

``` text
Spectral redundancy
        +
Functional sensitivity
        ↓
Select reallocation target
```

------------------------------------------------------------------------

## 6. Fixed-Budget Search

The canonical target was:

``` text
47,437,768 parameters
```

The reallocation search separated:

``` text
Frozen components = 2,899,456 parameters
```

and searched the remaining allocation space using the calibrated
parameter equation:

``` text
P = 12011080
  + (36912 × r)
  + (30744 × n)
  + (12300 × h)
  + (12 × f × r)
  + (6144 × f)

f = min(32, r, n)
```

Candidates were subsequently checked using the actual parameter tree
rather than relying only on analytical estimates.

The canonical model passed an exact recursive recount, and 25 candidate
architectures were instantiated and verified during the allocation
search.

------------------------------------------------------------------------

## 7. PB1: Q/K Factorization

PB1 factorized **only Q and K**.

For dense Q/K:

``` text
Wq ∈ R^(512×512)
Wk ∈ R^(512×512)
```

The C1 factorization is:

``` text
Wq ≈ Aq Bq

Aq ∈ R^(512×128)
Bq ∈ R^(128×512)

Wk ≈ Ak Bk

Ak ∈ R^(512×128)
Bk ∈ R^(128×512)
```

The actual forward path computes:

``` text
Q = (H Aq) Bq
K = (H Ak) Bk
```

rather than reconstructing dense Q/K matrices.

This distinction is important: the intervention is a genuine
architectural low-rank implementation, not merely checkpoint
compression.

------------------------------------------------------------------------

## 8. Final Candidate Architectures

  -------------------------------------------------------------------------------
  Candidate       Q rank     K rank     Hidden      Mamba   Parameters     Budget
                                                    state                   error
  ----------- ---------- ---------- ---------- ---------- ------------ ----------
  C1                 128        128       1664        608   47,440,328     +2,560

  C2                  64         64       2048        640   47,441,864     +4,096

  C3                 128         64       1536        640   47,440,840     +3,072
  -------------------------------------------------------------------------------

All three are effectively fixed-budget relative to the 47.44M target.

### C1

``` text
Q rank        = 128
K rank        = 128
hidden_dim    = 1664
mamba_state   = 608
ax_res        = 512
```

### C2

``` text
Q rank        = 64
K rank        = 64
hidden_dim    = 2048
mamba_state   = 640
ax_res        = 512
```

### C3

``` text
Q rank        = 128
K rank        = 64
hidden_dim    = 1536
mamba_state   = 640
ax_res        = 512
```

------------------------------------------------------------------------

## 9. PB1-B Functional Smoke

All three candidates passed the functional gate.

  -----------------------------------------------------------------------
  Candidate        Q reconstruction   K reconstruction       Final logits
                             metric             metric        relative L2
  -------------- ------------------ ------------------ ------------------
  C1                     0.61362884         0.61382307         1.08748608

  C2                     0.78987367         0.78962360           1.165422

  C3                     0.61362884         0.78962360           1.120252
  -----------------------------------------------------------------------

The gate checked:

-   exact parameter count
-   expected tensor shapes
-   dense Q/K removal
-   forward execution
-   finite outputs
-   direct factorized path
-   DeepSupervision outputs

------------------------------------------------------------------------

## 10. PB1-C Final TPU Protocol

C1/C2/C3 were trained under the same final protocol on TPU v5e-8:

``` text
Devices:              8 TPU devices
Batch:                1
Sequence length:      512
Loss tail:            512
Train sequence:       32
Characters/step:      32
Target:               10,000,000 chars
Steps:                312,500
Checkpoint interval:  31,250 steps
Eval chunks:          128
Eval batch:           8
Learning rate:        0.0002
Weight decay:         0.0001
Auxiliary weight:     0.05
Future target weight: 0.5
Future target count:  2
Optimizer:            AdamW
Global gradient clip: 1
Auxiliary layer:      6
Seed:                 1
True low-rank matmul:  true
Recurrent chunk:      32
Gradient horizon:     32
```

Training semantics:

``` text
32-token-truncated-BPTT-continuous-forward-state
state_carry_across_updates = true
full_sequence_backward      = false
```

The identical protocol makes the C1/C2/C3 comparison substantially
cleaner.

------------------------------------------------------------------------

## 11. Final TPU Results

### C1

``` text
Final validation BPC = 2.381268139964033
Final test BPC       = 2.508946329389158
Best validation      = 2.2113771206031867 @ 9M chars
Runtime              ≈ 18,807 s ≈ 5.22 h
```

Validation trajectory:

    Characters   Validation BPC
  ------------ ----------------
            1M          2.61857
            2M          2.56001
            3M          2.46210
            4M          2.36367
            5M          2.35147
            6M          2.29700
            7M          2.23656
            8M          2.35412
            9M          2.21138
           10M          2.38127

### C2

``` text
Final validation BPC = 2.394753600587825
Final test BPC       = 2.551723832112241
Best validation      = 2.215498340865812 @ 9M chars
Runtime              ≈ 18,423 s ≈ 5.12 h
```

### C3

``` text
Final validation BPC = 2.3722945044497235
Final test BPC       = 2.5548515300986634
Best validation      = 2.221576272240798 @ 7M chars
Runtime              ≈ 19,928 s ≈ 5.54 h
```

### Comparison

  Metric                    C1           C2           C3
  --------------- ------------ ------------ ------------
  Q/K                  128/128        64/64       128/64
  Hidden                  1664         2048         1536
  Mamba                    608          640          640
  Parameters        47,440,328   47,441,864   47,440,840
  Final val BPC         2.3813       2.3948   **2.3723**
  Test BPC          **2.5089**       2.5517       2.5549

**C1 is selected on held-out test BPC.**

C3's better final validation score is a useful secondary observation,
but it did not translate into the best test result.

------------------------------------------------------------------------

## 12. C1 Native Owner Architecture

The final owner architecture is:

``` text
Modus_X_C1_DeepSupervision
```

It contains:

-   current hidden-state stream
-   archive hidden-state stream
-   recurrent Mamba-style state
-   memory feedback
-   vector router
-   DeepSupervision / auxiliary path
-   future prediction head
-   native factorized Q/K

C1 factor tensors:

``` text
m_wq_A = (12, 512, 128)
m_wq_B = (12, 128, 512)

m_wk_A = (12, 512, 128)
m_wk_B = (12, 128, 512)
```

Dense `m_wq` and `m_wk` are absent from the C1 owner architecture.

Exact owner validation:

``` text
PARAMS=47440328
DENSE_Q=False
DENSE_K=False
```

Expected DeepSupervision outputs were produced and all outputs were
finite.

------------------------------------------------------------------------

## 13. Gradient Validation

A dedicated gradient test confirmed optimization through every factor:

``` text
Q_A_GRAD = 0.0612347163
Q_B_GRAD = 0.0625220984
K_A_GRAD = 0.0610094406
K_B_GRAD = 0.0624606423
ALL_FINITE = True
```

Therefore the low-rank matrices are genuinely trainable parts of the
architecture.

------------------------------------------------------------------------

## 14. Vector Router Architecture Finding

An important architecture-consistency issue was discovered during
integration.

The research metadata had indicated:

``` text
vector_router = False
```

but the existing `Modus_X_MemoryFeedbackArchive` construction path
explicitly enables the vector router and supplies a router hidden size.

This behavior was already part of the actual canonical
47,437,768-parameter architecture.

For C1:

``` text
vector_router = True
```

is required to reach the intended:

``` text
47,440,328 parameters
```

Disabling it produced:

``` text
47,237,972 parameters
```

Therefore the final C1 architecture deliberately retains the vector
router.

This finding is important because it prevents a silent
metadata/implementation mismatch in future reproduction.

------------------------------------------------------------------------

## 15. Research Trainer Freeze

The validated PB1-C research trainer was frozen as:

``` text
research/parameter_allocation/pb1_c_train_final_research.py
```

The frozen trainer hash was:

``` text
EDF8412A4F194555D7AA49607F08AB20CBFBE9B7BB71BC4587DA5A4679BA33B7
```

The native final C1 trainer removes candidate selection from the normal
interface and fixes C1 as the final architecture.

------------------------------------------------------------------------

## 16. Implementation Validation

The C1 owner integration passed:

``` text
Python compilation
Exact parameter recount
Dense Q removal
Dense K removal
Q factor shape validation
K factor shape validation
Forward pass
Finite outputs
DeepSupervision outputs
Gradient flow
Short TPU implementation smoke
```

The short TPU smoke produced finite outputs and the exact C1 parameter
count. It was **not** used as a language-model performance result.

------------------------------------------------------------------------

## 17. Additional Experimental Findings

### 17.1 Parameter allocation is more informative than simple compression

The project demonstrates a different framing of parameter efficiency.

Instead of:

``` text
compress → smaller model
```

the useful sequence is:

``` text
compress → recover budget → reinvest budget
```

This creates a parameter-allocation problem.

### 17.2 Spectral and functional evidence should be combined

SVD energy retention is only a structural signal.

The project strengthened the decision by combining:

``` text
matrix spectral analysis
+
model-level BPC sensitivity
```

before constructing final candidates.

### 17.3 Moderate compression appears more promising than maximum compression

C2 used rank 64 for both Q and K and allocated more capacity elsewhere,
but it did not produce the best test BPC.

C1 retained rank 128 for both and won on test.

This suggests that aggressive compression can remove useful Q/K capacity
even when substantial spectral redundancy exists.

### 17.4 Q and K may have asymmetric capacity requirements

C3 used Q=128 and K=64.

Its best final validation BPC suggests that asymmetric allocation is
worth studying further, even though C3 did not win the held-out test
comparison.

A future study could sweep:

``` text
Q rank × K rank
```

rather than assuming equal ranks.

### 17.5 Validation can be noisy

C1 and C2 reached their best validation scores at 9M characters; C3
reached its best at 7M.

The final 10M validation checkpoint was not the best checkpoint for any
candidate.

Future experiments should define checkpoint selection rules before
training and evaluate the selected checkpoint on an untouched test set.

------------------------------------------------------------------------

## 18. Dense Control and Causal Attribution

The original dense baseline achieved approximately:

``` text
1.465 BPC
```

but was trained using a different protocol and a much larger character
budget.

Therefore:

``` text
Original dense baseline
        ≠
matched short-budget dense control
```

A short dense control using the same protocol is useful to isolate the
effect of architecture.

If the pending 4M dense control completes, it should be added here as:

``` text
Matched dense control:
Characters: 4M
Protocol: same short-budget C1/C2/C3 protocol
Validation BPC: [insert measured value]
Test BPC: [insert measured value]
```

This result should be described as a **matched short-budget control**,
not as a replacement for the original long-run dense baseline.

------------------------------------------------------------------------

## 19. Scientific Claims

### Supported

1.  Q/K projections are empirically highly compressible in the studied
    Modus_X architecture.
2.  Moderate Q/K low-rank factorization can be implemented as a native
    architectural change.
3.  The recovered Q/K capacity can be reinvested into hidden/recurrent
    capacity while keeping the total model near 47.44M parameters.
4.  C1/C2/C3 are valid approximately fixed-budget architectures.
5.  C1 achieved the best held-out test BPC among the three final
    reallocation candidates.
6.  C1 has correct forward execution and gradient flow.

### Not yet established

The experiments do **not** establish that:

> C1 universally outperforms the original dense architecture.

The original dense result is confounded by a different training
budget/protocol.

A matched dense control and, ideally, multiple seeds are required for a
stronger causal performance claim.

------------------------------------------------------------------------

## 20. Limitations

-   One seed was used for the final candidate comparison.
-   The original dense baseline was trained under a different
    compute/training regime.
-   The final candidate experiments used 10M characters and may not
    represent full convergence.
-   Only three final allocation points were tested.
-   The complete Q-rank × K-rank × hidden/recurrent Pareto frontier was
    not exhaustively searched.
-   No multi-seed confidence intervals were obtained.
-   A short dense control, even if completed, would still be limited by
    its smaller training budget.

------------------------------------------------------------------------

## 21. Final Architecture Snapshot

``` text
                  Modus_X_C1_DeepSupervision
                           │
                     Token Embedding
                         dim = 512
                           │
                           ▼
                ┌──────────────────────┐
                │ 12 × Modus_X Layer  │
                │                      │
                │ Hidden = 1664        │
                │ Mamba state = 608    │
                │ Archive state        │
                │ Memory feedback      │
                │ Vector router        │
                │                      │
                │ Q: 512→128→512      │
                │ K: 512→128→512      │
                └──────────┬───────────┘
                           │
                 ┌─────────┼─────────┐
                 ▼         ▼         ▼
              LM Head   Aux/DS   Future Head
                 │         │         │
                 └─────────┼─────────┘
                           ▼
                         BPC
```

------------------------------------------------------------------------

## 22. Final Conclusion

Modus_X provides evidence for a practical fixed-budget parameter
reallocation strategy.

The study did not simply shrink the model. It first identified
redundancy, then converted that redundancy into an architectural budget
that could be reinvested.

The resulting C1 architecture:

``` text
Q rank        = 128
K rank        = 128
hidden_dim    = 1664
mamba_state   = 608
total params  = 47,440,328
```

achieved:

``` text
Test BPC = 2.508946
```

and was the best held-out candidate among C1/C2/C3.

The strongest research conclusion is:

> **Q/K capacity in Modus_X can be substantially reduced through
> low-rank factorization, and the recovered capacity can be reallocated
> into hidden/recurrent capacity while maintaining an approximately
> fixed 47.44M parameter budget. Under the common 10M-character TPU
> protocol, the C1 allocation produced the best held-out test BPC among
> the tested candidates.**

The remaining question is causal:

> **Does this reallocation itself improve performance over an
> equivalently trained dense model?**

That question should be answered with the matched dense control before
claiming a definitive improvement over the dense architecture.

------------------------------------------------------------------------

## 23. Repository / Reproducibility Checklist

``` text
[✓] Exact canonical parameter census
[✓] Matrix structural analysis
[✓] Q/K SVD compressibility analysis
[✓] Functional sensitivity analysis
[✓] Fixed-budget candidate search
[✓] Exact candidate recount
[✓] PB1-B functional smoke
[✓] PB1-C C1 10M run
[✓] PB1-C C2 10M run
[✓] PB1-C C3 10M run
[✓] Native C1 owner integration
[✓] Exact C1 parameter validation
[✓] Dense Q/K removal validation
[✓] Forward validation
[✓] DeepSupervision validation
[✓] Gradient validation
[✓] Research trainer freeze
[ ] Add matched dense-control result if available
[ ] Finalize report after dense-control result
[ ] Final documentation commit / PR
```
