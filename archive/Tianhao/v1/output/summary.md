# Method 1 — Pick-One-Out Generalization Results

**Dataset:** BPI Challenge 2017 (15,930 unique variants)
**PM4Py Baseline:** 0.9485 (built-in generalization, computed once on full log with Inductive Miner)

| Sampling | # Variants | Miner | Baseline (PM4Py) | Method 1 (Pure) | Method 1 (Joint) | Pure–Joint Gap | Runtime |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **2%** | 319 | Alpha Miner | 0.9485 | 0.3905 | 0.3904 | 0.0001 | 73 s |
| | | Heuristics Miner | 0.9485 | 0.9608 | 0.9612 | 0.0004 | 131 s |
| | | Inductive Miner (IM) | 0.9485 | 1.0000 | 1.0000 | 0.0000 | 1,427 s |
| | | Inductive Miner (IMf) | 0.9485 | 0.9772 | 0.9773 | 0.0001 | 1,069 s |
| **10%** | 1,593 | Alpha Miner | 0.9485 | 0.3922 | 0.3920 | 0.0001 | 238 s |
| | | Heuristics Miner | 0.9485 | 0.9594 | 0.9598 | 0.0003 | 523 s |
| | | Inductive Miner (IM) | 0.9485 | 1.0000 | 1.0000 | 0.0000 | 7,229 s |
| | | Inductive Miner (IMf) | 0.9485 | 0.9776 | 0.9777 | 0.0001 | 5,479 s |
| **20%** | 3,186 | Alpha Miner | 0.9485 | 0.3912 | 0.3911 | 0.0001 | 1,095 s |
| | | Heuristics Miner | 0.9485 | 0.9594 | 0.9598 | 0.0004 | 2,579 s |
| | | Inductive Miner (IM) | 0.9485 | 1.0000 | 1.0000 | 0.0000 | 40,969 s |
| | | Inductive Miner (IMf) | 0.9485 | 0.9773 | 0.9774 | 0.0001 | 19,086 s |

**Notes:**
- Baseline is PM4Py's built-in generalization metric — computed once on the full log using Inductive Miner; it is independent of sampling rate and miner choice.
- All runs completed. IM consistently achieves perfect 1.0000 across all sampling rates; IMf stabilizes around 0.9773–0.9776.
- The Pure–Joint gap remains extremely small (< 0.0005) across all runs — consistent with the v1→v2 diagnosis in `Method1Log.md`.

# Analysis
+ Empirical testing across escalating sampling rates confirmed that the arithmetic-logarithmic joint weighting collapses into a constant multiplier, neutralizing its discriminative power.

+ IM (Flower/Underfitting): Scores a perfect 1.000 across the board. Because v1 metric currently lacks the structural penalty, IM's massive, hyper-permissive models easily swallow any variant you throw at them.
+ The Baseline's Blindness: The PM4Py baseline sits at a static 0.9485, completely failing to capture the catastrophic rigidity of the Alpha Miner. v1 metric, even in its flawed state, already provides a much more realistic assessment of Alpha Miner than the baseline.