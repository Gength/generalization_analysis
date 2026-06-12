# What changed in v25 / v26 — technical summary

*Plain-language summary for the project team. Full technical spec: [`Method_GenShadow.md`](Method_GenShadow.md); benchmark integration: [`BenchmarkDesign_v2.md`](BenchmarkDesign_v2.md).*

## TL;DR

v25 and v26 are small, additive extensions of the v23/v24 design — no redesign, same
Good–Turing core, same N=6, same backoff machinery, same deduplication, same protocol and
runtime, drop-in compatible interface. v25 **completes the Katz backoff idea that v2.3
introduced**, extending it from the mutation *rate* to the mutation *proposal*. v26 adds an
acceptance-based reading of the score and a switchable sampling mode. Benchmarked against
the k-fold ground truth on two datasets, the extensions cut the calibration error to about
a third and resolve a ranking disagreement on BPI 2013 — with v24 results staying in the
benchmark unchanged as the baseline for comparison.

---

## Background: the mutation mechanism has two halves

When the shadow-log walker is at some context, two separate questions decide a mutation:

1. **How often should a never-seen continuation happen here?** (the mutation *rate*)
2. **When it happens, which activity gets inserted?** (the mutation *proposal*)

**Half 1 was solved in v2/v2.3** — and solved well. The Dynamic Backoff Smoothing
mechanism (resolve the deepest context with enough support, fall back otherwise) prevents
sparse high-order contexts from producing falsely inflated mutation rates. Without it,
Good–Turing at N=6 would mutate constantly and the shadow log *would* be alphabet noise.
This fix is fully intact in v24 and in everything after it — v25/v26 build directly on it.

**Half 2 had simply never been on anyone's plate.** From v1 through v24, the inserted
activity is drawn uniformly from the whole alphabet (`random.choice(alphabet)`). That was a
reasonable maximum-entropy placeholder while the rate side was being engineered. It only
became visible as the binding constraint once we (a) adopted the strict construct definition
("shadow traces must be plausible *valid* future behavior") and (b) measured how large the
mutated share actually is on sparse logs: **~50 % of shadow traces on Sepsis** carry at
least one mutation (vs. ~4 % on BPI 2017, where the original calibration was done). On logs
like Sepsis, the proposal distribution is therefore not a footnote — it shapes the score.

## What v25 does

One change: the mutation proposal now uses **the same backoff principle as everything
else**. When a mutation fires, the new activity is drawn from the next-shorter context's
continuation distribution, restricted to activities never seen after the current context —
*"this exact 6-step history never saw X next, but similar shorter histories did."* That is
precisely how Katz backoff redistributes unseen probability mass in language models, so v25
is the v2.3 backoff design carried to its logical conclusion. Injected events are now
plausible-but-novel by construction.

v25 also adds two transparency counters (no behavior change): `duplicates_kept` reports if
the dedup retry cap (100) was ever exhausted (it never triggers on D1/D2 — now we can
*prove* that instead of assuming it), and `truncated_traces` reports walks cut by the length
cap.

## What v26 does

Three additions on top of v25:

1. **Acceptance rate** (`gen_accept`): besides mean replay fitness (which grants partial
   credit), report the fraction of shadow traces the model replays *perfectly*. Fitness
   says "how smoothly does it replay"; acceptance says "does it actually accept". Reported
   alongside — nothing is replaced. (This also gives the new Trace-Model pole an exact 0.0
   and the Flower Model an exact 1.0 — clean bounds; see `BenchmarkDesign_v2.md`.)
2. **Data-driven length cap**: `2 × longest observed trace` instead of the fixed 100
   (Sepsis has traces up to length 185).
3. **`successor_weighting` switch**: `'log'` keeps the v2.1 ln-damped sampling — a
   deliberate design choice that up-weights rare paths, retained as the stress-test mode —
   and `'mle'` samples proportionally to raw frequencies, so the shadow log represents the
   *expected* future and uses the same statistics as the Good–Turing estimate. The
   trade-off between the two modes only became measurable now that the R1 ground-truth
   baselines exist; the data (below) favors `'mle'` for the headline score.

**Unchanged throughout:** `p_unseen` estimation (so the v2.3 rate fix and the N=6
calibration carry over), backoff resolution, deduplication, the 5×1000 protocol, seeding,
runtime (within ~6 %), and the `evaluate_miner` interface.

## The evidence (vs. variant-based 5-fold CV ground truth, R1)

| | D1 Sepsis (4 seeds) | | D2 BPI 2013 (2 seeds) | |
|---|---|---|---|---|
| **Version** | **MAE ↓** | **rank corr.** | **MAE ↓** | **rank corr.** |
| v24 | 0.077 | 1.00 | 0.047 | 0.943 |
| v25 / v26-log | 0.067 | 1.00 | 0.045 | 0.943 |
| v26-mle | **0.024** | 1.00 | **0.021** | **1.000** |

On D2, R1 ranks the Heuristics model above Inductive-Infrequent (0.996 vs 0.982); the
ln-damped modes invert this pair, the mle mode matches R1 exactly. The Flower model scores
1.0 under every version (correct under the construct definition — `Method_GenShadow.md`
§1), discriminative spread increases, and all numbers are stable across seeds (std ≤ 0.001).

## Impact on the lock-in

Minimal by design. v24 remains in the benchmark unchanged (method M1), so every existing
number stays valid and comparable; v25/v26 enter as additional methods (M1d–M1f) next to
the existing ablations. The only open decision is which configuration is the *headline*
(proposal: M1f = v26-mle — best calibration and rank fidelity on both datasets at identical
cost; decision log in `BenchmarkDesign_v2.md`, pending team agreement). Re-running the full
M1 family on a dataset is one command and ~5 min:

```
$env:PYTHONUTF8 = '1'
uv run python benchmark/run_m1_family.py --dataset D1
```
