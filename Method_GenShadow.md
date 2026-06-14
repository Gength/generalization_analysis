# Gen_shadow — Method Specification

> **Status: authoritative.** This document describes the generalization metric as currently intended.
> Where it conflicts with `Method2Log.md` or framing in `BenchmarkDesign.md` (flower model expectations,
> Gen_struct motivation), **this document takes precedence** — those passages reflect an earlier,
> contested design stance (see §8).
> Code: `HybridGen/algorithm/v24.py`. The package name "HybridGen" is legacy (see §9).

---

## 1. What this metric measures — and what it deliberately does not

**Definition.** Generalization is the degree to which a discovered process model accepts
*future, valid* behavior of the underlying process that is not present in the recorded event log.
The event log is a sample; the model should describe the process, not the sample.

**Strict construct separation.** Generalization is one of four orthogonal quality dimensions
(fitness, precision, generalization, simplicity), and this metric measures *only* generalization:

- **Generalization is not precision.** Detecting over-permissiveness (accepting *invalid* behavior)
  is the precision dimension's job. Folding a permissiveness penalty into a generalization score
  makes any mid-range score ambiguous — one cannot tell "rejects future valid behavior" from
  "accepts invalid behavior" — and destroys the diagnostic value of the quality quadrant.
  You do not redefine recall just because a degenerate model achieves maximum recall;
  you report precision next to it.
- Metrics that mix the two measure a different construct. AVATAR's score, for instance, is the
  harmonic mean of fitness and precision against generated traces — literally an F-measure.
  An F-measure is a useful *aggregate*, but it must not be labeled "generalization".

**Consequence — the flower model scores 1.0, and that is correct.** The flower model accepts all
future behavior of any process over its alphabet; its generalization is maximal *by definition*.
It is a useless model because of its precision and simplicity, not its generalization.

**Flower model as construct-purity litmus.** This gives the benchmark a sharp interpretation:
a metric with pure "accepts-future-valid-behavior" semantics must assign the flower model ≈ 1.0.
A lower flower score reveals contamination by precision or structural notions.
Observed on Sepsis (D1): Gen_shadow 1.00, Bootstrap 1.00, k-fold CV 1.00 (pure);
PM4Py built-in 0.91, SpeciAL 0.80, AVATAR 0.40 (contaminated / different construct).

**Intended usage.** Gen_shadow is to be read *alongside* precision (and fitness, simplicity),
never alone. A model with Gen ≈ 1 and precision ≈ 0 is diagnosed as over-permissive —
by the precision column, which is exactly where that information belongs.

---

## 2. Score definition

```
Gen  =  Gen_shadow  =  mean trace-level token-replay fitness of a synthetic
                       "shadow log" replayed on the model under evaluation,
                       averaged over 5 independent generation runs (mean ± std)
```

There is **no structural term**. The historical formula
`Gen_total = w·Gen_shadow + (1−w)·Gen_struct` survives in the code signature with the
structural component defunct (`w` effectively 1); see §8.

---

## 3. Generation algorithm (v2.4)

The shadow log is produced by a stochastic trace generator learned **from the log only** —
the model under evaluation is consulted exclusively at replay time, keeping the probe
independent of the artifact being judged.

**3.1 Variant compression.** The log is reduced to its variants with case counts;
all statistics are frequency-weighted. Preprocessing cost is independent of case count.

**3.2 Variable-order context statistics.** For every order `n = 1..max_n` (default 6),
record per observed n-gram context `s`: the frequency of each successor activity, and the
frequency with which `s` occurs at trace end.

**3.3 Katz-style backoff.** At each step the walker resolves its decision context by trying
the deepest order first: the longest suffix of the partial trace whose total successor count
meets the support threshold `τ = 5` (and whose unseen-mass estimate is informative, i.e. < 1)
is used; otherwise the order is reduced, falling back ultimately to `n = 1`.
`max_n` is therefore an **upper bound**, not a fixed operating point — sparse logs
degrade gracefully to lower effective orders.

**3.4 Good–Turing novelty mass.** For the resolved context `s`:

```
p_unseen(s) = N1(s) / N(s)
```

where `N(s)` is the total number of observed continuations of `s` and `N1(s)` the number of
successor activities observed exactly once. This is the estimated probability that the next
event is something never observed after `s`.

**3.5 Step decision.**
- With probability `p_unseen(s)`: **mutate** — append an activity the context has not produced
  before. Proposal distribution: v2.4 draws uniformly from the alphabet; **v2.5 draws
  Katz-consistently** from the deepest lower-order successor distribution restricted to
  activities unseen at the resolved order (see §6).
- Otherwise: **exploit** — sample a successor with probability ∝ `ln(f + 1)`.
  The logarithmic damping prevents dominant mainstream paths from drowning out rare but
  legitimate continuations.

**3.6 Context-aware termination.** Trace end is decided by `p_end(s) = #(s at trace end) / #(s)`,
resolved with the same backoff scheme. Conditioning termination on the n-gram context (v2.3+)
rather than the last activity alone fixes premature termination when "ending activities"
are generated early in a trace.

**3.7 Deduplication.** Every generated trace is checked against the set of original variants;
exact copies are discarded and regenerated (≤ 100 retries). The shadow log contains only
sequences that do not occur in the log — every shadow trace is unseen behavior by construction.

**3.8 Parameters (fixed across all datasets; never tuned per dataset).**

| Parameter | Value | Rationale |
|---|---|---|
| `max_n` | 6 | Empirical mutation peak on BPI 2017 (N-gram sweep, `Method2Log_Geng.md`); backoff protects sparser logs |
| `safe_threshold` τ | 5 | Minimum support for a context to be trusted |
| `num_shadow_traces` | min(1000, \|L\|) | Stable means; enough mutated traces for stratified analysis |
| `iterations` | 5 | Tight mean ± std |
| `seed` | 42 | Reproducibility |

---

## 4. What the shadow log represents

**Recombination dominance is a feature.** At N=6 on BPI 2017, ~96 % of shadow traces contain
no mutation event: they are novel recombinations of observed local patterns. This matches what
future behavior empirically looks like: on Sepsis, TLRA = 0.19, i.e. ~81 % of new cases arrive
as never-before-seen variants composed of familiar fragments. A probe dominated by unseen
recombinations, with a thin tail of genuinely novel events, is a reasonable sample of
"future valid traces".

**The mutation tail is the estimated unseen-event mass.** Good–Turing tells us *how much*
probability future behavior places on never-observed continuations per context. The mutation
mechanism injects exactly that mass. What it cannot tell us is *what* those unseen events are —
which is the open question in §6.

---

## 5. Validation methodology

**Ground truth: variant-based hold-out (R1, R2).** 5-fold cross-validation and
leave-one-variant-out, both **variant-based** (all traces of a variant stay on one side of the
split; the discovery algorithm never sees the held-out sequences). These are the most literal
operationalization of the definition in §1: discover on a subset of behavior, test whether
genuinely unseen-but-real traces are accepted. Gen_shadow tries to approximate this signal at
a fraction of the cost — R2 needs one discovery run per variant.

What hold-out *cannot* test: behavior beyond the log's variant set (valid recombinations and
novel events that were never recorded). That residual space is exactly what the shadow log's
generator reaches into — the argument for Gen_shadow over hold-out is therefore both cost
*and* coverage.

**Agreement criteria — use all three, they answer different questions:**

| Criterion | Question answered |
|---|---|
| Spearman ρ vs R1 | "Do the metric and hold-out share the same *idea* (same ordering of models)?" |
| MAE / calibration vs R1 | "Are the absolute values trustworthy, not just the ordering?" |
| Discriminative spread (max−min over real miners) | "Does the metric actually separate good from bad models?" |

**R3, the random floor — purpose and reading.** R3 replays uniformly random traces and defines
the *minimum*: a metric scoring a model below what random garbage achieves is broken. R3 is not
a competitor and its rank agreement with R1 carries no validation weight; on permissiveness-spread
miners, *any* replay probe ranks them similarly. The floor's softness (random traces reach 0.77
token-replay fitness on Inductive-strict due to partial credit) suggests a complementary strict
floor: the fraction of random traces that replay *perfectly* (near 0 for real models, 1.0 for
the flower model).

**D1 snapshot (Sepsis, quick calculation from `benchmark/results/configs/`, flower excluded):**

| Method | Spearman vs R1 | MAE vs R1 | Spread |
|---|---|---|---|
| Gen_shadow (M1) | 1.00 | ~0.076 | 0.67 |
| Bootstrap (M6) | 1.00 | ~0.015 | 0.74 |
| PM4Py (M2) | −0.43 | ~0.17 | 0.08 |
| SpeciAL (M7) | −0.37 | ~0.21 | 0.26 |
| AVATAR (M5) | 0.37 | ~0.24 | 0.41 |
| Random floor (R3) | 1.00 | ~0.28 | 0.49 |

Reading: Spearman alone cannot separate Gen_shadow from the floor (both 1.00) — MAE and spread
do (0.076 vs 0.28; 0.67 vs 0.49). This is why all three criteria are reported.

---

## 6. Open design question: the mutation proposal distribution

**Current behavior (v2.4).** When the walker decides to mutate, the inserted activity is drawn
**uniformly from the entire alphabet** (`random.choice(alphabet)`). The metric then treats the
resulting trace as a regular sample of future behavior: mean fitness implicitly weights mutated
traces by their empirical share (~4 % at N=6 on BPI 2017; less on sparse logs where backoff
lowers the effective order).

**Why this is defensible.** Good–Turing provides the *amount* of unseen mass but no information
about its *shape*. Uniform is the maximum-entropy choice under total ignorance: we assert only
"something unseen happens here" and test the model's openness in the least biased way.

**Why it is in tension with §1.** A uniformly drawn activity can be process-impossible at that
position (e.g. a registration event mid-treatment). Such a trace is arguably not *valid* future
behavior — and a metric defined over valid future behavior should not reward a model for
accepting it. With uniform proposals, a small fraction of the score (bounded by the mutated-trace
share) rewards openness to noise rather than openness to plausible novelty.

**Candidate refinements — (1) and (3) are implemented in v2.5
(`HybridGen/algorithm/v25.py`), not yet benchmarked (benchmark M1 remains v2.4):**

1. **Katz-consistent proposal** *(preferred direction)*: draw the mutated activity from the
   backed-off lower-order successor distribution, restricted to activities unseen at the current
   order. This is exactly how Katz smoothing redistributes unseen mass in language models:
   "behavior plausible in a related (shorter) context, but never observed in this specific one."
   It uses the structural information we have instead of discarding it, and turns the generator
   into a proper backoff language model end to end.
2. **Global-frequency proposal**: draw proportionally to global activity frequency. Weaker than
   (1) — ignores context entirely — but removes the most implausible insertions cheaply.
3. **Keep uniform, report stratified**: the `had_mutation` flags already allow splitting the
   score into (fitness_regular, fitness_mutated). Reporting the pair instead of blending preserves
   the diagnostic separation argued for in §1 — the same logic that forbids mixing generalization
   with precision also suggests not silently blending "recombination acceptance" with
   "noise tolerance". The pair forms an *openness profile* of the model.

Options (1) and (3) compose, and v2.5 implements both: Katz-consistent proposal plus separate
`gen_shadow_regular` / `gen_shadow_mutated` reporting, along with two transparency counters —
`duplicates_kept` (dedup retry cap exhausted; v2.4 silently kept the duplicate) and
`truncated_traces` (walks cut at `max_trace_length`, which are incomplete process instances).
The mutation *rate* is untouched (`p_unseen` is identical), so the N-sweep calibration carries
over up to second-order trajectory effects.

**Impact is dataset-dependent — the "thin tail" framing only holds on dense logs.** The
mutated-trace share is ≈ 4 % at N=6 on BPI 2017, but **≈ 49 % on Sepsis** (v2.5 smoke test,
500 traces × 3 iterations): on sparse logs the backoff resolves at low orders where singleton
continuations are common, so the per-step Good–Turing mass is high and compounds over ~14
decisions per trace. On such logs the proposal distribution materially shapes the score.
v2.4 → v2.5 on Sepsis: Inductive-strict 0.959 → 0.965, Heuristics 0.833 → 0.846; under the
Katz-consistent proposal, mutated traces replay slightly *better* than regular ones
(IM: 0.970 vs 0.959) — injected events are now plausible by construction, as intended.
The stratified BPI 2017 analysis under uniform proposals (`Method2Log_Geng.md`) showed the
per-miner sensitivity the other way: IM Δ = +0.004, Heuristics Δ = +0.033, Alpha Δ = −0.004
(regular − mutated fitness).

**v2.6 — acceptance and probe integrity** (`HybridGen/algorithm/v26.py`) adds, on top of v2.5:
(a) `gen_accept` — the fraction of shadow traces the model replays *perfectly*
(pm4py `trace_is_fit`), the direct operationalization of "accepts future valid behavior";
mean fitness stays as `gen_total`, both reported stratified;
(b) a data-driven trace-length cap `min(max(100, 2 × longest observed trace), 1000)`
replacing the hard 100 (Sepsis has traces of length 185);
(c) `successor_weighting='log'|'mle'` — `'mle'` samples ∝ raw frequency, i.e. from the
estimated future-trace distribution itself, resolving the inconsistency of computing
`p_unseen` from raw counts while sampling from ln-damped ones; `'log'` remains the
deliberate rare-behavior stress-test mode.

**Version-comparison results (`benchmark/version_comparison.py`, master-benchmark protocol,
vs variant-based 5-fold R1; D1 Sepsis aggregated over seeds {42, 1, 7, 99}, D2 BPI 2013
Incidents over seeds {42, 7}; flower litmus = 1.0 for every version on both datasets):**

| Version | D1 Pearson | D1 Spearman | D1 MAE | D2 Pearson | D2 Spearman | D2 MAE | s/cell |
|---|---|---|---|---|---|---|---|
| v2.4 (uniform, ln-damped) | 0.970±.001 | 1.00 | 0.077±.001 | 0.985 | 0.943 | 0.047 | 2.4–4.9 |
| v2.5 / v2.6-log (Katz) | 0.975±.000 | 1.00 | 0.067±.000 | 0.987 | 0.943 | 0.045 | 2.4–5.1 |
| **v2.6-mle** | **0.996±.001** | 1.00 | **0.024±.001** | **0.993** | **1.000** | **0.021** | 2.4–5.2 |

Readings:
- The Katz-consistent proposal improves calibration moderately; **MLE sampling improves it
  ~3× on both datasets** — the ln-damping was probing a rare-tilted distribution rather than
  the expected future, and it was costing real calibration.
- **On D2, mle is the only mode that ranks correctly**: R1 puts Heuristics (0.996) above
  Inductive-Infrequent (0.982); all ln-damped versions invert this pair (Spearman 0.943),
  mle restores it (Spearman 1.000). The damping penalized the Heuristics model for
  rare-branch behavior the expected future rarely exercises.
- Seed stability: std ≤ 0.001 on Pearson/MAE across 4 seeds — none of this is a seed artifact.
- `duplicates_kept = 0` on D2 despite TLRA 0.80 — the dedup retry cap did not bite;
  re-check on D5 (TLRA 0.95).
- Runtime is version-independent (~2.4 s/cell on D2, ~5 s/cell on D1 for the full
  5×1000-trace protocol); v2.6's additions cost ≤ 6 % over v2.4.
- v2.5 ≡ v2.6-log behaviorally (the old length cap never binds on D1/D2; `truncated = 0`).

**Decision (recommended): make `'mle'` the headline mode** — it dominates on every measured
criterion on both datasets with no runtime cost — and keep `'log'` as the documented
stress-test variant for rare-behavior robustness analysis.

**Acceptance validated against its own ground truth** (`benchmark/r1_accept.py`: fraction of
*held-out* traces perfectly replayed, variant-based 5-fold × 3, seed 42). Acceptance numbers
must never be compared against fitness-R1 (apples vs oranges); against acceptance-R1 they
hold up:

| | D1 Pearson | D1 MAE | D2 Pearson | D2 MAE |
|---|---|---|---|---|
| M1e (log) `gen_accept` | 0.999 | 0.173 | 0.847 | 0.242 |
| **M1f (mle) `gen_accept`** | **0.998** | **0.076** | **0.997** | **0.021** |

Key facts this resolves:
- The near-zero acceptances on D1 are **true properties of the nets**, not metric failures:
  the acceptance ground truth itself is ≈ 0 for Alpha (0.000), Alpha+ (0.000), Heuristics
  (0.028), Heuristics-strict (0.043) — these models *never strictly accept* unseen real
  traces; their respectable fitness scores are pure partial credit. Only the Inductive
  family genuinely accepts unseen behavior on Sepsis (IM-infrequent 0.78, IM-strict 1.00).
- Alpha+ tying the Trace model at 0.0 under acceptance is therefore **correct**, not a bug.
- The poles anchor exactly under acceptance: Trace 0.0000, Flower 1.0000 (both datasets,
  ground truth and metric alike).
- Spearman is uninformative under acceptance on these logs (mass ties at zero); use
  Pearson/MAE.
- Remaining bias: `gen_accept` underestimates acceptance-R1 in the mid-range on D1
  (IM-strict 0.74 vs 1.00) — shadow traces are *harder* than held-out real traces because
  they include genuinely novel recombinations and mutations; acceptance-R1 is an upper
  reference, not an identity target.

Conclusion: fitness (`gen_total`) and acceptance (`gen_accept`) are two valid readings of
the same probe, each validated against its matching ground truth (fitness↔fitness-R1 MAE
0.023/0.022; acceptance↔acceptance-R1 MAE 0.076/0.021 under mle). Report both; the
fitness–acceptance gap per model is itself diagnostic (large gap = the model's score is
partial-credit-driven).

---

## 7. Known limitations (honest list)

- **Shared measurement core with the ground truth.** Gen_shadow and R1/R2 both score via token
  replay; part of any agreement stems from the shared fitness notion. Mitigation: R1/R2 rediscover
  models per fold; planned cross-check with alignment-based fitness on a subsample.
- **Generator trained on the full log** while models are also discovered on the full log.
  A variant-split (generator statistics from one share, discovery on another) would remove this
  coupling; cost: weaker statistics. Not currently implemented.
- **Single-dataset evidence** for the validation results so far (D1); D2–D5 pending.
- **`max_n = 6` calibrated on one log** (BPI 2017) via mutation-rate peak — a proxy criterion,
  not direct agreement with ground truth. Held fixed everywhere by design (a metric must not be
  tuned per dataset).
- **Mutation proposal** — see §6.

---

## 8. Gen_struct — historical note (defunct)

`Gen_struct` (structural frequency analysis: replay the original log, penalize rarely used model
parts) was designed by the practical partner under a different stance on the flower model —
the view that a generalization metric must itself penalize maximal permissiveness. Under the
construct definition in §1, that role belongs to precision, and the component was removed from
the score in v2.3. It is **defunct as a method**: do not reactivate it as an anti-flower device.

It may still be worth *discussing* in the report as an explored design direction, with two honest
observations: (a) its anti-flower motivation conflates generalization with precision (§1);
(b) its anti-overfitting role (penalizing branches that memorize outlier traces) targets behavior
that Gen_shadow already detects through replay failures of recombined traces. Passages in
`Method2Log.md` §2.2, `BenchmarkDesign.md` (flower "expected lowest"), and the per-cell config
JSON notes reflect the superseded stance.

---

## 9. Naming

"HybridGen" describes the defunct two-component design and is now a misnomer — the metric is the
pure generative component. The code package keeps the legacy name until a rename is decided.
Candidates: **GenShadow**, **ShadowGen**, **SLG (Shadow-Log Generalization)**.
On rename: update package name, report, and benchmark method label (M1) together.
