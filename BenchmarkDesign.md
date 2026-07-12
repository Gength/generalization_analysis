# Generalization Benchmark Design — Methodology v2

> **Status.** This document defines **Methodology v2** of the generalization benchmark.
> It **supersedes** the previous `BenchmarkDesign.md` (Methodology v1). All v1 sections
> not explicitly restated below — external methods M2–M8, reference metrics R1–R3, datasets
> D1–D21, statistical protocol, SLURM execution plan, risks — are **inherited unchanged**,
> with v2-specific overrides noted inline.
>
> Construct definition follows [`Method_GenShadow.md`](Method_GenShadow.md) (authoritative):
> generalization = acceptance of future *valid* behavior, strictly separated from precision.
>
> v1 results in `benchmark/results/configs/` remain valid; v2 results live in
> **`benchmark/results/configs_v2/`**.

---

## What changed vs. Methodology v1 (summary)

| # | Change | Why |
|---|--------|-----|
| 1 | Tier 1 extended: **M1e (v2.5), M1f (v2.6-log), M1g (v2.6-mle)** added | New algorithm versions fixing probe defects found in v2.4 (see [`WhatChanged_v25_v26.md`](WhatChanged_v25_v26.md)) |
| 2 | Eighth miner added: **Filtered Trace Model** (top-50 variants) | The 0.0 pole (memorization), opposite of the Flower Model's 1.0 pole |
| 3 | Pole interpretation corrected | Flower ≈ 1.0 is **correct** for a pure generalization metric (construct-purity litmus), not a failure; Trace low is the overfitting pole |
| 4 | Reporting: **mean ± std for every M1 version** + acceptance + probe-integrity counters | Transparency; v2.6 metrics expose `gen_accept`, `duplicates_kept`, `truncated_traces` |
| 5 | Agreement protocol: **Spearman + MAE + spread** vs R1, poles excluded | Spearman alone is a low bar (even the random floor achieves 1.0 on D1) |
| 6 | New runner: `benchmark/run_m1_family.py` writes official config JSONs to `configs_v2/` | One command per dataset, model discovery cached; R1 computed separately by `r1.sh` |
| 7 | M1g (v2.6-mle) recommended as headline candidate | Best calibration across all criteria; only mode ranking D2 correctly |

---

## Objective

Compare **HybridGen v2.4–v2.6** against 7 external generalization baselines across all 21 real-world event logs in the catalog and 8 process discovery configurations. The benchmark produces a structured CSV of per-method generalization scores, enabling quantitative ranking, correlation analysis, and qualitative assessment of each method's discriminative power.

---

## Methods Under Comparison

### Tier 1 — Our Method (v2)

| # | Method | Algorithm | Key property |
|---|--------|-----------|--------------|
| M1a | HybridGen v1 | 1-gram DFG + Good–Turing | simplest ablation |
| M1b | HybridGen v2.1, N=3 | flat termination | isolates context-aware termination |
| M1c | HybridGen v2.1, N=6 | flat termination | isolates N=3→6 upgrade |
| M1d | HybridGen v2.4 | uniform mutation proposal, ln-damped sampling | v1-methodology baseline (unchanged, for continuity) |
| M1e | HybridGen v2.5 | Katz-consistent mutation proposal | mutations drawn from backed-off lower-order context instead of uniform alphabet noise; probe-integrity counters |
| M1f | HybridGen v2.6 (log) | v2.5 + acceptance rate + data-driven length cap | ln-damped sampling retained (stress-test mode) |
| M1g | HybridGen v2.6 (mle) | v2.6 with `successor_weighting='mle'` | samples the estimated future distribution itself — **headline candidate** |

All M1a–M1g versions report **mean ± std over 5 iterations** of 1,000 shadow traces, seed 42,
`max_n=6` (except M1a/M1b by design), `safe_threshold=5`. M1f/M1g additionally report
`gen_accept` (perfect-replay rate), the regular/mutated openness profile, and the
probe-integrity counters `duplicates_kept` / `truncated_traces`.

### Tier 2 — External Generalization Baselines

| # | Method | Paradigm | Approach | Output | Runtime |
|---|--------|----------|----------|--------|---------|
| M2 | **PM4Py Built-in** | Structural counting | Counts invisible transitions, token-replay deviations, and visited states in the reachability graph. `pm4py.algo.evaluation.generalization.apply()`. | Scalar [0,1] | < 1 s |
| M3 | **Entropic Relevance** (Polyvyanyy et al. 2020) | Entropy-based | Computes entropic relevance of a stochastic process model to an event log. Currently approximated via per-miner DFG simulation (PNML → play\_out → DFG JSON → JDFG2Aut → automaton JSON → Relevance). Uses open-source `relevance.jar` from [promtecmx/relevance](https://github.com/promtecmx/relevance). Full SDFA conversion pending. | Scalar [0,1] | ~seconds (Java subprocess) |
| M4 | **Anti-Alignment Generalization** (van Dongen, 2017) | Adversarial | ❌ **Archived** — infeasible on real-life logs (see [Archived Methods](#archived-methods)). | Scalar [0,1] | N/A |
| M5 | **AVATAR** (Theis & Darabi, 2020) | GAN-based | ✅ D1 completed. RelGAN trained on 80% variants → samples → harmonic mean of token-replay fitness and ET Conformance precision. Multi-word activity fix applied. | Scalar [0,1] | ~4h (GAN training) |
| M6 | **Bootstrap Generalization** (Polyvyanyy et al. 2022) | Entropia `-bgen` eigenvalue-based precision & recall | ✅ D1/D2 complete. Uses fixed JAR `jbpt-pm-entropia-1.7.1.jar` (`k=2`). Entry point: `benchmark/job_m6.py` (core: `benchmark/bridges/run_m6_bgen.py`). See [`BenchmarkGuide.md §M6 Implementation Note`](BenchmarkGuide.md#m6-implementation-note). | Precision & recall [0,1]; F1 in summary table | ~45–60 s/cell |
| M7 | **SpeciAL4PM** (Kabierski et al. 2023) | Species diversity | ✅ D1 completed. C1 coverage ratio between simulated and original log species profiles. | C1 ratio [0,1] | ~12 s/cell |
| M8 | **Pattern-based Generalization** (Reißner et al. 2020) | Pattern matching | ❌ **Archived** — too slow/unstable for real logs (see [Archived Methods](#archived-methods)). | Scalar [0,1] | N/A |

#### Repository & Local Path References

| Method | Repository | Local Path |
|--------|-----------|------------|
| Entropic Relevance | [promtecmx/relevance](https://github.com/promtecmx/relevance) | `./src/relevance/` |
| Anti-Alignment Generalization | [ProcessMPUT/processm](https://github.com/ProcessMPUT/processm) → `processm.conformance/src/main/kotlin/processm/conformance/models/antialignments` | `./src/processm/` |
| AVATAR | [Julian-Theis/AVATAR](https://github.com/Julian-Theis/AVATAR) | `./src/AVATAR/` |
| Bootstrap Generalization | [lgbanuelos/bsgen](https://github.com/lgbanuelos/bsgen) + [jbpt/codebase](https://github.com/jbpt/codebase/tree/master/jbpt-pm/gen/bootstrap) | `./src/bsgen/` (Python), `./src/codebase/jbpt-pm/entropia/` (Java `-bgen`) |
| SpeciAL4PM | [MartinKabierski/SpeciAL-core](https://github.com/MartinKabierski/SpeciAL-core) | `./src/SpeciAL-core/` |
| Pattern-based Gen. | [reissnda/AutomataConformance](https://github.com/reissnda/AutomataConformance) | `./src/AutomataConformance/` |

### Tier 3 — Reference / Sanity-Check Metrics

These are NOT generalization metrics per se, but provide empirical anchors for interpreting the generalization scores:

| # | Method | Approach | Role |
|---|--------|----------|------|
| R1 | **K-Fold Cross-Validation Fitness** (5-fold, variant-based) | Split log by variants → discover on train → replay test on model. Average over 5 folds. | Empirical "ground truth" — measures actual generalization to held-out data. |
| R2 | **Leave-One-Variant-Out Fitness** | Each variant held out in turn, model discovered on rest, fitness on held-out variant. | Finer-grained CV; catches fragile models that K-fold might miss. **Parallelize across SLURM `krater` partition (MaxJobs=15, MaxSubmit=30).** |
| R3 | **Naive Random Baseline** | Generate N random traces (uniform activity sampling, random length 1–max_trace_len) → token-replay on model. | Lower bound — any reasonable method should score above this. |

---

## Datasets

All datasets reside under `data/` with per-directory `summary.txt` files containing full statistics (cases, events, variants, trace length distribution, N-gram sparsity, TLRA). Ordered below from smallest to largest by memory pressure.

### Full Dataset Catalog

| # | Dataset | Path | Cases | Events | Acts | Variants | Avg Len | Size | TLRA |
|---|---------|------|-------|--------|------|----------|---------|------|------|
| D6 | BPI 2013 Problem Open | `data/BPI-Challenge_2013/` | 819 | 2,351 | 3 | 108 | 2.9 | 69 KB | 0.87 |
| D7 | BPI 2013 Problem Closed | `data/BPI-Challenge_2013/` | 1,487 | 6,660 | 4 | 183 | 4.5 | 187 KB | 0.88 |
| D8 | BPI 2015 Municipality 2 | `data/BPI-Challenge_2015/` | 832 | 44,354 | 410 | 828 | 53.3 | 34 MB | 0.005 |
| D9 | BPI 2015 Municipality 4 | `data/BPI-Challenge_2015/` | 1,053 | 47,293 | 356 | 1,049 | 44.9 | 37 MB | 0.004 |
| D1 | **Sepsis** | `data/Sepsis Cases - Event Log_1_all/` | 1,050 | 15,214 | 16 | 846 | 14.5 | 0.2 MB | 0.19 |
| D10 | BPI 2015 Municipality 1 | `data/BPI-Challenge_2015/` | 1,199 | 52,217 | 398 | 1,170 | 43.6 | 41 MB | 0.02 |
| D11 | BPI 2011 Hospital | `data/BPI-Challenge_2011/` | 1,143 | 150,291 | 624 | 981 | 131.5 | 2.4 MB | 0.14 |
| D12 | BPI 2015 Municipality 5 | `data/BPI-Challenge_2015/` | 1,156 | 59,083 | 389 | 1,153 | 51.1 | 46 MB | 0.003 |
| D13 | BPI 2015 Municipality 3 | `data/BPI-Challenge_2015/` | 1,409 | 59,681 | 383 | 1,349 | 42.4 | 47 MB | 0.04 |
| D14 | BPI 2020 PrepaidTravel | `data/BPI-Challenge_2020/` | 2,099 | 18,246 | 29 | 202 | 8.7 | — | 0.90 |
| D15 | BPI 2020 InternationalDecl. | `data/BPI-Challenge_2020/` | 6,449 | 72,151 | 34 | 753 | 11.2 | — | 0.88 |
| D16 | BPI 2020 RequestForPayment | `data/BPI-Challenge_2020/` | 6,886 | 36,796 | 19 | 89 | 5.3 | — | 0.99 |
| D17 | BPI 2020 PermitLog | `data/BPI-Challenge_2020/` | 7,065 | 86,581 | 51 | 1,478 | 12.3 | — | 0.79 |
| D2 | **BPI 2013 Incident** | `data/BPI-Challenge_2013/` | 7,554 | 65,533 | 4 | 1,511 | 8.7 | 1.3 MB | 0.80 |
| D18 | BPI 2020 DomesticDecl. | `data/BPI-Challenge_2020/` | 10,500 | 56,437 | 17 | 99 | 5.4 | — | 0.99 |
| D19 | BPI 2012 | `data/BPI-Challenge_2012/` | 13,087 | 262,200 | 24 | 4,366 | 20.0 | 3.3 MB | 0.67 |
| D3 | **BPI 2017** | `data/BPI-Challenge_2017/` | 31,509 | 1,202,267 | 26 | 15,930 | 38.2 | 29.6 MB | 0.49 |
| D4 | **BPI 2018** | `data/BPI-Challenge_2018/` | 43,809 | 2,514,266 | 41 | 28,457 | 57.4 | 158 MB | 0.35 |
| D20 | Hospital Billing | `data/Hospital Billing - Event Log_1_all/` | 100,000 | 451,359 | 18 | 1,020 | 4.5 | 6.6 MB | 0.99 |
| D21 | Road Traffic Fine | `data/Road Traffic Fine Management Process_1_all/` | 150,370 | 561,470 | 11 | 231 | 3.7 | 3.5 MB | 0.998 |
| D5 | **BPI 2019** | `data/BPI-Challenge_2019/` | 251,734 | 1,595,923 | 42 | 11,973 | 6.3 | 16.9 MB | 0.95 |

> **Acts** = unique activity labels. **Variants** = unique trace sequences. **TLRA** = 1 − (variants/cases), the probability an additional trace has been seen before. Higher = more repetitive.

### Miner Availability (Model Discovery Timing)

Model discovery was timed on all 21 datasets × 8 miners (1h timeout per miner, HPC).
See `benchmark/statistics/_miner_availability.json` for the canonical record.

| Dataset | Available | Unavailable |
|---------|-----------|-------------|
| D1–D7, D14–D21 | **8/8** (all miners) | — |
| D8 (BPI2015_Municipality_2) | 5/8: Flower, Trace_Filtered, Heuristics, Heuristics_Strict, Inductive_Infrequent | Alpha (timeout), Alpha+ (timeout), Inductive_Strict (RecursionError) |
| D9 (BPI2015_Municipality_4) | 5/8: same as D8 | same as D8 |
| D10 (BPI2015_Municipality_1) | 5/8: same as D8 | same as D8 |
| D11 (BPI2011_Hospital) | 5/8: same miner set as D8 | Alpha (timeout), Alpha+ (timeout), Inductive_Strict (timeout) |
| D12 (BPI2015_Municipality_5) | 5/8: same as D8 | same as D8 |
| D13 (BPI2015_Municipality_3) | 4/8: Flower, Trace_Filtered, Heuristics, Heuristics_Strict | Alpha (timeout), Alpha+ (timeout), Inductive_Infrequent (RecursionError), Inductive_Strict (RecursionError) |

> Benchmark jobs should skip unavailable miners
> rather than failing at runtime.

### Benchmark Dataset Scope

The initial design (v1) selected 5 representative datasets (D1–D5) spanning diverse process characteristics. **Following reviewer feedback, the benchmark now runs on all 21 datasets in the full catalog above.** The original D1–D5 set remains informative for understanding the benchmark's design rationale:

| # | Dataset | Cases | Events | Variants | Avg Len | TLRA | Why Selected |
|---|---------|-------|--------|----------|---------|------|-------------|
| D1 | **Sepsis** | 1,050 | 15,214 | 846 | 14.5 | 0.19 | Smallest — ideal smoke test. Hospital process, high attribute richness. |
| D2 | **BPI 2013 Incident** | 7,554 | 65,533 | 1,511 | 8.7 | 0.80 | Small but diverse (1.5K variants from only 4 activities). Very low structural complexity tests whether metrics over-penalize simple models. |
| D3 | **BPI 2017** | 31,509 | 1,202,267 | 15,930 | 38.2 | 0.49 | Variant explosion (87% singletons) + deep traces — stress-tests HybridGen N-gram states and ILP-based methods. |
| D4 | **BPI 2018** | 43,809 | 2,514,266 | 28,457 | 57.4 | 0.35 | Largest variant count (28K) + deepest traces (avg 57) + lowest TLRA (0.35) — hardest generalization challenge. |
| D5 | **BPI 2019** | 251,734 | 1,595,923 | 11,973 | 6.3 | 0.95 | Largest case count — tests PM4Py memory scaling. Structured purchase-to-pay with rare branches. |

> **Coverage rationale (full 21-dataset catalog):** The expanded set covers every available real-life log from BPI Challenges 2011–2020 plus Hospital Billing and Road Traffic Fine, spanning:
> - **Trace depth**: from 2.9 (D6) to 131.5 (D11) avg events
> - **Variant diversity**: from 89 (D16) to 28,457 (D4)
> - **Log size**: from 819 (D6) to 251K (D5) cases
> - **TLRA**: from 0.003 (D8) to 0.998 (D21) — coverage from extreme novelty to near-determinism
> - **Structural complexity**: from 3 activities (D6) to 624 (D11) — tests whether metrics conflate simplicity with generalization

### Two-Phase Execution Plan

```
Phase A: Local Machine (small to medium datasets — D1, D2, D6, D7, D14–D19)
  ├── D1  Sepsis (1,050 cases)        — smoke test, ~14 min for fast methods
  ├── D2  BPI 2013 Inc. (7,554 cases)  — validates pipeline on 4-activity log
  ├── D6  BPI 2013 Problem Open (819 cases)  — smallest, 3 activities
  ├── D7  BPI 2013 Problem Closed (1,487)     — companion to D6
  ├── D14 BPI 2020 PrepaidTravel (2,099)      — moderate size
  ├── D15 BPI 2020 InternationalDecl. (6,449) — larger but still local
  ├── D16 BPI 2020 RequestForPayment (6,886)  — simple (19 acts, 89 variants)
  ├── D17 BPI 2020 PermitLog (7,065)          — diverse (51 acts, 1.5K variants)
  ├── D18 BPI 2020 DomesticDecl. (10,500)     — large case count, simple structure
  └── D19 BPI 2012 (13,087 cases)             — largest local-friendly log
      Goal: run all fast methods (M1–M3, M5–M7, R1–R3) on these 10 datasets.
      M5 (AVATAR) may be skipped on local if GPU is unavailable.

Phase B: CIP-Pool 128GB Machine (large / memory-heavy datasets — D3–D5, D8–D13, D20–D21)
  ├── D8  BPI 2015 Muni. 2 (832 cases, 410 acts, 34 MB)    — high act count
  ├── D9  BPI 2015 Muni. 4 (1,053 cases, 356 acts, 37 MB)   — high act count
  ├── D10 BPI 2015 Muni. 1 (1,199 cases, 398 acts, 41 MB)   — high act count
  ├── D11 BPI 2011 Hospital (1,143 cases, 624 acts, 131.5 avg len) — deep traces
  ├── D12 BPI 2015 Muni. 5 (1,156 cases, 389 acts, 46 MB)   — high act count
  ├── D13 BPI 2015 Muni. 3 (1,409 cases, 383 acts, 47 MB)   — high act count
  ├── D20 Hospital Billing (100K cases, 6.6 MB)              — large case count
  ├── D21 Road Traffic Fine (150K cases, 3.5 MB)             — largest case count
  ├── D3  BPI 2017 (31K cases, 16K variants)                 — variant explosion
  ├── D4  BPI 2018 (44K cases, 28K variants, 2.5M events)   — heaviest
  └── D5  BPI 2019 (252K cases)                              — largest raw count
```

**Note:** D8–D13 (BPI 2015 municipalities + BPI 2011 Hospital) are included in Phase B despite modest case counts because their high activity counts (356–624) and deep traces (avg 42–131) cause memory pressure during model discovery and N-gram pre-computation. Additionally, model discovery timing (1h timeout) revealed that **Alpha and Alpha+ timeout on all six datasets**, **Inductive_Strict fails with RecursionError on D8/D9/D10/D12/D13 and times out on D11**, and **Inductive_Infrequent also fails on D13**. Only 4–5 of 8 miners are available for benchmark runs on these datasets. See [Miner Availability](#miner-availability-model-discovery-timing) above.

---

**Partial infrastructure worth noting:**
- `gen/bootstrap/inputs_r.csv` provides validated parameter configurations for Sepsis (`VE_30_Sepsis_*`), Road Traffic (`PE_11_Road_Traffic_*`), and BPI Challenge (`PE_01_BPI_Challenge_*`, `VE_01_BPI_Challenge_*`) models. These can serve as a reference for `-bgen` parameter selection (log_size, generations, k_value, breeding_factor).
- Entropia `examples/` (legacy jbpt-pm-entropia) contains `sepsis.xes.gz` and a pre-built SDFA model `sdfa_sepsis_1.000.json` — can be used to validate the `relevance.jar` bridge against a known-good SDFA before running on our own discovered models.
- SpeciAL4PM eval scripts have hardcoded paths to BPI 2012, Road Traffic, Sepsis, BPI 2018, BPI 2019 — confirming these datasets are compatible with the species estimation pipeline.

---

## Process Discovery Configurations (v2: eight, spanning both poles)

| # | Miner | Construction | Role |
|---|-------|--------------|------|
| 0 | **Filtered Trace Model** | one isolated path per variant, **top-50 variants by frequency** (identical to `master_benchmark_v24.py`) | **0.0 pole** — pure memorization; accepts nothing unseen |
| 1 | Alpha Miner | `pm4py.discover_petri_net_alpha()` | Low (overfits to noise on real logs) |
| 2 | Alpha+ Miner | `pm4py.discover_petri_net_alpha_plus()` | Low–Moderate |
| 3 | Heuristics (Default) | `pm4py.discover_petri_net_heuristics(log)` | Moderate |
| 4 | Heuristics (Strict) | `pm4py.discover_petri_net_heuristics(log, dependency_threshold=0.99)` | Moderate–High |
| 5 | Inductive (Strict) | `pm4py.discover_petri_net_inductive(log, noise_threshold=0.0)` | High (block-structured) |
| 6 | Inductive (Infrequent) | `pm4py.discover_petri_net_inductive(log, noise_threshold=0.2)` | High (filters infrequent paths) |
| 7 | Flower Model | Manual construction (all activities in one loop) | **1.0 pole** — accepts everything |

**Why top-50 for the Trace Model.** A full trace model has one branch per variant
(Sepsis: ~12,000 transitions; BPI 2017: ~600,000), which makes token replay intractable —
this is why v1 archived it. Capping at the 50 most frequent variants keeps the net at
~750 transitions (seconds per cell) while preserving the semantics that matter: the model
memorizes a fixed set of observed traces and rejects everything else. The cap is recorded
in every config JSON (`trace_model_top_k: 50`).

**Pole interpretation (corrected vs v1).** Under the pure-generalization construct
(see [`Method_GenShadow.md`](Method_GenShadow.md)):

- **Flower ≈ 1.0 is the expected, correct score** — the litmus for construct purity.
  A metric scoring Flower < 1 is contaminated with precision/structure.
- **Trace Model low is the expected, correct score** — the memorization pole. It will not
  reach exactly 0.0 under token replay (partial credit grants unseen traces some fitness;
  the v1 "ultimate" runs measured ~0.53–0.63 on Sepsis), so it is a *low anchor*, not a
  literal zero. Its perfect-replay acceptance (`gen_accept`, M1f/M1g) **is** ≈ 0.
- Both poles are **excluded from agreement statistics** (Pearson/Spearman/MAE/spread are
  computed over the six real miners) and reported separately as litmus checks.

### Mapping to Model Morphology Archetypes

The eight miners span the full generalization spectrum defined in
[`archive/Tianhao/ExperimentDesign.md`](archive/Tianhao/ExperimentDesign.md) (§2.1.1),
covering all six morphological archetypes:

| # | Benchmark Miner | Morphology Archetype | Confidence | Rationale |
|---|----------------|---------------------|------------|-----------|
| 0 | **Filtered Trace Model** (top‑50) | **Trace Model** | Direct | Practical approximation of "one path per variant." Top‑50 cap preserves memorization semantics while keeping token replay tractable. |
| 1 | **Alpha Miner** | **Spaghetti Model** | Direct | ExperimentDesign §2.1.1‑B explicitly names Alpha Miner as the canonical generator of spaghetti models — "attempts to accommodate every low-frequency, long-tail anomaly." |
| 2 | **Alpha+ Miner** | **Spaghetti Model** (milder) | High | Adds limited arc pruning but still lacks frequency-based filtering. On real logs it produces tangled nets, though less extreme than raw Alpha. |
| 3 | **Heuristics (Default)** | **Causal / Heuristics Net** | Direct | The archetype's representative algorithm. Dependency‑driven arc pruning with default thresholds produces exactly the "probability‑driven pragmatism" described in §2.1.1‑D. |
| 4 | **Heuristics (Strict)** (`dependency_threshold=0.99`) | **Causal Net → Lasagna boundary** | High | Aggressively prunes low‑confidence arcs, yielding a cleaner structure that approaches the Lasagna ideal. Still tolerates deadlocks (non‑block‑structured). |
| 5 | **Inductive (Strict)** (`noise_threshold=0.0`) | **Strict Block‑Structured Model** | Direct | Zero noise filtering, pure recursive cut detection. Guarantees block‑structured output — the definitive "algorithmic discipline" archetype. |
| 6 | **Inductive (Infrequent)** (`noise_threshold=0.2`) | **Lasagna Model** — "Holy Grail" | Direct | Noise filtering prunes infrequent exception branches into adjacent constructs, creating the "crisp backbone + encapsulated exceptions" signature (ExperimentDesign: "Run Inductive Miner with noise threshold ≈ 0.2–0.4"). |
| 7 | **Flower Model** | **Flower Model** | Direct | Identical construction (all activities in one concurrent block). |

**Key observations:**

1. **Full archetype coverage** — The six archetypes are covered by the eight miners. Two miners (Alpha, Alpha+) map to Spaghetti at different severities; two (Heuristics Default, Heuristics Strict) map to Causal Net with different proximity to Lasagna.
2. **Quadrant diagram alignment** — The [quadrant visualization](archive/Tianhao/ExperimentDesign.md#211-the-model-morphology-catalog) maps directly onto benchmark results: the x‑axis (Structural Complexity) corresponds to the structural penalties anchored by the Trace/Flower poles; the y‑axis (Behavioral Permissiveness) corresponds to the generalization score. Excluding the two poles, the six real miners trace the inverted‑U trajectory the ExperimentDesign predicts.
3. **Decomposition opportunity** — The Gen\_Shadow vs. Gen\_Struct decomposition that ExperimentDesign §2.1.1 prescribes per archetype is directly testable: M1f/M1g already report `gen_shadow_regular`/`gen_shadow_mutated`; re‑adding structural decomposition for the final analysis would complete the picture.

---

## Evaluation Protocol

### Per Cell: (Dataset × Miner × Method) — v2

1. **Discover model** (self-contained in /tmp workdir) or via `--output` to `configs_v2/`.
2. **Evaluate** with the per-method protocol; **record mean, std, and raw per-iteration scores**.
3. **Write one config JSON per cell** to the output directory
   (`{Dataset}__{Miner}__{Method}.json`, v1 schema + new optional result fields:
   `gen_accept`, `gen_accept_std`, `gen_shadow_regular`, `gen_shadow_mutated`,
   `duplicates_kept`, `truncated_traces`, `max_trace_length_used`).
4. **Ground truth**: R1 (variant-based 5-fold CV, 3 shuffles, seed 42) — copied from v1 configs
   where present, computed fresh for new miners (e.g. Trace_Filtered), and re-written to
   `configs_v2/` so v2 is self-contained.
5. **Agreement reporting** per method: **Pearson, Spearman, MAE, spread** over the six real
   miners, plus the two pole litmus values. Never report Spearman alone.

### Statistical Rigor

| Method(s) | Iterations | Guide Default | Reporting |
|-----------|-----------|---------------|-----------|
| M1a–M1g (HybridGen) | 5 | [`job_m1.py`](BenchmarkGuide.md#m1-family-runner-v2-methodology) | Mean ± std; M1f/M1g additionally add `gen_accept`, `duplicates_kept`, `truncated_traces` |
| M2 (PM4Py built-in) | 1 (deterministic) | [`job_m2.py`](BenchmarkGuide.md#self-contained-jobs) | Single value |
| M3 (Entropic Relevance) | 1 (deterministic) | [`job_m3.py`](BenchmarkGuide.md#self-contained-jobs) | Single value |
| M4 (Anti-Alignment) | — | — | ❌ Archived — infeasible |
| M5 (AVATAR) | 2 sampling runs (target: 3–5) | [`job_m5.py`](BenchmarkGuide.md#self-contained-jobs) | Mean ± std (D1); single score (D2) |
| M6 (Bootstrap Gen) | 10 bootstrap replicates (target: 100) | [`job_m6.py --m 10`](BenchmarkGuide.md#m6-implementation-note) | Mean ± std; gen_score = F1(p,r) |
| M7 (SpeciAL4PM) | 1 per N-gram (1–3 gram + trace variant) | [`job_m7.py`](BenchmarkGuide.md#self-contained-jobs) | Profile; C1 ratio |
| M8 (Pattern-based) | — | — | ❌ Archived — JAR crashes |
| R1 (K-Fold CV) | k=5 folds × 3 shuffles | [`job_r1.py`](BenchmarkGuide.md#r-family-runners-r1r3-reference-metrics) | Mean ± std |
| R2 (Leave-One-Variant-Out) | 1 pass per variant | [`job_r2.py`](BenchmarkGuide.md#r-family-runners-r1r3-reference-metrics) | Single value per variant; aggregate by mean |
| R3 (Random baseline) | 5 | [`job_r3.py`](BenchmarkGuide.md#r-family-runners-r1r3-reference-metrics) | Mean ± std |

Seed all random number generators for reproducibility (`seed=42`).

### K-Fold CV: Why k=5

Variant-based k-fold partitions trace variants (not individual traces) into k groups. All traces sharing the same activity sequence stay together in train or test — never split across folds.

| k | Training fraction | Per-fold test size (D1) | Per-fold test size (D3) | Assessment |
|---|-------------------|------------------------|------------------------|------------|
| 3 | 67% | ~282 variants | ~5,310 variants | Coarse — test set is 33% of variants, leaving only 67% for model discovery |
| 5 | 80% | ~169 variants | ~3,186 variants | **Recommended** — standard in process mining; 80/20 split per fold; 5 scores averaged |
| 6 | 83% | ~141 variants | ~2,655 variants | Slightly more training data per fold; acceptable for variant-rich datasets (D3–D5) |
| 10 | 90% | ~85 variants | ~1,593 variants | Too small test folds for D1/D2; model discovery on 90% of variants is nearly the full log |

**Choice: k=5.** Provides an 80/20 train/test split per fold (standard in ML), works for all D1–D5 datasets (D1's 846 variants → ~169 per test fold is sufficient, D3's 15,930 → ~3,186 per test fold), and produces 5 scores to average. k=6 is also defensible but k=5 is more conventional.

### HybridGen Hyperparameters

The choice of `max_n` was determined empirically. See `analysis/Mutation/MutationReport.md` for the full N-gram sweep (N=1..8) on BPI 2017. **Key conclusion: N=6 is the mutation peak (49× V1 baseline, 3.8% mutated traces).** Beyond N=6 the curse of dimensionality overtakes context benefit (mutation rate declines, backoffs double). N=6 also resolves a small-sample artifact in Alpha Miner that appeared at N=3.

The Katz backoff mechanism makes `max_n` an **upper bound**, not a fixed operating point — sparse datasets (D4/D5) gracefully degrade to lower effective N.

**Per-dataset viability:**

| Dataset | N=3 Safe States | N=6 Projected | Verdict |
|---------|----------------|---------------|---------|
| D1 Sepsis | 55.9% | ~30–40% top-order | Katz backoff → effective ≈ N=2–3 |
| D2 BPI 2013 | 88.9% | >75% top-order | Excellent (4 activities) |
| D3 BPI 2017 | 67.0% | **80.0%** (measured) | Empirical peak |
| D4 BPI 2018 | 52.0% | ~25–35% | Katz backoff → effective ≈ N=2 |
| D5 BPI 2019 | 51.5% | ~25–35% | Katz backoff → effective ≈ N=2 |

**Recommended configuration (fixed across all datasets):**

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `max_n` | **6** | Empirical mutation peak on BPI 2017. Katz backoff degrades gracefully on sparser datasets. |
| `safe_threshold` | **5** | Well-tested; applies to total transition frequency, not unique outgoing count. |
| `num_shadow_traces` | **min(1000, len(log))** | 1,000 traces → ~38 mutated at N=6 for stratified analysis. |
| `iterations` | **5** | Algorithm default; 5 iterations produce tight mean ± std. |

**Ablation experiments (M1a–M1c) — all run on D1–D5:**

| Ablation | Configuration | Isolates |
|----------|--------------|----------|
| M1a (v1) | 1-gram DFG only, no `max_n` | Simplest baseline |
| M1b (v2.1 N=3) | `max_n=3`, flat termination | **Context-aware termination** (M1b vs M1d) |
| M1c (v2.1 N=6) | `max_n=6`, flat termination | **N=3→6 upgrade** (M1c vs M1b) and **v2.4 fix on top of N=6** (M1c vs M1d) |

### Output Format

**Config JSONs** (one per cell):

- v1 methodology: `benchmark/results/configs/{Dataset}__{Miner}__{Method}.json`
- v2 methodology default: `/tmp/benchmark_{METHOD}_{DS}_*/results/{Dataset}__{Miner}__{Method}.json`
- v2 methodology production (`--output benchmark/results/configs_v2`): `benchmark/results/configs_v2/{Dataset}__{Miner}__{Method}.json`

Every (dataset, miner, method) cell produces a JSON file recording the exact configuration and results. Config JSONs are the **source of truth** — the CSVs below are derived from them.

v2 schema adds these optional result fields to the v1 schema:
`gen_accept`, `gen_accept_std`, `gen_shadow_regular`, `gen_shadow_mutated`,
`duplicates_kept`, `truncated_traces`, `max_trace_length_used`.

**Required fields per method (v1 schema):**

```json
{
  "dataset": "Sepsis",
  "miner": "Inductive (Strict)",
  "method": "M1d",
  "method_label": "HybridGen v2.4",
  "timestamp": "2026-06-05T14:30:00Z",
  "host": "local|cip-pool",
  "seed": 42,
  "parameters": {
    "max_n": 6,
    "safe_threshold": 5,
    "num_shadow_traces": 1000,
    "iterations": 5
  },
  "results": {
    "mean": 0.8523,
    "std": 0.0142,
    "raw_iterations": [0.841, 0.855, 0.867, 0.842, 0.856],
    "runtime_s": 45.2
  },
  "notes": ""
}
```

**Method-specific parameter schemas:**

| Method(s) | Parameters to Record |
|-----------|---------------------|
| M1a–M1d | `max_n`, `safe_threshold`, `num_shadow_traces`, `iterations` |
| M1e (v2.5) | same as M1d + `duplicates_kept`, `truncated_traces` |
| M1f, M1g (v2.6) | same as M1e + `successor_weighting` (`"log"` or `"mle"`), `gen_accept`, `gen_accept_std`, `gen_shadow_regular`, `gen_shadow_mutated`, `max_trace_length_used` |
| M2 | (none — deterministic) |
| M3 | `jar_version` (relevance.jar), `dfg_simulation_method` |
| M4 | `jar_version`, `timeout_s` |
| M5 | `checkpoint_epoch`, `temperature`, `strategy` (naive/mh), `n_samples` |
| M6 | `n`, `m`, `g`, `k`, `p`, `jar_version` |
| M7 | `n_gram_range` (e.g., [1,5]), `simulation_repeats` |
| M8 | `oracle` (global/local), `matching` (exact/partial), `noise_threshold`, `occurrence`, `balance` |
| R1, R2 | `k` (folds), `shuffles`, `variant_based` (true) |
| R3 | `num_traces`, `max_trace_length` |

**Primary CSV**: `benchmark/results/generalization_benchmark_v24.csv`

```
Dataset, Miner, Method, N, Mean, Std, CI_Lower, CI_Upper, Runtime_s, Notes
```

**Raw CSV**: `benchmark/results/generalization_benchmark_v24_raw.csv` — per-iteration scores.

**Reference CSV**: `benchmark/results/reference_metrics_v24.csv` — R1–R3 scores.

---

## How to Run (v2)

Please refer to `BenchmarkGuide.md` for detailed instructions on setting up the environment, running the benchmark, and interpreting results.

---

## Integration Plan per External Method

### M2 — PM4Py Built-in Generalization

```python
from pm4py.algo.evaluation.generalization import algorithm as generalization_eval
score = generalization_eval.apply(log, net, im, fm)
```

Zero integration cost. Already used in existing benchmarks.

---

### M3 — Entropic Relevance (relevance.jar)

**What it computes:** Entropic relevance measures how well a stochastic process model (DFG→automaton with probabilities) explains the event log in information-theoretic terms. The current implementation uses a **per-miner DFG (simulated via PM4Py `play_out`, 5000 traces)** → **JDFG2Aut** (converts DFG JSON to a probability-annotated automaton) → **Relevance** (computes entropic relevance). Previously used a single log-level DFG via the closed-source Entropia JAR, returning identical non-discriminating values; fixed 2026-06-26 via per-miner DFGs, and switched 2026-06-27 to open-source `relevance.jar` (github.com/promtecmx/relevance).

**Dependencies:**
- JDK 21+
- `relevance.jar` + `OpenXES-20180810.jar` in `./src/relevance/`

**CLI:**
```bash
# Step 1: DFG → probability automaton
java -cp relevance.jar:OpenXES.jar org.jbpt.relevance.JDFG2Aut <dfg.json> <outdir/>
# Step 2: entropic relevance
java -cp relevance.jar:OpenXES.jar org.jbpt.relevance.Relevance <automaton.json> <log.xes>
```

**Integration Strategy:**

1. **Prepare models** via `benchmark/job_prepare.py` with `mode="pnml"` — mines all 8 PNMLs and a log-level DFG.
2. **For each miner**: load PNML → simulate (5000 traces via `pm4py.play_out()`) → generate model-level DFG JSON → JDFG2Aut → Relevance.
3. **Parse** the 6th CSV column from Relevance stdout.
4. **Clean up** temporary files.

**Bridge script:** `benchmark/bridges/run_m3.py`

**Estimated runtime per cell:** ~30-120 s (PNML simulation dominates).

**Note:** The ideal approach (Petri net → SDFA conversion) is not yet implemented. The DFG-via-simulation + JDFG2Aut approach is a sound approximation that gives discriminating scores per miner on all evaluated datasets.

---



### M5 — AVATAR (RelGAN)

✅ D1/D2 complete (2 runs, mean±std). 

For setup (Docker build, dataset preparation) and execution, see [`BenchmarkGuide.md §M5`](BenchmarkGuide.md#per-method-scripts).

| Aspect | Detail |
|--------|--------|
| Docker images | `avatar-tf1` (TF1.15) / `avatar-tf2` (TF2) |
| Training | 5000 adv steps, checkpoint suffix varies by dataset (tp_eval-based: D1=3901, D2=3781) |
| Sampling | 10000 traces, naive strategy, greedy longest-match decoding |
| Multi-word fix | GAN token output → greedy longest-match reconstruction (see [BenchmarkGuide.md §M5](BenchmarkGuide.md#per-method-scripts)) |
| Results (D1, 2 runs) | See [BenchmarkGuide.md §5](BenchmarkGuide.md#5-results-d1-sepsis) table |
| Results (D2, 1 run) | See [BenchmarkGuide.md §5b](BenchmarkGuide.md#5b-results-d2-bpi2013-incidents) table |

---

### M6 — Bootstrap Generalization

✅ D1/D2 complete. Uses fixed JAR `jbpt-pm-entropia-1.7.1.jar` (k=2, 10 bootstrap replicates).
Entry point: `benchmark/job_m6.py` (core algorithm: `benchmark/bridges/run_m6_bgen.py`).

For setup, parameters, and execution, see [`BenchmarkGuide.md §M6 Implementation Note`](BenchmarkGuide.md#m6-implementation-note). Results in [§5 (D1)](BenchmarkGuide.md#5-results-d1-sepsis) and [§5b (D2)](BenchmarkGuide.md#5b-results-d2-bpi2013-incidents).

Notable fixes:
- **JAR NPE fix**: `EventLogSampling.java:101` null guard → enabled k=2 on all datasets (see [Src Repositories](BenchmarkGuide.md#2-src-repositories)).
- **Runner script**: `benchmark/job_m6.py` (core: `benchmark/bridges/run_m6_bgen.py`) automates -bgen invocation, output parsing, and config JSON writing.

| Parameter | Design Value | Guide Default |
|-----------|-------------|---------------|
| `m` (replicates) | 10 (target: 100) | `--m 10` |
| `n` (sample size) | 200 | `--n 200` |
| `g` (generations) | 10 | `--g 10` |
| `k` (subtrace length) | 2 | `--k 2` |

---

---

### M7 — SpeciAL4PM (Species Diversity)

**What it computes:** Compares the species diversity/coverage profile of the original event log against the profile of traces simulated from the model. A well-generalizing model should produce simulated traces with similar diversity to the original log.

The primary generalization proxy is the **coverage (C1) ratio**:
```
gen_score = C1_simulated / C1_original
```
Values close to 1.0 indicate the model captures the log's diversity. Values < 1.0 suggest the model overfits (insufficient diversity in simulated traces). Values > 1.0 suggest underfitting (model simulates too much unobserved behavior).

**Dependencies:**
- Python 3.8+ (compatible with our environment).
- PM4Py (already installed).
- Source: `./src/SpeciAL-core/special4pm/`

**Integration Strategy:**

1. **Profile the original log**: Use `SpeciesEstimator` with 1-to-5-gram species retrieval, compute C1 (coverage) at full log size.
2. **Simulate traces from the model**: Use `simulate_model(net, im, fm, size=len(log))` to generate a log of equal size.
3. **Profile the simulated log**: Same estimator, same N-grams, record C1.
4. **Compute generalization score**: `gen = C1_simulated / C1_original` (clipped to [0,1]).
5. **Repeat 3×** (model simulation is stochastic) and report mean ± std.

**Bridge script:** `benchmark/bridges/special4pm_bridge.py`

**Estimated runtime per cell:** 1–5 min (model simulation + species estimation).

**Fallback:** If model simulation produces empty traces (deadlocked model), exclude that miner. For very small models (e.g., Flower Model), the simulation may loop infinitely — impose a `maxTraceLength=1000` cap.

---



## Execution Order & Dependencies

### SLURM Resource Constraints (CIP-Pool `krater` Partition)

From `sacctmgr`:
- **MaxJobs**: 15 (running jobs)
- **MaxSubmit**: 30 (queued + running)
- **QOS**: normal

R2 (Leave-One-Variant-Out) on variant-heavy datasets uses SLURM **array jobs** with `--array=0-29` (30 sub-jobs = MaxSubmit cap). Each array task processes a slice of variants:

| Dataset | Variants | Array size | Variants/task | Wall time estimate |
|---------|----------|-----------|---------------|-------------------|
| D3 BPI 2017 | 15,930 | 30 | ~531 | ~10–20 min per task (10 min discovery × 531 variants) |
| D4 BPI 2018 | 28,000 | 30 | ~933 | ~30–60 min per task |
| D5 BPI 2019 | 11,973 | 30 | ~399 | ~8–15 min per task |

Other methods (M1–M8) run as single-node jobs within the same partition. No array parallelization needed.

### JSON Config Recording Requirement

**Every experiment run MUST produce a sidecar JSON file** recording the exact configuration used. Without this, a result is untrustworthy — you cannot know whether the score came from `max_n=3` or `max_n=6`, `iterations=5` or `iterations=1`, etc. Results without matching config JSONs are treated as invalid.

Config JSONs are written to:

- v1: `benchmark/results/configs/{Dataset}__{Miner}__{Method}.json`
- v2: `benchmark/results/configs_v2/{Dataset}__{Miner}__{Method}.json`

Example: `Sepsis__Inductive_Strict__M1.json`

### Phase A: Local Machine — Small/Medium Datasets (D1, D2, D6, D7, D14–D19)

All fast methods (M1–M3, M5–M7, R1–R3) can be computed on these 10 datasets locally.
M5 (AVATAR) may be skipped if no GPU is available.

```
Step 1: D1 Sepsis — Completed ✅
  ├── M1a–M1g, M2, M3, M5, M6, M7, R1–R3 — all complete
  └── Configs written to configs_v2/

Step 2: D2 BPI 2013 Incident — Completed ✅
  ├── M1a–M1g, M2, M3, M5, M6, M7, R1–R3 — all complete
  └── Configs written to configs_v2/

Step 3: Small BPI 2013 datasets (Phase A)
  ├── D6  BPI 2013 Problem Open    — very small (819 cases, 3 activities)
  ├── D7  BPI 2013 Problem Closed  — small (1,487 cases, 4 activities)
  └── All methods expected to run in < 10 min each

Step 4: BPI 2020 family (Phase A)
  ├── D14 BPI 2020 PrepaidTravel     (2,099 cases, 29 acts)
  ├── D15 BPI 2020 InternationalDecl. (6,449 cases, 34 acts)
  ├── D16 BPI 2020 RequestForPayment  (6,886 cases, 19 acts, 89 variants — simplest)
  ├── D17 BPI 2020 PermitLog          (7,065 cases, 51 acts, 1.5K variants)
  ├── D18 BPI 2020 DomesticDecl.      (10,500 cases, 17 acts, 99 variants)
  └── All methods run in < 30 min per dataset

Step 5: D19 BPI 2012 (Phase A)
  ├── 13,087 cases, 24 acts, 4,366 variants
  ├── Fast methods: ~30 min
  └── R2 (Leave-One-Variant-Out) may need SLURM array (~146 variants/job)

Step 6: Validate pipeline
  ├── D1, D2: 8 miners × 11 methods = 88 configs each
  ├── All Phase A datasets: verify config JSONs written
  └── Config JSONs are source of truth
```

### Phase B: CIP-Pool 128GB Machine — Heavy Datasets (D3–D5, D8–D13, D20–D21)

Transfer codebase to the 128GB machine. All methods computed from scratch.

**Heavy datasets split into sub-phases by memory profile:**

- **High-activity (Phase B1):** D8–D13 (BPI 2015 municipalities + BPI 2011 Hospital) — modest case counts but 356–624 unique activities, deep traces (avg 42–131), causing memory pressure during N-gram pre-computation and SDFA conversion.
- **High-volume (Phase B2):** D20 Hospital Billing (100K cases) + D21 Road Traffic Fine (150K cases) — large case counts stress token replay and R2.
- **Heaviest (Phase B3):** D3 (31K cases, 16K variants), D4 (44K cases, 28K variants), D5 (252K cases) — original heavy trio.

```
Step 7: Re-run Environment Setup on CIP-Pool machine

Step 8a: D8  BPI 2015 Muni. 2  (832 cases, 410 acts, 34 MB)
  ├── Available miners (5/8): Flower, Trace_Filtered, Heuristics, Heuristics_Strict,
  │     Inductive_Infrequent
  ├── Unavailable: Alpha (timeout), Alpha+ (timeout), Inductive_Strict (RecursionError)
  ├── M1a–M1g (HybridGen)          — ~5–10 min (N-gram state blowup from 410 acts)
  ├── M2 (PM4Py), M3, M5–M7, R1–R3 — ~30 min combined
  └── Write config JSON for every cell

Step 8b: D9  BPI 2015 Muni. 4  (1,053 cases, 356 acts, 37 MB)
  ├── Same miner availability as D8 (5/8)
  ├── M1a–M1g (HybridGen)          — ~5–10 min
  ├── M2–M7, R1–R3               — ~30 min combined
  └── Write config JSON

Step 8c: D10 BPI 2015 Muni. 1  (1,199 cases, 398 acts, 41 MB)
  ├── Same miner availability as D8 (5/8)
  ├── M1a–M1g                     — ~5–10 min
  ├── M2–M7, R1–R3               — ~30 min combined
  └── Write config JSON

Step 8d: D11 BPI 2011 Hospital   (1,143 cases, 624 acts, 131.5 avg len)
  ├── Available miners (5/8): Flower, Trace_Filtered, Heuristics, Heuristics_Strict,
  │     Inductive_Infrequent
  ├── Unavailable: Alpha (timeout), Alpha+ (timeout), Inductive_Strict (timeout)
  ├── M1a–M1g                     — ~10–20 min (deepest traces → large N-gram states)
  ├── M2–M7, R1–R3               — ~45 min combined
  └── Write config JSON

Step 8e: D12 BPI 2015 Muni. 5  (1,156 cases, 389 acts, 46 MB)
  ├── Same miner availability as D8 (5/8)
  ├── M1a–M1g                     — ~5–10 min
  ├── M2–M7, R1–R3               — ~30 min combined
  └── Write config JSON

Step 8f: D13 BPI 2015 Muni. 3  (1,409 cases, 383 acts, 47 MB)
  ├── Available miners (4/8): Flower, Trace_Filtered, Heuristics, Heuristics_Strict
  ├── Unavailable: Alpha (timeout), Alpha+ (timeout), Inductive_Infrequent (RecursionError),
  │     Inductive_Strict (RecursionError)
  ├── M1a–M1g                     — ~5–10 min
  ├── M2–M7, R1–R3               — ~30 min combined
  └── Write config JSON

Step 8g: D20 Hospital Billing     (100,000 cases, 6.6 MB)
  ├── M1a–M1g                     — ~10–15 min
  ├── M2–M7, R1–R3               — ~1 h combined (token replay on 100K traces)
  ├── R2                          — SLURM array (MaxSubmit=30 → ~34 variants/job)
  └── Write config JSON

Step 8h: D21 Road Traffic Fine    (150,370 cases, 3.5 MB)
  ├── M1a–M1g                     — ~10–15 min
  ├── M2–M7, R1–R3               — ~1.5 h combined (largest case count)
  ├── R2                          — SLURM array (MaxSubmit=30 → ~8 variants/job)
  └── Write config JSON

Step 9: D3 BPI 2017 (heavy: variant explosion + deep traces)
  ├── M1a–M1g (HybridGen) — ~15–30 min (N-gram state blowup)
  ├── M2 (PM4Py) — ~1 s
  ├── M3, M4, M6, M7, M8 — ~30–90 min
  ├── R1 (K-Fold CV, k=5) — ~5 min
  ├── R2 (Leave-One-Variant-Out) — parallelize via SLURM array job (MaxSubmit=30 → ~531 variants/job, MaxJobs=15 running concurrently)
  ├── R3 (Random baseline) — ~1 min
  ├── M5 (AVATAR) — ~4–8 hours
  └── Write config JSON for every cell

Step 10: D4 BPI 2018 (heaviest: 28K variants, 2.5M events, 158 MB compressed)
  ├── M1a–M1g (HybridGen) — ~30–60 min (massive N-gram state space)
  ├── M2–M8 — ~2–6 hours combined
  ├── R1 (K-Fold CV, k=5) — ~10 min
  ├── R2 (Leave-One-Variant-Out) — parallelize via SLURM array job (MaxSubmit=30 → ~933 variants/job, MaxJobs=15 running)
  ├── R3 (Random baseline) — ~1 min
  ├── M5 (AVATAR) — ~6–12 hours
  ├── ⚠️ Risk: PM4Py read_xes on 2.5M events may OOM even on 128GB
  └── Write config JSON for every cell

Step 11: D5 BPI 2019 (heavy: 251K cases in RAM)
  ├── M1a–M1g (HybridGen) — ~15–30 min
  ├── M2–M8 — ~1–3 hours combined
  ├── R1 (K-Fold CV, k=5) — ~5 min
  ├── R2 (Leave-One-Variant-Out) — parallelize via SLURM array job (MaxSubmit=30 → ~399 variants/job, MaxJobs=15 running)
  ├── R3 (Random baseline) — ~1 min
  ├── M5 (AVATAR) — ~4–8 hours
  └── Write config JSON for every cell

Step 12: Aggregate results across all 21 datasets
  ├── Validate all config JSONs are present and consistent
  ├── Compile primary CSV from config JSON pool
  └── Produce analysis deliverables (correlation matrix, leaderboard, etc.)
```

**Total estimated wall-clock time:**
- Local (D1, D2, D6, D7, D14–D19): ~20–30 hours total (M5 AVATAR may be skipped on local; without AVATAR ~6–12 h)
- CIP-Pool (D3–D5, D8–D13, D20–D21): ~80–200 hours (M5 AVATAR + M6 Bootstrap Gen dominate; D4 BPI 2018 is the worst case; high-activity municipality logs also add overhead)

---

## Analysis Deliverables (Post-Benchmark)

1. **Leaderboard table**: Rank all methods (M1–M8) by mean score per dataset, with miner-level breakdown.
2. **Correlation matrix**: Pairwise Pearson/Spearman correlation between all generalization methods (M1–M8) plus reference metrics (R1–R2). Cluster methods by paradigm (structural / entropy / adversarial / generative / pattern-based / diversity).
3. **Agreement with ground truth**: Scatter plot of each method vs. R1 (K-Fold CV fitness). Methods correlating most strongly with K-fold fitness capture "true" generalization. Compute **Pearson, Spearman, MAE, spread** over the six real miners; exclude poles.
4. **Discriminative power**: Per method, compute the spread (max − min) across miners on the same dataset. A good metric cleanly separates Trace Model (low) from Flower Model (high).
5. **Ablation delta table**: M1d vs. M1a vs. M1b — quantify the incremental contribution of Katz backoff, log weighting, and context-aware termination. Extended to M1e–M1g for v2.5/v2.6 deltas.
6. **Runtime comparison**: Bar chart of per-method wall-clock time. Highlight cost-to-value ratio of heavy methods (AVATAR, Bootstrap Gen) vs. lightweight methods (HybridGen, PM4Py, Entropic Relevance).
7. **Paradigm agreement analysis**: Do methods within the same paradigm (e.g., M3 + M6 entropy-based, M5 + M4 adversarial, M7 + M8 pattern-based) agree more with each other than with methods from other paradigms?

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| AVATAR TF 1.15 vs. current Python 3.12 | High | Docker image `avatar-tf1` (nvcr.io TF 1.15 + pm4py 1.2.6) |
| AVATAR GAN training too slow for 21 datasets | High | Pre-train one GAN per dataset; reuse across miners |
| AVATAR multi-word activity names | High | Greedy longest-match decoder (see M5 section) |
| Anti-Alignment ILP very slow on large logs | High | ❌ **Archived** — 14h per miner, infeasible |
| BPI 2018 (D4) OOM on local machine (28K variants, 2.5M events, 158 MB) | High | Skip D4 on local; run exclusively on CIP-Pool 128GB machine |
| BPI 2017 (D3) OOM on local machine (15,930 variants + deep traces) | High | Skip D3 on local; run exclusively on CIP-Pool 128GB machine |
| BPI 2019 (D5) OOM on local machine (251K cases) | High | Skip D5 on local; run exclusively on CIP-Pool 128GB machine |
| HybridGen N-gram state explosion on BPI 2017 and BPI 2018 at N=6 (pre-computing 6-gram states) | Medium | Cap `max_n=6`; the Katz backoff mechanism already falls back to N=1 for collapsed states. Monitor memory during N-gram pre-computation on D4 (28K variants). |
| BPI 2018 158 MB compressed — PM4Py read_xes may exceed memory | High | Use chunked XES parsing if available; skip Alpha/Alpha+ miners on D4 if discovery exceeds 30 min |
| Results from unknown configurations (e.g., prior runs without JSON provenance) are treated as invalid | Low | All scores must be regenerated with known, recorded configurations; never import results without matching config JSON |
| Pattern-based Gen lpsolve native lib unavailable | Medium | Detect at setup; skip M8 entirely if lib missing |
| Entropic Relevance SDFA conversion from Petri net | Medium | Current per-miner DFG simulation is a viable approximation; full SDFA conversion (PM4Py reachability graph + token replay probabilities) is future work |
| Bootstrap Gen entropy JAR fails on certain models | Medium | Fallback to alignment-based fitness only |
| Bootstrap Gen breeding produces no valid traces for small logs (e.g., Sepsis) | Medium | Fall back to nonparametric bootstrap (`-p=0`) for logs with < 100 traces |
| Model simulation deadlocks (SpeciAL4PM, AVATAR) | Medium | Impose maxTraceLength; catch empty simulations; exclude miner if consistently deadlocking |
| relevance.jar SDFA validation — no known-good score to verify bridge script | Low | Use pre-built `sdfa_sepsis_1.000.json` + `sepsis.xes.gz` from Entropia `examples/` as a smoke test for `promtecmx/relevance` |
| Alpha/Alpha+/Inductive_Strict miners timeout or crash on D8–D13 (high-activity logs) | High | Skip unavailable miners per `benchmark/statistics/_miner_availability.json`; accept 4–5 miner coverage on these datasets |
| JDK 1.8 unavailable | Low | All Java methods blocked; pre-compute offline or skip Java-dependent methods |
| Memory blowup from storing all raw scores | Low | Stream to CSV incrementally |

---

## Archived Methods

### M4 — Anti-Alignment Generalization (AntiAlignments JAR)

> ❌ **Archived** — The algorithm is inherently single-threaded O(n²~n³). On D1 Sepsis (1,050 traces), Alpha+ ran for 14 hours without completing a single miner. Verified working on mini dataset (10 traces, Gen=0.7125, 53ms) but infeasible on any real-life log. Scripts moved to `archive/Tianhao/benchmark/`.
>
> **2026-06-27 re-evaluation**: Tested with [ProcessMPUT/processm](https://github.com/ProcessMPUT/processm) Kotlin implementation (`antialignments` module). On D1 Sepsis with Alpha miner: still infeasible — no result after 10+ hours. The algorithm's exponential complexity remains the fundamental blocker.

### M8 — Pattern-based Generalization (AutomataConformance)

> ❌ **Archived** — The JAR catches all exceptions internally and returns "t/out". After fixing xvfb (`--auto-servernum`) and enabling stderr capture, all miners still return "t/out" within seconds. The underlying algorithm (ILP-based pattern matching) is too unstable for real-life logs. Scripts moved to `archive/Tianhao/benchmark/`.

---

## Decision Log

- **2026-06-11** — M1g (v2.6-mle) is the recommended headline configuration: it dominates all
  other M1a–M1g versions on every agreement criterion on D1 (4 seeds) and D2 (2 seeds), is the
  only mode that ranks D2 correctly (Spearman 1.0 vs 0.943), and costs the same runtime.
  `'log'` weighting is retained as M1f for rare-behavior stress-testing.
  *Pending: practical partner sign-off before the report/benchmark headline switches from M1d (v2.4) to M1g.*

---

## References

- **Entropic Relevance**: Polyvyanyy, A., et al. (2020). "Entropic Relevance: A Mechanism for Measuring Stochastic Process Model Quality." *arXiv:2007.09310*. Implementation: [promtecmx/relevance](https://github.com/promtecmx/relevance); original: [jbpt/codebase](https://github.com/jbpt/codebase/tree/master/jbpt-pm/entropia)
- **Anti-Alignment Generalization**: van Dongen, B. (2017). "Computing Alignments of Event Data and Process Models." *Transactions on Petri Nets and Other Models of Concurrency*. Original: [ProM AntiAlignments](https://github.com/promworkbench/AntiAlignments); Kotlin reimplementation: [ProcessMPUT/processm](https://github.com/ProcessMPUT/processm) (`processm.conformance/src/main/kotlin/processm/conformance/models/antialignments`)
- **AVATAR**: Theis, J. & Darabi, H. (2020). "Adversarial System Variant Approximation to Quantify Process Model Generalization." *IEEE Access*, 8, 194410–194427. [Julian-Theis/AVATAR](https://github.com/Julian-Theis/AVATAR)
- **Bootstrap Generalization**: Polyvyanyy, A., et al. (2022). "Bootstrapping Generalization of Process Models." *Information Systems*. [lgbanuelos/bsgen](https://github.com/lgbanuelos/bsgen)
- **SpeciAL4PM**: Kabierski, M., et al. (2023). "Addressing the Log Representativeness Problem Using Species Discovery." *ICPM 2023*. [MartinKabierski/SpeciAL-core](https://github.com/MartinKabierski/SpeciAL-core)
- **Pattern-based Generalization**: Reißner, D., et al. (2020). "Scalable Conformance Checking of Process Models." *Journal of Systems and Software*. [reissnda/AutomataConformance](https://github.com/reissnda/AutomataConformance)
- **PM4Py baseline**: van der Aalst, W. M. P. (2016). *Process Mining: Data Science in Action*. Springer.
- `Method2Log.md`, `Method2Log_Geng.md` — Method 2 development logs.
- `ExperimentDesign.md` — archived at `archive/Tianhao/ExperimentDesign.md`
- `Method_GenShadow.md` — Gen_shadow metric specification (authoritative).
- `WhatChanged_v25_v26.md` — v2.5/v2.6 technical summary.
