# Generalization Benchmark — Design Specification v2

> Construct definition follows [`Method_GenShadow.md`](Method_GenShadow.md):
> generalization = acceptance of future *valid* behavior, strictly separated from precision.
>
> v2 methodology results in **`benchmark/results/configs_v2/`**.

---

## Objective

Compare **HybridGen v2.4–v2.6** against 7 external generalization baselines across 21 real-world event logs (D1–D21) and 8 process discovery configurations. The benchmark produces structured JSON scores per (dataset, miner, method), enabling quantitative ranking, correlation analysis, and assessment of each method's discriminative power.

---

## Methods Under Comparison

### Tier 1 — HybridGen (Our Method)

| # | Method | Algorithm | Key property |
|---|--------|-----------|--------------|
| M1a | HybridGen v1 | 1-gram DFG + Good–Turing | Simplest ablation |
| M1b | HybridGen v2.1, N=3 | Flat termination | Isolates context-aware termination |
| M1c | HybridGen v2.1, N=6 | Flat termination | Isolates N=3→6 upgrade |
| M1d | HybridGen v2.4 | Uniform mutation, ln-damped sampling | v1 methodology baseline |
| M1e | HybridGen v2.5 | Katz-consistent mutation | Mutations drawn from backed-off context |
| M1f | HybridGen v2.6 (log) | v2.5 + acceptance rate + length cap | Stress-test mode |
| M1g | HybridGen v2.6 (mle) | v2.6 with `successor_weighting='mle'` | **Headline candidate** |

All M1 variants report mean ± std over 5 iterations × 1000 shadow traces, seed 42, `max_n=6` (except M1a/M1b).

### Tier 2 — External Baselines

| # | Method | Paradigm | Output | Runtime |
|---|--------|----------|--------|---------|
| M2 | **PM4Py Built-in** | Structural counting | Scalar [0,1] | < 1 s |
| M3 | **Entropic Relevance** | Entropy-based | Scalar [0,1] | ~30–120 s |
| M5 | **AVATAR (RelGAN)** | GAN-based | Scalar [0,1] | ~4 h (GPU) |
| M6 | **Bootstrap Generalization** | Entropia `-bgen` eigenvalue | Precision & recall [0,1]; F1 | ~45–60 s |
| M7 | **SpeciAL4PM** | Species diversity | C1 ratio [0,1] | ~12 s |

M4 (Anti-Alignment) and M8 (Pattern-based) are archived — infeasible on real-life logs.
See `BenchmarkGuide.md` for implementation details and setup of each method.

### Tier 3 — Reference Metrics (Ground Truth)

| # | Method | Approach | Role |
|---|--------|----------|------|
| R1 | **K-Fold CV** (5-fold, variant-based) | Split variants → discover on train → replay on test | Empirical ground truth |
| R2 | **Leave-One-Variant-Out** | Hold out each variant in turn | Finer-grained CV |
| R3 | **Naive Random Baseline** | Uniform random traces → token replay | Lower bound |

---

## Datasets

21 datasets spanning BPI Challenges 2011–2020, Sepsis, Hospital Billing, and Road Traffic Fine Management. All reside under `data/`.

| # | Dataset | Cases | Events | Acts | Variants | Avg Len |
|---|---------|-------|--------|------|----------|---------|
| D6 | BPI 2013 Problem Open | 819 | 2,351 | 3 | 108 | 2.9 |
| D7 | BPI 2013 Problem Closed | 1,487 | 6,660 | 4 | 183 | 4.5 |
| D8 | BPI 2015 Mun. 2 | 832 | 44,354 | 410 | 828 | 53.3 |
| D9 | BPI 2015 Mun. 4 | 1,053 | 47,293 | 356 | 1,049 | 44.9 |
| **D1** | **Sepsis** | **1,050** | **15,214** | **16** | **846** | **14.5** |
| D10 | BPI 2015 Mun. 1 | 1,199 | 52,217 | 398 | 1,170 | 43.6 |
| D11 | BPI 2011 Hospital | 1,143 | 150,291 | 624 | 981 | 131.5 |
| D12 | BPI 2015 Mun. 5 | 1,156 | 59,083 | 389 | 1,153 | 51.1 |
| D13 | BPI 2015 Mun. 3 | 1,409 | 59,681 | 383 | 1,349 | 42.4 |
| D14 | BPI 2020 PrepaidTravel | 2,099 | 18,246 | 29 | 202 | 8.7 |
| D15 | BPI 2020 InternationalDecl. | 6,449 | 72,151 | 34 | 753 | 11.2 |
| D16 | BPI 2020 RequestForPayment | 6,886 | 36,796 | 19 | 89 | 5.3 |
| D17 | BPI 2020 PermitLog | 7,065 | 86,581 | 51 | 1,478 | 12.3 |
| **D2** | **BPI 2013 Incidents** | **7,554** | **65,533** | **4** | **1,511** | **8.7** |
| D18 | BPI 2020 DomesticDecl. | 10,500 | 56,437 | 17 | 99 | 5.4 |
| D19 | BPI 2012 | 13,087 | 262,200 | 24 | 4,366 | 20.0 |
| **D3** | **BPI 2017** | **31,509** | **1,202,267** | **26** | **15,930** | **38.2** |
| **D4** | **BPI 2018** | **43,809** | **2,514,266** | **41** | **28,457** | **57.4** |
| D20 | Hospital Billing | 100,000 | 451,359 | 18 | 1,020 | 4.5 |
| D21 | Road Traffic Fine | 150,370 | 561,470 | 11 | 231 | 3.7 |
| **D5** | **BPI 2019** | **251,734** | **1,595,923** | **42** | **11,973** | **6.3** |

> **Acts** = unique activity labels. **Variants** = unique trace sequences. Datasets in **bold** (D1–D5) are the original representative selection; the benchmark runs on all 21.

Miner availability per dataset is recorded in `benchmark/statistics/_miner_availability.json`. High-activity datasets (D8–D13) have fewer available miners (Alpha/Alpha+/Inductive_Strict timeout or crash).

---

## Process Discovery Configurations (8 Miners)

| # | Miner | Construction | Role |
|---|-------|--------------|------|
| 0 | **Filtered Trace Model** (top-50 variants) | One path per variant, capped at 50 most frequent | **0.0 pole** — pure memorization |
| 1 | Alpha Miner | `pm4py.discover_petri_net_alpha()` | Low (overfits) |
| 2 | Alpha+ Miner | `pm4py.discover_petri_net_alpha_plus()` | Low–Moderate |
| 3 | Heuristics (Default) | `pm4py.discover_petri_net_heuristics(log)` | Moderate |
| 4 | Heuristics (Strict) | `dependency_threshold=0.99` | Moderate–High |
| 5 | Inductive (Strict) | `noise_threshold=0.0` | High (block-structured) |
| 6 | Inductive (Infrequent) | `noise_threshold=0.2` | High (filters infrequent paths) |
| 7 | **Flower Model** | All activities in one loop | **1.0 pole** — accepts everything |

**Pole interpretation:** Flower ≈ 1.0 is the expected correct score for a pure generalization metric (construct-purity litmus). Trace Model low anchors the memorization pole. Both poles are excluded from agreement statistics; reported separately as litmus checks.

---

## Evaluation Protocol

1. **Discover model** (self-contained in /tmp workdir)
2. **Evaluate** with per-method protocol; record mean, std, and raw per-iteration scores
3. **Write one config JSON per cell** to output directory: `{Dataset}__{Miner}__{Method}.json`
4. **Ground truth**: R1 (variant-based 5-fold CV, 3 shuffles, seed 42)
5. **Agreement reporting**: Pearson, Spearman, MAE, spread over the six real miners

### Statistical Rigor

| Method(s) | Iterations | Reporting |
|-----------|------------|-----------|
| M1a–M1g | 5 | Mean ± std |
| M2, M3 | 1 (deterministic) | Single value |
| M5 (AVATAR) | 2 sampling runs | Mean ± std |
| M6 (Bootstrap Gen) | 10 bootstrap replicates | Mean ± std; F1(p,r) |
| M7 (SpeciAL4PM) | 1 per N-gram | Profile; C1 ratio |
| R1 (K-Fold CV) | k=5 × 3 shuffles | Mean ± std |
| R2 (LOVO) | 1 pass per variant | Aggregate by mean |
| R3 (Random) | 5 | Mean ± std |

Seed all RNGs with `seed=42`.

---

## Analysis Deliverables

1. **Leaderboard table**: Rank all methods by mean score per dataset.
2. **Correlation matrix**: Pairwise Pearson/Spearman between all methods plus reference metrics.
3. **Agreement with ground truth**: Scatter plots per method vs. R1; compute Pearson, Spearman, MAE, spread.
4. **Discriminative power**: Per-method spread (max − min) across miners on the same dataset.
5. **Ablation delta table**: M1d vs. M1a–M1g — quantify incremental contributions.
6. **Runtime comparison**: Bar chart of per-method wall-clock time.
7. **Paradigm agreement analysis**: Do methods within the same paradigm agree more than across paradigms?

---

## Decision Log

- **2026-06-11** — M1g (v2.6-mle) is the recommended headline configuration: dominates all other M1 variants on every agreement criterion, the only mode that ranks D2 correctly (Spearman 1.0), and costs the same runtime.
- **2026-06-12** — Methodology v2: M1 family expanded to M1a–M1g; Trace_Filtered miner added (0.0 pole).
- **2026-06-26** — M3 switched from single log-level DFG to per-miner simulated DFGs (open-source `relevance.jar`), producing discriminating scores.
- **2026-06-18** — M6 switched from token-replay fitness to Entropia eigenvalue-based scoring (`-bgen`).

---

## References

- **HybridGen** — `Method_GenShadow.md` (this project)
- **Entropic Relevance** — Polyvyanyy et al. (2020). [promtecmx/relevance](https://github.com/promtecmx/relevance)
- **AVATAR** — Theis & Darabi (2020). *IEEE Access*. [Julian-Theis/AVATAR](https://github.com/Julian-Theis/AVATAR)
- **Bootstrap Generalization** — Polyvyanyy et al. (2022). *Information Systems*. [lgbanuelos/bsgen](https://github.com/lgbanuelos/bsgen)
- **SpeciAL4PM** — Kabierski et al. (2023). *ICPM 2023*. [MartinKabierski/SpeciAL-core](https://github.com/MartinKabierski/SpeciAL-core)
- **PM4Py** — van der Aalst (2016). *Process Mining: Data Science in Action*. Springer.
