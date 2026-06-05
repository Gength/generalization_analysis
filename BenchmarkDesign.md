# Generalization Benchmark Design

## Objective

Compare **HybridGen v24** against 7 external generalization baselines across 5 real-world event logs (selected from a catalog of 20+) and 7 process discovery algorithms. The benchmark produces a structured CSV of per-method generalization scores, enabling quantitative ranking, correlation analysis, and qualitative assessment of each method's discriminative power.

---

## Methods Under Comparison

### Tier 1 — Our Method

| # | Method | Approach | Output | Key Innovation |
|---|--------|----------|--------|----------------|
| M1 | **HybridGen v24** | N-gram (N=6) + Katz backoff + context-aware termination + Good-Turing estimation. Generates shadow log by DFS walk, replays on model via token replay. Deduplication ensures no shadow trace copies an original. | Scalar [0,1] | N=6 is the empirical mutation peak on BPI 2017 (49× V1 baseline, 3.8% mutated traces); Katz backoff gracefully degrades on sparser datasets. |

**Ablation baselines** (our own earlier versions, quantifying v24's incremental gains):

| # | Method | Approach | Why Include |
|---|--------|----------|-------------|
| M1a | **HybridGen v1** | DFG + Good-Turing (1-gram, no backoff, flat termination). | Establishes the simplest baseline in our method family. |
| M1b | **HybridGen v2.1 (N=3)** | N-gram + Katz backoff + log-weighted probabilities. Pre-v23, flat per-activity termination, `max_n=3`. | Isolates the contribution of **context-aware termination** (v24's key fix) by holding N=3 constant. |
| M1c | **HybridGen v2.1 (N=6)** | Same as M1b but with `max_n=6`. Pre-v23, flat termination, deeper context. | Isolates the contribution of **N=3→6 upgrade**. Comparing M1c vs M1 reveals the marginal benefit of context-aware termination ON TOP OF N=6. |

### Tier 2 — External Generalization Baselines

| # | Method | Paradigm | Approach | Output | Runtime |
|---|--------|----------|----------|--------|---------|
| M2 | **PM4Py Built-in** | Structural counting | Counts invisible transitions, token-replay deviations, and visited states in the reachability graph. `pm4py.algo.evaluation.generalization.apply()`. | Scalar [0,1] | < 1 s |
| M3 | **Entropic Relevance** (Polyvyanyy et al. 2020) | Entropy-based | Computes the relevance of a stochastic process model (SDFA) to an event log via entropic relevance measure. Part of the Entropia tool. Requires model in SDFA JSON format. | Scalar [0,1] | ~seconds (Java subprocess) |
| M4 | **Anti-Alignment Generalization** (van Dongen, 2017) | Adversarial | Constructs anti-alignments — model-valid traces maximally deviating from the log — then computes a generalization score from the fitness/precision trade-off. Implemented in ProM's AntiAlignments package. | Scalar [0,1] | ~minutes (Java subprocess) |
| M5 | **AVATAR** (Theis & Darabi, 2020) | GAN-based | Trains a RelGAN on observed trace variants → generates synthetic variants → harmonic mean of token-replay fitness and alignment precision. | Scalar [0,1] | ~hours (GAN training + sampling + replay) |
| M6 | **Bootstrap Generalization** (Polyvyanyy et al. 2022) | Bootstrap + breeding | Bootstrap resampling + genetic trace breeding (crossover at shared k-grams) + entropy-based precision/recall. Available in both Python (`bsgen_eval.py`) and Java (Entropia `-bgen`). | Scalar [0,1] per replicate; mean ± 95% CI | ~hours (Java subprocess per replicate) |
| M7 | **SpeciAL4PM** (Kabierski et al. 2023) | Species diversity | Extracts N-gram "species" from the log → estimates Hill diversity profiles (D0/D1/D2) and coverage (C1) → simulates traces from the model → compares simulated vs. original diversity profiles. Generalization score = coverage ratio or profile divergence. | Coverage ratio or scalar [0,1] | ~minutes (PM4Py simulation + species estimation) |
| M8 | **Pattern-based Generalization** (Reißner et al. 2020) | Pattern matching | Identifies concurrent patterns (via concurrency oracle + partial orders) and repetitive patterns (via tandem repeats) in the log → tests each pattern against the model's parallel blocks and loops → weighted average of pattern fulfillments. | Scalar [0,1] | ~minutes (Java subprocess) |

#### Repository & Local Path References

| Method | Repository | Local Path |
|--------|-----------|------------|
| Entropic Relevance | [jbpt/codebase/jbpt-pm/entropia](https://github.com/jbpt/codebase/tree/master/jbpt-pm/entropia) | `./src/codebase/jbpt-pm/entropia/` |
| Anti-Alignment Generalization | [promworkbench/AntiAlignments](https://github.com/promworkbench/AntiAlignments) | `./src/prom_workspace_link/packages/antialignments-6.14.4/AntiAlignments.jar` |
| | | Deps: `./src/prom_workspace_link/packages/`, `./src/prom_workspace_link/lib/`, `./src/prom_workspace_link/dist/` |
| AVATAR | [Julian-Theis/AVATAR](https://github.com/Julian-Theis/AVATAR) | `./src/AVATAR/` |
| Bootstrap Generalization | [lgbanuelos/bsgen](https://github.com/lgbanuelos/bsgen) + [jbpt/codebase](https://github.com/jbpt/codebase/tree/master/jbpt-pm/gen/bootstrap) | `./src/bsgen/` (Python), `./src/codebase/jbpt-pm/entropia/` (Java `-bgen`) |
| SpeciAL4PM | [MartinKabierski/SpeciAL-core](https://github.com/MartinKabierski/SpeciAL-core) | `./src/SpeciAL-core/` |
| Pattern-based Gen. | [reissnda/AutomataConformance](https://github.com/reissnda/AutomataConformance) | `./src/AutomataConformance/` |

### Tier 3 — Reference / Sanity-Check Metrics

These are NOT generalization metrics per se, but provide empirical anchors for interpreting the generalization scores:

| # | Method | Approach | Role |
|---|--------|----------|------|
| R1 | **K-Fold Cross-Validation Fitness** (5-fold, variant-based) | Split log by variants → discover on train → replay test on model. Average over 5 folds. | Empirical "ground truth" — measures actual generalization to held-out data. |
| R2 | **Leave-One-Variant-Out Fitness** | Each variant held out in turn, model discovered on rest, fitness on held-out variant. | Finer-grained CV; catches fragile models that 3-fold might miss. |
| R3 | **Naive Random Baseline** | Generate N random traces (uniform activity sampling, random length 1–max_trace_len) → token-replay on model. | Lower bound — any reasonable method should score above this. |

---

## Datasets

All datasets reside under `data/` with per-directory `summary.txt` files containing full statistics (cases, events, variants, trace length distribution, N-gram sparsity, TLRA). Ordered below from smallest to largest by memory pressure.

### Full Dataset Catalog

| # | Dataset | Path | Cases | Events | Acts | Variants | Avg Len | Size | TLRA |
|---|---------|------|-------|--------|------|----------|---------|------|------|
| — | BPI 2013 Problem Open | `data/BPI-Challenge_2013/` | 819 | 2,351 | 3 | 108 | 2.9 | 69 KB | 0.87 |
| — | BPI 2013 Problem Closed | `data/BPI-Challenge_2013/` | 1,487 | 6,660 | 4 | 183 | 4.5 | 187 KB | 0.88 |
| — | BPI 2015 Municipality 2 | `data/BPI-Challenge_2015/` | 832 | 44,354 | 410 | 828 | 53.3 | 34 MB | 0.005 |
| — | BPI 2015 Municipality 4 | `data/BPI-Challenge_2015/` | 1,053 | 47,293 | 356 | 1,049 | 44.9 | 37 MB | 0.004 |
| D1 | **Sepsis** | `data/Sepsis Cases - Event Log_1_all/` | 1,050 | 15,214 | 16 | 846 | 14.5 | 0.2 MB | 0.19 |
| — | BPI 2015 Municipality 1 | `data/BPI-Challenge_2015/` | 1,199 | 52,217 | 398 | 1,170 | 43.6 | 41 MB | 0.02 |
| — | BPI 2011 Hospital | `data/BPI-Challenge_2011/` | 1,143 | 150,291 | 624 | 981 | 131.5 | 2.4 MB | 0.14 |
| — | BPI 2015 Municipality 5 | `data/BPI-Challenge_2015/` | 1,156 | 59,083 | 389 | 1,153 | 51.1 | 46 MB | 0.003 |
| — | BPI 2015 Municipality 3 | `data/BPI-Challenge_2015/` | 1,409 | 59,681 | 383 | 1,349 | 42.4 | 47 MB | 0.04 |
| — | BPI 2020 PrepaidTravel | `data/BPI-Challenge_2020/` | 2,099 | 18,246 | 29 | 202 | 8.7 | — | 0.90 |
| — | BPI 2020 InternationalDecl. | `data/BPI-Challenge_2020/` | 6,449 | 72,151 | 34 | 753 | 11.2 | — | 0.88 |
| — | BPI 2020 RequestForPayment | `data/BPI-Challenge_2020/` | 6,886 | 36,796 | 19 | 89 | 5.3 | — | 0.99 |
| — | BPI 2020 PermitLog | `data/BPI-Challenge_2020/` | 7,065 | 86,581 | 51 | 1,478 | 12.3 | — | 0.79 |
| D2 | **BPI 2013 Incident** | `data/BPI-Challenge_2013/` | 7,554 | 65,533 | 4 | 1,511 | 8.7 | 1.3 MB | 0.80 |
| — | BPI 2020 DomesticDecl. | `data/BPI-Challenge_2020/` | 10,500 | 56,437 | 17 | 99 | 5.4 | — | 0.99 |
| — | BPI 2012 | `data/BPI-Challenge_2012/` | 13,087 | 262,200 | 24 | 4,366 | 20.0 | 3.3 MB | 0.67 |
| D3 | **BPI 2017** | `data/BPI-Challenge_2017/` | 31,509 | 1,202,267 | 26 | 15,930 | 38.2 | 29.6 MB | 0.49 |
| D4 | **BPI 2018** | `data/BPI-Challenge_2018/` | 43,809 | 2,514,266 | 41 | 28,457 | 57.4 | 158 MB | 0.35 |
| — | Hospital Billing | `data/Hospital Billing - Event Log_1_all/` | 100,000 | 451,359 | 18 | 1,020 | 4.5 | 6.6 MB | 0.99 |
| — | Road Traffic Fine | `data/Road Traffic Fine Management Process_1_all/` | 150,370 | 561,470 | 11 | 231 | 3.7 | 3.5 MB | 0.998 |
| D5 | **BPI 2019** | `data/BPI-Challenge_2019/` | 251,734 | 1,595,923 | 42 | 11,973 | 6.3 | 16.9 MB | 0.95 |

> **Acts** = unique activity labels. **Variants** = unique trace sequences. **TLRA** = 1 − (variants/cases), the probability an additional trace has been seen before. Higher = more repetitive.

### Selected Benchmark Datasets (D1–D5)

From the full catalog, 5 datasets are selected for the actual benchmark, chosen to span diverse process characteristics while keeping runtime feasible:

| # | Dataset | Cases | Events | Variants | Avg Len | Why Selected |
|---|---------|-------|--------|----------|---------|-------------|
| D1 | **Sepsis** | 1,050 | 15,214 | 846 | 14.5 | Smallest — ideal smoke test. Hospital process, high attribute richness. |
| D2 | **BPI 2013 Incident** | 7,554 | 65,533 | 1,511 | 8.7 | Small but diverse (1.5K variants from only 4 activities). Very low structural complexity tests whether metrics over-penalize simple models. |
| D3 | **BPI 2017** | 31,509 | 1,202,267 | 15,930 | 38.2 | Variant explosion (87% singletons) + deep traces — stress-tests HybridGen N-gram states and ILP-based methods. |
| D4 | **BPI 2018** | 43,809 | 2,514,266 | 28,457 | 57.4 | Largest variant count (28K) + deepest traces (avg 57) + lowest TLRA (0.35) — hardest generalization challenge. |
| D5 | **BPI 2019** | 251,734 | 1,595,923 | 11,973 | 6.3 | Largest case count — tests PM4Py memory scaling. Structured purchase-to-pay with rare branches. |

**Why these 5?** They span the axes that matter for generalization evaluation:
- **Trace depth**: from 8.7 (D2) to 57.4 (D4) avg events
- **Variant diversity**: from 846 (D1) to 28,457 (D4)
- **Log size**: from 1K (D1) to 251K (D5) cases
- **TLRA**: from 0.19 (D1, low representativeness) to 0.80+ (D2, moderate)
- **Structural complexity**: D2 has only 4 activities but 1,511 variants — tests whether metrics conflate simplicity with generalization

### Two-Phase Execution Plan

```
Phase A: Local Machine (development, debugging, smoke test)
  ├── D1 Sepsis — 1,050 cases, ~14 min for all fast methods
  └── D2 BPI 2013 Incident — 7,554 cases, still runs locally
      Goal: validate all 8 methods + 3 reference metrics + 7 miners
            end-to-end before scaling up. Both D1 and D2 are small enough
            for local iteration.

Phase B: CIP-Pool 128GB Machine (full benchmark)
  ├── D3 BPI 2017 — 31K cases, 15,930 variants, deep traces → memory-heavy
  ├── D4 BPI 2018 — 44K cases, 28,457 variants, 2.5M events → heaviest
  └── D5 BPI 2019 — 251K cases, 1.6M events → largest raw case count
```

---

**Partial infrastructure worth noting:**
- `gen/bootstrap/inputs_r.csv` provides validated parameter configurations for Sepsis (`VE_30_Sepsis_*`), Road Traffic (`PE_11_Road_Traffic_*`), and BPI Challenge (`PE_01_BPI_Challenge_*`, `VE_01_BPI_Challenge_*`) models. These can serve as a reference for `-bgen` parameter selection (log_size, generations, k_value, breeding_factor).
- Entropia `examples/` contains `sepsis.xes.gz` and a pre-built SDFA model `sdfa_sepsis_1.000.json` — this can be used to validate the Entropic Relevance bridge script against a known-good SDFA before running on our own discovered models.
- SpeciAL4PM eval scripts have hardcoded paths to BPI 2012, Road Traffic, Sepsis, BPI 2018, BPI 2019 — confirming these datasets are compatible with the species estimation pipeline.

---

## Process Discovery Algorithms (Miners)

Covering the generalization spectrum from underfitting to overfitting:

| # | Miner | PM4Py Call | Expected Generalization |
|---|-------|-----------|------------------------|
| 1 | Alpha Miner | `pm4py.discover_petri_net_alpha()` | Low (overfits to noise on real logs) |
| 2 | Alpha+ Miner | `pm4py.discover_petri_net_alpha_plus()` | Low–Moderate |
| 3 | Heuristics (Default) | `pm4py.discover_petri_net_heuristics(log)` | Moderate |
| 4 | Heuristics (Strict) | `pm4py.discover_petri_net_heuristics(log, dependency_threshold=0.99)` | Moderate–High |
| 5 | Inductive (Strict) | `pm4py.discover_petri_net_inductive(log, noise_threshold=0.0)` | High (block-structured) |
| 6 | Inductive (Infrequent) | `pm4py.discover_petri_net_inductive(log, noise_threshold=0.2)` | High (filters infrequent paths) |
| 7 | Flower Model | Manual construction (all activities in one concurrent block) | Lowest (maximum permissiveness) |

---

## Evaluation Protocol

### Per Cell: (Dataset × Miner × Method)

1. **Discover model** on full event log.
2. **Compute PM4Py fitness** (token replay) — sanity check; flag models with fitness < 0.5.
3. **Compute generalization score** for each applicable method.
4. **Record runtime** per method (wall-clock, excluding model discovery).

### Statistical Rigor

| Method(s) | Iterations | Reporting |
|-----------|-----------|-----------|
| M1, M1a–M1c (HybridGen) | 5 | Mean ± std |
| M2 (PM4Py built-in) | 1 (deterministic) | Single value |
| M3 (Entropic Relevance) | 1 (deterministic) | Single value |
| M4 (Anti-Alignment) | 1 (deterministic) | Single value |
| M5 (AVATAR) | 3–5 sampling runs per checkpoint | Mean ± std |
| M6 (Bootstrap Gen) | 100 bootstrap replicates | Mean ± 95% CI |
| M7 (SpeciAL4PM) | 1 per N-gram (1–5 gram) | Profile; use C1 ratio as scalar |
| M8 (Pattern-based) | 1 (deterministic given thresholds) | Single value |
| R1, R2 (K-Fold CV) | **k=5 folds** × 3 shuffles | Mean ± std |
| R3 (Random baseline) | 5 | Mean ± std |

Seed all random number generators for reproducibility (`seed=42`).

### K-Fold CV: Why k=5

Variant-based k-fold partitions trace variants (not individual traces) into k groups. All traces sharing the same activity sequence stay together in train or test — never split across folds.

| k | Training fraction | Per-fold test size (D1) | Per-fold test size (D3) | Assessment |
|---|-------------------|------------------------|------------------------|------------|
| 3 | 67% | ~282 variants | ~5,310 variants | Coarse — test set is 33% of variants, leaving only 67% for model discovery |
| 5 | 80% | ~169 variants | ~3,186 variants | **Recommended** — standard in process mining; 80/20 split per fold; 5 scores averaged |
| 6 | 83% | ~141 variants | ~2,655 variants | Slightly more training data per fold; acceptable for variant-rich datasets (D3–D5) |
| 10 | 90% | ~85 variants | ~1,593 variants | Too small test folds for D1/D2; model discovery on 90% of variants is nearly the full log |

**Choice: k=5.** Provides an 80/20 train/test split per fold (standard in ML), works for all 5 datasets (D1's 846 variants → ~169 per test fold is sufficient), and produces 5 scores to average. k=6 is also defensible but k=5 is more conventional.

### HybridGen v24 Hyperparameters

#### Why max_n=6? Empirical Evidence from Mutation Analysis

The choice of `max_n` was revised from 3 to 6 based on a systematic N-gram sweep experiment on BPI Challenge 2017 (see `analysis/Mutation/MutationReport.md`). The key finding: **N=6 is the empirical mutation peak**, not N=3.

The algorithm's Katz backoff mechanism makes `max_n` an **upper bound**, not a fixed operating point. When states are sparse, the algorithm gracefully degrades to lower N — so increasing `max_n` never produces *worse* results than a lower value; it only adds richer context where the data supports it.

**Empirical N-gram sweep on BPI 2017** (1,000 shadow traces, `safe_threshold=5`, seed=42):

| Metric | N=1 | N=3 | **N=6** | N=8 |
|--------|-----|-----|---------|-----|
| Top-order usage | 100% | 92.6% | **80.0%** | 72.4% |
| Mutation rate (×10⁻³) | 0.027 | 0.376 | **1.333** | 1.208 ↓ |
| Mutated traces (of 1000) | 0 | 12 | **38** | 40 |
| × V1 baseline | 1× | 14× | **49×** | 44× |
| Backoff count | 0 | 466 | 3,788 | 7,514 |

The mutation rate **peaks at N=6** (0.001333) and then *declines* at N=7 (0.001255) and N=8 (0.001208). This is the "Curse of Dimensionality overtaking Context Benefit": beyond N=6, states become so sparse that Katz backoff increasingly falls back to lower orders, reducing effective mutation. The 80% top-order usage at N=6 is still healthy — 4 out of 5 decisions use the richest available context.

**Why this matters for generalization evaluation:**

N=6 provides **38 mutated traces per 1,000** (3.8%) vs. only 12 (1.2%) at N=3. This larger mutation sample enables statistically robust stratified analysis:

| Miner | N=3 Δ (Reg−Mut) | N=6 Δ (Reg−Mut) | Improvement |
|-------|-----------------|-----------------|-------------|
| Heuristics Miner | +0.0256 | **+0.0325** | +27% — mutation vulnerability becomes clearer |
| Alpha Miner | −0.0195 | −0.0035 | Converged — N=3 result was a small-sample artifact |

At N=3, Alpha Miner appeared to have an *inverse* mutation effect (mutated traces scored higher than regular traces — statistically implausible). With N=6's larger sample, this artifact disappeared. N=6 provides more reliable stratified metrics.

**Per-dataset viability:**

| Dataset | N=3 Safe States | N=6 Projected | Verdict |
|---------|----------------|---------------|---------|
| D1 Sepsis | 55.9% | ~30–40% top-order | Katz backoff handles sparsity; effective behavior ≈ N=2–3 |
| D2 BPI 2013 Incident | 88.9% | >75% top-order | Excellent — 4 activities → tiny state space |
| D3 BPI 2017 | 67.0% | **80.0% top-order** (measured) | N=6 is the empirical peak |
| D4 BPI 2018 | 52.0% | ~25–35% top-order | Katz backoff handles sparsity; effective behavior ≈ N=2 |
| D5 BPI 2019 | 51.5% | ~25–35% top-order | Katz backoff handles sparsity; effective behavior ≈ N=2 |

For D4/D5, N=6 will use lower effective N due to backoff — but this is the *correct* behavior. The algorithm automatically uses less context when the data doesn't support more. Setting `max_n=6` does not force N=6; it merely *allows* N=6 when the data justifies it.

**Recommended configuration (fixed across all datasets):**

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `max_n` | **6** | Empirical mutation peak on BPI 2017 (49× V1 baseline). Katz backoff gracefully degrades on sparser datasets — `max_n` is an upper bound, not a fixed operating point. |
| `safe_threshold` | **5** | Well-tested in existing benchmarks. Threshold applies to total transition frequency (not unique outgoing count), so even 4-activity datasets (D2) are safe. |
| `num_shadow_traces` | **min(1000, len(log))** | 1,000 shadow traces for D2–D5. For D1 (1,050 original traces), generate 1,000 — close to 1:1 ratio. At N=6, 1,000 traces yields ~38 mutated traces for stratified analysis. |
| `iterations` | **5** | Algorithm default (`evaluate_miner` in v23.py:186). Each iteration generates a fresh shadow log → replays → scores. 5 iterations gives a tight mean ± std. |

**Ablation experiments (M1a–M1c):**

| Ablation | Configuration | Purpose |
|----------|--------------|---------|
| M1a (v1) | no `max_n` (1-gram DFG only) | Simplest baseline — no N-gram context at all |
| M1b (v2.1 N=3) | `max_n=3`, flat per-activity termination | Isolate **context-aware termination**: compare M1b vs M1 to measure the v24 fix |
| M1c (v2.1 N=6) | `max_n=6`, flat per-activity termination | Isolate **N=3→6 upgrade**: compare M1c vs M1b to measure N-gram depth benefit; compare M1c vs M1 to measure context-aware termination ON TOP OF N=6 |

All three ablations run on **all 5 datasets** with the same `safe_threshold=5`, `num_shadow_traces=min(1000, len(log))`, `iterations=5`.

### Output Format

**Config JSONs** (one per cell): `benchmark/results/configs/{Dataset}__{Miner}__{Method}.json`

Every (dataset, miner, method) cell produces a JSON file recording the exact configuration and results. See the JSON schema in the Execution Order section. Config JSONs are the **source of truth** — the CSVs below are derived from them.

**Primary CSV**: `benchmark/results/generalization_benchmark_v24.csv`

```
Dataset, Miner, Method, N, Mean, Std, CI_Lower, CI_Upper, Runtime_s, Notes
```

**Raw CSV**: `benchmark/results/generalization_benchmark_v24_raw.csv` — per-iteration scores.

**Reference CSV**: `benchmark/results/reference_metrics_v24.csv` — R1–R3 scores.

---

## Integration Plan per External Method

### M2 — PM4Py Built-in Generalization

```python
from pm4py.algo.evaluation.generalization import algorithm as generalization_eval
score = generalization_eval.apply(log, net, im, fm)
```

Zero integration cost. Already used in existing benchmarks.

---

### M3 — Entropic Relevance (Entropia JAR)

**What it computes:** Entropic relevance measures how well a stochastic process model (SDFA) explains the event log in information-theoretic terms. It requires the model as a **stochastic deterministic finite automaton (SDFA) in JSON format**, not a Petri net.

**Dependencies:**
- JDK 1.8+
- Entropia JAR: `./src/codebase/jbpt-pm/entropia/jbpt-pm-entropia-1.8.jar`

**CLI:**
```bash
java -jar jbpt-pm-entropia-1.8.jar -r -rel=<log.xes> -ret=<model.sdfa.json>
```

**Integration Strategy:**

1. **Export event log** as XES via `pm4py.write_xes()`.
2. **Convert Petri net to SDFA**: This is the main challenge. The Petri net must be converted to a stochastic automaton. Options:
   - Use the event log to annotate transition probabilities on the Petri net reachability graph, then export as SDFA JSON.
   - Use PM4Py's `discover_stochastic_petri_net()` or a manual frequency-based annotation.
   - Fallback: Use the Entropia tool's `-emp` / `-emr` (exact matching precision/recall) which accept `.pnml` directly, as an approximation.
3. **Shell out** via `subprocess.run()` to the Entropia JAR.
4. **Parse** the single numeric score from stdout.

**Bridge script:** `benchmark/bridges/entropia_bridge.py`

**Estimated runtime per cell:** < 10 s (JVM startup + computation).

**Fallback:** If SDFA conversion is infeasible, use Entropia's exact matching precision (`-emp`) on PNML directly — while not the "Entropic Relevance" measure per se, it provides a related non-deterministic precision score.

---

### M4 — Anti-Alignment Generalization (AntiAlignments JAR)

**What it computes:** Anti-alignments are model-valid traces that maximally deviate from the observed log. The generalization score is derived from the fitness/precision tension captured by these adversarial traces.

**Dependencies:**
- JDK 1.8+
- AntiAlignments JAR: `./src/prom_workspace_link/packages/antialignments-6.14.4/AntiAlignments.jar`
- ProM libraries: `./src/prom_workspace_link/packages/`, `./src/prom_workspace_link/lib/`, `./src/prom_workspace_link/dist/`

**Integration Strategy:**

The `PatternGeneralization.java` in `AutomataConformance` already wraps the Anti-Alignment computation via:
```bash
java -cp ... main.PatternGeneralization <path/> <log.xes> <model.pnml> AntiAlignmentsGeneralization [timeLimit] [timeUnit]
```

1. **Export** Petri net as PNML via `pm4py.write_pnml()`.
2. **Export** event log as XES.
3. **Invoke** via `subprocess.run()` with classpath pointing to all required ProM JARs.
4. **Parse** the generalization score from stdout (CSV line: `log,model,approach,execution_time,generalization`).

**Bridge script:** `benchmark/bridges/antialign_bridge.py`

**Estimated runtime per cell:** 1–10 min (ILP-based anti-alignment construction).

**Fallback:** Timeout at 10 min per cell; record as "timed out" and exclude from analysis. For large logs (BPI 2017), anti-alignment computation may be infeasible — mark as "not applicable."

---

### M5 — AVATAR (RelGAN)

**What it computes:** Harmonic mean of token-replay fitness and alignment precision on GAN-generated synthetic trace variants.

**Dependencies:**
- **Isolated environment**: Python 3.7, TensorFlow 1.15, PM4Py 1.2.6
- Source: `./src/AVATAR/`

**Integration Strategy:**

1. **Environment isolation**: Conda env `avatar-env` with Python 3.7 + TF 1.15 + PM4Py 1.2.6.
2. **Pre-train one GAN per dataset**: AVATAR's generator learns the log distribution, not the model. Train once per dataset, reuse checkpoint across all 7 miners.
3. **Per-miner replay only**: Export Petri net as PNML → run `generalization.py` with the pre-trained checkpoint and the specific PNML.
4. **Bridge**: `benchmark/bridges/avatar_bridge.py` — shells out to `avatar-env`, invokes the AVATAR pipeline, parses the score.

**Estimated runtime:**
- GAN training per dataset: 2–6 hours (one-time, reusable).
- Per-miner replay: 5–15 min.

---

### M6 — Bootstrap Generalization

**What it computes:** Bootstrap resampling + genetic trace breeding + entropy-based precision/recall, aggregated with confidence intervals.

Two implementations available:
- **Python** (`./src/bsgen/bsgen_eval.py`): Full pipeline with multiprocessing. Calls Entropia JAR for entropy.
- **Java** (`./src/codebase/jbpt-pm/entropia/`, `-bgen` flag): Simpler CLI, requires DFG JSON model format.

**Dependencies:**
- JDK 1.8+
- Entropia JAR: `./src/codebase/jbpt-pm/entropia/jbpt-pm-entropia-1.8.jar` (or `jbpt-pm-entropia-1.6.jar` from `./src/bsgen/`)
- Python dependencies: PM4Py, pandas, numpy

**Integration Strategy (Java path, recommended for simplicity):**

```bash
java -jar jbpt-pm-entropia-1.8.jar -bgen -rel=<log.xes> -ret=<model.json> -n=<sample_size> -m=<replicates> -g=<generations> -k=2 -p=1.0 -s
```

1. **Convert Petri net to DFG JSON**: Build a JSON with nodes (activity + frequency + id) and arcs (from, to, frequency). Labels `INPUT` and `OUTPUT` mark process boundaries. Compute frequencies from the event log's directly-follows relations.
2. **Export** log as XES.
3. **Invoke** Entropia `-bgen`.
4. **Parse** model-system precision/recall from stdout, compute generalization = 2 × (precision × recall) / (precision + recall).

**Parameters:**
- `n` = number of traces in original log (bootstrap sample size).
- `m` = 100 (replicates).
- `g` = 100 (breeding generations).
- `k` = 2 (crossover subtrace length).
- `p` = 1.0 (breeding probability).
- `-s` for silent mode (score only).

**Bridge script:** `benchmark/bridges/bsgen_bridge.py`

**Estimated runtime per cell:** 5–30 min (dominated by Java subprocess calls for entropy per replicate).

**Fallback:** For small logs (Sepsis, < 100 traces), skip genetic breeding — the crossover algorithm may produce no valid new traces. Use the Entropia `-bgen` with `-p=0` (nonparametric bootstrap only). If the entropy JAR is unavailable, fall back to alignment-based fitness only.

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

### M8 — Pattern-based Generalization (AutomataConformance)

**What it computes:** Extracts concurrent patterns (via concurrency oracle + partial orders) and repetitive patterns (via tandem repeats) from the event log, then tests each pattern against the model's parallel blocks and loops. Each pattern is assigned a partial fulfillment score; the final generalization is the weighted average by trace count.

**Dependencies:**
- JDK 1.8+
- Compiled JAR from `./src/AutomataConformance/`
- lpsolve native library (`liblpsolve55j.jnilib` for macOS, `.so`/`.dll` for Linux/Windows)

**CLI** (from `main.PatternGeneralization`):
```bash
java -cp <classpath> main.PatternGeneralization <path/> <log.xes> <model.pnml> PatternBasedGeneralization [global/local] [PartialMatching/ExactMatching] [noiseThreshold] [occurrence] [balance] [timeLimit] [timeUnit]
```

Four variants:
- **Global + ExactMatching**: Uses global concurrency oracle, exact pattern matching.
- **Global + PartialMatching**: Global oracle, partial matching (transitive closure).
- **Local + ExactMatching**: Local concurrency oracle (threshold-based), exact matching.
- **Local + PartialMatching**: Local oracle, partial matching.

**Integration Strategy:**

1. **Export** Petri net as PNML, log as XES.
2. **Invoke** all four variants via `subprocess.run()` with a timeout.
3. **Parse** the CSV output: `generalization concurrent pattern, generalization repetitive pattern, overall generalization`.
4. **Primary score**: Use the `overall generalization` column. Optionally analyze the concurrent vs. repetitive decomposition.

**Default parameters** (from the paper):
- Global oracle: `noiseThreshold=0.02`
- Local oracle: `occurrence=0.55`, `balance=0.1`
- Partial matching: enabled (transitive closure)
- Time limit: 10 min per cell

**Bridge script:** `benchmark/bridges/pattern_gen_bridge.py`

**Estimated runtime per cell:** 1–10 min (ILP-based pattern matching).

**Fallback:** If lpsolve native library is unavailable on the platform, skip the method entirely (no fallback — pattern matching depends on ILP solver). For very large logs, set a 10-minute timeout.

---

## Execution Order & Dependencies

### JSON Config Recording Requirement

**Every experiment run MUST produce a sidecar JSON file** recording the exact configuration used. Without this, a result is untrustworthy — you cannot know whether the score came from `max_n=3` or `max_n=6`, `iterations=5` or `iterations=1`, etc. Results without matching config JSONs are treated as invalid.

Config JSONs are written to `benchmark/results/configs/` with the naming convention:

```
{Dataset}__{Miner}__{Method}.json
```

Example: `Sepsis__Inductive_Strict__M1.json`

**Required fields per method:**

```json
{
  "dataset": "Sepsis",
  "miner": "Inductive (Strict)",
  "method": "M1",
  "method_label": "HybridGen v24",
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
| M1, M1a–M1c | `max_n`, `safe_threshold`, `num_shadow_traces`, `iterations` |
| M2 | (none — deterministic) |
| M3 | `jar_version`, `sdfa_conversion_method` |
| M4 | `jar_version`, `timeout_s` |
| M5 | `checkpoint_epoch`, `temperature`, `strategy` (naive/mh), `n_samples` |
| M6 | `n`, `m`, `g`, `k`, `p`, `jar_version` |
| M7 | `n_gram_range` (e.g., [1,5]), `simulation_repeats` |
| M8 | `oracle` (global/local), `matching` (exact/partial), `noise_threshold`, `occurrence`, `balance` |
| R1, R2 | `k` (folds), `shuffles`, `variant_based` (true) |
| R3 | `num_traces`, `max_trace_length` |

### Phase A: Local Machine — Smoke Test (D1 + D2)

All methods computed from scratch. No results reused from prior runs.

```
Step 0: Environment Setup (local)
  ├── Ensure JDK 1.8+ is available (for M3, M4, M6, M8)
  ├── Build AutomataConformance JAR (M8)
  ├── Create AVATAR conda env (Python 3.7 + TF 1.15) (M5)
  ├── Install SpeciAL4PM as editable package (M7):
  │     pip install -e ./src/SpeciAL-core/
  └── Verify Entropia JAR (jbpt-pm-entropia-1.8.jar) works (M3, M6)

Step 1: D1 Sepsis — All Methods (current env + JDK + isolated AVATAR env)
  ├── M1, M1a–M1c (HybridGen variants) — ~2–5 min
  ├── M2 (PM4Py built-in) — < 1 s
  ├── M7 (SpeciAL4PM) — ~3–5 min
  ├── R1, R2 (K-Fold CV) — ~5 min
  ├── R3 (Random baseline) — ~30 s
  ├── M3 (Entropic Relevance) — ~30 s
  ├── M4 (Anti-Alignment) — ~5–10 min
  ├── M6 (Bootstrap Gen) — ~10–20 min
  ├── M8 (Pattern-based Gen) — ~5–10 min
  ├── M5 (AVATAR) — ~2–3 hours (GAN training)
  └── Write config JSON for every cell

Step 2: D2 BPI 2013 Incident — All Methods
  ├── M1–M1c (HybridGen) — ~5–10 min
  ├── M2 (PM4Py) — < 1 s
  ├── M7 (SpeciAL4PM) — ~5–10 min
  ├── R1, R2 (K-Fold CV) — ~5–10 min
  ├── R3 (Random baseline) — ~30 s
  ├── M3, M4, M6, M8 (Java methods) — ~20–40 min
  ├── M5 (AVATAR) — ~2–4 hours (GAN training)
  └── Write config JSON for every cell

Step 3: Validate pipeline
  ├── All methods return valid scores for all 7 miners on D1, D2
  ├── Config JSONs exist for every (dataset, miner, method) cell
  ├── Config JSONs match the documented parameter schema
  └── Fix any integration issues before Phase B
```

### Phase B: CIP-Pool 128GB Machine — Heavy Datasets (D3–D5)

Transfer codebase to the 128GB machine. All methods computed from scratch.

```
Step 4: Re-run Environment Setup on CIP-Pool machine

Step 5: D3 BPI 2017 (heavy: variant explosion + deep traces)
  ├── M1–M1c (HybridGen) — ~15–30 min (N-gram state blowup)
  ├── M2 (PM4Py) — ~1 s
  ├── M3, M4, M6, M7, M8 — ~30–90 min
  ├── R1, R2, R3 — ~10–20 min
  ├── M5 (AVATAR) — ~4–8 hours
  └── Write config JSON for every cell

Step 6: D4 BPI 2018 (heaviest: 28K variants, 2.5M events, 158 MB compressed)
  ├── M1–M1c (HybridGen) — ~30–60 min (massive N-gram state space)
  ├── M2–M8 — ~2–6 hours combined
  ├── M5 (AVATAR) — ~6–12 hours
  ├── ⚠️ Risk: PM4Py read_xes on 2.5M events may OOM even on 128GB
  └── Write config JSON for every cell

Step 7: D5 BPI 2019 (heavy: 251K cases in RAM)
  ├── M1–M1c (HybridGen) — ~15–30 min
  ├── M2–M8 — ~1–3 hours combined
  ├── M5 (AVATAR) — ~4–8 hours
  └── Write config JSON for every cell

Step 8: Aggregate results across all 5 datasets
  ├── Validate all config JSONs are present and consistent
  ├── Compile primary CSV from config JSON pool
  └── Produce analysis deliverables (correlation matrix, leaderboard, etc.)
```

**Total estimated wall-clock time:**
- Local (D1+D2): ~6–12 hours (M5 AVATAR dominates; everything else is fast)
- CIP-Pool (D3–D5): ~30–80 hours (M5 AVATAR + M6 Bootstrap Gen dominate; D4 BPI 2018 is the worst case)

---

## Analysis Deliverables (Post-Benchmark)

1. **Leaderboard table**: Rank all methods (M1–M8) by mean score per dataset, with miner-level breakdown.
2. **Correlation matrix**: Pairwise Pearson/Spearman correlation between all generalization methods (M1–M8) plus reference metrics (R1–R2). Cluster methods by paradigm (structural / entropy / adversarial / generative / pattern-based / diversity).
3. **Agreement with ground truth**: Scatter plot of each method vs. R1 (K-Fold CV fitness). Methods correlating most strongly with K-fold fitness capture "true" generalization.
4. **Discriminative power**: Per method, compute the spread (max − min) across miners on the same dataset. A good metric cleanly separates Flower Model (low) from Inductive Miner (high).
5. **Ablation delta table**: M1 vs. M1a vs. M1b — quantify the incremental contribution of Katz backoff, log weighting, and context-aware termination.
6. **Runtime comparison**: Bar chart of per-method wall-clock time. Highlight cost-to-value ratio of heavy methods (AVATAR, Bootstrap Gen) vs. lightweight methods (HybridGen, PM4Py, Entropic Relevance).
7. **Paradigm agreement analysis**: Do methods within the same paradigm (e.g., M3 + M6 entropy-based, M5 + M4 adversarial, M7 + M8 pattern-based) agree more with each other than with methods from other paradigms?

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| AVATAR TF 1.15 + Python 3.7 vs. current Python 3.12 | High | Isolated conda env; bridge via subprocess + file I/O |
| AVATAR GAN training too slow for 5 datasets | High | Pre-train one GAN per dataset (log-dependent, not model-dependent); reuse across miners |
| Anti-Alignment ILP times out on large logs (BPI 2017) | High | 10-min timeout per cell; mark as "timed out"; exclude from large-log subset analysis |
| BPI 2018 (D4) OOM on local machine (28K variants, 2.5M events, 158 MB) | High | Skip D4 on local; run exclusively on CIP-Pool 128GB machine |
| BPI 2017 (D3) OOM on local machine (15,930 variants + deep traces) | High | Skip D3 on local; run exclusively on CIP-Pool 128GB machine |
| BPI 2019 (D5) OOM on local machine (251K cases) | High | Skip D5 on local; run exclusively on CIP-Pool 128GB machine |
| HybridGen N-gram state explosion on BPI 2017 and BPI 2018 at N=6 (pre-computing 6-gram states) | Medium | Cap `max_n=6`; the Katz backoff mechanism already falls back to N=1 for collapsed states. Monitor memory during N-gram pre-computation on D4 (28K variants). |
| BPI 2018 158 MB compressed — PM4Py read_xes may exceed memory | High | Use chunked XES parsing if available; skip Alpha/Alpha+ miners on D4 if discovery exceeds 30 min |
| Results from unknown configurations (e.g., prior runs without JSON provenance) are treated as invalid | Low | All scores must be regenerated with known, recorded configurations; never import results without matching config JSON |
| Pattern-based Gen lpsolve native lib unavailable | Medium | Detect at setup; skip M8 entirely if lib missing |
| Entropic Relevance SDFA conversion from Petri net | Medium | Use PM4Py stochastic map + reachability graph → SDFA; fallback to `-emp` exact matching precision on PNML |
| Bootstrap Gen entropy JAR fails on certain models | Medium | Fallback to alignment-based fitness only |
| Bootstrap Gen breeding produces no valid traces for small logs (e.g., Sepsis) | Medium | Fall back to nonparametric bootstrap (`-p=0`) for logs with < 100 traces |
| Model simulation deadlocks (SpeciAL4PM, AVATAR) | Medium | Impose maxTraceLength; catch empty simulations; exclude miner if consistently deadlocking |
| Entropia SDFA validation — no known-good score to verify bridge script | Low | Use pre-built `sdfa_sepsis_1.000.json` + `sepsis.xes.gz` from Entropia `examples/` as a smoke test for the bridge script |
| JDK 1.8 unavailable | Low | All Java methods blocked; pre-compute offline or skip Java-dependent methods |
| Memory blowup from storing all raw scores | Low | Stream to CSV incrementally |

---

## References

- **Entropic Relevance**: Polyvyanyy, A., et al. (2020). "Entropic Relevance: A Mechanism for Measuring Stochastic Process Model Quality." *arXiv:2007.09310*. [jbpt/codebase](https://github.com/jbpt/codebase/tree/master/jbpt-pm/entropia)
- **Anti-Alignment Generalization**: van Dongen, B. (2017). "Computing Alignments of Event Data and Process Models." *Transactions on Petri Nets and Other Models of Concurrency*. [ProM AntiAlignments](https://github.com/promworkbench/AntiAlignments)
- **AVATAR**: Theis, J. & Darabi, H. (2020). "Adversarial System Variant Approximation to Quantify Process Model Generalization." *IEEE Access*, 8, 194410–194427. [Julian-Theis/AVATAR](https://github.com/Julian-Theis/AVATAR)
- **Bootstrap Generalization**: Polyvyanyy, A., et al. (2022). "Bootstrapping Generalization of Process Models." *Information Systems*. [lgbanuelos/bsgen](https://github.com/lgbanuelos/bsgen)
- **SpeciAL4PM**: Kabierski, M., et al. (2023). "Addressing the Log Representativeness Problem Using Species Discovery." *ICPM 2023*. [MartinKabierski/SpeciAL-core](https://github.com/MartinKabierski/SpeciAL-core)
- **Pattern-based Generalization**: Reißner, D., et al. (2020). "Scalable Conformance Checking of Process Models." *Journal of Systems and Software*. [reissnda/AutomataConformance](https://github.com/reissnda/AutomataConformance)
- **PM4Py baseline**: van der Aalst, W. M. P. (2016). *Process Mining: Data Science in Action*. Springer.
- `ExperimentDesign.md` — Our experimental strategy document.
- `Method2Log.md`, `Method2Log_Geng.md` — Method 2 development logs.
