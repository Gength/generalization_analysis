# Process Model Generalization — HybridGen Metric & Benchmark

> **Quantifying process model generalization** via generative N-gram analysis.
>
> Generalization — the ability of a discovered process model to accept *future, valid*
> behavior absent from the recorded event log — is the least understood of the four
> established process model quality dimensions (fitness, precision, generalization,
> simplicity). This project builds **HybridGen**, a metric that derives a stochastic
> trace generator from the event log via variable-order N-gram statistics with
> Katz-style backoff and Good–Turing novelty estimation, and scores a model by
> replaying the resulting synthetic **shadow log**. A cross-paradigm benchmark
> confronts the metric against seven published baselines anchored by variant-based
> cross-validation as ground truth.

---

## Project Structure

```
generalization_analysis/
├── README.md                         # This file
├── pyproject.toml                    # Python project config (uv package manager)
│
├── Method_GenShadow.md               # ★ Authoritative metric specification
├── WhatChanged_v25_v26.md            # v25/v26 technical summary
├── BenchmarkDesign.md                # Benchmark methodology v2 (supersedes v1)
├── BenchmarkGuide.md                 # Quick-start guide for running benchmarks
│
├── HybridGen/                        # ★ Core package
│   ├── __init__.py
│   ├── algorithm/                    # Versioned algorithm files (v1.py … v26.py)
│   │   ├── v1.py                     #   DFG + Good–Turing (1-gram, no backoff)
│   │   ├── v2.py … v22_eval.py      #   N-gram evolution
│   │   ├── v23.py … v24.py          #   Context-aware termination, Katz backoff
│   │   ├── v25.py                    #   Katz-consistent mutation proposal
│   │   └── v26.py                    #   Acceptance rate + MLE successor weighting
│   └── utils.py                     # Module auto-discovery registry
│
├── data/                             # Event logs (XES .gz)
│   ├── Sepsis Cases - Event Log_1_all/
│   ├── BPI-Challenge_2013/
│   ├── BPI-Challenge_2017/
│   ├── BPI-Challenge_2018/
│   └── …
│
├── benchmark/                        # Benchmark scripts, models, results
│   ├── run_m1_family.py             # ★ M1-family runner (v2 methodology)
│   ├── version_comparison.py        #   Multi-seed cross-version comparison
│   ├── version_comparison_analysis.ipynb  # Post-hoc analysis notebook
│   ├── demo_d1.py                   #   D1 Sepsis full run (M1–M1f + M2 + R3)
│   ├── r1_accept.py                 #   R1 ground truth (acceptance-based)
│   ├── 01_prepare_models.py         #   Model discovery for all miners
│   ├── run_all.sh                   #   Full pipeline entry point
│   ├── m1.sh / m3.sh / … / m7.sh   #   Per-method shell scripts
│   ├── models/                      #   Pre-discovered PNML + DFG JSON
│   └── results/
│       ├── configs/                 #   v1 methodology results
│       ├── configs_v2/             #   ★ v2 methodology results
│       ├── version_comparison_D1.csv
│       └── version_comparison_D2.csv
│
├── analysis/                         # Exploration notebooks & reports
│   └── Mutation/                    # N-gram sweep analysis
│
├── report/                           # Preliminary LaTeX paper
│   ├── main.tex                     # "Quantifying Process Model Generalization"
│   ├── main.pdf
│   └── references.bib
│
├── archive/                          # Historical / deprecated code
│   ├── v1/                          # Method 1 (pick-one-out, abandoned)
│   └── Tianhao/                     # Archived experiments & logs
│
└── output/                           # Legacy experiment results
```

---

## HybridGen Metric — How It Works

The metric has three stages:

### 1. N-gram Statistics
The log is reduced to its variants with frequency weights. For every order n = 1..6, record per observed n-gram context the frequency of each successor activity and the frequency with which the context occurs at trace end.

### 2. Shadow Log Generation
A stochastic walker generates synthetic traces:

| Step | Mechanism |
|------|-----------|
| **Context resolution** | Katz-style backoff — try the deepest context first; fall back if support < τ=5 |
| **Novelty estimation** | Good–Turing `p_unseen(s) = N₁(s) / N(s)` — probability the next event is unseen |
| **Mutation** | With probability `p_unseen`, insert a novel activity. **v25+**: drawn from backed-off lower-order context (Katz-consistent); **v24-**: uniform over alphabet |
| **Exploitation** | Otherwise, sample successor proportionally to `ln(f+1)` (log) or raw frequency (MLE, v26+) |
| **Termination** | Context-aware `p_end(s)` resolved with same backoff |

### 3. Replay & Score
Replay the shadow log on the model via PM4Py token replay.
`Gen_shadow = mean trace-level replay fitness` over 5 independent runs.

---

## Algorithm Versions

| Version | Key Innovation | Status |
|---------|---------------|--------|
| v1 | 1-gram DFG + Good–Turing | Historical |
| v2–v22 | N-gram + backoff evolution | Historical |
| **v23** | Context-aware termination; fixed mutation rate over-estimation | Historical |
| **v24** | Stable baseline; uniform mutation proposal, ln-damped sampling | **v1 benchmark baseline** |
| **v25** | Katz-consistent mutation proposal; probe-integrity counters | ✅ |
| **v26 (log)** | v25 + acceptance rate + data-driven length cap | ✅ Stress-test mode |
| **v26 (mle)** | v26 with `successor_weighting='mle'` | **🏆 Headline candidate** |

> **Latest recommendation (2026-06-11):** v26-mle (M1f) dominates all other versions
> on every agreement criterion vs. ground truth, is the only mode that ranks D2
> correctly (Spearman 1.0 vs 0.943), and costs the same runtime.

---

## Benchmark Overview

**8 miner configurations spanning the generalization spectrum:**

| # | Miner | Role |
|---|-------|------|
| 0 | **Trace_Filtered** (top-50 variants) | **0.0 pole** — pure memorization |
| 1–6 | Alpha, Alpha+, Heuristics (default/strict), Inductive (strict/infrequent) | Six "real" miners |
| 7 | **Flower Model** | **1.0 pole** — accepts everything |

**3 tiers of methods:**

| Tier | Methods | What |
|------|---------|------|
| **T1** | M1–M1f (HybridGen v1→v26) | Our method family + ablations |
| **T2** | M2–M7 (PM4Py, Entropic, AVATAR, Bootstrap, SpeciAL4PM) | External baselines |
| **T3** | R1–R3 (K-Fold CV, Leave-One-Out, Random) | Reference / sanity checks |

**5 datasets:** D1 Sepsis → D2 BPI 2013 → D3 BPI 2017 → D4 BPI 2018 → D5 BPI 2019

---

## Quick Start

```bash
# Environment
uv sync

# D1 Sepsis — M1 family benchmark (all 7 versions, 8 miners)
uv run python benchmark/run_m1_family.py --dataset D1

# D2 BPI 2013 Incidents
uv run python benchmark/run_m1_family.py --dataset D2

# Only new versions (v25/v26)
uv run python benchmark/run_m1_family.py --dataset D1 --methods M1d M1e M1f

# Multi-seed robustness check
uv run python benchmark/version_comparison.py --dataset D1 --seeds 42 1 7 99

# Legacy demo (M1–M1f via shell script)
bash benchmark/m1.sh
```

Results → `benchmark/results/configs_v2/` per (dataset, miner, method).

---

## Key Findings (Preliminary)

| Metric | Pearson vs R1 | Spearman vs R1 | MAE |
|--------|:------------:|:--------------:|:---:|
| HybridGen v26-mle | **0.996** | **0.943** | **0.023** |
| HybridGen v24 | 0.989 | 0.886 | 0.031 |
| PM4Py Built-in | **−0.429** | **−0.371** | 0.167 |

- PM4Py's built-in generalization metric correlates **negatively** with held-out fitness.
- v25's Katz-consistent mutation proposal reduces calibration error by ≈3×.
- v26-mle is the only mode that ranks D2 correctly (Spearman 1.0).
- The flower model scores 1.0 under Gen_shadow — this is **correct** for a pure generalization construct (see `Method_GenShadow.md` §1).

---

## Documentation

| Document | What |
|----------|------|
| `Method_GenShadow.md` | **Authoritative** metric specification & design rationale |
| `BenchmarkDesign.md` | Benchmark methodology v2 — full protocol, methods, datasets |
| `BenchmarkGuide.md` | Quick-start guide for running experiments |
| `WhatChanged_v25_v26.md` | v25/v26 plain-language technical summary |
| `report/main.pdf` | Preliminary LaTeX paper (LNCS format) |

Archived / historical:
- `archive/v1/` — Original Method 1 (pick-one-out, abandoned)
- `archive/Tianhao/` — Legacy experiment logs and deprecated methods
- `Method1Log.md`, `Method2Log.md`, `Method2Log_Geng.md` — Development logs (historical reference)

---

## References

- **HybridGen** — `Method_GenShadow.md` (this project)
- **Entropic Relevance** — Polyvyanyy et al. (2020). [jbpt/codebase](https://github.com/jbpt/codebase/tree/master/jbpt-pm/entropia)
- **AVATAR** — Theis & Darabi (2020). *IEEE Access*. [Julian-Theis/AVATAR](https://github.com/Julian-Theis/AVATAR)
- **Bootstrap Generalization** — Polyvyanyy et al. (2022). *Information Systems*. [lgbanuelos/bsgen](https://github.com/lgbanuelos/bsgen)
- **SpeciAL4PM** — Kabierski et al. (2023). *ICPM 2023*. [MartinKabierski/SpeciAL-core](https://github.com/MartinKabierski/SpeciAL-core)
- **PM4Py** — van der Aalst (2016). *Process Mining: Data Science in Action*. Springer.
