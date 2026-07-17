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

## Project Layout

```
.
├── README.md                         # This file
├── pyproject.toml                    # Python project config (uv package manager)
│
├── Method_GenShadow.md               # ★ Authoritative metric specification
├── BenchmarkDesign.md                # ★ Benchmark methodology v2 (coauthored)
├── BenchmarkGuide.md                 # Quick-start guide for running benchmarks
│
├── HybridGen/                        # ★ Core metric package
│   ├── algorithm/                    #   Versioned algorithm files (v1.py … v26.py)
│   └── experiment/                   #   Experiment runners (v1.py, v2.py)
│
├── benchmark/                        # Benchmark scripts, miners, results
│   ├── job_m1.py … job_m7.py         #   Self-contained job wrappers (M1–M7)
│   ├── job_r1.py, job_r2.py, job_r3.py  #   Self-contained reference wrappers (R1–R3)
│   ├── run_m1_family.py              #   M1-family core implementation
│   ├── run_m2.py                     #   M2 core implementation
│   ├── run_r_family.py               #   R1/R2/R3 core implementation
│   ├── bridges/                      #   Bridge scripts for external methods
│   │   ├── run_m3.py                 #   M3 (Entropic Relevance)
│   │   ├── run_m6_bgen.py            #   M6 (Bootstrap Gen)
│   │   └── run_m7.py                 #   M7 (SpeciAL4PM)
│   ├── docker/                       #   Docker infrastructure for AVATAR (M5)
│   │   ├── Dockerfile.avatar / .tf2
│   │   └── run_avatar.py
│   ├── shell/                        #   SLURM-ready shell scripts
│   │   ├── m1.sh … m7.sh             #   Per-method (sbatch/bash dual-use)
│   │   ├── r1.sh, r2.sh, r3.sh       #   Reference methods
│   │   └── run_all.sh                #   Full pipeline entry point
│   ├── miners.py                     #   Miner definitions
│   ├── datasets.py                   # ★ Canonical D1–D21 dataset registry
│   ├── utils.py                      #   Shared utilities
│   ├── models/                       #   Pre-discovered PNML + DFG JSON
│   └── results/                      #   methodology results
│
├── data/                             # Event logs (XES .gz)
├── report/                           # LaTeX paper (main.tex, main.pdf, references.bib)
├── output/                           # HybridGen experiment JSON results
```

---

## HybridGen Metric — How It Works

The metric has three stages:

### 1. N-gram Statistics

The log is reduced to its variants with frequency weights. For every order n = 1..6, the algorithm records per observed n-gram context the frequency of each successor activity and the frequency with which the context occurs at trace end.

### 2. Shadow Log Generation

A stochastic walker generates synthetic traces:

| Step | Mechanism |
|------|-----------|
| **Context resolution** | Katz-style backoff — try the deepest context first; fall back if support < τ = 5 |
| **Novelty estimation** | Good–Turing `p_unseen(s) = N₁(s) / N(s)` — probability the next event is unseen |
| **Mutation** | With probability `p_unseen`, insert a novel activity. **v2.5+**: drawn from backed-off lower-order context (Katz-consistent); **v2.4−**: uniform over alphabet |
| **Exploitation** | Otherwise, sample successor proportionally to `ln(f+1)` (log damping) or raw frequency (MLE, v2.6+) |
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
| **v2.4** | Stable baseline; uniform mutation, ln-damped sampling | **v1 benchmark baseline** |
| **v2.5** | Katz-consistent mutation proposal; probe-integrity counters | ✅ |
| **v2.6 (log)** | v2.5 + acceptance rate + data-driven length cap | ✅ Stress-test mode |
| **v2.6 (mle)** | v2.6 with `successor_weighting='mle'` | **🏆 Headline candidate** |

> **Latest recommendation (2026-06-11):** v2.6-mle (M1g) dominates all other versions
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

| Tier | Methods | Description |
|------|---------|-------------|
| **T1** | M1a–M1g (HybridGen v1 → v2.6) | Our method family + ablations |
| **T2** | M2–M7 (PM4Py, Entropic Relevance, AVATAR, Bootstrap Gen, SpeciAL4PM) | External baselines |
| **T3** | R1–R3 (K-Fold CV, Leave-One-Out, Random) | Reference / sanity checks |

**21 datasets (D1–D21):** Sepsis, BPI Challenges 2011–2020, Hospital Billing, Road Traffic Fine Management — see `benchmark/datasets.py` for the full catalog.

---

## Quick Start

```bash
# Install dependencies (auto-creates venv)
uv sync

# D1 Sepsis — M1 family benchmark (all 7 versions, all 8 miners)
uv run python benchmark/job_m1.py --dataset D1

# Single methods
uv run python benchmark/job_m2.py --dataset D1
uv run python benchmark/job_m3.py --dataset D1
uv run python benchmark/job_m6.py --dataset D1
uv run python benchmark/job_m7.py --dataset D1
uv run python benchmark/job_r1.py --dataset D1
uv run python benchmark/job_r2.py --dataset D1
uv run python benchmark/job_r3.py --dataset D1

# Using shell wrappers (bash or sbatch)
bash benchmark/shell/m1.sh --dataset D1

# Full pipeline (all methods, all datasets)
bash benchmark/shell/run_all.sh

# Production run (results → benchmark/results/configs_v2/):
OUTPUT_DIR=benchmark/results/configs_v2 bash benchmark/shell/run_all.sh D1
```

Results go to `/tmp/<workdir>/results/` by default, or `benchmark/results/configs_v2/` with `--output`. Each produces one JSON file per (dataset, miner, method) pair.

**Result extraction:**
```bash
uv run python benchmark/extract_results.py --dataset D3     # single dataset
uv run python benchmark/extract_results.py --all             # all datasets
```

Outputs a Markdown table with MAE, Pearson, and Spearman vs. ground truth (R1).

---

## Key Findings (Preliminary)

| Metric | Pearson vs R1 | Spearman vs R1 | MAE |
|--------|:------------:|:--------------:|:---:|
| HybridGen v2.6-mle | **0.996** | **0.943** | **0.023** |
| HybridGen v2.4 | 0.989 | 0.886 | 0.031 |
| PM4Py Built-in | **−0.429** | **−0.371** | 0.167 |

- PM4Py's built-in generalization metric correlates **negatively** with held-out fitness.
- v2.5's Katz-consistent mutation proposal reduces calibration error by ≈3×.
- v2.6-mle is the only mode that ranks D2 correctly (Spearman 1.0).
- The flower model scores 1.0 under Gen_shadow — this is **correct** for a pure generalization construct (see `Method_GenShadow.md` §1).

---

## Documentation

| Document | Description |
|----------|-------------|
| `Method_GenShadow.md` | **Authoritative** metric specification & design rationale |
| `BenchmarkDesign.md` | Benchmark methodology v2 — full protocol, methods, datasets |
| `BenchmarkGuide.md` | Quick-start guide for running experiments |
| `report/main.pdf` | LaTeX paper (LNCS format) |
