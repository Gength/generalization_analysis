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
│   ├── run_m1_family.py              #   M1-family core + CLI
│   ├── run_m2.py                     #   M2 core + CLI
│   ├── run_r_family.py               #   R1/R2/R3 core + CLI
│   ├── bridges/
│   │   ├── run_m3.py                 #   M3 (Entropic Relevance) + CLI
│   │   ├── run_m4.py                 #   M4 (Anti-Alignment) + CLI
│   │   ├── run_m6_adapted.py         #   M6 adapted (bsgen) + CLI
│   │   ├── run_m6_bgen.py            #   M6 (Bootstrap Gen) + CLI
│   │   └── run_m7.py                 #   M7 (SpeciAL4PM) + CLI
│   ├── docker/
│   │   ├── Dockerfile.avatar / .tf2
│   │   └── run_avatar.py             #   M5 (AVATAR) + CLI
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
| v2–v2.2 | N-gram + backoff evolution | Historical |
| **v2.3** | Context-aware termination; fixed mutation rate over-estimation | Historical |
| **v2.4** | Stable baseline; uniform mutation, ln-damped sampling | **v1 benchmark baseline** |
| **v2.5** | Katz-consistent mutation proposal; probe-integrity counters | ✅ |
| **v2.6 (log)** | v2.5 + acceptance rate + data-driven length cap | ✅ Stress-test mode |
| **v2.6 (mle)** | v2.6 with `successor_weighting='mle'` | **🏆 Headline candidate** |

> **Latest recommendation (2026-06-11):** v2.6-mle (M1g) dominates all other versions
> on every agreement criterion vs. ground truth, is the only mode that ranks D2
> correctly (Spearman 1.0 vs 0.943), and costs the same runtime.

---

## Quick Start

```bash
# Install dependencies (auto-creates venv)
uv sync
```

---

## Benchmark

Full documentation — layout, job model, commands, dataset quick-index, result extraction, and gotchas — is at **[`benchmark/README.md`](benchmark/README.md)**.

Key documents:
- [`BenchmarkDesign.md`](BenchmarkDesign.md) — methodology and metric definitions
- [`BenchmarkGuide.md`](BenchmarkGuide.md) — step-by-step operations and result formats

---

## Documentation

| Document | Description |
|----------|-------------|
| `Method_GenShadow.md` | **Authoritative** metric specification & design rationale |
| `BenchmarkDesign.md` | Benchmark methodology v2 — full protocol, methods, datasets |
| `BenchmarkGuide.md` | Quick-start guide for running experiments |
| `report/main.pdf` | LaTeX paper (LNCS format) |
