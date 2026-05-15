# Process Model Generalization Analysis

> Evaluating the **generalization capability** of process mining algorithms on unseen behavior.

---

## Project Structure

```
generalization_analysis/
├── README.md                       # This file — project overview & usage guide
├── pyproject.toml                  # Python project config (uv package manager)
├── requirements.txt                # Python dependencies
│
├── pick_one_out_algorithm.py       # ★ Method 1 core algorithm
├── pick_one_out_experiment.py      # ★ Method 1 experiment runner (CLI front-end)
├── pick_one_out.sh                 # SLURM batch job submission script
│
├── ExperimentDesign.md             # Experimental design (three-tier framework)
├── Method1Log.md                   # Method 1 formula iteration log (v1 → v2 evolution)
├── Method2Log.md                   # Method 2 documentation (teammate's hybrid approach)
│
├── data/
│   └── BPI-Challenge_2017/         # Input data (BPI Challenge 2017 event log)
│
└── output/                         # Experiment results (JSON format)
    ├── method1_*.json              # Method 1 results (summary + details)
    └── baseline_*.json             # PM4Py baseline results
```

---

## Two Generalization Metrics

### Method 1 — Pick-One-Out (Leave-One-Out)

**Core idea:** Remove each variant one at a time, rediscover the model from the remaining log, and check whether the held-out variant can be replayed.

- **Weighting formula (v2, current):**
  $$w_{v2}(v_i) = \ln\big(f(v_i) + 1\big) \times \frac{HMean(E_{v_i})}{MaxGlobalFreq(E_{\log})}, \quad HMean(E_{v_i}) = \frac{|E_{v_i}|}{\sum_{e \in E_{v_i}} \frac{1}{GlobalFreq(e)}}$$
  Uses the **harmonic mean** (dominated by the smallest edge frequency — the "weakest link") instead of the arithmetic mean, preventing frequent path edges from diluting rare deviation edges. This sharply distinguishes true noise from concurrency-induced rare variants.
- **Replay strategy:** Token-based replay (fast) → Alignment-based (fallback).
- **Docs:** `Method1Log.md` tracks the full design evolution from v1 (double-log arithmetic-mean weighting → cancelled out) to v2 (harmonic-mean normalized joint weighting).

### Method 2 — Hybrid Generative–Structural

**Core idea:** Combines two complementary dimensions:
- **$Gen_{shadow}$ (generative behavioral analysis):** Uses Good-Turing estimation on local states to generate a synthetic shadow log, then evaluates replay fitness to measure generative flexibility.
- **$Gen_{struct}$ (structural frequency analysis):** Replays the original log and penalizes rarely-used structural paths to prevent overfitting.
- **Final score:** $Gen_{Total} = w \times Gen_{shadow} + (1-w) \times Gen_{struct}$

- **Docs:** `Method2Log.md`

---

## Quick Start

### Environment Setup

```bash
# After cloning, use uv to manage the environment
cd generalization_analysis

# Install dependencies (uv auto-creates a virtual environment)
uv sync

# Or with pip
pip install -r requirements.txt
```

### Running Experiments

```bash
# Method 1: sample 2% of variants, Inductive Miner only
uv run python pick_one_out_experiment.py --method method1 -t 2 -m IM

# Method 1: sample 10% of variants, all 4 miners
uv run python pick_one_out_experiment.py --method method1 -t 10 -m all

# PM4Py baseline only (built-in generalization metric)
uv run python pick_one_out_experiment.py --method baseline
```

**CLI arguments:**

| Argument | Options | Description |
|----------|---------|-------------|
| `--method` | `method1`, `baseline` | Evaluation method |
| `-t` / `--max-variant-percentage` | integer (1–100) | Percentage of variants to sample |
| `-m` / `--miner` | `IM`, `IMf`, `Heuristics`, `Alpha`, `all` | Miner selection |
| `-w` / `--workers` | integer | Number of parallel workers (default: 1) |

### Cluster Batch Submission

```bash
# Edit PERCENTAGE and NUM_WORKERS in pick_one_out.sh, then:
bash pick_one_out.sh
```

This script submits Method 1 and baseline jobs via SLURM on the Krater partition.

---

## Output Files

Each run produces two JSON files under `output/`:

- **`*_summary.json`** — Lightweight summary: run config + Pure/Joint scores per miner
- **`*_details.json`** — Full data: per-variant frequency, replay score, weights, etc.

Naming convention: `{method}_{miner}_{pct}pct_{timestamp}_{type}.json`

---

## Experiment Design Overview

See `ExperimentDesign.md` for the full three-tier experimental framework:

| Tier | Scope | Goal |
|------|-------|------|
| **I** | Single-dataset deep analysis | Ranking attribution, scatter-plot correlation vs. baseline, internal ablation study |
| **II** | Cross-method benchmarking | Method 1 vs. Method 2 rank consistency analysis |
| **III** | Stress testing | Model morphology confrontation (Trace → Spaghetti → Causal → Block → Lasagna → Flower), noise injection, data-characteristic sensitivity, scalability profiling |

---

## Data Sources

- Primary dataset: **BPI Challenge 2017** (`data/BPI-Challenge_2017/`)
- Event log format: XES (`.xes.gz`)
- PM4Py miners used: Inductive Miner (IM/IMf), Heuristics Miner, Alpha Miner
