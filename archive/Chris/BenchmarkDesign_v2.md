# Generalization Benchmark Design — Methodology v2

> **Status.** This document defines **Methodology v2** of the generalization benchmark.
> It does **not** replace [`BenchmarkDesign.md`](BenchmarkDesign.md) (Methodology v1): every
> section of v1 that is not explicitly restated below — external methods M2–M8, reference
> metrics R1–R3, datasets D1–D5, statistical protocol, SLURM execution plan, risks — is
> **inherited unchanged**. v1 results in `benchmark/results/configs/` remain valid v1 results
> and are never overwritten; v2 results live in **`benchmark/results/configs_v2/`**.
>
> Construct definition follows [`Method_GenShadow.md`](Method_GenShadow.md) (authoritative):
> generalization = acceptance of future *valid* behavior, strictly separated from precision.

---

## What changed vs. Methodology v1 (summary)

| # | Change | Why |
|---|--------|-----|
| 1 | Tier 1 extended: **M1d (v25), M1e (v26-log), M1f (v26-mle)** added | New algorithm versions fixing probe defects found in v24 (see [`WhatChanged_v25_v26.md`](WhatChanged_v25_v26.md)) |
| 2 | Eighth miner added: **Filtered Trace Model** (top-50 variants) | The 0.0 pole (memorization), opposite of the Flower Model's 1.0 pole |
| 3 | Pole interpretation corrected | Flower ≈ 1.0 is **correct** for a pure generalization metric (construct-purity litmus), not a failure; Trace low is the overfitting pole |
| 4 | Reporting: **mean ± std for every M1 version** + acceptance + probe-integrity counters | Transparency; v26 metrics expose `gen_accept`, `duplicates_kept`, `truncated_traces` |
| 5 | Agreement protocol: **Spearman + MAE + spread** vs R1, poles excluded | Spearman alone is a low bar (even the random floor achieves 1.0 on D1) |
| 6 | New runner: `benchmark/run_m1_family.py` writes official config JSONs to `configs_v2/` | One command per dataset, model discovery cached, R1 copied/computed automatically |

---

## Tier 1 — Our Method (v2)

| # | Method | Algorithm | Key property |
|---|--------|-----------|--------------|
| M1  | HybridGen v24 | uniform mutation proposal, ln-damped sampling | v1-methodology baseline (unchanged, for continuity) |
| M1a | HybridGen v1 | 1-gram DFG + Good–Turing | simplest ablation |
| M1b | HybridGen v2.1, N=3 | flat termination | isolates context-aware termination |
| M1c | HybridGen v2.1, N=6 | flat termination | isolates N=3→6 upgrade |
| M1d | **HybridGen v25** | **Katz-consistent mutation proposal** | mutations drawn from backed-off lower-order context instead of uniform alphabet noise; probe-integrity counters |
| M1e | **HybridGen v26 (log)** | v25 + acceptance rate + data-driven length cap | ln-damped sampling retained (stress-test mode) |
| M1f | **HybridGen v26 (mle)** | v26 with `successor_weighting='mle'` | samples the estimated future distribution itself — **headline candidate** (best calibration & only mode ranking D2 correctly) |

All M1 versions report **mean ± std over 5 iterations** of 1,000 shadow traces, seed 42,
`max_n=6` (except M1a/M1b by design), `safe_threshold=5`. M1e/M1f additionally report
`gen_accept` (perfect-replay rate), the regular/mutated openness profile, and the
probe-integrity counters `duplicates_kept` / `truncated_traces`.

## Discovery Algorithms (v2: eight, spanning both poles)

| # | Miner | Construction | Role |
|---|-------|--------------|------|
| 0 | **Filtered Trace Model** | one isolated path per variant, **top-50 variants by frequency** (identical to `master_benchmark_v24.py`) | **0.0 pole** — pure memorization; accepts nothing unseen |
| 1–6 | Alpha, Alpha+, Heuristics (default/strict), Inductive (strict/infrequent) | PM4Py, same parameters as v1 | the six "real" miners |
| 7 | Flower Model | all activities in one loop | **1.0 pole** — accepts everything |

**Why top-50 for the Trace Model.** A full trace model has one branch per variant
(Sepsis: ~12,000 transitions; BPI 2017: ~600,000), which makes token replay intractable —
this is why v1 archived it. Capping at the 50 most frequent variants keeps the net at
~750 transitions (seconds per cell) while preserving the semantics that matter: the model
memorizes a fixed set of observed traces and rejects everything else. The cap is recorded
in every config JSON (`trace_model_top_k: 50`).

**Pole interpretation (corrected vs v1).** Under the pure-generalization construct:

- **Flower ≈ 1.0 is the expected, correct score** — the litmus for construct purity.
  A metric scoring Flower < 1 is contaminated with precision/structure.
- **Trace Model low is the expected, correct score** — the memorization pole. It will not
  reach exactly 0.0 under token replay (partial credit grants unseen traces some fitness;
  the v1 "ultimate" runs measured ~0.53–0.63 on Sepsis), so it is a *low anchor*, not a
  literal zero. Its perfect-replay acceptance (`gen_accept`, M1e/M1f) **is** ≈ 0.
- Both poles are **excluded from agreement statistics** (Pearson/Spearman/MAE/spread are
  computed over the six real miners) and reported separately as litmus checks.

## Evaluation & Reporting Protocol (v2)

Per (dataset, miner, method) cell — unchanged from v1 except where noted:

1. Discover the model once per miner on the full log (cached across methods).
2. Evaluate with the per-method protocol; **record mean, std, and raw per-iteration scores**.
3. Write one config JSON per cell to `benchmark/results/configs_v2/`
   (`{Dataset}__{Miner}__{Method}.json`, v1 schema + new optional result fields:
   `gen_accept`, `gen_accept_std`, `gen_shadow_regular`, `gen_shadow_mutated`,
   `duplicates_kept`, `truncated_traces`, `max_trace_length_used`).
4. Ground truth: R1 (variant-based 5-fold CV, 3 shuffles, seed 42) — copied from v1 configs
   where present, computed fresh otherwise (e.g. Trace_Filtered), and re-written to
   `configs_v2/` so v2 is self-contained.
5. Agreement reporting per method: **Pearson, Spearman, MAE, spread** over the six real
   miners, plus the two pole litmus values. Never report Spearman alone.

## How to Run

```bash
# Everything (all 7 M1-family methods, 8 miners, config JSONs + summary):
uv run python benchmark/run_m1_family.py --dataset D1
uv run python benchmark/run_m1_family.py --dataset D2

# Only the new versions:
uv run python benchmark/run_m1_family.py --dataset D1 --methods M1d M1e M1f

# Quick multi-seed robustness check (CSV output, no config JSONs):
uv run python benchmark/version_comparison.py --dataset D1 --seeds 42 1 7 99

# Post-hoc analysis / plots:
#   benchmark/version_comparison_analysis.ipynb
```

Expected runtime: ~5–8 min per dataset locally (D1; D2 faster). External methods (M2–M7)
and R2/R3 are unchanged — run them exactly as in Methodology v1 if v2 cells for them are
needed; until then, comparisons against M2–M7 use the v1 configs (valid: same models, same
log, same seed).

## Decision Log

- **2026-06-11** — M1f (v26-mle) is the recommended headline configuration: it dominates all
  other M1 versions on every agreement criterion on D1 (4 seeds) and D2 (2 seeds), is the
  only mode that ranks D2 correctly (Spearman 1.0 vs 0.943), and costs the same runtime.
  `'log'` weighting is retained as M1e for rare-behavior stress-testing.
  *Pending: practical partner sign-off before the report/benchmark headline switches from M1 (v24) to M1f.*
