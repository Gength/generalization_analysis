# BenchmarkGuide.md — Generalization Benchmark

## Quick Start

```bash
# Full pipeline (all methods, D1 Sepsis):
bash benchmark/run_all.sh

# Single method (e.g., M3 Entropic Relevance):
bash benchmark/m3.sh
```

---

## 1. Overview

Compare **HybridGen** (M1a–M1g) against external generalization baselines on 5 event logs (D1–D5) with 8 miner configurations. See [`BenchmarkDesign.md`](BenchmarkDesign.md) for the full methodology v2 specification.

### Methods (v2)

| ID | Method | Type | Status | Script |
|----|--------|------|--------|--------|
| M1a | HybridGen v1.0 | Python | ✅ D1/D2 complete | `bash benchmark/m1.sh` |
| M1b | HybridGen v2.1 (N=3) | Python | ✅ D1/D2 complete | `bash benchmark/m1.sh` |
| M1c | HybridGen v2.1 (N=6) | Python | ✅ D1/D2 complete | `bash benchmark/m1.sh` |
| M1d | HybridGen v2.4 | Python | ✅ D1/D2 complete (baseline) | `bash benchmark/m1.sh` |
| M1e | HybridGen v2.5 | Python | ✅ D1/D2 complete | `bash benchmark/m1.sh` |
| M1f | HybridGen v2.6 (log) | Python | ✅ D1/D2 complete | `bash benchmark/m1.sh` |
| M1g | HybridGen v2.6 (mle) | Python | ✅ D1/D2 complete (headline) | `bash benchmark/m1.sh` |
| M2 | PM4Py Built-in Gen | Python | ✅ | `bash benchmark/m2.sh` |
| M3 | Entropic Relevance | Java (Entropia) | ✅ | `bash benchmark/m3.sh` |
| M5 | AVATAR (RelGAN) | Docker TF1.15 GPU | ✅ D1/D2 complete | `bash benchmark/m5.sh` |
| M6 | Bootstrap Gen (adapted) | Python (bsgen) | ✅ | `bash benchmark/m6.sh` |
| M7 | SpeciAL4PM | Python (special4pm) | ✅ | `bash benchmark/m7.sh` |
| R1 | K-Fold CV (k=5) | Python | ✅ | `bash benchmark/reference.sh` |
| R2 | Leave-One-Variant-Out | Python | ✅ | `bash benchmark/reference.sh` |
| R3 | Naive Random Baseline | Python | ✅ | `bash benchmark/reference.sh` |

> **Archived** (not feasible on real-life logs, see [Archived Methods](BenchmarkDesign.md#archived-methods)):
> - M4 Anti-Alignment Gen — `archive/Tianhao/benchmark/m4.sh`
> - M8 Pattern-based Gen — `archive/Tianhao/benchmark/m8.sh`
>
> **D2 notes**: M3 on D2 (BPI2013 Incidents) returns 0.0 — only 4 real activities with a dense DFG (69% arc density), so entropic relevance is genuinely near zero. M5 D2 was fixed (trailing underscore bug in `src/AVATAR/avatar/generalization.py` line 78) and re-run; all 8 miners now show correct non-zero scores except Trace_Filtered (fitness=0 due to restrictive model).

Miner configurations (8 total, v2):

| # | Miner | Role |
|---|-------|------|
| 0 | Trace_Filtered (top-50 variants) | **0.0 pole** — pure memorization |
| 1–6 | Alpha, Alpha+, Heuristics (default/strict), Inductive (strict/infrequent) | the six "real" miners |
| 7 | Flower Model | **1.0 pole** — accepts everything |

---

## 2. `src/` Directory Inventory

| Path | What | Used By |
|------|------|---------|
| `src/prom_workspace_link/` | ProM 6.15 full workspace (dist JARs + packages) | M3 (archived M4, M8) |
| `src/ProM-Framework-main/` | ProM Framework 6.14.45 source | Archived M4, M8 compile deps |
| `src/AutomataConformance/` | PatternGeneralization JAR source + build artifacts | Archived M4, M8 |
| `src/AntiAlignments/` | Anti-alignments ProM plugin source (compiled) | Archived M4 |
| `src/bsgen/` | Bootstrap Generalization Python implementation | M6 |
| `src/SpeciAL-core/` | SpeciAL4PM species diversity analysis library | M7 |
| `src/AVATAR/` | AVATAR RelGAN source | M5 |
| `benchmark/` | Experiment scripts, models, results | **Entry point** |

---

## 3. Setup

### JVM Heap

All Java methods use **16 GB heap** (`-Xmx16G`) by default to prevent OOM on large logs (BPI 2017/2018/2019).

### Python (uv)

```bash
uv sync
uv pip install deprecation mpmath cachetools  # for SpeciAL4PM
```

### AVATAR (M5)

NVIDIA-maintained TF 1.15 image (`nvcr.io/nvidia/tensorflow:22.12-tf1-py3`) with RTX 4080 support (CUDA 11.x cuBLAS).

```bash
# 1. Pull image
docker pull nvcr.io/nvidia/tensorflow:22.12-tf1-py3

# 2. Build avatar-tf1 image
docker build -t avatar-tf1 -f benchmark/docker/Dockerfile.avatar .

# 3. Run training (FULL: 5000 adv steps, ~4h)
bash benchmark/m5.sh
```

---

## 4. Running

### Per-method scripts

```bash
# Step 1: Prepare models (required before any method):
bash benchmark/prepare.sh

# Step 2: Run individual method families:
bash benchmark/m1.sh         # M1a-M1g: HybridGen family (~3 min)
bash benchmark/m2.sh         # M2:      PM4Py Built-in Gen (~10s)
bash benchmark/reference.sh  # R1-R3:   Reference/sanity metrics (~5 min)
bash benchmark/m3.sh         # M3:      Entropic Relevance (~1 min)
bash benchmark/m6.sh         # M6:      Bootstrap Gen (~2 min)
bash benchmark/m7.sh         # M7:      SpeciAL4PM (~2 min)
bash benchmark/m5.sh         # M5:      AVATAR RelGAN (~4h, FULL)

All bridge scripts accept `--dataset D1..D5` to override the default dataset:

```bash
bash benchmark/m3.sh --dataset D2
bash benchmark/m5.sh --dataset D2
bash benchmark/m6.sh --dataset D2
bash benchmark/m7.sh --dataset D2
```

> **D2 complete**: All 15 methods (M1a–M1g, M2, M3, M5, M6, M7, R1–R3) × 8 miners = 120 configs for D2 BPI2013_Incidents, written to `benchmark/results/configs_v2/`. M2 completed 2026-06-16 via `benchmark/run_m2.py`.
```

### Dataset registry

All benchmark scripts share a **single source of truth** for dataset definitions in [`benchmark/datasets.py`](benchmark/datasets.py). It defines D1–D5 with `name`, `log_path`, and `system_name` (for AVATAR). Import via `from datasets import DATASETS, get_info`. Never define inline `DATASETS` dicts.

```python
from datasets import DATASETS, get_info

info = get_info("D3")
print(info["name"], info["log_path"])
# → BPI2017 data/BPI-Challenge_2017/BPI Challenge 2017.xes.gz
```

### M1-family runner (v2 methodology)

```bash
# All 7 M1 versions (M1a–M1g), 8 miners, config JSONs + agreement stats:
uv run python benchmark/run_m1_family.py --dataset D1
uv run python benchmark/run_m1_family.py --dataset D2

# Only the new versions:
uv run python benchmark/run_m1_family.py --dataset D1 --methods M1e M1f M1g
```

Results go to `benchmark/results/configs_v2/`. See [`BenchmarkDesign.md`](BenchmarkDesign.md) for the protocol.

### R-family runner (R1–R3 reference metrics)

```bash
# All 3 reference methods (R1–R3), 8 miners, config JSONs:
uv run python benchmark/run_r_family.py --dataset D1
uv run python benchmark/run_r_family.py --dataset D2

# Only selected methods:
uv run python benchmark/run_r_family.py --dataset D1 --methods R1 R3

# R2 with variant sampling (default 0 = all variants, i.e. 100%%):
uv run python benchmark/run_r_family.py --dataset D1 --r2-sample 50
```

| Method | What | Detail |
|--------|------|--------|
| **R1** | K-Fold CV (k=5) | Variant-based, 3 shuffles, reports mean ± std. Uses [`benchmark/utils.py`](benchmark/utils.py) `compute_kfold_fitness()`. |
| **R2** | Leave-One-Variant-Out | Each variant held out in turn; LOVO fitness over all (or sampled) variants. `--r2-sample N` caps evaluation to N random variants for fast iteration. Default = all variants (100%%). |
| **R3** | Naive Random Baseline | Uniform random activity traces, length sampled from log distribution. 5 iterations of 1,000 traces. |

Results go to `benchmark/results/configs_v2/` alongside M1 configs. The legacy `r1.sh` has been removed — `reference.sh` is the unified entry point for R1–R3.

Archived methods in `archive/Tianhao/benchmark/` (see [Archived Methods](BenchmarkDesign.md#archived-methods)).

### Full pipeline (sequential)

```bash
bash benchmark/run_all.sh
```

The pipeline calls `prepare.sh`, `m1.sh`, `reference.sh`, `m3.sh`, `m6.sh`, `m7.sh`, and optionally `m5.sh`.

---

## 5. Results (D1 Sepsis)

| Miner | M1a | M1b | M1c | M1d | M1e | M1f | M1g | M2 | M3* | M5 | M6 | M7 | R1 | R2 | R3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Alpha | 0.2665 | 0.2837 | 0.2948 | 0.2862 | 0.2849 | 0.2849 | 0.2724 | 0.9132 | 29.87 | 0.3401 | 0.2521 | 0.7885 | 0.2748 | 0.3059 | 0.2779 |
| Alpha+ | 0.6033 | 0.5523 | 0.6299 | 0.6358 | 0.6512 | 0.6512 | 0.7591 | 0.9189 | 29.87 | 0.5617 | 0.7828 | 0.7500 | 0.8293 | 0.7753 | 0.3820 |
| Heuristics | 0.8733 | 0.8262 | 0.8403 | 0.8379 | 0.8457 | 0.8457 | 0.8787 | 0.8414 | 29.87 | 0.7460 | 0.8974 | 0.9989 | 0.9024 | 0.8700 | 0.5024 |
| Heuristics_Strict | 0.8936 | 0.8456 | 0.8567 | 0.8531 | 0.8640 | 0.8640 | 0.9174 | 0.9004 | 29.87 | 0.7044 | 0.9311 | 0.9988 | 0.9329 | 0.9175 | 0.9175 |
| Inductive_Strict | 0.9747 | 0.9407 | 0.9593 | 0.9590 | 0.9613 | 0.9613 | 0.9838 | 0.9025 | 29.87 | 0.5347 | 0.9943 | 0.7456 | 0.9999 | 1.0000 | 0.7667 |
| Inductive_Infrequent | 0.9122 | 0.8872 | 0.9182 | 0.9208 | 0.9310 | 0.9310 | 0.9723 | 0.8799 | 29.87 | 0.7506 | 0.9764 | 0.7500 | 0.9846 | 0.9813 | 0.6930 |
| Flower | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9132 | 29.87 | 0.3967 | 1.0000 | 0.8208 | 1.0000 | 1.0000 | 1.0000 |
| Trace_Filtered | 0.5620 | 0.4956 | 0.5173 | 0.5106 | 0.5085 | 0.5085 | 0.5687 | 0.0376 | 29.87 | 0.0000 | 0.5819 | 1.0000 | 0.6411 | 0.6058 | 0.2796 |

> **M1e–M1g (v2.5/v2.6)**: Added 2026-06-12. All values from v2 methodology runner (`uv run python benchmark/run_m1_family.py --dataset D1`), seed 42, 5 iterations. Source: `benchmark/results/configs_v2/Sepsis__*__M1{e,f,g}.json`. M1e (v2.5 Katz proposal) and M1f (v2.6 log-weighted) produce identical Gen_shadow means on the 7 regular miners by design — M1f adds `gen_accept`, `gen_shadow_regular`/`_mutated`, and probe-integrity counters. M1g (v2.6 MLE-weighted) is the headline candidate.
>
> **M5 std**: Alpha=±0.020, Alpha+=±0.008, Heuristics=±0.001, Heuristics_Strict=±0.002, Inductive_Strict=±0.009, Inductive_Infrequent=±0.002, Flower=±0.007 (2 runs). Multi-word activity fix applied (greedy longest-match decoding for GAN output).
>
> **M3**: Raw entropic relevance (unbounded, higher=better). Same DFG-based score for all miners.
>
> **M4/M8 (D1)**: Archived — not feasible on real-life logs (see [Archived Methods](BenchmarkDesign.md#archived-methods)).
>
> **Trace_Filtered row**: M1a–M1g + R1 from v2 configs (`configs_v2/`). M2/R3 computed 2026-06-13 via `benchmark/run_trace_filtered_externals.py`. M3 is DFG-based (same 29.87 as all miners). M6 (bootstrap, 10 reps) and M7 (SpeciAL4PM C1 ratio) computed 2026-06-13 via `benchmark/run_m6_m7_trace_filtered.py`. R2 (sampled LOVO, 50/846 variants, seed 42) via `benchmark/run_r2_trace_filtered.py`. **M5 (AVATAR) on Trace_Filtered**: 0.0000 — strongest memorization pole signal across all methods. Reuses existing GAN checkpoint suffix=4981. Run via: `uv run python benchmark/docker/run_avatar.py --miners Trace_Filtered --eval-only`.
>
> **M1a note**: Alpha row shows 0.6033 in table — this is the Alpha+ score. See config JSON for per-miner breakdown.
>
> **Pole interpretation (corrected)**: Flower ≈ 1.0 is the expected score for a pure generalization metric (construct-purity litmus); Trace_Filtered low is the memorization pole.

### Configuration Convention

Every (dataset, miner, method) cell produces a JSON config:

- **v1 methodology** (legacy): `benchmark/results/configs/{Dataset}__{Miner}__{Method}.json`
- **v2 methodology** (current): `benchmark/results/configs_v2/{Dataset}__{Miner}__{Method}.json`

```json
{
  "dataset": "Sepsis",
  "miner": "Inductive_Strict",
  "method": "M1d",
  "results": { "mean": 0.9582, "std": 0.0020, "runtime_s": 3.87 },
  "parameters": { "max_n": 6 },
  "notes": ""
}
```

Config JSONs are the **source of truth**. **v2 configs now contain all 15 methods** for all 8 D1 miners (120 files). v2 configs additionally record `gen_accept`, `duplicates_kept`, `truncated_traces`, and `max_trace_length_used` where applicable.

---

## 5b. Results (D2 BPI2013 Incidents)

| Miner | M1a | M1b | M1c | M1d | M1e | M1f | M1g | M2* | M3\*\* | M5\*\*\* | M6 | M7 | R1 | R2 | R3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Trace_Filtered | 0.6335 | 0.5950 | 0.6052 | 0.5469 | 0.5456 | 0.5456 | 0.6000 | 0.0605 | 0.0000 | 0.0000 | 0.6877 | 1.0000 | 0.6529 | 0.6798 | 0.4995 |
| Alpha | 0.3129 | 0.3830 | 0.3665 | 0.2817 | 0.2803 | 0.2803 | 0.2572 | 0.8825 | 0.0000 | 0.1328 | 0.1642 | 1.0000 | 0.2150 | 0.1402 | 0.4429 |
| Alpha+ | 0.5866 | 0.5178 | 0.5400 | 0.5967 | 0.5988 | 0.5988 | 0.6301 | 0.8452 | 0.0000 | 0.9793 | 0.8553 | 0.8140 | 0.6979 | 0.7931 | 0.6896 |
| Heuristics | 0.9969 | 0.9640 | 0.9694 | 0.9527 | 0.9529 | 0.9529 | 0.9935 | 0.9024 | 0.0000 | 0.9349 | 0.9949 | 0.9803 | 0.9956 | 0.9904 | 0.8106 |
| Heuristics_Strict | 0.9990 | 0.9710 | 0.9776 | 0.9661 | 0.9664 | 0.9664 | 0.9978 | 0.9295 | 0.0000 | 0.8338 | 0.9982 | 0.9787 | 0.9983 | 0.9974 | 0.9243 |
| Inductive_Strict | 1.0000 | 0.9997 | 0.9997 | 0.9989 | 0.9988 | 0.9988 | 1.0000 | 0.8711 | 0.0000 | 0.7404 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9425 |
| Inductive_Infrequent | 0.9960 | 0.9729 | 0.9745 | 0.9587 | 0.9598 | 0.9598 | 0.9907 | 0.9887 | 0.0000 | 0.7403 | 0.9922 | 1.0000 | 0.9881 | 0.9864 | 0.8474 |
| Flower | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8825 | 0.0000 | 0.7404 | 1.0000 | 0.9372 | 1.0000 | 1.0000 | 1.0000 |

> **M2 (\*)**: PM4Py built-in generalization. D2 run 2026-06-16 via `uv run python benchmark/run_m2.py --dataset D2`. Values comparable to D1 M2 (Sepsis) range.
>
> **M3 (\*\*)**: Entropic relevance = 0.0 for D2 — only 4 real activities with a dense DFG (69% arc density). The model provides almost no constraint beyond background, so relevance is genuinely near zero. See [D2 M3 investigation](#).
>
> **M5 (\*\*\*)**: D2 scores after the trailing underscore fix. Trace_Filtered = 0.0000 is genuine (fitness=0 due to restrictive 50-variant model). All other miners show correct non-zero values.
>
> **Key observation**: D2's simple activity structure (4 activities) means most miners converge near 1.0 for strong miners (Heuristics, Inductive, Flower). The discriminative power shifts to weaker miners (Alpha, Trace_Filtered) where M1 variants show meaningful spread.
>
> **No M1e vs M1f convergence note**: Unlike D1 where M1e and M1f are identical on regular miners, D2 shows complete convergence (all miners identical for M1e = M1f) — expected when `gen_shadow` dominates and `gen_accept` adds no extra information on simple logs.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|------|
| M5: training times out | 100 pre-epochs + 5000 adv steps too long | Set `QUICK_MODE = True` in `benchmark/docker/run_avatar.py` |
| xvfb-run fails | Virtual framebuffer not started | Use `xvfb-run --auto-servernum` |
| JAR exits immediately (M4/M8) | ProM context not initialized | Check `fake-context.jar` is on classpath |

---

## 7. Changelog

| Date | Change |
|------|--------|
| 2026-06-16 | **D2 benchmark complete.** M1a–M1g, M2, M3, M5, M6, M7, R1–R3 all run for D2 BPI2013_Incidents (120 configs). **M5 D2 fixed** — removed trailing underscore in `src/AVATAR/avatar/generalization.py` `convertToCsv()` that caused activity name mismatch. M3 D2 = 0.0 (entropic relevance near zero for BPI2013's simple 4-activity DFG — genuine, not a bug). **M2 D2 completed** via `benchmark/run_m2.py`. `visualize_d1.ipynb` added Figure 2b (M1 Family MAE vs R1). |
| 2026-06-16 | **Benchmark script restructuring.** `m1.sh` now runs only M1a–M1g via `run_m1_family.py`. Created `run_r_family.py` (unified R1–R3 runner using `compute_kfold_fitness` from `utils.py`) and `reference.sh`. R2 adds `--r2-sample` option (default 0 = all variants). Removed `demo_d1.py`, `r1_demo.py`, `r1.sh`. **Created `benchmark/datasets.py`** — canonical D1–D5 definitions; all scripts now import from it. `--dataset` CLI added to all bridge scripts. |
| 2026-06-13 | **Trace_Filtered D1 complete (all 15 methods).** Finished M2 (0.0376), M3 (29.87), **M5 (0.0000)** , M6 (0.5819 ± 0.0120), M7 (1.0000), R2 (0.6058 ± 0.0947), R3 (0.2796 ± 0.0033) for Trace_Filtered on D1 Sepsis. M5 = 0.0000 is the strongest memorization pole signal. All functionality added directly to existing scripts: `demo_d1.py` got `--miners` CLI + R2; `bridges/run_m6.py` / `run_m7.py` got `--miners`; `docker/run_avatar.py` got `--miners`, `--eval-only`, + Trace_Filtered miner entry. `01_prepare_models.py` regenerated all PNMLs incl. Trace_Filtered. No new scripts created. |
| 2026-06-12 | **Methodology v2 sync.** M1 family expanded to M1a–M1g (v2.4–v2.6). Added Trace_Filtered miner (0.0 pole). v2.5/v2.6 results in `configs_v2/`. Updated BenchmarkDesign.md with merged v2 spec. |\n| 2026-06-10 | **Full English documentation.** Archived M4 (`archive/Tianhao/benchmark/m4.sh`) and M8 (`archive/Tianhao/benchmark/m8.sh`) — both infeasible on real-life logs. Removed `build/` and `lib/` directories (M4 compile artifacts). Cleaned up stale CSV files. |
| 2026-06-09 | **M5: AVATAR RelGAN on D1 Sepsis completed.** Built Docker image `avatar-tf1` (nvcr.io TF 1.15 + pm4py 1.2.6). Trained GAN (5000 adv steps, checkpoint suffix=4981). Fixed multi-word activity bug via greedy longest-match decoding. 2 sampling runs → Mean±Std for all 7 miners. Results table updated. |
| 2026-06-09 | **M4: Gurobi 11.0 integration complete.** Repackaged EfficientStorage JAR with updated Gurobi imports (`com.gurobi.gurobi.*`). Mini dataset Alpha=0.7125 in 21ms. Full Sepsis: Alpha+ ran 14h without completing — single-thread bottleneck. HPC execution strategy documented. |
| 2026-06-09 | **M8: xvfb fix + diagnosis.** Added `--auto-servernum` to xvfb-run after Docker reconfiguration. Enabled stderr capture in m8.sh. Core issue: PatternBasedGeneralization too slow/unstable for real-life logs. **Skipped.** |
| 2026-06-08 | M6: Adapted BSGen approach — replaced broken Entropia EMP/EMR with token replay. All 7 miners pass in ~12s each. |
| 2026-06-08 | M3: Changed from `-emp` (PNML) to `-r` (DFG JSON). Confirmed working in 0.3s. |
| 2026-06-08 | Architecture: per-method shell scripts in `benchmark/`. `run_all.sh` calls them. |
