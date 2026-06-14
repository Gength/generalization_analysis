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

Compare **HybridGen v2.4–v2.6** (M1–M1f) against external generalization baselines on 5 event logs (D1–D5) with 8 miner configurations. See [`BenchmarkDesign.md`](BenchmarkDesign.md) for the full methodology v2 specification.

### Methods (v2)

| ID | Method | Type | Status | Script |
|----|--------|------|--------|--------|
| M1–M1f | **HybridGen v2.4–v2.6** | Python | ✅ D1/D2 complete (all 8 miners incl. Trace_Filtered) | `bash benchmark/m1.sh` (v1) or `uv run python benchmark/run_m1_family.py` (v2) |
| M2 | PM4Py Built-in Gen | Python | ✅ | `bash benchmark/m1.sh` |
| M3 | Entropic Relevance | Java (Entropia) | ✅ | `bash benchmark/m3.sh` |
| M5 | AVATAR (RelGAN) | Docker TF1.15 GPU | ✅ D1 complete (2 runs) | `bash benchmark/m5.sh` |
| M6 | Bootstrap Gen (adapted) | Python (bsgen) | ✅ | `bash benchmark/m6.sh` |
| M7 | SpeciAL4PM | Python (special4pm) | ✅ | `bash benchmark/m7.sh` |
| R1 | K-Fold CV (k=5) | Python | ✅ | `bash benchmark/r1.sh` |
| R2 | Leave-One-Variant-Out | Python | ✅ | Included in `m1.sh` |
| R3 | Naive Random Baseline | Python | ✅ | Included in `m1.sh` |

> **Archived** (not feasible on real-life logs, see [Archived Methods](BenchmarkDesign.md#archived-methods)):
> - M4 Anti-Alignment Gen — `archive/Tianhao/benchmark/m4.sh`
> - M8 Pattern-based Gen — `archive/Tianhao/benchmark/m8.sh`

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

### Per-method scripts (v1 methodology)

```bash
# Step 1: Prepare models (required before any method):
bash benchmark/prepare.sh

# Step 2: Run individual methods:
bash benchmark/m1.sh   # M1-M1f, M2, R3  (~3 min)
bash benchmark/r1.sh   # R1 K-Fold CV     (~3 min)
bash benchmark/m3.sh   # M3 Entropic      (~1 min)
bash benchmark/m6.sh   # M6 Bootstrap     (~2 min)
bash benchmark/m7.sh   # M7 SpeciAL4PM    (~2 min)
bash benchmark/m5.sh   # M5 AVATAR        (~4h, FULL)
```

### M1-family runner (v2 methodology)

```bash
# All 7 M1 versions (M1–M1f), 8 miners, config JSONs + agreement stats:
uv run python benchmark/run_m1_family.py --dataset D1
uv run python benchmark/run_m1_family.py --dataset D2

# Only the new versions:
uv run python benchmark/run_m1_family.py --dataset D1 --methods M1d M1e M1f
```

Results go to `benchmark/results/configs_v2/`. See [`BenchmarkDesign.md`](BenchmarkDesign.md) for the protocol.

Archived methods in `archive/Tianhao/benchmark/` (see [Archived Methods](BenchmarkDesign.md#archived-methods)).

### Full pipeline (sequential)

```bash
bash benchmark/run_all.sh
```

---

## 5. Results (D1 Sepsis)

| Miner | M1a | M1b | M1c | M1 | M1d | M1e | M1f | M2 | M3* | M5 | M6 | M7 | R1 | R2 | R3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Alpha | 0.2665 | 0.2837 | 0.2948 | 0.2862 | 0.2849 | 0.2849 | 0.2724 | 0.9132 | 29.87 | 0.3401 | 0.2521 | 0.7885 | 0.2748 | 0.3059 | 0.2779 |
| Alpha+ | 0.6033 | 0.5523 | 0.6299 | 0.6358 | 0.6512 | 0.6512 | 0.7591 | 0.9189 | 29.87 | 0.5617 | 0.7828 | 0.7500 | 0.8293 | 0.7753 | 0.3820 |
| Heuristics | 0.8733 | 0.8262 | 0.8403 | 0.8379 | 0.8457 | 0.8457 | 0.8787 | 0.8414 | 29.87 | 0.7460 | 0.8974 | 0.9989 | 0.9024 | 0.8700 | 0.5024 |
| Heuristics_Strict | 0.8936 | 0.8456 | 0.8567 | 0.8531 | 0.8640 | 0.8640 | 0.9174 | 0.9004 | 29.87 | 0.7044 | 0.9311 | 0.9988 | 0.9329 | 0.9175 | 0.9175 |
| Inductive_Strict | 0.9747 | 0.9407 | 0.9593 | 0.9590 | 0.9613 | 0.9613 | 0.9838 | 0.9025 | 29.87 | 0.5347 | 0.9943 | 0.7456 | 0.9999 | 1.0000 | 0.7667 |
| Inductive_Infrequent | 0.9122 | 0.8872 | 0.9182 | 0.9208 | 0.9310 | 0.9310 | 0.9723 | 0.8799 | 29.87 | 0.7506 | 0.9764 | 0.7500 | 0.9846 | 0.9813 | 0.6930 |
| Flower | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9132 | 29.87 | 0.3967 | 1.0000 | 0.8208 | 1.0000 | 1.0000 | 1.0000 |
| Trace_Filtered | 0.5620 | 0.4956 | 0.5173 | 0.5106 | 0.5085 | 0.5085 | 0.5687 | 0.0376 | 29.87 | 0.0000 | 0.5819 | 1.0000 | 0.6411 | 0.6058 | 0.2796 |

> **M1d–M1f (v2.5/v2.6)**: Added 2026-06-12. All values from v2 methodology runner (`uv run python benchmark/run_m1_family.py --dataset D1`), seed 42, 5 iterations. Source: `benchmark/results/configs_v2/Sepsis__*__M1{d,e,f}.json`. M1d (v2.5 Katz proposal) and M1e (v2.6 log-weighted) produce identical Gen_shadow means on the 7 regular miners by design — M1e adds `gen_accept`, `gen_shadow_regular`/`_mutated`, and probe-integrity counters. M1f (v2.6 MLE-weighted) is the headline candidate.
>
> **M5 std**: Alpha=±0.020, Alpha+=±0.008, Heuristics=±0.001, Heuristics_Strict=±0.002, Inductive_Strict=±0.009, Inductive_Infrequent=±0.002, Flower=±0.007 (2 runs). Multi-word activity fix applied (greedy longest-match decoding for GAN output).
>
> **M3**: Raw entropic relevance (unbounded, higher=better). Same DFG-based score for all miners.
>
> **M4/M8 (D1)**: Archived — not feasible on real-life logs (see [Archived Methods](BenchmarkDesign.md#archived-methods)).
>
> **Trace_Filtered row**: M1–M1f + R1 from v2 configs (`configs_v2/`). M2/R3 computed 2026-06-13 via `benchmark/run_trace_filtered_externals.py`. M3 is DFG-based (same 29.87 as all miners). M6 (bootstrap, 10 reps) and M7 (SpeciAL4PM C1 ratio) computed 2026-06-13 via `benchmark/run_m6_m7_trace_filtered.py`. R2 (sampled LOVO, 50/846 variants, seed 42) via `benchmark/run_r2_trace_filtered.py`. **M5 (AVATAR) on Trace_Filtered**: 0.0000 — strongest memorization pole signal across all methods. Reuses existing GAN checkpoint suffix=4981. Run via: `uv run python benchmark/docker/run_avatar.py --miners Trace_Filtered --eval-only`.
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
  "method": "M1",
  "results": { "mean": 0.9582, "std": 0.0020, "runtime_s": 3.87 },
  "parameters": { "max_n": 6 },
  "notes": ""
}
```

Config JSONs are the **source of truth**. **v2 configs now contain all 15 methods** for all 8 D1 miners (120 files). v2 configs additionally record `gen_accept`, `duplicates_kept`, `truncated_traces`, and `max_trace_length_used` where applicable.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|------|
| M5: training times out | 100 pre-epochs + 5000 adv steps too long | Set `QUICK_MODE = True` in `benchmark/docker/run_avatar.py` |
| M5: activity names don't match | GAN generates lowercase tokens; model has mixed-case multi-word names | Greedy longest-match decoding applied in `run_m5_final.py` |
| xvfb-run fails | Virtual framebuffer not started | Use `xvfb-run --auto-servernum` |
| JAR exits immediately (M4/M8) | ProM context not initialized | Check `fake-context.jar` is on classpath |

---

## 7. Changelog

| Date | Change |
|------|--------|
| 2026-06-13 | **Trace_Filtered D1 complete (all 15 methods).** Finished M2 (0.0376), M3 (29.87), **M5 (0.0000)** , M6 (0.5819 ± 0.0120), M7 (1.0000), R2 (0.6058 ± 0.0947), R3 (0.2796 ± 0.0033) for Trace_Filtered on D1 Sepsis. M5 = 0.0000 is the strongest memorization pole signal. All functionality added directly to existing scripts: `demo_d1.py` got `--miners` CLI + R2; `bridges/run_m6.py` / `run_m7.py` got `--miners`; `docker/run_avatar.py` got `--miners`, `--eval-only`, + Trace_Filtered miner entry. `01_prepare_models.py` regenerated all PNMLs incl. Trace_Filtered. No new scripts created. |
| 2026-06-12 | **Methodology v2 sync.** M1 family expanded to M1–M1f (v2.4–v2.6). Added Trace_Filtered miner (0.0 pole). v2.5/v2.6 results in `configs_v2/`. Updated BenchmarkDesign.md with merged v2 spec. |\n| 2026-06-10 | **Full English documentation.** Archived M4 (`archive/Tianhao/benchmark/m4.sh`) and M8 (`archive/Tianhao/benchmark/m8.sh`) — both infeasible on real-life logs. Removed `build/` and `lib/` directories (M4 compile artifacts). Cleaned up stale CSV files. |
| 2026-06-09 | **M5: AVATAR RelGAN on D1 Sepsis completed.** Built Docker image `avatar-tf1` (nvcr.io TF 1.15 + pm4py 1.2.6). Trained GAN (5000 adv steps, checkpoint suffix=4981). Fixed multi-word activity bug via greedy longest-match decoding. 2 sampling runs → Mean±Std for all 7 miners. Results table updated. |
| 2026-06-09 | **M4: Gurobi 11.0 integration complete.** Repackaged EfficientStorage JAR with updated Gurobi imports (`com.gurobi.gurobi.*`). Mini dataset Alpha=0.7125 in 21ms. Full Sepsis: Alpha+ ran 14h without completing — single-thread bottleneck. HPC execution strategy documented. |
| 2026-06-09 | **M8: xvfb fix + diagnosis.** Added `--auto-servernum` to xvfb-run after Docker reconfiguration. Enabled stderr capture in m8.sh. Core issue: PatternBasedGeneralization too slow/unstable for real-life logs. **Skipped.** |
| 2026-06-08 | M6: Adapted BSGen approach — replaced broken Entropia EMP/EMR with token replay. All 7 miners pass in ~12s each. |
| 2026-06-08 | M3: Changed from `-emp` (PNML) to `-r` (DFG JSON). Confirmed working in 0.3s. |
| 2026-06-08 | Architecture: per-method shell scripts in `benchmark/`. `run_all.sh` calls them. |
