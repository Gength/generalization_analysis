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

Compare **HybridGen v24** against external generalization baselines on 5 event logs (D1–D5).

### Methods

| ID | Method | Type | Status | Script |
|----|--------|------|--------|--------|
| M1 | HybridGen v24 (N=6) | Python | ✅ | `bash benchmark/m1.sh` |
| M1a | HybridGen v1 (1-gram) | Python | ✅ | `bash benchmark/m1.sh` |
| M1b | HybridGen v2.1 (N=3) | Python | ✅ | `bash benchmark/m1.sh` |
| M1c | HybridGen v2.1 (N=6) | Python | ✅ | `bash benchmark/m1.sh` |
| M2 | PM4Py Built-in Gen | Python | ✅ | `bash benchmark/m1.sh` |
| M3 | Entropic Relevance | Java (Entropia) | ✅ | `bash benchmark/m3.sh` |
| M4 | Anti-Alignment Gen | Java (ProM, Gurobi) | ❌ Single-thread bottleneck, 14h+ per miner, archived | `archive/Tianhao/benchmark/m4.sh` |
| M5 | AVATAR (RelGAN) | Docker TF1.15 GPU | ✅ D1 complete (2 runs) | `bash benchmark/m5.sh` |
| M6 | Bootstrap Gen (adapted) | Python (bsgen) | ✅ | `bash benchmark/m6.sh` |
| M7 | SpeciAL4PM | Python (special4pm) | ✅ | `bash benchmark/m7.sh` |
| M8 | Pattern-based Gen | Java (ProM, Gurobi) | ❌ Infeasible on real-life logs, archived | `archive/Tianhao/benchmark/m8.sh` |
| R1 | K-Fold CV (k=5) | Python | ✅ | `bash benchmark/r1.sh` |
| R2 | Leave-One-Variant-Out | Python | ✅ | Included in `m1.sh` |
| R3 | Naive Random Baseline | Python | ✅ | Included in `m1.sh` |

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

# Step 2: Run individual methods:
bash benchmark/m1.sh   # M1-M1c, M2, R3  (~2 min)
bash benchmark/r1.sh   # R1 K-Fold CV     (~3 min)
bash benchmark/m3.sh   # M3 Entropic      (~1 min)
bash benchmark/m6.sh   # M6 Bootstrap     (~2 min)
bash benchmark/m7.sh   # M7 SpeciAL4PM    (~2 min)
bash benchmark/m5.sh   # M5 AVATAR        (~4h, FULL)
```

Archived (not feasible on real-life logs):
- `archive/Tianhao/benchmark/m4.sh` — M4 Anti-Alignment
- `archive/Tianhao/benchmark/m8.sh` — M8 Pattern-based

### Full pipeline (sequential)

```bash
bash benchmark/run_all.sh
```

---

## 5. Results (D1 Sepsis)

| Miner | M1 | M1a | M1b | M1c | M2 | M3* | M4 | **M5** | **M6** | M7 | M8 | R1 | R2 | R3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Alpha | 0.2862 | 0.2665 | 0.2837 | 0.2948 | 0.9132 | 29.87 | -1.0000 | **0.3401** | **0.2521** | 0.7885 | -1.0000 | 0.2748 | 0.3059 | 0.2779 |
| Alpha+ | 0.6358 | 0.6033 | 0.5523 | 0.6299 | 0.9189 | 29.87 | -1.0000 | **0.5617** | **0.7828** | 0.7500 | -1.0000 | 0.8293 | 0.7753 | 0.3820 |
| Heuristics | 0.8379 | 0.8733 | 0.8262 | 0.8403 | 0.8414 | 29.87 | -1.0000 | **0.7460** | **0.8974** | 0.9989 | -1.0000 | 0.9024 | 0.8700 | 0.5024 |
| Heuristics_Strict | 0.8531 | 0.8936 | 0.8456 | 0.8567 | 0.9004 | 29.87 | -1.0000 | **0.7044** | **0.9311** | 0.9988 | -1.0000 | 0.9329 | 0.9175 | 0.9175 |
| Inductive_Strict | 0.9590 | 0.9747 | 0.9407 | 0.9593 | 0.9025 | 29.87 | -1.0000 | **0.5347** | **0.9943** | 0.7456 | -1.0000 | 0.9999 | 1.0000 | 0.7667 |
| Inductive_Infrequent | 0.9208 | 0.9122 | 0.8872 | 0.9182 | 0.8799 | 29.87 | -1.0000 | **0.7506** | **0.9764** | 0.7500 | -1.0000 | 0.9846 | 0.9813 | 0.6930 |
| Flower | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9132 | 29.87 | -1.0000 | **0.3967** | **1.0000** | 0.8208 | -1.0000 | 1.0000 | 1.0000 | 1.0000 |

> **M5 std**: Alpha=±0.020, Alpha+=±0.008, Heuristics=±0.001, Heuristics_Strict=±0.002, Inductive_Strict=±0.009, Inductive_Infrequent=±0.002, Flower=±0.007 (2 runs). Multi-word activity fix applied (greedy longest-match decoding for GAN output).
>
> **M3**: Raw entropic relevance (unbounded, higher=better). Same DFG-based score for all miners.
>
> **M4 (D1)**: All -1 — Skipped. Algorithm is inherently single-threaded O(n²~n³); Alpha+ ran for 14h without completing one miner. Verified on mini dataset (10 traces, Gen=0.7125, 53ms) but infeasible on full Sepsis.
>
> **M8 (D1)**: All -1 — Skipped. PatternBasedGeneralization algorithm too slow/unstable for real-life logs.
>
> **M1a note**: Alpha row shows 0.6033 in table — this is the Alpha+ score. See config JSON for per-miner breakdown.

### Configuration Convention

Every (dataset, miner, method) cell produces a JSON config in `benchmark/results/configs/`:

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

Config JSONs are the **source of truth**.

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
| 2026-06-10 | **Full English documentation.** Archived M4 (`archive/Tianhao/benchmark/m4.sh`) and M8 (`archive/Tianhao/benchmark/m8.sh`) — both infeasible on real-life logs. Removed `build/` and `lib/` directories (M4 compile artifacts). Cleaned up stale CSV files. |
| 2026-06-09 | **M5: AVATAR RelGAN on D1 Sepsis completed.** Built Docker image `avatar-tf1` (nvcr.io TF 1.15 + pm4py 1.2.6). Trained GAN (5000 adv steps, checkpoint suffix=4981). Fixed multi-word activity bug via greedy longest-match decoding. 2 sampling runs → Mean±Std for all 7 miners. Results table updated. |
| 2026-06-09 | **M4: Gurobi 11.0 integration complete.** Repackaged EfficientStorage JAR with updated Gurobi imports (`com.gurobi.gurobi.*`). Mini dataset Alpha=0.7125 in 21ms. Full Sepsis: Alpha+ ran 14h without completing — single-thread bottleneck. HPC execution strategy documented. |
| 2026-06-09 | **M8: xvfb fix + diagnosis.** Added `--auto-servernum` to xvfb-run after Docker reconfiguration. Enabled stderr capture in m8.sh. Core issue: PatternBasedGeneralization too slow/unstable for real-life logs. **Skipped.** |
| 2026-06-08 | M6: Adapted BSGen approach — replaced broken Entropia EMP/EMR with token replay. All 7 miners pass in ~12s each. |
| 2026-06-08 | M3: Changed from `-emp` (PNML) to `-r` (DFG JSON). Confirmed working in 0.3s. |
| 2026-06-08 | Architecture: per-method shell scripts in `benchmark/`. `run_all.sh` calls them. |
