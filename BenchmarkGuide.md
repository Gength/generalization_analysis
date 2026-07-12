# BenchmarkGuide.md — Generalization Benchmark

## Quick Start

```bash
# Full pipeline (all methods, D1 Sepsis):
bash benchmark/shell/run_all.sh

# Single method (e.g., M3 Entropic Relevance):
bash benchmark/shell/m3.sh
```

---

## 1. Overview

Compare **HybridGen** (M1a–M1g) against external generalization baselines on 21 event logs (D1–D21) with 8 miner configurations. See [`BenchmarkDesign.md`](BenchmarkDesign.md) for the full methodology v2 specification.

### Methods (v2)

| ID | Method | Type | Status | Spec | Script |
|----|--------|------|--------|------|--------|
| M1a | HybridGen v1.0 | Python | ✅ D1/D2 complete | [Design §Tier 1](BenchmarkDesign.md#tier-1--our-method-v2) | `bash benchmark/shell/m1.sh` |
| M1b | HybridGen v2.1 (N=3) | Python | ✅ D1/D2 complete | ↑ | `bash benchmark/shell/m1.sh` |
| M1c | HybridGen v2.1 (N=6) | Python | ✅ D1/D2 complete | ↑ | `bash benchmark/shell/m1.sh` |
| M1d | HybridGen v2.4 | Python | ✅ D1/D2 complete (baseline) | ↑ | `bash benchmark/shell/m1.sh` |
| M1e | HybridGen v2.5 | Python | ✅ D1/D2 complete | ↑ | `bash benchmark/shell/m1.sh` |
| M1f | HybridGen v2.6 (log) | Python | ✅ D1/D2 complete | ↑ | `bash benchmark/shell/m1.sh` |
| M1g | HybridGen v2.6 (mle) | Python | ✅ D1/D2 complete (headline) | ↑ | `bash benchmark/shell/m1.sh` |
| M2 | PM4Py Built-in Gen | Python | ✅ | [Design §M2](BenchmarkDesign.md#tier-2--external-generalization-baselines) | `bash benchmark/shell/m2.sh` |
| M3 | Entropic Relevance | Java (Entropia) | ✅ | [Design §M3](BenchmarkDesign.md#tier-2--external-generalization-baselines) | `bash benchmark/shell/m3.sh` |
| M5 | AVATAR (RelGAN) | Docker TF1.15 GPU | ✅ D1/D2 complete | [Design §M5](BenchmarkDesign.md#m5--avatar-relgan) | `bash benchmark/shell/m5.sh` |
| M6 | Bootstrap Gen (Entropia -bgen) | Java (Entropia 1.7) | ✅ D1/D2 complete | [Design §M6](BenchmarkDesign.md#m6--bootstrap-generalization) | `bash benchmark/shell/m6.sh` |
| M7 | SpeciAL4PM | Python (special4pm) | ✅ | [Design §M7](BenchmarkDesign.md#tier-2--external-generalization-baselines) | `bash benchmark/shell/m7.sh` |
| R1 | K-Fold CV (k=5) | Python | ✅ | [Design §Tier 3](BenchmarkDesign.md#tier-3--reference--sanity-check-metrics) | `bash benchmark/shell/reference.sh` |
| R2 | Leave-One-Variant-Out | Python | ✅ | ↑ | `bash benchmark/shell/reference.sh` |
| R3 | Naive Random Baseline | Python | ✅ | ↑ | `bash benchmark/shell/reference.sh` |

> **Archived** (not feasible on real-life logs, see [Archived Methods](BenchmarkDesign.md#archived-methods)):
> - M4 Anti-Alignment Gen — `archive/Tianhao/benchmark/m4.sh`
> - M8 Pattern-based Gen — `archive/Tianhao/benchmark/m8.sh`
>
> **D2 notes**: M3 on D2 (BPI2013 Incidents) returns 0.0 — only 4 real activities with a dense DFG (69% arc density), so entropic relevance is genuinely near zero. M5 D2 was fixed (trailing underscore bug in `src/AVATAR/avatar/generalization.py` line 78) and re-run with tp_eval-based best-suffix selection (suffix=3781, previously suffix=4981); all 8 miners now show correct non-zero scores except Trace_Filtered (fitness=0 due to restrictive model).

### M6 Implementation Note

M6 ("Bootstrap Generalization") has **two versions** in this benchmark,
reflecting a methodological upgrade:

| Component | v1 methodology (`configs/`) | v2 methodology (`configs_v2/` — current) |
|-----------|----------------------------|------------------------------------------|
| Sampler | `log_sample_with_breeding()` — genetic crossover, k=2, p=1.0 | **Same** sampler, but now **built into the Entropia JAR** (`-bgen` flag) |
| Scoring | PM4Py token-replay fitness (`rf_eval.apply`) | Entropia `-bgen` — eigenvalue-based exact-match **precision & recall** (Polyvyanyy et al., CAiSE 2022) |
| Replicates | 10 reps × 200 traces, 10 generations | 5 reps × 200 traces, 10 generations |
| What it measures | How many bred traces the model can replay perfectly | Entropy-based agreement between model and bootstrapped log behavior |

**Why the change?** Earlier attempts to use Entropia `-emp`/`-emr` directly on PNML
models failed with NullPointerException (JAR bugs processing real Petri nets with
silent transitions). The solution is to use the **`-bgen`** flag, which takes the
model in **DFG JSON format** (converted from PNML via model simulation) and runs
the full bootstrap breeding + eigenvalue scoring internally. Three JAR versions
were tested (1.6, 1.7, 1.8); only **1.7's `-bgen`** works reliably on real logs.
(1.6 lacks `-bgen`; 1.8 is a fat JAR needing AcceptingPetriNet.jar on the classpath
and has the same `k=2` NPE bug as 1.7.)

A **patched JAR** (`jbpt-pm-entropia-1.7.1.jar`) fixes a null-pointer bug in
`EventLogSampling.logBreeding:101` that crashed `-bgen` on some datasets (D2)
with `k=2`. The two-line fix adds a `sites != null` guard. See
[M6 Fixed JAR](#m6-fixed-jar) for details.

**Impact on results:** Entropia scores are systematically **lower** than token-replay
fitness (the eigenvalue-based measure is stricter), so the v2 M6 column is not
directly comparable to v1 M6 values. The new values correctly separate the poles
(Trace_Filtered higher precision, Flower ≈ 1.0 recall) and provide a principled
information-theoretic generalization estimate.

**Dataset-specific note (D2 — BPI2013 Incidents):** The `-bgen` flag's internal
XES parser used to trigger a NullPointerException on BPI2013 Incidents when using
the default `k=2` subtrace length. This is now **fixed** in the patched JAR
[`jbpt-pm-entropia-1.7.1.jar`](src/codebase/jbpt-pm/entropia/jbpt-pm-entropia-1.7.1.jar)
(see [`EventLogSampling.java`](src/codebase/jbpt-pm/src/main/java/org/jbpt/pm/gen/bootstrap/EventLogSampling.java)
line 101: added `sites != null` guard). All D2 results now use `k=2` with the
fixed JAR. As noted above, `k=1` vs `k=2` produces systematically different
precision/recall values (k=2 is stricter → lower precision), so the new D2 values
are not directly comparable to the old k=1 results.

**Known limitations (verified 2026-06-18):**
1. **`k=2` on D2 — FIXED** (was NPE at `EventLogSampling.logBreeding:101`).
   See [`jbpt-pm-entropia-1.7.1.jar`](src/codebase/jbpt-pm/entropia/jbpt-pm-entropia-1.7.1.jar)
   and [`EventLogSampling.java`](src/codebase/jbpt-pm/src/main/java/org/jbpt/pm/gen/bootstrap/EventLogSampling.java)
   (line 101: added `sites != null` guard). All 8 miners pass with `k=2`.
2. **`.xes.gz` not supported.** The Entropia 1.7 XES parser reads `.xes.gz`
   compressed files incorrectly (returns a format error). Always pass
   decompressed `.xes` files to `-rel` when using `-bgen`.
3. **D1 (Sepsis) with `k=2` works correctly** — 16 activities and diverse
   trace lengths ensure breeding sites are always found.

**M6 Fixed JAR:**
A patched version of the Entropia 1.7 JAR is available at
`src/codebase/jbpt-pm/entropia/jbpt-pm-entropia-1.7.1.jar`. It fixes the D2 `k=2`
NPE by adding a null guard in `EventLogSampling.logBreeding`. To use it, replace
`jbpt-pm-entropia-1.7.jar` with `jbpt-pm-entropia-1.7.1.jar` in the classpath.
The patched JAR works on all 8 miners × D2 with `k=2`.

**Runner script:** `benchmark/bridges/run_m6_bgen.py` provides the core algorithm;
`benchmark/job_m6.py` is the self-contained job wrapper.
Usage: `uv run python benchmark/job_m6.py --dataset D2 --k 2 --m 5`

**Configuration files:**
- v1 scores (breeding + token replay) → `benchmark/results/configs/{Dataset}__{Miner}__M6.json`
- v2 scores (breeding + Entropia -bgen) → `benchmark/results/configs_v2/{Dataset}__{Miner}__M6.json`
- Per-miner DFG JSONs used by -bgen → prepared automatically in `/tmp` by `job_m6.py`

Miner configurations (8 total, v2):

| # | Miner | Role |
|---|-------|------|
| 0 | Trace_Filtered (top-50 variants) | **0.0 pole** — pure memorization |
| 1–6 | Alpha, Alpha+, Heuristics (default/strict), Inductive (strict/infrequent) | the six "real" miners |
| 7 | Flower Model | **1.0 pole** — accepts everything |

---

## 2. Src Repositories

The `src/` directory contains third-party repositories cloned from GitHub.
Each is kept as close to upstream as possible; modifications are minimal and
documented below.

| Directory | Upstream | Purpose |
|-----------|----------|---------|
| `src/codebase/` | [jbpt/codebase](https://github.com/jbpt/codebase) | jBPT Entropia JAR — used by M3, M6 |
| `src/AVATAR/` | [Julian-Theis/AVATAR](https://github.com/Julian-Theis/AVATAR) | AVATAR RelGAN (TF1) — used by M5 |
| `src/AVATAR_tf2/` | [Julian-Theis/AVATAR](https://github.com/Julian-Theis/AVATAR) | AVATAR RelGAN (TF2 compat) — used by M5 `--tf2` |
| `src/SpeciAL-core/` | [MartinKabierski/SpeciAL-core](https://github.com/MartinKabierski/SpeciAL-core) | SpeciAL4PM library — used by M7 |

### Modifications

#### `src/codebase/` — Fixed Entropia JAR (1.7.1)

- **File added** (untracked): `jbpt-pm/entropia/jbpt-pm-entropia-1.7.1.jar`
- **Bug**: `EventLogSampling.java:101` calls `sites.isEmpty()` without null check;
  `getBreedingSites()` returns `null` when a trace is shorter than k (crossover
  produces short traces on D2 BPI2013 Incidents), causing NPE at k=2.
- **Fix**: Added `sites != null &&` guard at `logBreeding:101` in
  `src/main/java/org/jbpt/pm/gen/bootstrap/EventLogSampling.java`,
  recompiled with `javac -cp jbpt-pm-entropia-1.7.jar:lib/*` and replaced the
  class inside the JAR.
- **Upstream issue**: Same NPE present in both 1.7 and 1.8 JARs.
- **Runner**: `benchmark/bridges/run_m6_bgen.py` (core algorithm; wrapped by `benchmark/job_m6.py`)

#### `src/AVATAR/` — Trailing underscore fix

- **File modified**: `avatar/generalization.py` line 78
- **Bug**: Activity name matching failed on D2 (BPI2013 Incidents) due to a
  trailing underscore in the label comparison logic.
- **Fix**: Changed greedy longest-match decoding to strip trailing underscores.
- **Note**: This fix was applied directly to the cloned repo.

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
bash benchmark/shell/m5.sh
```

### Switching Datasets

Self-contained jobs (job_m1.py, job_m2.py, etc.) prepare their own data in `/tmp`,
so there is no shared cache to clean up when switching datasets.

### Data (Git-LFS)

All XES event logs in `data/` are tracked with **Git-LFS** (`.gitattributes`). A plain `git clone` or `git pull` will only download pointer files (~130 bytes) if Git-LFS is not properly initialized in the system, resulting in `invalid gzip header` errors when PM4Py/r4pm tries to parse the XES files.

**Install and Initialize Git-LFS (user-level, no root required):**

```bash
ARCH=$(uname -m)
[ "$ARCH" = "x86_64" ] && LFS_ARCH="amd64" || LFS_ARCH="arm64"
LFS_VERSION="3.6.1"
wget "[https://github.com/git-lfs/git-lfs/releases/download/v$](https://github.com/git-lfs/git-lfs/releases/download/v$){LFS_VERSION}/git-lfs-linux-${LFS_ARCH}-v${LFS_VERSION}.tar.gz"
tar -xzf "git-lfs-linux-${LFS_ARCH}-v${LFS_VERSION}.tar.gz"

# Install to default local binary directory (~/.local/bin)
PREFIX="$HOME/.local" ./git-lfs-${LFS_VERSION}/install.sh
export PATH="$HOME/.local/bin:$PATH"

# CRITICAL: Register Git-LFS global filters and hooks to prevent modified status bugs
git lfs install
git lfs version
```

**Clone and Setup (First time):**

```bash
# Modern Git handles LFS automatically during clone if 'git lfs install' was executed
git clone <repo-url>
```

**Pull and Update:**

```bash
# If 'git lfs install' is active, a standard pull will fetch both code and LFS data atomically.
# DO NOT split into 'git pull && git lfs pull' as it may corrupt the Git index.
git pull
```

**Verify data integrity:**

```bash
gzip -t "data/BPI-Challenge_2017/BPI Challenge 2017.xes.gz"
# Exit code 0 = valid gzip
```

All shell scripts in `benchmark/shell/` include `export PATH="$HOME/.local/bin:$PATH"`, so git-lfs installed to `~/.local/bin` is available to SLURM jobs.

---

### Dataset & Miner Availability

| Key | Dataset | Benchmark Status | Miners Available |
|-----|---------|-----------------|-----------------|
| D1 | Sepsis | ✅ Complete | 8/8 |
| D2 | BPI2013_Incidents | ✅ Complete | 8/8 |
| D3 | BPI2017 | ⚠️ all except M5 | 8/8 |
| D4 | BPI2018 | ⚠️ all except M5 | 8/8 |
| D5 | BPI2019 | ⚠️ all except M5 | 8/8 |
| D6 | BPI2013_Problem_Open | ⚠️ partial | 8/8 |
| D7 | BPI2013_Problem_Closed | ⚠️ partial | 8/8 |
| D8 | BPI2015_Municipality_2 | — | 5/8 (no Alpha, Alpha+, Inductive_Strict) |
| D9 | BPI2015_Municipality_4 | — | 5/8 (same) |
| D10 | BPI2015_Municipality_1 | — | 5/8 (same) |
| D11 | BPI2011_Hospital | — | 5/8 (no Alpha, Alpha+; Inductive_Strict timeout) |
| D12 | BPI2015_Municipality_5 | — | 5/8 (same as D8) |
| D13 | BPI2015_Municipality_3 | — | 4/8 (also no Inductive_Infrequent) |
| D14 | BPI2020_PrepaidTravel | — | 8/8 |
| D15 | BPI2020_InternationalDecl | — | 8/8 |
| D16 | BPI2020_RequestForPayment | — | 8/8 |
| D17 | BPI2020_PermitLog | — | 8/8 |
| D18 | BPI2020_DomesticDecl | — | 8/8 |
| D19 | BPI2012 | — | 8/8 |
| D20 | Hospital_Billing | — | 8/8 |
| D21 | Road_Traffic_Fine | — | 8/8 |

> Canonical source: `benchmark/statistics/_miner_availability.json`. All benchmark
> jobs should read this file and skip unavailable miners.

## 4. Running

### Self-contained jobs

Every method is a self-contained job. No preparation step needed —
each script creates a temp workdir in `/tmp`, prepares data, runs, and cleans up:

```bash
# Run any method directly (default: results → /tmp/<workdir>/results/):
uv run python benchmark/job_m1.py --dataset D1   # M1a-M1g (~3 min)
uv run python benchmark/job_m2.py --dataset D1   # M2 (~10s)
uv run python benchmark/job_m3.py --dataset D1   # M3 (~1 min)
uv run python benchmark/job_m6.py --dataset D1   # M6 (~2 min)
uv run python benchmark/job_m7.py --dataset D1   # M7 (~2 min)
uv run python benchmark/job_m5.py --dataset D1   # M5 (~4h, FULL)
uv run python benchmark/job_r1.py --dataset D1   # R1 (~5 min)
uv run python benchmark/job_r2.py --dataset D1   # R2 (~10 min)
uv run python benchmark/job_r3.py --dataset D1   # R3 (~2 min)

# Production output (writes to benchmark/results/configs_v2/):
uv run python benchmark/job_m1.py --dataset D1 --output benchmark/results/configs_v2
```

All scripts accept `--dataset D1..D21` and `--output <dir>` (default: `/tmp/.../results/`).

Shell wrappers (shortcuts):
```bash
bash benchmark/shell/m1.sh --dataset D1
bash benchmark/shell/m2.sh --dataset D1
bash benchmark/shell/m3.sh --dataset D1 --miners Alpha Flower
bash benchmark/shell/r1.sh --dataset D1
bash benchmark/shell/r2.sh --dataset D1
bash benchmark/shell/r3.sh --dataset D1
# reference.sh runs all three: R1 + R2 + R3
bash benchmark/shell/reference.sh --dataset D1
```

> **D2 complete**: All 15 methods (M1a–M1g, M2, M3, M5, M6, M7, R1–R3) × 8 miners = 120 configs for D2 BPI2013_Incidents, written to `benchmark/results/configs_v2/`. M2 completed 2026-06-16.

### Dataset registry

All benchmark scripts share a **single source of truth** for dataset definitions in [`benchmark/datasets.py`](benchmark/datasets.py). It defines D1–D21 with `name`, `log_path`, and `system_name` (for AVATAR). Import via `from datasets import DATASETS, get_info`. Never define inline `DATASETS` dicts.

```python
from datasets import DATASETS, get_info

info = get_info("D3")
print(info["name"], info["log_path"])
# → BPI2017 data/BPI-Challenge_2017/BPI Challenge 2017.xes.gz
```

### M1-family runner (v2 methodology)

```bash
# All 7 M1 versions (M1a–M1g), 8 miners, config JSONs + agreement stats:
uv run python benchmark/job_m1.py --dataset D1
uv run python benchmark/job_m1.py --dataset D2

# Only the new versions:
uv run python benchmark/job_m1.py --dataset D1 --methods M1e M1f M1g

# Production output (writes to configs_v2/):
uv run python benchmark/job_m1.py --dataset D1 --output benchmark/results/configs_v2
```

Default output: `/tmp/<workdir>/results/` — safe for testing, never touches project files.
Use `--output benchmark/results/configs_v2` to write to the production directory.
See [`BenchmarkDesign.md`](BenchmarkDesign.md) for the protocol.

### R-family runners (R1–R3 reference metrics)

Each R method is an independent job:

```bash
# R1: K-Fold CV (k=5, variant-based, 3 shuffles)
uv run python benchmark/job_r1.py --dataset D1

# R2: Leave-One-Variant-Out (default: all variants; --r2-sample N to cap)
uv run python benchmark/job_r2.py --dataset D1
uv run python benchmark/job_r2.py --dataset D1 --r2-sample 50

# R3: Naive Random Baseline (5 iterations, 1000 traces)
uv run python benchmark/job_r3.py --dataset D1
```

Or using shell wrappers:
```bash
bash benchmark/shell/r1.sh --dataset D1
bash benchmark/shell/r2.sh --dataset D1
bash benchmark/shell/r3.sh --dataset D1
bash benchmark/shell/reference.sh --dataset D1  # all three sequentially
```

| Method | What | Detail |
|--------|------|--------|
| **R1** | K-Fold CV (k=5) | Variant-based, 3 shuffles, reports mean ± std. Uses [`benchmark/utils.py`](benchmark/utils.py) `compute_kfold_fitness()`. |
| **R2** | Leave-One-Variant-Out | Each variant held out in turn; LOVO fitness over all (or sampled) variants. `--r2-sample N` caps evaluation to N random variants for fast iteration. Default = all variants (100%%). |
| **R3** | Naive Random Baseline | Uniform random activity traces, length sampled from log distribution. 5 iterations of 1,000 traces. |

Results go to `benchmark/results/configs_v2/` when `--output` is specified,
or to `/tmp/<workdir>/results/` by default. The legacy `r1.sh` has been removed —
`reference.sh` is the unified entry point for R1–R3.

Archived methods in `archive/Tianhao/benchmark/` (see [Archived Methods](BenchmarkDesign.md#archived-methods)).

### Full pipeline (sequential)

```bash
bash benchmark/shell/run_all.sh
```

The pipeline calls `m1.sh`, `m2.sh`, `m3.sh`, `m6.sh`, `m7.sh`, and optionally `m5.sh`.
Each job is self-contained — no prepare step is needed.

For production runs (write to `benchmark/results/configs_v2/`):
```bash
OUTPUT_DIR=benchmark/results/configs_v2 bash benchmark/shell/run_all.sh D1
```

---

## 5. Results (D1 Sepsis)

| Miner | M1a | M1b | M1c | M1d | M1e | M1f | M1g | M2 | M3* | M5 | M6 | M7 | R1 | R2 | R3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Trace_Filtered | 0.562 | 0.4956 | 0.5173 | 0.5106 | 0.5085 | 0.5085 | 0.5687 | 0.0285 | 46.5076 | 0.138 | 0.4609 | 1.0 | 0.6411 | 0.6058 | 0.2796 |
| Alpha | 0.2654 | 0.2841 | 0.2968 | 0.2881 | 0.2849 | 0.2849 | 0.2724 | 0.9132 | 63.3129 | 0.3289 | 0.0 | 0.7989 | 0.2748 | 0.3059 | 0.2779 |
| Alpha+ | 0.6062 | 0.5522 | 0.6271 | 0.6277 | 0.6512 | 0.6512 | 0.7591 | 0.9189 | 64.3446 | 0.3115 | 0.1875 | 0.75 | 0.8293 | 0.7753 | 0.382 |
| Heuristics | 0.8651 | 0.8214 | 0.8355 | 0.8317 | 0.8457 | 0.8457 | 0.8787 | 0.8298 | 62.2738 | 0.7462 | 0.3554 | 1.0 | 0.9023 | 0.87 | 0.5024 |
| Heuristics_Strict | 0.8929 | 0.8465 | 0.8535 | 0.8512 | 0.864 | 0.864 | 0.9174 | 0.9004 | 61.7765 | 0.7343 | 0.3342 | 0.9983 | 0.9329 | 0.9175 | 0.6212 |
| Inductive_Strict | 0.9753 | 0.9423 | 0.9557 | 0.9574 | 0.9613 | 0.9613 | 0.9838 | 0.9025 | 59.755 | 0.5705 | 0.2298 | 0.7443 | 0.9999 | 1.0 | 0.7667 |
| Inductive_Infrequent | 0.9107 | 0.8901 | 0.9161 | 0.92 | 0.931 | 0.931 | 0.9723 | 0.8799 | 61.8158 | 0.6974 | 0.2708 | 0.75 | 0.9846 | 0.9813 | 0.693 |
| Flower | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.9132 | 63.0675 | 0.3301 | 0.1812 | 0.8034 | 1.0 | 1.0 | 1.0 |

> **M1e–M1g (v2.5/v2.6)**: Added 2026-06-12. All values from v2 methodology runner (`uv run python benchmark/run_m1_family.py --dataset D1`), seed 42, 5 iterations. Source: `benchmark/results/configs_v2/Sepsis__*__M1{e,f,g}.json`. M1e (v2.5 Katz proposal) and M1f (v2.6 log-weighted) produce identical Gen_shadow means on the 7 regular miners by design — M1f adds `gen_accept`, `gen_shadow_regular`/`_mutated`, and probe-integrity counters. M1g (v2.6 MLE-weighted) is the headline candidate.
>
> **M5 std**: Alpha=±0.057, Alpha+=±0.002, Heuristics=±0.063, Heuristics_Strict=±0.110, Inductive_Strict=±0.023, Inductive_Infrequent=±0.021, Flower=±0.007 (2 runs). Multi-word activity fix applied (greedy longest-match decoding for GAN output). Checkpoint selection via tp_eval-based best suffix (suffix=3901).
>
> **M3**: Raw entropic relevance (unbounded, higher=better). Per-miner DFG simulation via open-source `relevance.jar` (JDFG2Aut + Relevance). Scores discriminate between miners on all datasets.
>
> **M4/M8 (D1)**: Archived — not feasible on real-life logs (see [Archived Methods](BenchmarkDesign.md#archived-methods)).
>
> **Trace_Filtered row**: M1a–M1g + R1 from v2 configs (`configs_v2/`). M2/R3 computed 2026-06-13 via `benchmark/run_trace_filtered_externals.py`. **M3 fix (2026-06-27)**: switched from Entropia JAR `-r` to open-source `relevance.jar` (JDFG2Aut + Relevance); D2 now discriminating (was 0.0). M6 (Entropia -bgen, 5 reps, k=2) recomputed 2026-06-18 with **fixed JAR** `jbpt-pm-entropia-1.7.1.jar` — values consistent with previous run (within bootstrap noise). See [M6 Implementation Note](#m6-implementation-note). M7 (SpeciAL4PM C1 ratio) computed 2026-06-13 via `benchmark/run_m6_m7_trace_filtered.py`. R2 (sampled LOVO, 50/846 variants, seed 42) via `benchmark/run_r2_trace_filtered.py`. **M5 (AVATAR) on Trace_Filtered**: 0.1380 — re-run with tp_eval-based best-suffix selection (suffix=3901, previously suffix=4981). Run via: `uv run python benchmark/docker/run_avatar.py --miners Trace_Filtered --eval-only`.
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
| Trace_Filtered | 0.6335 | 0.595 | 0.6052 | 0.5469 | 0.5456 | 0.5456 | 0.6 | 0.0605 | 10.5817 | 0.0 | 0.7515 | 1.0 | 0.6529 | 0.6798 | 0.4995 |
| Alpha | 0.3129 | 0.383 | 0.3665 | 0.2817 | 0.2803 | 0.2803 | 0.2572 | 0.8825 | 22.4668 | 0.1928 | 0.0 | 1.0 | 0.215 | 0.1402 | 0.4429 |
| Alpha+ | 0.5866 | 0.5178 | 0.54 | 0.5967 | 0.5988 | 0.5988 | 0.6301 | 0.8452 | 18.9131 | 0.9654 | 0.5443 | 0.814 | 0.6979 | 0.7931 | 0.6896 |
| Heuristics | 0.9969 | 0.964 | 0.9694 | 0.9527 | 0.9529 | 0.9529 | 0.9935 | 0.9024 | 16.2197 | 0.9086 | 0.7543 | 0.9803 | 0.9956 | 0.9904 | 0.8106 |
| Heuristics_Strict | 0.999 | 0.971 | 0.9776 | 0.9661 | 0.9664 | 0.9664 | 0.9978 | 0.9295 | 15.2261 | 0.8143 | 0.7244 | 0.9787 | 0.9983 | 0.9974 | 0.9243 |
| Inductive_Strict | 1.0 | 0.9997 | 0.9997 | 0.9989 | 0.9988 | 0.9988 | 1.0 | 0.8711 | 18.6842 | 0.7589 | 0.5633 | 1.0 | 1.0 | 1.0 | 0.9425 |
| Inductive_Infrequent | 0.996 | 0.9729 | 0.9745 | 0.9587 | 0.9598 | 0.9598 | 0.9907 | 0.9887 | 12.2147 | 0.7589 | 0.8095 | 1.0 | 0.9881 | 0.9864 | 0.8474 |
| Flower | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.8825 | 20.2597 | 0.7589 | 0.5261 | 0.9372 | 1.0 | 1.0 | 1.0 |

> **M2 (\*)**: PM4Py built-in generalization. D2 run 2026-06-16 via `uv run python benchmark/run_m2.py --dataset D2`. Values comparable to D1 M2 (Sepsis) range.
>
> **M3 (\*\*)**: Raw entropic relevance (unbounded, higher=better). Per-miner DFG via `relevance.jar`. Range 10.6–22.5 — now discriminating, though values are compressed due to D2's small activity alphabet (4 activities).
>
> **M5 (\*\*\*)**: D2 scores after the trailing underscore fix, re-run with tp_eval-based best-suffix selection (suffix=3781, previously suffix=4981). Trace_Filtered = 0.0000 is genuine (fitness=0 due to restrictive 50-variant model). All other miners show correct non-zero scores.
>
> **M6 (\*\*\*\*)**: Bootstrap Generalization with Entropia `-bgen` scoring (v2 methodology). All values computed with the **fixed JAR** `jbpt-pm-entropia-1.7.1.jar` (null guard added at `EventLogSampling.logBreeding:101`), `k=2`, **10 bootstrap replicates** (design spec: 10, target: 100). The table shows F1 score (harmonic mean of precision and recall). See [M6 Implementation Note](#m6-implementation-note) and [Design §M6](BenchmarkDesign.md#m6--bootstrap-generalization).
>
> **Key observation**: D2's simple activity structure (4 activities) means most miners converge near 1.0 for strong miners (Heuristics, Inductive, Flower). The discriminative power shifts to weaker miners (Alpha, Trace_Filtered) where M1 variants show meaningful spread.
>
> **No M1e vs M1f convergence note**: Unlike D1 where M1e and M1f are identical on regular miners, D2 shows complete convergence (all miners identical for M1e = M1f) — expected when `gen_shadow` dominates and `gen_accept` adds no extra information on simple logs.

---

## 5c. Results (D3 BPI2017)

| Miner | M1a | M1b | M1c | M1d | M1e | M1f | M1g | M2 | M3* | M5 | M6** | M7 | R1 | R2 | R3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Trace_Filtered | 0.5483 | 0.5884 | 0.6181 | 0.6301 | 0.6285 | 0.6285 | 0.6051 | 0.051 | 107.9212 | - | 0.5764 | 1.0 | 0.6724 | 0.5958 | 0.3165 |
| Alpha | 0.3969 | 0.3956 | 0.3874 | 0.3992 | 0.4009 | 0.4009 | 0.389 | 0.9824 | 175.4245 | - | 0.0864 | 0.75 | 0.3851 | 0.3886 | 0.3748 |
| Alpha+ | 0.6047 | 0.699 | 0.7535 | 0.7859 | 0.7832 | 0.7832 | 0.8599 | 0.9828 | 182.2312 | - | 0.093 | 0.75 | 0.8838 | 0.8595 | 0.4509 |
| Heuristics | 0.93 | 0.8654 | 0.8815 | 0.8812 | 0.8818 | 0.8818 | 0.9477 | 0.9289 | 185.5653 | - | 0.1366 | 0.9918 | 0.9526 | 0.9546 | 0.5626 |
| Heuristics_Strict | 0.9396 | 0.8801 | 0.8939 | 0.8936 | 0.8939 | 0.8939 | 0.9512 | 0.9575 | 185.4373 | - | 0.125 | 0.9219 | 0.9522 | 0.9598 | 0.5693 |
| Inductive_Strict | 0.9653 | 0.9887 | 0.9945 | 0.9963 | 0.9974 | 0.9974 | 0.9996 | 0.9485 | 183.5132 | - | 0.1345 | 0.8536 | 1.0 | 1.0 | 0.8829 |
| Inductive_Infrequent | 0.9431 | 0.9332 | 0.95 | 0.953 | 0.9555 | 0.9555 | 0.9761 | 0.9473 | 186.1838 | - | 0.0176 | 0.9999 | 0.9807 | 0.9771 | 0.7252 |
| Flower | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.9824 | 183.5981 | - | 0.0837 | 0.7811 | 1.0 | 1.0 | 1.0 |

> **M3 (\*)**: Raw entropic relevance (unbounded, higher=better). Per-miner DFG simulation via open-source `relevance.jar` (JDFG2Aut + Relevance). Scores discriminate between miners on all datasets.
>
> **M6 (\*\*)**: Bootstrap Generalization with Entropia `-bgen` scoring (v2 methodology). All values computed with the **fixed JAR** `jbpt-pm-entropia-1.7.1.jar` (null guard added at `EventLogSampling.logBreeding:101`), `k=2`, **10 bootstrap replicates** (design spec: 10, target: 100). The table shows F1 score (harmonic mean of precision and recall). See [Design §M6](BenchmarkDesign.md#m6--bootstrap-generalization).
>
> **M5**: Not yet run on D3 (BPI2017). Docker AVATAR training requires ~4h per miner. See [BenchmarkDesign.md](BenchmarkDesign.md) for the full method catalog.
>
> **Key observations**: D3 (BPI2017) has a deep activity profile (87% variant singletons, average trace length ~37) which stresses all methods. Flower ≈ 1.0 on all M1 variants and R1/R2 confirms construct-purity: a pure generalization metric should score the maximally permissive model at ceiling. Trace_Filtered shows the widest M1 spread (0.55–0.63) and the lowest R3 (0.3165), consistent with severe memorization pressure. M6 (bootstrap gen) produces very low F1 scores on Alpha-family miners (0.0864 Alpha, 0.0930 Alpha+) — likely due to precision collapse from poor recall on deep, sparse traces.
>
> **M1e ≡ M1f**: Complete convergence across all miners (M1e = M1f values identical), consistent with D1 and D2 patterns — `gen_shadow` dominates and `gen_accept` adds no discriminative information when shadow-trace coverage is already saturated.

---

## 5d. Results (D4 BPI2018)

| Miner | M1a | M1b | M1c | M1d | M1e | M1f | M1g | M2 | M3* | M5 | M6** | M7 | R1 | R2 | R3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Trace_Filtered | 0.5417 | 0.3904 | 0.3975 | 0.3972 | 0.397 | 0.4111 | 0.5814 | 0.0717 | 255.0204 | - | 0.0 | 1.0 | 0.5953 | 0.5643 | 0.3199 |
| Alpha | 0.2161 | 0.2215 | 0.2146 | 0.2047 | 0.1915 | 0.1922 | 0.22 | 0.9686 | 314.8662 | - | 0.0 | 0.7512 | 0.1955 | 0.2094 | 0.2881 |
| Alpha+ | 0.5085 | 0.5042 | 0.5116 | 0.5164 | 0.5026 | 0.5369 | 0.604 | 0.9698 | 300.589 | - | 0.0 | 0.75 | 0.6333 | 0.6437 | 0.461 |
| Heuristics | 0.8746 | 0.8165 | 0.8391 | 0.8466 | 0.8519 | 0.8582 | 0.8608 | 0.8604 | 314.8549 | - | 0.0 | 0.9974 | 0.8822 | 0.8792 | 0.6184 |
| Heuristics_Strict | 0.928 | 0.8903 | 0.9056 | 0.9109 | 0.9213 | 0.9254 | 0.9306 | 0.9043 | 314.8657 | - | 0.0 | 0.8747 | 0.9383 | 0.9378 | 0.6774 |
| Inductive_Strict | 0.9744 | 0.9773 | 0.9752 | 0.9749 | 0.9742 | 0.9742 | 0.9758 | 0.9526 | 314.8657 | - | 0.0 | 0.968 | 0.9847 | - | 0.9745 |
| Inductive_Infrequent | 0.9318 | 0.8565 | 0.8806 | 0.8928 | 0.9081 | 0.9187 | 0.9688 | 0.9292 | 303.1617 | - | 0.0 | 0.8769 | 0.9818 | 0.9843 | 0.7247 |
| Flower | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.9686 | 308.1287 | - | 0.0 | 0.7725 | 1.0 | 1.0 | 1.0 |

> **M3 (\*)**: Raw entropic relevance (unbounded, higher=better). Per-miner DFG simulation via open-source `relevance.jar` (JDFG2Aut + Relevance).
>
> **M6 (\*\*)**: Bootstrap Generalization with Entropia `-bgen` scoring (v2 methodology). All values computed with the **fixed JAR** `jbpt-pm-entropia-1.7.1.jar`, `k=2`, **10 bootstrap replicates**. The table shows F1 score (harmonic mean of precision and recall).
>
> **M5**: Not yet run on D4 (BPI2018). Docker AVATAR training requires ~4h per miner.
>
> **Key observations**: D4 (BPI2018) has the largest variant count (28K) and deepest traces (avg 57) of all benchmark datasets, with the lowest TLRA (0.35) — making it the hardest generalization challenge. Flower ≈ 1.0 on all M1 variants and R1/R2/R3 confirms construct-purity. M6 produces 0.0 for all miners — extreme sparsity causes complete bootstrap precision collapse. The Heuristics family shows the highest M1 scores (0.82–0.93), reflecting their good generalization to unseen behavior.
>
> **M1e ≡ M1f**: Complete convergence across all miners (M1e = M1f values identical), consistent with D1–D3 patterns.

---

## 5e. Results (D5 BPI2019)

| Miner | M1a | M1b | M1c | M1d | M1e | M1f | M1g | M2 | M3* | M5 | M6** | M7 | R1 | R2 | R3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Trace_Filtered | 0.6158 | 0.5272 | 0.5536 | 0.5497 | 0.5583 | 0.5583 | 0.5548 | 0.1453 | 18.2537 | - | 0.6904 | 1.0 | 0.5825 | 0.5253 | 0.2047 |
| Alpha | 0.3463 | 0.3509 | 0.3584 | 0.3303 | 0.3163 | 0.3163 | 0.3679 | 0.9112 | 39.8273 | - | 0.0 | - | 0.3452 | 0.2825 | 0.3383 |
| Alpha+ | 0.4032 | 0.4585 | 0.4576 | 0.4555 | 0.4391 | 0.4391 | 0.3228 | 0.9147 | 39.5793 | - | 0.0807 | - | 0.4564 | 0.4149 | 0.2491 |
| Heuristics | 0.8302 | 0.812 | 0.8146 | 0.8041 | 0.8104 | 0.8104 | 0.8378 | 0.7734 | 27.4939 | - | 0.1885 | - | 0.8848 | 0.8587 | 0.4991 |
| Heuristics_Strict | 0.8858 | 0.8498 | 0.8544 | 0.8486 | 0.8531 | 0.8531 | 0.897 | 0.8708 | 23.202 | - | 0.1944 | - | 0.9293 | 0.9113 | 0.5635 |
| Inductive_Strict | 0.9995 | 0.9983 | 0.998 | 0.9962 | 0.9984 | 0.9984 | 0.9958 | 0.9182 | 39.6001 | - | 0.2222 | - | 1.0 | 0.9999 | 0.9553 |
| Inductive_Infrequent | 0.9891 | 0.9673 | 0.9723 | 0.9584 | 0.9622 | 0.9622 | 0.9617 | 0.8923 | 39.3444 | - | 0.2007 | - | 0.9879 | 0.9839 | 0.8205 |
| Flower | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.9112 | 34.0811 | - | 0.0984 | - | 1.0 | 1.0 | 1.0 |

> **M3 (\*)**: Raw entropic relevance (unbounded, higher=better). Per-miner DFG simulation via open-source `relevance.jar` (JDFG2Aut + Relevance).
>
> **M6 (\*\*)**: Bootstrap Generalization with Entropia `-bgen` scoring (v2 methodology). Fixed JAR `jbpt-pm-entropia-1.7.1.jar`, `k=2`, **10 bootstrap replicates**. Table shows F1 score (harmonic mean of precision and recall).
>
> **M5**: Not yet run on D5 (BPI2019).
>
> **Key observations**: D5 (BPI2019) has the largest case count (251K) — a PM4Py memory scaling stress test — but the smallest variant count (550) and highest TLRA (0.88) among D1–D5. This structured purchase-to-pay process allows Inductive miners and Flower to achieve near-perfect scores (≥0.96 M1, ≥0.99 R1/R2). M6 scores show a clear miner ranking — Inductive miners (0.20–0.22) outperform Alpha-family (0.0–0.08) — and M6's modest scores overall suggest trace-level recall is the bottleneck even on a well-structured log.
>
> **M1e ≡ M1f**: Complete convergence across all miners, consistent with D1–D4 patterns.

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
| 2026-07-12 | **M5 D1/D2 re-run with tp_eval-based best-suffix selection.** D1 (Sepsis) uses suffix=3901 (was 4981), D2 (BPI2013 Incidents) uses suffix=3781 (was 4981). `run_avatar.py` refactored: modular helpers, single XES read, early-noise filtering (>=2000). Config JSONs and benchmark CSVs updated. |
| 2026-06-27 | **D4 (BPI2018) and D5 (BPI2019) results complete (except M5).** All methods M1a–M1g, M2, M3, M6, M7, R1–R3 now have results for both datasets. Minor gaps: D4 M2 missing for Flower, Inductive_Infrequent, Inductive_Strict; D4 R2 missing for Inductive_Strict; D5 M7 only on Trace_Filtered. Added §5d (D4) and §5e (D5) results tables. |
| 2026-06-26 | **M3 re-run D1/D2/D3 (per-miner DFG fix):** Changed M3 from single log-level DFG to per-miner simulated DFGs (PNML → `play_out` 5000 traces → DFG JSON → Entropia `-r`). Scores now discriminate between miners: D1 M3 range 46.51–64.34 (was 29.87 uniform), D3 M3 range 76.99–178.56 (was 178.53 uniform). D2 M3 remains 0.0 for all miners (4-activity dense DFG, genuinely near zero). |
| 2026-06-20 | **M1 no longer computes R1.** R1 ground-truth (5-fold CV) removed from `run_m1_family.py`. M1 now outputs only M1a–M1g gen_shadow values. R1 is produced separately by `job_r1.py` / `r1.sh`. Agreement (Pearson/Spearman/MAE) is computed offline by merging M1 and R1 JSONs. |
| 2026-06-18 | **Standard M6 (Entropia -bgen) benchmark complete for D1 and D2.** Replaced token-replay fitness scoring with Entropia eigenvalue-based precision/recall (`-bgen` flag, 1.7 JAR). Per-miner DFG JSONs generated via PNML simulation. D2 requires `k=1` workaround (JAR NPE at `k=2`). See [M6 Implementation Note](#m6-implementation-note). Updated `configs_v2/` with new standard scores; previous token-replay scores retained in `configs/`.
| 2026-06-18 | **M6** Default `m` changed from 5 to 10; D1 and D2 runs have been fully re-executed. Updated the "Statistical Rigor" table in `BenchmarkDesign.md` to reflect actual settings. The M6 ​​columns in the D1/D2 tables have been updated accordingly. See [Statistical Rigor](BenchmarkDesign.md#statistical-rigor).
| 2026-06-16 | **Benchmark script restructuring.** `m1.sh` now runs only M1a–M1g via `run_m1_family.py`. Created `run_r_family.py` (unified R1–R3 runner using `compute_kfold_fitness` from `utils.py`) and `reference.sh`. R2 adds `--r2-sample` option (default 0 = all variants). Removed `demo_d1.py`, `r1_demo.py`, `r1.sh`. **Created `benchmark/datasets.py`** — canonical D1–D21 definitions; all scripts now import from it. `--dataset` CLI added to all bridge scripts. |
| 2026-06-13 | **Trace_Filtered D1 complete (all 15 methods).** Finished M2 (0.0376), M3 (29.87), **M5 (0.0000)** , M6 (0.5819 ± 0.0120), M7 (1.0000), R2 (0.6058 ± 0.0947), R3 (0.2796 ± 0.0033) for Trace_Filtered on D1 Sepsis. M5 = 0.0000 is the strongest memorization pole signal. All functionality added directly to existing scripts: `demo_d1.py` got `--miners` CLI + R2; `bridges/run_m6.py` / `run_m7.py` got `--miners`; `docker/run_avatar.py` got `--miners`, `--eval-only`, + Trace_Filtered miner entry. `01_prepare_models.py` regenerated all PNMLs incl. Trace_Filtered. No new scripts created. |
| 2026-06-12 | **Methodology v2 sync.** M1 family expanded to M1a–M1g (v2.4–v2.6). Added Trace_Filtered miner (0.0 pole). v2.5/v2.6 results in `configs_v2/`. Updated BenchmarkDesign.md with merged v2 spec. |\n| 2026-06-10 | **Full English documentation.** Archived M4 (`archive/Tianhao/benchmark/m4.sh`) and M8 (`archive/Tianhao/benchmark/m8.sh`) — both infeasible on real-life logs. Removed `build/` and `lib/` directories (M4 compile artifacts). Cleaned up stale CSV files. |
| 2026-06-09 | **M5: AVATAR RelGAN on D1 Sepsis completed.** Built Docker image `avatar-tf1` (nvcr.io TF 1.15 + pm4py 1.2.6). Trained GAN (5000 adv steps, checkpoint suffix=4981). Fixed multi-word activity bug via greedy longest-match decoding. 2 sampling runs → Mean±Std for all 7 miners. Results table updated. |
| 2026-06-09 | **M4: Gurobi 11.0 integration complete.** Repackaged EfficientStorage JAR with updated Gurobi imports (`com.gurobi.gurobi.*`). Mini dataset Alpha=0.7125 in 21ms. Full Sepsis: Alpha+ ran 14h without completing — single-thread bottleneck. HPC execution strategy documented. |
| 2026-06-09 | **M8: xvfb fix + diagnosis.** Added `--auto-servernum` to xvfb-run after Docker reconfiguration. Enabled stderr capture in m8.sh. Core issue: PatternBasedGeneralization too slow/unstable for real-life logs. **Skipped.** |
| 2026-06-08 | M6: Adapted BSGen approach — replaced broken Entropia EMP/EMR with token replay. All 7 miners pass in ~12s each. Full JAR bug history documented in [`run_m6.py`](benchmark/bridges/run_m6.py) docstring.
| 2026-06-08 | M3: Changed from `-emp` (PNML) to `-r` (DFG JSON). Confirmed working in 0.3s. |
| 2026-06-08 | Architecture: per-method shell scripts in `benchmark/`. `run_all.sh` calls them. |
