# BenchmarkGuide.md — Generalization Benchmark Operations

---

## Quick Start

```bash
# Full pipeline (all methods, D1 Sepsis):
bash benchmark/shell/run_all.sh

# Single method:
bash benchmark/shell/m1.sh          # HybridGen variants
bash benchmark/shell/m2.sh          # PM4Py Built-in
bash benchmark/shell/m3.sh          # Entropic Relevance
bash benchmark/shell/m5.sh          # AVATAR (GPU, ~4h)
bash benchmark/shell/m6.sh          # Bootstrap Gen
bash benchmark/shell/m7.sh          # SpeciAL4PM
bash benchmark/shell/reference.sh   # R1–R3
```

All methods described in [`BenchmarkDesign.md`](BenchmarkDesign.md).

---

## Setup

### Python Environment

```bash
uv sync
uv pip install deprecation mpmath cachetools  # SpeciAL4PM dependencies
```

### JVM Heap

All Java methods use **16 GB heap** (`-Xmx16G`) by default.

### AVATAR (M5) — Docker

NVIDIA-maintained TF 1.15 image with RTX 4080 support.

```bash
docker pull nvcr.io/nvidia/tensorflow:22.12-tf1-py3
docker build -t avatar-tf1 -f benchmark/docker/Dockerfile.avatar .
# Training (~4h):
bash benchmark/shell/m5.sh
```

### Data — Git-LFS

Event logs in `data/` are tracked with Git-LFS. Plain `git clone` downloads only pointer files (~130 bytes), causing `invalid gzip header` errors.

```bash
# Install (user-level, no root required)
ARCH=$(uname -m)
[ "$ARCH" = "x86_64" ] && LFS_ARCH="amd64" || LFS_ARCH="arm64"
LFS_VERSION="3.6.1"
wget "https://github.com/git-lfs/git-lfs/releases/download/v${LFS_VERSION}/git-lfs-linux-${LFS_ARCH}-v${LFS_VERSION}.tar.gz"
tar -xzf "git-lfs-linux-${LFS_ARCH}-v${LFS_VERSION}.tar.gz"
PREFIX="$HOME/.local" ./git-lfs-${LFS_VERSION}/install.sh
export PATH="$HOME/.local/bin:$PATH"
git lfs install
git lfs version

# Verify integrity
gzip -t "data/BPI-Challenge_2017/BPI Challenge 2017.xes.gz"
```

---

## Rebuilding `src/`

The `src/` directory contains external repositories used by benchmark methods. If missing or corrupted, clone them as follows:

```bash
mkdir -p src && cd src

# AVATAR (M5) — Docker-based GAN
git clone https://github.com/Julian-Theis/AVATAR.git

# SpeciAL4PM (M7) — species diversity
git clone https://github.com/MartinKabierski/SpeciAL-core.git

# jBPT codebase (M3, M6) — Entropia JARs
git clone https://github.com/jbpt/codebase.git

# Entropic Relevance JARs (M3)
git clone https://github.com/promtecmx/relevance.git
# After cloning, place the required JARs (relevance.jar, OpenXES.jar) in src/relevance/

# Bootstrap Gen bridge (M6) — Python utils for JAR invocation
# Provided as part of this repo at benchmark/bridges/run_m6_bgen.py
```

**Active dependencies:**

| Directory | Upstream | Used by | Required |
|-----------|----------|---------|----------|
| `src/AVATAR/` | [Julian-Theis/AVATAR](https://github.com/Julian-Theis/AVATAR) | M5 | ✅ |
| `src/SpeciAL-core/` | [MartinKabierski/SpeciAL-core](https://github.com/MartinKabierski/SpeciAL-core) | M7 | ✅ |
| `src/codebase/` | [jbpt/codebase](https://github.com/jbpt/codebase) | M3, M6 | ✅ |
| `src/relevance/` | [promtecmx/relevance](https://github.com/promtecmx/relevance) | M3 | ✅ |

**Known modifications (tracked in git history):**
- `src/codebase/` — Patched Entropia JAR (`jbpt-pm-entropia-1.7.1.jar`) fixes a null-pointer bug in `EventLogSampling.java:101` for D2 BPI2013 Incidents at `k=2`.
- `src/AVATAR/` — Trailing underscore fix in `avatar/generalization.py:78` for D2 activity name matching.

---

## Running Experiments

### Self-contained jobs

Every method is a self-contained job. No preparation step needed — each script creates a temp workdir in `/tmp`, prepares data, runs, and cleans up:

```bash
# Default: results → /tmp/<workdir>/results/
uv run python benchmark/job_m1.py --dataset D1   # M1a–M1g (~3 min)
uv run python benchmark/job_m2.py --dataset D1   # M2 (~10 s)
uv run python benchmark/job_m3.py --dataset D1   # M3 (~1 min)
uv run python benchmark/job_m6.py --dataset D1   # M6 (~2 min)
uv run python benchmark/job_m7.py --dataset D1   # M7 (~2 min)
uv run python benchmark/job_m5.py --dataset D1   # M5 (~4 h, FULL)
uv run python benchmark/job_r1.py --dataset D1   # R1 (~5 min)
uv run python benchmark/job_r2.py --dataset D1   # R2 (~10 min)
uv run python benchmark/job_r3.py --dataset D1   # R3 (~2 min)

# Production output:
uv run python benchmark/job_m1.py --dataset D1 --output benchmark/results/configs_v2
```

All scripts accept `--dataset D1..D21` and `--output <dir>`. Each produces one JSON file per (dataset, miner, method).

### Shell wrappers (bash or SLURM)

```bash
# Bash
bash benchmark/shell/m1.sh --dataset D1

# SLURM (requires CIP-Pool krater partition)
sbatch benchmark/shell/m1.sh --dataset D1

# Full pipeline (all methods, all datasets)
bash benchmark/shell/run_all.sh

# Production run:
OUTPUT_DIR=benchmark/results/configs_v2 bash benchmark/shell/run_all.sh D1
```

### Dataset & Miner Availability

| Datasets | Available Miners |
|----------|-----------------|
| D1–D7, D14–D21 | 8/8 |
| D8–D12 | 5/8 (Alpha, Alpha+, Inductive_Strict timeout) |
| D13 | 4/8 (also no Inductive_Infrequent) |

> Canonical source: `benchmark/statistics/_miner_availability.json`. All jobs read this file and skip unavailable miners.

---

## Results

### Extraction

```bash
uv run python benchmark/extract_results.py --dataset D3     # single dataset
uv run python benchmark/extract_results.py --all             # all datasets
```

Outputs a Markdown table with MAE, Pearson, and Spearman vs. R1 (ground truth).

### Output Structure

Results directory (`benchmark/results/configs_v2/`):
- One JSON per cell: `{Dataset}__{Miner}__{Method}.json`
- Example: `Sepsis__Inductive_Strict__M1g.json`

### Visualization

Open `visualize_benchmark.ipynb`, set `DATASET_KEY`, and Run All. Saves heatmap CSVs and PNGs to `analysis/benchmark/{DATASET_NAME}/`.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|------|
| M5 training timeout | 5000 adv steps too long | Set `QUICK_MODE = True` in `benchmark/docker/run_avatar.py` |
| xvfb-run fails | Virtual framebuffer | Use `xvfb-run --auto-servernum` |
| `invalid gzip header` | Git-LFS pointers not fetched | Run `git lfs install` and `git pull` |
| JVM OutOfMemoryError | Heap too small | Set `-Xmx16G` in the shell script |
| Java method returns 0.0 | Missing JAR or classpath | Check `src/relevance/` or `src/codebase/` |
