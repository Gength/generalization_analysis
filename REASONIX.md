# REASONIX.md — generalization-analysis

## Stack
- **Python 3.12** (`.python-version`)
- **uv** package manager (`pyproject.toml`, `uv.lock`)
- **pm4py** — process mining (miners, token replay, conformance) [pyproject.toml:52]
- **numpy, pandas, scipy, scikit-learn** — numerical / ML
- **matplotlib, seaborn** — plotting
- **networkx** — graph analysis
- **Jupyter** — notebook-driven analysis (`.ipynb` files at root)

## Layout
- `README.md` — project overview
- `Method2Log.md` — detailed log of Method 2 development, challenges, and insights (to be expanded), coauthored by both teammates
- `Method_GenShadow.md` — authoritative construct definition: generalization = acceptance of future *valid* behavior, strictly separated from precision
- `BenchmarkDesign.md` — **Coauthored**, methodological framework: defines benchmark design philosophy, metric definitions, method taxonomy, dataset selection rationale, and evaluation criteria.
- `BenchmarkGuide.md` — benchmark operations guide: step-by-step instructions (how to run benchmarks, command reference), result format specification, and output file interpretation.
- `visualize_benchmark.ipynb` — primary benchmark visualization notebook: MAE-to-R1 heatmaps per method×miner, saves CSVs and PNGs to `analysis/benchmark/{DATASET_NAME}/`
- `visualize_gen_shadow.ipynb` — GenShadow construct visualization notebook
- `report/` — LaTeX report source and compiled PDF (`main.tex`, `main.pdf`, `references.bib`)
- `analysis/` — notebooks and scripts for analyzing algorithms
  - `Mutation/MutationReport.md` + `analyze_mutation.py` — mutation analysis
  - `Structure/StructMetricAnalysis.md` + `analyze_struct_metrics.py` — structural metrics analysis
  - `benchmark/` — benchmark visualization outputs (CSVs + PNGs, organized by dataset name)
- `HybridGen/` — Method 2 package: algorithm versions (`algorithm/v1.py`…`v26.py`) + experiment runners (`experiment/v1.py`, `v2.py`), registry-based loading
- `data/` — XES event logs (BPI Challenge 2011–2020, Hospital Billing, Sepsis, Road Traffic)
- `output/` — HybridGen experiment JSON results and extensive evaluation outputs
- `archive/` — legacy scripts and notebooks from development, kept for historical reference; not actively maintained or updated
  - `Chris/` — teammate's archived work (WhatChanged_v25_v26.md, BenchmarkDesign_v2.md, leaderboard data, etc.)
  - `Tianhao/` — your archived work (ExperimentDesign.md, v1 scripts, presentation notebook, etc.)
  - `src_deps/` — external dependency backups
- `benchmark/` — scripts and runners for executing and analyzing benchmarks
  - `01_prepare_models.py` + `prepare.sh` — model discovery (PNML + DFG JSON export for all miners)
  - `02_gen_per_miner_dfgs.py` — per-miner DFG generation
  - `miners.py` — definitions of all miners used in benchmarks
  - `datasets.py` — canonical D1–D21 dataset definitions (single source of truth; all scripts import from here)
  - `utils.py` — shared benchmark utilities
  - `run_m1_family.py` — **teammate**, M1-family runner (Methodology v2): one-command M1a–M1g for a dataset, writes to `results/configs_v2/`
  - `run_m2.py` — M2 runner (PM4Py built-in generalization)
  - `run_r_family.py` — R-family runner (R1/R2/R3 reference methods)
  - `version_comparison.py` — **teammate**, multi-seed cross-version comparison: v2.4 vs v2.5 vs v2.6 vs v2.6-mle
  - `r1_accept.py` — **teammate**, R1 acceptance rate computation
  - `version_comparison_analysis.ipynb` — **teammate**, notebook for analyzing `version_comparison_D*.csv`
  - `bridges/` — bridge scripts for external methods (M3, M6, M7, AVATAR)
    - `avatar_bridge.py`, `run_m3.py`, `run_m6_bgen.py`, `run_m7.py`
  - `docker/` — Docker infrastructure for AVATAR (RelGAN) method M5
    - `Dockerfile.avatar`, `Dockerfile.avatar.tf2`, `run_avatar.py`
  - `models/` — pre-discovered PNML models, DFG JSON, manifest, and dataset XES cache
  - `results/`
    - `configs/` — benchmark config JSONs (Methodology v1)
    - `configs_v2/` — benchmark config JSONs (Methodology v2, M1a–M1g + M2–M7 + R1–R3)
    - `version_comparison_D1.csv`, `version_comparison_D2.csv` — version comparison results
  - Shell runners: `m1.sh`, `m2.sh`, `m3.sh`, `m5.sh`, `m6.sh`, `m7.sh`, `reference.sh`, `run_all.sh`
- `src/` — external code for benchmarking, not part of project
- `artifacts/` — conceptual diagrams, presentation slides, and other non-code deliverables
  - `katz-mutation.drawio` — flowchart of Katz proposal for mutation-based generalization estimation (v2.5)

## Commands
```bash
# Install dependencies (auto-creates venv)
uv sync

# --- HybridGen (Method 2) ---
# List registered HybridGen algorithms/experiments
uv run python -m HybridGen --list

# Run a specific HybridGen version
uv run python -m HybridGen -a v2.6 -e v2 --miner all --weight 0.5

# --- Benchmark: Model Discovery ---
# Export PNML + DFG JSON for all miners (run once per dataset)
uv run python benchmark/01_prepare_models.py

# --- Benchmark: M1 family (Methodology v2) ---
# Full M1a–M1g family on a dataset (writes to results/configs_v2/)
uv run python benchmark/run_m1_family.py --dataset D1
uv run python benchmark/run_m1_family.py --dataset D2
uv run python benchmark/run_m1_family.py --dataset D1 --methods M1e M1f M1g

# --- Benchmark: Version comparison (multi-seed) ---
uv run python benchmark/version_comparison.py --dataset D1 --seeds 42 1 7 99
uv run python benchmark/version_comparison.py --dataset D2 --seeds 42

# --- Benchmark: Full pipeline (sequential) ---
bash benchmark/run_all.sh
```

## Conventions
- **`uv run python`** prefix for python script invocations in this project except from `src/`
- **Registry pattern** in `HybridGen/`: modules auto-register via decorators; `import_modules()` in `HybridGen/utils.py` discovers them at import time
- **Versioned algorithm files**: `v1.py`…`v26.py` under `HybridGen/algorithm/` — newer versions (v2.3+) export `calculate_gen_shadow` only (Gen_struct deprecated); v22 and earlier export `calculate_gen_shadow_*` and `calculate_gen_struct`
- **Experiment runners** accept `args=None` (parse CLI) or a pre-built `Namespace` (API use)
- **Dataset registry** in `benchmark/datasets.py` — the canonical definition of dataset names and paths. All benchmark scripts import from here; never define inline `DATASETS` dicts.
- files labeled "teammate" are primarily authored by other teammate, should not be edited without permission, reference only; files labeled "coauthored" are jointly authored and can be edited by either teammate with proper care and communication.

## Role Boundaries — Planner vs Executor

**Planner (read‑only):**
- Plans, surveys code, reads `.md`/`.py`/`.json`/`.sh`/`.drawio` files via standard tools
- **MUST NOT read, view, or interact with `.ipynb` files through ANY channel** — not via `read_file`, not via `web_fetch`, not via `glob` on `.ipynb` paths (except to confirm file existence/paths for handover). Notebooks contain base64‑encoded images that cause truncation in non‑MCP tools and are inherently unreadable outside the Jupyter MCP.
- When notebook context is needed, describe WHAT information is needed from WHICH notebook; the executor will report back a summary

**Executor (read‑write):**
- The executor owns all notebook interaction. Notebooks MUST be accessed exclusively through Jupyter MCP tools:

  - `mcp__jupyter__use_notebook` → connect and activate a notebook
  - `mcp__jupyter__read_notebook` (brief mode + pagination) → survey cells
  - `mcp__jupyter__insert_cell` / `mcp__jupyter__edit_cell_source` / `mcp__jupyter__overwrite_cell_source` → edit
  - `mcp__jupyter__execute_cell` / `mcp__jupyter__insert_execute_code_cell` → run code
  - `mcp__jupyter__read_cell` → inspect outputs
  - `mcp__jupyter__execute_code` → temporary debug (not saved to notebook)

- There is NO fallback. If the Jupyter MCP server is not connected or tools fail, report the problem and wait for the user to resolve it.

## Watch out for
- **Data files are .xes.gz** — PM4Py reads them directly; don't unzip
- **`archive/` is historical** — the current Method 1 entry points are at repo root

## Notes
- Read `BenchmarkDesign.md` and `BenchmarkGuide.md` for detailed instructions on running benchmarks, handling data, and reporting results. The benchmark has strict requirements to ensure data integrity and experiment validity.
- After completing the experiments, you should output the raw measurement data in a structured format. Do not output redundant debug (Debug) information to the console. Before deleting any folders, you must list all files and subfolders within the current folder and indicate the reason and basis for the deletion operation in the output. The deletion operation must go through a human review and approval process to ensure data security and compliance.
- All project documentation should be written in English.
- When starting the m5 docker container, you need to check if the m5 container is clean. Do not have two or more m5 containers running simultaneously.
- When starting the experiment m1-r3 independently, you need to call `benchmark/01_prepare_models.py` to ensure that the model is consistent with the dataset used in the current experiment.
- The waiting time for running experiments should increase exponentially, for example, 5 minutes followed by 10 minutes, and multiple experiments cannot be run at the same time. Before each experiment startup, you need to check the number of currently running experiments to ensure it does not exceed one.