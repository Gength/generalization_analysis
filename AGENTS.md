# AGENTS.md — generalization-analysis
---
## Role Boundaries — Planner vs Executor

**Planner (read‑only):**
- Plans, surveys code, reads `.md`/`.py`/`.json`/`.sh`/`.drawio` files via standard tools
- **May read `.ipynb` files ONLY through Jupyter MCP read tools listed in `reasonix.toml`'s `planner_allowed_tools`** (currently `mcp__jupyter__use_notebook`, `mcp__jupyter__read_notebook`, `mcp__jupyter__read_cell`, `mcp__jupyter__list_notebooks`, `mcp__jupyter__list_files`, `mcp__jupyter__list_kernels`, `mcp__jupyter__restart_notebook`). **MUST NOT** read notebooks via `read_file`, `web_fetch`, or any non-MCP tool. `glob` on `.ipynb` paths is allowed only to confirm file existence/handover. This restriction exists because notebooks contain base64‑encoded images that cause truncation in non‑MCP tools.
- When notebook context is needed, describe WHAT information is needed from WHICH notebook in the plan; the executor will read the notebook via Jupyter MCP and report findings in its final summary message. **Optionally**, if the required information is purely read-only (surveying cell structure, reading source with brief mode), the planner may use the allowed Jupyter MCP tools directly; the executor remains responsible for executing code and reporting runtime state in its summary message.

**Executor (read‑write):**
- The executor owns all notebook interaction. Notebooks MUST be accessed exclusively through Jupyter MCP tools:

  - `mcp__jupyter__use_notebook` → connect and activate a notebook
  - `mcp__jupyter__read_notebook` (brief mode + pagination) → survey cells
  - `mcp__jupyter__insert_cell` / `mcp__jupyter__edit_cell_source` / `mcp__jupyter__overwrite_cell_source` → edit
  - `mcp__jupyter__execute_cell` / `mcp__jupyter__insert_execute_code_cell` → run code
  - `mcp__jupyter__read_cell` → inspect outputs
  - `mcp__jupyter__execute_code` → temporary debug (not saved to notebook)

- There is NO fallback. If the Jupyter MCP server is not connected or tools fail, report the problem and wait for the user to resolve it.
---

## Stack
- **Python 3.12** (`.python-version`)
- **uv** package manager (`pyproject.toml`, `uv.lock`)
- **pm4py** — process mining (miners, token replay, conformance) [pyproject.toml:52]
- **Jupyter** — notebook-driven analysis

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
- `benchmark/` — benchmark scripts, jobs, and results
  See [benchmark/README.md](benchmark/README.md) — full layout, job model, and result structure.
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

See [benchmark/README.md](benchmark/README.md) — full benchmark command reference.
```

## Conventions
- **`uv run python`** prefix for python script invocations in this project except from `src/`
- **Registry pattern** in `HybridGen/`: modules auto-register via decorators; `import_modules()` in `HybridGen/utils.py` discovers them at import time
- **Versioned algorithm files**: `v1.py`…`v26.py` under `HybridGen/algorithm/` — newer versions (v2.3+) export `calculate_gen_shadow` only (Gen_struct deprecated); v22 and earlier export `calculate_gen_shadow_*` and `calculate_gen_struct`
- **Experiment runners** accept `args=None` (parse CLI) or a pre-built `Namespace` (API use)
- **Dataset registry** in `benchmark/datasets.py` — the canonical definition of dataset names and paths. All benchmark scripts import from here; never define inline `DATASETS` dicts.
- files labeled "teammate" are primarily authored by other teammate, should not be edited without permission, reference only; files labeled "coauthored" are jointly authored and can be edited by either teammate with proper care and communication.
- Read `BenchmarkDesign.md` and `BenchmarkGuide.md` for detailed instructions on running benchmarks, handling data, and reporting results. The benchmark has strict requirements to ensure data integrity and experiment validity.
- **⚠️ Before running any script under `benchmark/`, you MUST read [`benchmark/README.md`](benchmark/README.md).** It defines the output policy (`benchmark/results/configs_v2/` is the mandatory staging area), the job model, dataset quick-index, JSON key mapping per method, extraction commands, gotchas, and experiment safety rules. Skipping this document will cause misrouted output and data inconsistency.
- All project documentation should be written in English.

## Benchmark quick-index

Docs: [`BenchmarkDesign.md`](BenchmarkDesign.md) (methodology) · [`BenchmarkGuide.md`](BenchmarkGuide.md) (operations, results).

See [benchmark/README.md](benchmark/README.md) — full dataset table, JSON key mapping, extraction commands, and gotchas.

### Extraction

`benchmark/extract_results.py` — `--dataset D3` or `--all`, outputs Markdown table.

## Watch out for
- **Data files are .xes.gz** — PM4Py reads them directly; don't unzip
- **`archive/` is historical** — the current Method 1 entry points are at repo root

## Notes

- After completing the experiments, you should output the raw measurement data in a structured format. Do not output redundant debug (Debug) information to the console. Before deleting any folders, you must list all files and subfolders within the current folder and indicate the reason and basis for the deletion operation in the output. The deletion operation must go through a human review and approval process to ensure data security and compliance.
- See [benchmark/README.md](benchmark/README.md) for experiment safety guidelines (M5 Docker, waiting times, concurrent runs).