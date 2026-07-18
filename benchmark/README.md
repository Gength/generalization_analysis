# Benchmark — Layout & Quick Reference

> See [`BenchmarkDesign.md`](../BenchmarkDesign.md) for methodology.
> See [`BenchmarkGuide.md`](../BenchmarkGuide.md) for operations and result formats.

---

## Layout

### Job scripts (directly callable)

Each script creates a unique `/tmp/benchmark_{METHOD}_{DS}_{TIMESTAMP}_{RAND}/` workdir,
prepares its own data, runs, writes results, and cleans up. `--output <dir>` overrides
the output destination.

- `run_m1_family.py` — M1 family (M1a–M1g): calculates `mean` in results. CLI: `uv run python benchmark/run_m1_family.py --dataset D1`
- `run_m2.py` — M2 (PM4Py Built-in Gen): calculates `score` in results. CLI: `uv run python benchmark/run_m2.py --dataset D1`
- `run_r_family.py` — R-family (R1/R2/R3): calculates `mean` in results. CLI: `uv run python benchmark/run_r_family.py --method R1 --dataset D1`
- `bridges/run_m3.py` — M3 entropic relevance, CLI: `uv run python benchmark/bridges/run_m3.py --dataset D1`
- `bridges/run_m4.py` — M4 anti-alignment, CLI: `uv run python benchmark/bridges/run_m4.py --dataset D1`
- `bridges/run_m6_adapted.py` — M6 adapted variant (bsgen + token replay), CLI: `uv run python benchmark/bridges/run_m6_adapted.py --dataset D1`
- `bridges/run_m6_bgen.py` — M6 bootstrap gen (Entropia -bgen), CLI: `uv run python benchmark/bridges/run_m6_bgen.py --dataset D1`
- `bridges/run_m7.py` — M7 SpeciAL4PM, CLI: `uv run python benchmark/bridges/run_m7.py --dataset D1`
- `docker/run_avatar.py` — M5 AVATAR via Docker, CLI: `uv run python benchmark/docker/run_avatar.py --dataset D1`
- `job_prepare.py` — `prepare_workdir(workdir, dataset_key, mode)` with 4 modes:
  `minimal`, `log_dfg`, `pnml`, `per_miner_dfgs`

### Bridges (external tool integrations)

- `bridges/run_m3.py` — M3 entropic relevance. CLI: `uv run python benchmark/bridges/run_m3.py --dataset D1`
- `bridges/run_m4.py` — M4 anti-alignment. CLI: `uv run python benchmark/bridges/run_m4.py --dataset D1`
- `bridges/run_m6_bgen.py` — M6 bootstrap gen (Entropia -bgen). CLI: `uv run python benchmark/bridges/run_m6_bgen.py --dataset D1`
- `bridges/run_m6_adapted.py` — M6 adapted (bsgen + token replay). CLI: `uv run python benchmark/bridges/run_m6_adapted.py --dataset D1`
- `bridges/run_m7.py` — M7 SpeciAL4PM. CLI: `uv run python benchmark/bridges/run_m7.py --dataset D1`
- `bridges/m4_export.py` — M4 model export helper
- `docker/run_avatar.py` — M5 AVATAR via Docker. CLI: `uv run python benchmark/docker/run_avatar.py --dataset D1`
- `bridges/avatar_bridge.py` — AVATAR (M5) native bridge (deprecated; use docker/run_avatar.py)
- `bridges/bsgen_eval.py` — BSGen evaluation utilities

### Shared infrastructure

- `datasets.py` — **canonical** D1–D21 dataset definitions (single source of truth)
- `miners.py` — miner definitions used across all methods
- `utils.py` — shared benchmark utilities

### Results & extraction

- `extract_results.py` — universal result extractor: `--dataset D3` or `--all`, outputs Markdown table
- `extract_runtime.py` — runtime extraction utility
- `results/`
  - `configs/` — **formal archive** of finalized benchmark results. Once results are
    validated and accepted, they are moved here. Do not write directly to this folder
    from job scripts.
  - `configs_v2/` — **temporary staging area** for benchmark experiment output.
    Each run produces slightly different results; all method output goes here first
    (`{Dataset}__{Miner}__{Method}.json`). After validation, results are promoted to
    `configs/`. All shell scripts in `shell/` default `--output` to this directory.
  - `version_comparison_D1.csv`, `version_comparison_D2.csv` — multi-seed version comparison results
  - `alignment_spotcheck.json`, `generator_validation.json` — spot-check / validation outputs

### Models (cached)

- `models/` — pre-discovered PNML models on datasets to accelerate metrics. 

### Shell runners

- `shell/m1.sh` … `shell/m7.sh` — SLURM-ready shell wrappers (`#SBATCH` headers)
- `shell/m6_adapted.sh` — M6 adapted
- `shell/r1.sh`, `shell/r2.sh`, `shell/r3.sh` — R-family
- `shell/miner_time.sh` — miner timing benchmark
- `shell/run_all.sh` — full pipeline (sequential)

### Statistics

- `statistics/` — per-dataset JSON statistics (case count, variant count, activity count, etc.)
- `statistics/_miner_availability.json` — miner availability matrix across datasets

### Analysis & utilities

- `version_comparison.py` — **teammate**, multi-seed cross-version comparison: v2.4 vs v2.5 vs v2.6 vs v2.6-mle
- `version_comparison_analysis.ipynb` — **teammate**, notebook for analyzing `version_comparison_D*.csv`
- `r1_accept.py` — **teammate**, R1 acceptance rate computation
- `make_figures.py` — figure generation for paper/report
- `stat_timings.py` — timing statistics aggregation
- `subsample_scaling.py` — subsample scaling analysis
- `compare_runtimes.py` — cross-method runtime comparison
- `alignment_spotcheck.py` — alignment quality spot-check
- `audit_configs.py` — config completeness audit
- `generator_validation.py` — generator output validation
- `RUNBOOK_d3d5_fixes.md` — runbook for D3/D5-specific fixes and workarounds

---

## Commands

```bash
# --- Run a method on a dataset (direct CLI) ---
uv run python benchmark/run_m1_family.py --dataset D1
uv run python benchmark/run_m2.py --dataset D1
uv run python benchmark/bridges/run_m3.py --dataset D1
uv run python benchmark/bridges/run_m6_bgen.py --dataset D1
uv run python benchmark/bridges/run_m7.py --dataset D1
uv run python benchmark/run_r_family.py --method R1 --dataset D1

# Production run (writes to configs_v2/)
uv run python benchmark/run_m2.py --dataset D1 --output benchmark/results/configs_v2

# --- Full pipeline (sequential) ---
bash benchmark/shell/run_all.sh
OUTPUT_DIR=benchmark/results/configs_v2 bash benchmark/shell/run_all.sh D1  # production

# --- Version comparison (multi-seed) ---
uv run python benchmark/version_comparison.py --dataset D1 --seeds 42 1 7 99
uv run python benchmark/version_comparison.py --dataset D2 --seeds 42
```

---

## Dataset quick-index

### Dataset key → name → status

| Key | Configs_v2 prefix | Status |
|-----|-------------------|--------|
| D1 | Sepsis | ✅ |
| D2 | BPI2013_Incidents | ✅ |
| D3 | BPI2017 | ⚠️ all except M5 |
| D4 | BPI2018 | ⚠️ all except M5 |
| D5 | BPI2019 | ⚠️ all except M5 |
| D6 | BPI2013_Problem_Open | ⚠️ partial |
| D7 | BPI2013_Problem_Closed | ⚠️ partial |
| D8 | BPI2015_Municipality_2 | — (5/8 miners) |
| D9 | BPI2015_Municipality_4 | — (5/8 miners) |
| D10 | BPI2015_Municipality_1 | — (5/8 miners) |
| D11 | BPI2011_Hospital | — (5/8 miners) |
| D12 | BPI2015_Municipality_5 | — (5/8 miners) |
| D13 | BPI2015_Municipality_3 | — (4/8 miners) |
| D14 | BPI2020_PrepaidTravel | — |
| D15 | BPI2020_InternationalDecl | — |
| D16 | BPI2020_RequestForPayment | — |
| D17 | BPI2020_PermitLog | — |
| D18 | BPI2020_DomesticDecl | — |
| D19 | BPI2012 | — |
| D20 | Hospital_Billing | — |
| D21 | Road_Traffic_Fine | — |

### JSON value key per method (configs_v2)

| Methods | Key in `results` |
|---------|-----------------|
| M1a–M1g, R1, R2, R3 | `mean` |
| M2 | `score` |
| M3 | `entropic_relevance_raw` (per-miner DFG-based) |
| M5 | `mean` (present for D1/D2, missing for D3) |
| M6, M7 | `gen_score` |

### Extraction

`benchmark/extract_results.py` — `--dataset D3` or `--all`, outputs Markdown table, missing → `-`.

---

## Gotchas

- **File prefix = dataset `name`, not key**: `BPI2017` not `D3`.
- **`benchmark/models/` is legacy** — self-contained jobs prepare models in `/tmp`.
- **M6 JAR**: must use patched `jbpt-pm-entropia-1.7.1.jar` (see BenchmarkGuide §1 M6 note).

---

## Experiment safety

- After completing experiments, output raw measurement data in a structured format. Do not output redundant debug information to the console. Before deleting any folders, list all files and subfolders within the current folder and indicate the reason and basis for the deletion operation in the output. Deletion operations must go through a human review and approval process.
- The waiting time for running experiments should increase exponentially (e.g., 5 minutes followed by 10 minutes). Do not run multiple experiments at the same time. Before each experiment startup, check the number of currently running experiments to ensure it does not exceed one.
