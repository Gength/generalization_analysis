#!/bin/bash
#SBATCH --partition=Krater
#SBATCH --job-name=bench_M6A
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=benchmark/logs/bench_M6A_%j.log
#
# M6 adapted  Bootstrap Generalization (bsgen breeding + PM4Py token replay)
# ─────────────────────────────────────────────────────────────────────────────
# The construct-faithful M6 of the report (NOT the Entropia -bgen F-measure;
# that one is benchmark/shell/m6.sh). Writes to benchmark/results/configs/
# by default, next to the D1 adapted-M6 configs. Do NOT redirect --output to
# configs_v2/ (the -bgen results live there under the same filenames).
#
# Requires src/bsgen/bsgen_eval.py on the runner (gitignored vendor code).
#
# CLI arguments:
#   --dataset D1..D21          Dataset key (required)
#   --output <dir>             Output directory (default: benchmark/results/configs)
#
# Examples:
#   sbatch benchmark/shell/m6_adapted.sh --dataset D3
#   sbatch benchmark/shell/m6_adapted.sh --dataset D5
#   sbatch benchmark/shell/m6_adapted.sh --dataset D4   # expect long replay times
# ─────────────────────────────────────────────────────────────────────────────

set -eo pipefail
export TMPDIR=/tmp
export PATH="$HOME/.local/bin:$PATH"
export PYTHONHASHSEED=0

# ── Miner configuration ──────────────────────────────────────────────────────
# Edit this array to subset miners for a run.
MINERS=(Trace_Filtered Alpha Alpha+ Heuristics Heuristics_Strict Inductive_Strict Inductive_Infrequent Flower)

uv run python benchmark/job_m6_adapted.py --miners "${MINERS[@]}" --workers 8 "$@"
