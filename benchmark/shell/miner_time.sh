#!/bin/bash
#SBATCH --partition=Krater
#SBATCH --job-name=miner_time
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=benchmark/logs/miner_time_%j.log
#
# stat_timings — model-discovery timing benchmark
# ─────────────────────────────────────────────────────────────────────────────
# CLI:
#   --dataset D1..D21    Dataset key (required)
#   --all                Run on all datasets (skips D8–D13)
#   --workers N          Parallel workers (default: 8 on HPC)
#   --timeout N          Per-miner timeout seconds (default: 3600)
#   --summary            Print summary table from existing results
#
# Examples:
#   sbatch benchmark/shell/miner_time.sh --dataset D1
#   sbatch benchmark/shell/miner_time.sh --dataset D2 --workers 4
#   sbatch benchmark/shell/miner_time.sh --dataset D3 --workers 8 --timeout 7200
#   sbatch benchmark/shell/miner_time.sh --all --workers 8
# ─────────────────────────────────────────────────────────────────────────────

set -eo pipefail
export TMPDIR=/tmp
export PATH="$HOME/.local/bin:$PATH"
export PYTHONHASHSEED=0

echo "[miner_time] started: $(date)"
echo "[miner_time] args: $*"

uv run python benchmark/stat_timings.py "$@"

echo "[miner_time] finished: $(date)"
