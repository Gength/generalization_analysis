#!/bin/bash
#SBATCH --partition=Krater
#SBATCH --job-name=bench_M1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --time=08:00:00
#SBATCH --output=benchmark/logs/bench_M1_%j.log
#
# M1a–M1g  HybridGen family (v1.0 / v2.1-N3 / v2.1-N6 / v2.4 / v2.5 / v2.6-log / v2.6-mle)
# ─────────────────────────────────────────────────────────────────────────────
# CLI arguments (all optional except --dataset):
#   --dataset D1..D21          Dataset key (required)
#   --output <dir>             Output directory (default: /tmp/<workdir>/results/)
#   --methods M1a M1b ... M1g  Subset of methods (default: all 7)
#   --miners Alpha Flower ...  Subset of miners (default: all 8)
#   --workers N                Parallel worker processes (default: 8)
#
# Examples:
#   bash benchmark/shell/m1.sh --dataset D1
#   bash benchmark/shell/m1.sh --dataset D1 --methods M1f M1g
#   bash benchmark/shell/m1.sh --dataset D1 --output benchmark/results/configs_v2
#   bash benchmark/shell/m1.sh --dataset D2 --workers 4
# ─────────────────────────────────────────────────────────────────────────────

set -eo pipefail
export TMPDIR=/tmp
export PATH="$HOME/.local/bin:$PATH"
export PYTHONHASHSEED=0

# ── Miner configuration ──────────────────────────────────────────────────────
# Edit this array to subset miners for a run.
MINERS=(Trace_Filtered Alpha Alpha+ Heuristics Heuristics_Strict Inductive_Strict Inductive_Infrequent Flower)

# Parse --miners from CLI args; everything else passes through
PASSTHRU=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --miners) shift; MINERS=()
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do MINERS+=("$1"); shift; done ;;
        *) PASSTHRU+=("$1"); shift ;;
    esac
done

uv run python benchmark/job_m1.py --miners "${MINERS[@]}" "${PASSTHRU[@]}"
