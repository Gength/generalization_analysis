#!/bin/bash
#SBATCH --partition=Krater
#SBATCH --job-name=bench_M2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=benchmark/logs/bench_M2_%j.log
#
# M2  PM4Py Built-in Generalization (token-replay-based)
# ─────────────────────────────────────────────────────────────────────────────
# CLI arguments:
#   --dataset D1..D21          Dataset key (required)
#   --output <dir>             Output directory (default: /tmp/<workdir>/results/)
#   --miners Alpha Flower ...  Subset of miners (default: all 8)
#
# Examples:
#   bash benchmark/shell/m2.sh --dataset D1
#   bash benchmark/shell/m2.sh --dataset D1 --output benchmark/results/configs_v2
#   bash benchmark/shell/m2.sh --dataset D1 --miners Alpha Flower
# ─────────────────────────────────────────────────────────────────────────────

set -eo pipefail
export TMPDIR=/tmp
export PATH="$HOME/.local/bin:$PATH"

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

uv run python benchmark/job_m2.py --miners "${MINERS[@]}" "${PASSTHRU[@]}"
