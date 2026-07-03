#!/bin/bash
#SBATCH --partition=Krater
#SBATCH --job-name=bench_M4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=benchmark/logs/bench_M4_%j.log
#
# M4  Anti-Alignment Generalization (ProcessM TwoPhaseDFS)
# ─────────────────────────────────────────────────────────────────────────────
# CLI arguments:
#   --dataset D1..D21          Dataset key (required)
#   --miners [Alpha Flower]   Optional: space-separated miner names (default: all 8)
#   --output <dir>             Output directory (default: /tmp/<workdir>/results/)
#
# Examples:
#   bash benchmark/shell/m4.sh --dataset D1
#   bash benchmark/shell/m4.sh --dataset D1 --miners Alpha Flower
#   bash benchmark/shell/m4.sh --dataset D1 --output benchmark/results/configs_v2
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

uv run python benchmark/job_m4.py --miners "${MINERS[@]}" "${PASSTHRU[@]}"
