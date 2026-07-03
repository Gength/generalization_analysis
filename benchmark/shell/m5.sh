#!/bin/bash
#SBATCH --partition=Krater
#SBATCH --job-name=bench_M5
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=benchmark/logs/bench_M5_%j.log
#
# M5  AVATAR RelGAN — Docker GPU required (TF 1.15)
# ─────────────────────────────────────────────────────────────────────────────
# CLI arguments (all optional except --dataset):
#   --dataset D1..D21          Dataset key (required)
#   --output <dir>             Output directory (default: /tmp/<workdir>/results/)
#   --miners Alpha Flower ...  Subset of miners (default: all 8)
#   --quick                    3 pre-epochs + 100 adv steps (fast demo ~30 min)
#   --eval-only                Skip training/sampling; reuse existing checkpoint
#   --tf2                      Use TF2 container (avatar-tf2) instead of TF1
#
# Examples:
#   bash benchmark/shell/m5.sh --dataset D1
#   bash benchmark/shell/m5.sh --dataset D1 --quick              # fast demo
#   bash benchmark/shell/m5.sh --dataset D1 --eval-only           # reuse checkpoint
#   bash benchmark/shell/m5.sh --dataset D1 --output benchmark/results/configs_v2
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

uv run python benchmark/job_m5.py --miners "${MINERS[@]}" "${PASSTHRU[@]}"
