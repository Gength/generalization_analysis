#!/bin/bash
#SBATCH --partition=Krater
#SBATCH --job-name=bench_M5
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=benchmark/logs/bench_M5_%j.log
#
# M5 — AVATAR RelGAN via Docker (TF 1.15 GPU)
# ─────────────────────────────────────────────────────────────────────────────
# Prerequisites:
#   Docker image built: docker build -t avatar-tf1 -f benchmark/docker/Dockerfile.avatar .
#   GPU + nvidia-container-toolkit required
#
# CLI arguments (all optional except --dataset):
#   --dataset D1..D21          Dataset key (required)
#   --output <dir>             Output directory (default: /tmp/<workdir>/results/)
#   --miners Alpha Flower ...  Subset of miners (default: all 8)
#   --eval-only                Skip training/sampling; reuse existing checkpoint
#   AVATAR_BATCH_SIZE=64        GAN batch size for training (default: 16; override via env)
#
# Examples:
#   bash benchmark/shell/m5.sh --dataset D1
#   bash benchmark/shell/m5.sh --dataset D1 --eval-only           # reuse checkpoint
#   bash benchmark/shell/m5.sh --dataset D1 --output benchmark/results/configs_v2
#   AVATAR_BATCH_SIZE=32 bash benchmark/shell/m5.sh --dataset D1  # reduced memory
# ─────────────────────────────────────────────────────────────────────────────

set -eo pipefail
export TMPDIR=/tmp
export PATH="$HOME/.local/bin:$PATH"
export PYTHONHASHSEED=0
# To override batch size, set AVATAR_BATCH_SIZE env var (default: 16, matching 16a1a1e pipeline).
# Example: AVATAR_BATCH_SIZE=64 bash benchmark/shell/m5.sh --dataset D1

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

uv run python benchmark/docker/run_avatar.py --miners "${MINERS[@]}" "${PASSTHRU[@]}"
