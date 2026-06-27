#!/bin/bash
# Full pipeline — submits one SLURM job per method (excl. M5 GPU)
# ─────────────────────────────────────────────────────────────────────────────
# Usage:
#   bash benchmark/shell/run_all.sh          # D1 (default)
#   bash benchmark/shell/run_all.sh D2       # D2
#
# Each sub-script has its own #SBATCH header; they are submitted as
# independent SLURM jobs. All results land in benchmark/results/configs_v2/.
# To include M5 (AVATAR GPU), uncomment the line below.
# ─────────────────────────────────────────────────────────────────────────────

set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

DATASET="${1:-D1}"
OUTPUT="benchmark/results/configs_v2"

echo "=========================================="
echo "Submitting all benchmark jobs — $DATASET"
echo "=========================================="
echo "Started: $(date)"

sbatch benchmark/shell/m1.sh --dataset "$DATASET" --output "$OUTPUT"
sbatch benchmark/shell/m2.sh --dataset "$DATASET" --output "$OUTPUT"
sbatch benchmark/shell/m3.sh --dataset "$DATASET" --output "$OUTPUT"
sbatch benchmark/shell/m6.sh --dataset "$DATASET" --output "$OUTPUT"
sbatch benchmark/shell/m7.sh --dataset "$DATASET" --output "$OUTPUT"
# sbatch benchmark/shell/m5.sh --dataset "$DATASET" --output "$OUTPUT"  # requires Docker GPU
sbatch benchmark/shell/r1.sh --dataset "$DATASET" --output "$OUTPUT"
bash benchmark/shell/r2.sh --dataset "$DATASET" --output "$OUTPUT"
sbatch benchmark/shell/r3.sh --dataset "$DATASET" --output "$OUTPUT"

echo ""
echo "All submitted: $(date)"
echo "Check: squeue -u $USER"
