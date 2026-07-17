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
#
# Edit the MINERS array below to change which miners are benchmarked.
# ─────────────────────────────────────────────────────────────────────────────

set -eo pipefail
export PYTHONHASHSEED=0
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

DATASET="${1:-D1}"
OUTPUT="benchmark/results/configs_v2"

# ── Miner configuration — edit this to control the default set ───────────────
MINERS=(Trace_Filtered Alpha Alpha+ Heuristics Heuristics_Strict Inductive_Strict Inductive_Infrequent Flower)
# MINERS=(Trace_Filtered Heuristics Heuristics_Strict Inductive_Infrequent Flower)

echo "=========================================="
echo "Submitting all benchmark jobs — $DATASET"
echo "=========================================="
echo "Started: $(date)"

bash benchmark/shell/m1.sh --dataset "$DATASET" --output "$OUTPUT" --miners "${MINERS[@]}"
bash benchmark/shell/m2.sh --dataset "$DATASET" --output "$OUTPUT" --miners "${MINERS[@]}"
bash benchmark/shell/m3.sh --dataset "$DATASET" --output "$OUTPUT" --miners "${MINERS[@]}"
bash benchmark/shell/m6.sh --dataset "$DATASET" --output "$OUTPUT" --miners "${MINERS[@]}"
bash benchmark/shell/m6_adapted.sh --dataset "$DATASET" --output "$OUTPUT" --miners "${MINERS[@]}"
bash benchmark/shell/m7.sh --dataset "$DATASET" --output "$OUTPUT" --miners "${MINERS[@]}"
bash benchmark/shell/r1.sh --dataset "$DATASET" --output "$OUTPUT" --miners "${MINERS[@]}"
bash benchmark/shell/r2.sh --dataset "$DATASET" --output "$OUTPUT" --miners "${MINERS[@]}"
bash benchmark/shell/r3.sh --dataset "$DATASET" --output "$OUTPUT" --miners "${MINERS[@]}"
bash benchmark/shell/m5.sh --dataset "$DATASET" --output "$OUTPUT" --miners "${MINERS[@]}"  # requires Docker GPU

echo ""
echo "All submitted: $(date)"
echo "Check: squeue -u $USER"
