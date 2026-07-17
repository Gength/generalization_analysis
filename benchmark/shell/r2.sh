#!/bin/bash
# R2 launcher — submits one SLURM job per miner (8 miners → 8 nodes).
# Each miner job uses variant-level parallelism internally (ProcessPoolExecutor).
#
# Usage:
#   bash benchmark/shell/r2.sh --dataset D1
#   bash benchmark/shell/r2.sh --dataset D1 --output benchmark/results/configs_v2
#   bash benchmark/shell/r2.sh --dataset D1 --r2-sample 100
#   bash benchmark/shell/r2.sh --dataset D1 --miners Alpha Flower
# ─────────────────────────────────────────────────────────────────────────────

set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# ── Parse arguments ──────────────────────────────────────────────────────────
DATASET=""; OUTPUT=""; SAMPLE=""

# Miner configuration — override via --miners
MINERS=(Trace_Filtered Alpha Alpha+ Heuristics Heuristics_Strict Inductive_Strict Inductive_Infrequent Flower)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)   DATASET="$2"; shift 2 ;;
        --dataset=*) DATASET="${1#*=}"; shift ;;
        --output)    OUTPUT="$2"; shift 2 ;;
        --output=*)  OUTPUT="${1#*=}"; shift ;;
        --r2-sample) SAMPLE="$2"; shift 2 ;;
        --r2-sample=*) SAMPLE="${1#*=}"; shift ;;
        --miners)    shift; MINERS=(); while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do MINERS+=("$1"); shift; done ;;
        --miners=*)  IFS=' ' read -ra MINERS <<< "${1#*=}"; shift ;;
        *) shift ;;
    esac
done

if [[ -z "$DATASET" ]]; then
    echo "ERROR: --dataset is required" >&2
    exit 1
fi

OUTPUT_ARG=""; [[ -n "$OUTPUT" ]] && OUTPUT_ARG="--output $OUTPUT"
SAMPLE_ARG=""; [[ -n "$SAMPLE" ]] && SAMPLE_ARG="--r2-sample $SAMPLE"

echo "Submitting ${#MINERS[@]} R2 jobs for dataset $DATASET …"

for miner in "${MINERS[@]}"; do
    sbatch \
        --job-name="R2_${miner}" \
        --partition=Krater \
        --nodes=1 --ntasks=1 --cpus-per-task=14 \
        --time=08:00:00 \
        --output="benchmark/logs/bench_R2_${miner}_%j.log" \
        --wrap="export TMPDIR=/tmp PYTHONHASHSEED=0 PATH=\"\$HOME/.local/bin:\$PATH\"; uv run python benchmark/run_r_family.py --method R2 --dataset ${DATASET} ${OUTPUT_ARG} ${SAMPLE_ARG} --miners ${miner}"
done

echo "Done. Check: squeue -u \$USER | grep R2_"
