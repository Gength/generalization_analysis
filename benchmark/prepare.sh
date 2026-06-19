#!/bin/bash
# Prepare models: discover all 7 miners, export PNML + DFG JSON + per-miner DFGs
set -e

# ── Detect dataset switch — warn if stale models exist ──────────────────────
# Extract dataset key from original args (default: D1)
ORIG_ARGS=("$@")
DATASET_KEY="D1"
i=0
while [ $i -lt ${#ORIG_ARGS[@]} ]; do
    arg="${ORIG_ARGS[$i]}"
    case "$arg" in
        --dataset=*) DATASET_KEY="${arg#*=}";;
        --dataset)   i=$((i+1)); DATASET_KEY="${ORIG_ARGS[$i]}";;
    esac
    i=$((i+1))
done

if [ -f "benchmark/models/manifest.json" ]; then
    CURRENT_DS=$(python3 -c "import json; print(json.load(open('benchmark/models/manifest.json')).get('dataset','unknown'))" 2>/dev/null || echo "unknown")
    if [ "$CURRENT_DS" != "unknown" ] && [ "$CURRENT_DS" != "$DATASET_KEY" ]; then
        echo "⚠️  WARNING: benchmark/models/ contains models for '$CURRENT_DS', but you requested '$DATASET_KEY'."
        echo "   Stale XES/DFG caches from previous datasets may cause errors."
        echo "   Either clean the directory first:"
        echo "     ls -la benchmark/models/   # review current files"
        echo "     rm -rf benchmark/models/*  # remove all cached models"
        echo "   Or proceed (old files will be overwritten/ignored where possible)."
        echo ""
    fi
fi

uv run python benchmark/01_prepare_models.py "${ORIG_ARGS[@]}"
uv run python benchmark/02_gen_per_miner_dfgs.py "${ORIG_ARGS[@]}"
