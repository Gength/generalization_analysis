#!/bin/bash
# Prepare models: discover all 7 miners, export PNML + DFG JSON + per-miner DFGs
uv run python benchmark/01_prepare_models.py "$@"
uv run python benchmark/02_gen_per_miner_dfgs.py "$@"
