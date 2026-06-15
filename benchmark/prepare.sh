#!/bin/bash
# Prepare models: discover all 7 miners, export PNML + DFG JSON
uv run python benchmark/01_prepare_models.py "$@"
