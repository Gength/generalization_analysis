#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
NUM_WORKERS=14
PERCENTAGE=2

submit_job() {
	local job_name="$1"
	local command="$2"

	sbatch \
		--partition=Krater \
		--job-name="$job_name" \
		--nodes=1 \
		--ntasks=1 \
		--cpus-per-task="$NUM_WORKERS" \
		--output="./logs/%x_%j.log" \
		--chdir="$SCRIPT_DIR" \
		--wrap="/bin/bash -c 'export TMPDIR=/tmp; $command'"
}

submit_job "method1_IM" \
	"source ~/.bashrc && uv run python pick_one_out_experiment.py --method method1 -t ${PERCENTAGE} -m IM -w ${NUM_WORKERS}"
submit_job "method1_IMf" \
	"source ~/.bashrc && uv run python pick_one_out_experiment.py --method method1 -t ${PERCENTAGE} -m IMf -w ${NUM_WORKERS}"
submit_job "method1_Heuristics" \
	"source ~/.bashrc && uv run python pick_one_out_experiment.py --method method1 -t ${PERCENTAGE} -m Heuristics -w ${NUM_WORKERS}"
submit_job "method1_Alpha" \
	"source ~/.bashrc && uv run python pick_one_out_experiment.py --method method1 -t ${PERCENTAGE} -m Alpha -w ${NUM_WORKERS}"
submit_job "baseline_IM" \
	"source ~/.bashrc && uv run python pick_one_out_experiment.py --method baseline -t ${PERCENTAGE} -m IM -w ${NUM_WORKERS}"
submit_job "baseline_IMf" \
	"source ~/.bashrc && uv run python pick_one_out_experiment.py --method baseline -t ${PERCENTAGE} -m IMf -w ${NUM_WORKERS}"
submit_job "baseline_Heuristics" \
	"source ~/.bashrc && uv run python pick_one_out_experiment.py --method baseline -t ${PERCENTAGE} -m Heuristics -w ${NUM_WORKERS}"
submit_job "baseline_Alpha" \
	"source ~/.bashrc && uv run python pick_one_out_experiment.py --method baseline -t ${PERCENTAGE} -m Alpha -w ${NUM_WORKERS}"