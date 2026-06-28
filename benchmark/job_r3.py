#!/usr/bin/env python3
"""job_r3 — R3 (Naive Random Baseline). Self-contained /tmp job."""
import os, sys, shutil, secrets, argparse
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_r_family import run_r3
from job_prepare import prepare_workdir
ap = argparse.ArgumentParser(description="R3 job")
ap.add_argument("--dataset", required=True); ap.add_argument("--output", default=None)
ap.add_argument("--seed", type=int, default=42); ap.add_argument("--miners", nargs="*", default=None)
ap.add_argument("--num-traces", type=int, default=1000)
args = ap.parse_args()
from datasets import DATASETS
ds_name = DATASETS[args.dataset]["name"]
miner_list = ", ".join(args.miners) if args.miners else "all 8"
print(f"[R3] {args.dataset} ({ds_name}) | miners: {miner_list} | seed={args.seed} | num_traces={args.num_traces}")
workdir = f"/tmp/benchmark_R3_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}/"
output_dir = args.output or os.path.join(workdir, "results"); os.makedirs(output_dir, exist_ok=True)
prepare_workdir(workdir, args.dataset, copy_xes=True)
run_r3(args.dataset, workdir, output_dir, seed=args.seed, miners=args.miners, num_traces=args.num_traces)
shutil.rmtree(workdir); print(f"  [clean] removed {workdir}")
