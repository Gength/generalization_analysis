#!/usr/bin/env python3
"""job_r2 — R2 (Leave-One-Variant-Out). Self-contained /tmp job."""
import os, sys, shutil, secrets, argparse
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_r_family import run_r2
from job_prepare import prepare_workdir
ap = argparse.ArgumentParser(description="R2 job")
ap.add_argument("--dataset", required=True); ap.add_argument("--output", default=None)
ap.add_argument("--seed", type=int, default=42); ap.add_argument("--miners", nargs="*", default=None)
ap.add_argument("--r2-sample", type=int, default=0)
ap.add_argument("--workers", type=int, default=14)
args = ap.parse_args()
from datasets import DATASETS
ds_name = DATASETS[args.dataset]["name"]
miner_list = ", ".join(args.miners) if args.miners else "all 8"
print(f"[R2] {args.dataset} ({ds_name}) | miners: {miner_list} | seed={args.seed} | sample={args.r2_sample} | workers={args.workers}")
workdir = f"/tmp/benchmark_R2_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}/"
output_dir = args.output or os.path.join(workdir, "results"); os.makedirs(output_dir, exist_ok=True)
prepare_workdir(workdir, args.dataset, copy_xes=True)
run_r2(args.dataset, workdir, output_dir, seed=args.seed, miners=args.miners,
       sample_n=args.r2_sample, workers=args.workers)
shutil.rmtree(workdir); print(f"  [clean] removed {workdir}")
