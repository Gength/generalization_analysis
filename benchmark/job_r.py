#!/usr/bin/env python3
"""job_r — R1-R3 (Reference metrics). Self-contained /tmp job."""
import os, sys, shutil, secrets, argparse
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from job_prepare import prepare_workdir
from run_r_family import run
ap = argparse.ArgumentParser(description="R-family job")
ap.add_argument("--dataset", required=True); ap.add_argument("--output", default=None)
ap.add_argument("--methods", nargs="+", default=["R1","R2","R3"])
ap.add_argument("--seed", type=int, default=42); ap.add_argument("--miners", nargs="*", default=None)
ap.add_argument("--r2-sample", type=int, default=0); ap.add_argument("--num-traces", type=int, default=1000)
args = ap.parse_args()
workdir = f"/tmp/benchmark_R_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}/"
output_dir = args.output or os.path.join(workdir, "results"); os.makedirs(output_dir, exist_ok=True)
prepare_workdir(workdir, args.dataset, copy_xes=True)
run(args.dataset, workdir, output_dir, methods=args.methods, seed=args.seed, miners=args.miners, r2_sample=args.r2_sample, num_traces=args.num_traces)
shutil.rmtree(workdir); print(f"  [clean] removed {workdir}")
