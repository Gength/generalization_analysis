#!/usr/bin/env python3
"""job_r1 — R1 (K-Fold CV). Self-contained /tmp job."""
import os, sys, shutil, secrets, argparse
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_r_family import run_r1
from job_prepare import prepare_workdir
ap = argparse.ArgumentParser(description="R1 job")
ap.add_argument("--dataset", required=True); ap.add_argument("--output", default=None)
ap.add_argument("--seed", type=int, default=42); ap.add_argument("--miners", nargs="*", default=None)
args = ap.parse_args()
workdir = f"/tmp/benchmark_R1_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}/"
output_dir = args.output or os.path.join(workdir, "results"); os.makedirs(output_dir, exist_ok=True)
prepare_workdir(workdir, args.dataset, copy_xes=True)
run_r1(args.dataset, workdir, output_dir, seed=args.seed, miners=args.miners)
shutil.rmtree(workdir); print(f"  [clean] removed {workdir}")
