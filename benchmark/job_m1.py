#!/usr/bin/env python3
"""job_m1 — M1a-M1g (HybridGen family). Self-contained /tmp job. Parallelized."""
import os, sys, shutil, secrets, argparse
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_m1_family import run
from job_prepare import prepare_workdir
ap = argparse.ArgumentParser(description="M1-family job")
ap.add_argument("--dataset", required=True); ap.add_argument("--output", default=None)
ap.add_argument("--methods", nargs="+", default=None)
ap.add_argument("--workers", type=int, default=8, help="Parallel workers (default: 8)")
args = ap.parse_args()
workdir = f"/tmp/benchmark_M1_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}/"
output_dir = args.output or os.path.join(workdir, "results"); os.makedirs(output_dir, exist_ok=True)
prepare_workdir(workdir, args.dataset, copy_xes=True)
run(args.dataset, workdir, output_dir, methods=args.methods, workers=args.workers)
shutil.rmtree(workdir); print(f"  [clean] removed {workdir}")
