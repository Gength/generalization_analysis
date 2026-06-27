#!/usr/bin/env python3
"""job_m6 — M6 (Bootstrap Gen). Self-contained /tmp job."""
import os, sys, shutil, secrets, argparse
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from job_prepare import prepare_workdir
from bridges.run_m6_bgen import run
ap = argparse.ArgumentParser(description="M6 job")
ap.add_argument("--dataset", required=True); ap.add_argument("--output", default=None)
ap.add_argument("--jar", default="fixed"); ap.add_argument("--k", type=int, default=2)
ap.add_argument("--m", type=int, default=10); ap.add_argument("--n", type=int, default=200)
ap.add_argument("--g", type=int, default=10); ap.add_argument("--p", type=float, default=1.0)
ap.add_argument("--miners", nargs="+", default=None)
args = ap.parse_args()
workdir = f"/tmp/benchmark_M6_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}/"
output_dir = args.output or os.path.join(workdir, "results"); os.makedirs(output_dir, exist_ok=True)
prepare_workdir(workdir, args.dataset, copy_xes=True, decompress_xes=True, discover_pnmls=True, per_miner_dfgs=True)
run(args.dataset, workdir, output_dir, jar=args.jar, k=args.k, m=args.m, n=args.n, g=args.g, p=args.p, miners=args.miners)
shutil.rmtree(workdir); print(f"  [clean] removed {workdir}")
