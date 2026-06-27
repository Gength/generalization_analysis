#!/usr/bin/env python3
"""job_m4 — M4 (Anti-Alignment Generalization). Self-contained /tmp job."""
import os, sys, shutil, secrets, argparse
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from job_prepare import prepare_workdir
from bridges.run_m4 import run
ap = argparse.ArgumentParser(description="M4 job")
ap.add_argument("--dataset", required=True); ap.add_argument("--output", default=None); ap.add_argument("--miners", nargs="*", default=None)
args = ap.parse_args()
workdir = f"/tmp/benchmark_M4_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}/"
output_dir = args.output or os.path.join(workdir, "results"); os.makedirs(output_dir, exist_ok=True)
prepare_workdir(workdir, args.dataset, copy_xes=True, discover_pnmls=True)
run(args.dataset, workdir, output_dir, miners=args.miners)
shutil.rmtree(workdir); print(f"  [clean] removed {workdir}")
