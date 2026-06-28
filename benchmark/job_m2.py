#!/usr/bin/env python3
"""job_m2 — M2 (PM4Py Built-in Gen). Self-contained /tmp job."""
import os, sys, shutil, secrets, argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser(description="M2 job")
ap.add_argument("--dataset", required=True)
ap.add_argument("--output", default=None)
ap.add_argument("--miners", nargs="*", default=None)
args = ap.parse_args()
from datasets import DATASETS
ds_name = DATASETS[args.dataset]["name"]
miner_list = ", ".join(args.miners) if args.miners else "all 8"
print(f"[M2] {args.dataset} ({ds_name}) | miners: {miner_list}")

workdir = f"/tmp/benchmark_M2_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}/"
output_dir = args.output or os.path.join(workdir, "results")
os.makedirs(output_dir, exist_ok=True)

from job_prepare import prepare_workdir
from run_m2 import run
prepare_workdir(workdir, args.dataset, copy_xes=True)
run(args.dataset, workdir, output_dir, miners=args.miners)
shutil.rmtree(workdir)
print(f"  [clean] removed {workdir}")
