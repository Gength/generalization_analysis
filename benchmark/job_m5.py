#!/usr/bin/env python3
"""job_m5 — M5 (AVATAR RelGAN). Self-contained /tmp job."""
import os, sys, shutil, secrets, argparse
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docker.run_avatar import run
from job_prepare import prepare_workdir
ap = argparse.ArgumentParser(description="M5 job")
ap.add_argument("--dataset", required=True); ap.add_argument("--output", default=None)
ap.add_argument("--miners", nargs="*", default=None)
ap.add_argument("--eval-only", action="store_true"); ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
from datasets import DATASETS
ds_name = DATASETS[args.dataset]["name"]
miner_list = ", ".join(args.miners) if args.miners else "all 8"
mode = "EVAL-ONLY" if args.eval_only else ("QUICK" if args.quick else "FULL")
print(f"[M5] {args.dataset} ({ds_name}) | miners: {miner_list} | mode: {mode}")
workdir = f"/tmp/benchmark_M5_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}/"
output_dir = args.output or os.path.join(workdir, "results"); os.makedirs(output_dir, exist_ok=True)
prepare_workdir(workdir, args.dataset, copy_xes=True)
run(args.dataset, workdir, output_dir, quick=args.quick, eval_only=args.eval_only, miners=args.miners)
shutil.rmtree(workdir); print(f"  [clean] removed {workdir}")
