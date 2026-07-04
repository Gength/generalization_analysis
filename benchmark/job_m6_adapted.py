#!/usr/bin/env python3
"""job_m6_adapted — M6 adapted (bsgen breeding + token replay). Self-contained /tmp job."""
import os, sys, shutil, secrets, argparse
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from job_prepare import prepare_workdir
from bridges.run_m6_adapted import run
ap = argparse.ArgumentParser(description="M6 adapted job")
ap.add_argument("--dataset", required=True)
ap.add_argument("--output", default="benchmark/results/configs",
                help="Default: benchmark/results/configs (NOT configs, which holds -bgen)")
ap.add_argument("--miners", nargs="+", default=None)
ap.add_argument("--replicates", type=int, default=10)
ap.add_argument("--generations", type=int, default=10)
ap.add_argument("--k", type=int, default=2)
ap.add_argument("--p", type=float, default=1.0)
ap.add_argument("--sample-size", type=int, default=200)
ap.add_argument("--cell-timeout", type=int, default=3600,
                help="Per-cell METRIC budget in seconds (discovery excluded; "
                     "protocol default 3600, 0 = unlimited)")
ap.add_argument("--workers", type=int, default=1,
                help="Parallel miner processes (results identical to serial)")
args = ap.parse_args()
from datasets import DATASETS
ds_name = DATASETS[args.dataset]["name"]
miner_list = ", ".join(args.miners) if args.miners else "all 8"
print(f"[M6 adapted] {args.dataset} ({ds_name}) | miners: {miner_list} | "
      f"reps={args.replicates} gens={args.generations} k={args.k} p={args.p} n={args.sample_size}")
workdir = f"/tmp/benchmark_M6A_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}/"
os.makedirs(args.output, exist_ok=True)
prepare_workdir(workdir, args.dataset, copy_xes=True, discover_pnmls=True)
run(args.dataset, workdir, args.output, miners=args.miners,
    replicates=args.replicates, generations=args.generations,
    k=args.k, p=args.p, sample_size=args.sample_size,
    cell_timeout=args.cell_timeout, workers=args.workers)
shutil.rmtree(workdir); print(f"  [clean] removed {workdir}")
