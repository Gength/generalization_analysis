#!/usr/bin/env python3
"""job_m7 — M7 (SpeciAL4PM). Self-contained /tmp job."""
import os, sys, shutil, secrets, argparse
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from job_prepare import prepare_workdir
from bridges.run_m7 import run
ap = argparse.ArgumentParser(description="M7 job")
ap.add_argument("--dataset", required=True); ap.add_argument("--output", default=None); ap.add_argument("--miners", nargs="*", default=None)
ap.add_argument("--cell-timeout", type=int, default=3600,
                help="Per-cell METRIC budget in seconds (discovery excluded; "
                     "protocol default 3600, 0 = unlimited)")
args = ap.parse_args()
from datasets import DATASETS
ds_name = DATASETS[args.dataset]["name"]
miner_list = ", ".join(args.miners) if args.miners else "all 8"
print(f"[M7] {args.dataset} ({ds_name}) | miners: {miner_list}")
workdir = f"/tmp/benchmark_M7_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}/"
output_dir = args.output or os.path.join(workdir, "results"); os.makedirs(output_dir, exist_ok=True)
prepare_workdir(workdir, args.dataset, copy_xes=True, discover_pnmls=True)
run(args.dataset, workdir, output_dir, miners=args.miners, cell_timeout=args.cell_timeout)
shutil.rmtree(workdir); print(f"  [clean] removed {workdir}")
