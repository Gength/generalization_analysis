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
ap.add_argument("--miners", nargs="*", default=None)
ap.add_argument("--workers", type=int, default=8, help="Parallel workers (default: 8)")
ap.add_argument("--cell-timeout", type=int, default=3600,
                help="Per-cell METRIC budget in seconds (discovery excluded; "
                     "protocol default 3600, R1/R2 exempt, 0 = unlimited)")
ap.add_argument("--no-model-cache", action="store_true",
                help="Force rediscovery instead of using benchmark/models/<key>/")
args = ap.parse_args()
from datasets import DATASETS
ds_name = DATASETS[args.dataset]["name"]
methods_str = ", ".join(args.methods) if args.methods else "M1a–M1g"
miners_str = ", ".join(args.miners) if args.miners else "all 8"
print(f"[M1] {args.dataset} ({ds_name}) | methods: {methods_str} | miners: {miners_str} | workers: {args.workers}")
workdir = f"/tmp/benchmark_M1_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}/"
output_dir = args.output or os.path.join(workdir, "results"); os.makedirs(output_dir, exist_ok=True)
prepare_workdir(workdir, args.dataset, copy_xes=True)
run(args.dataset, workdir, output_dir, methods=args.methods, miners=args.miners,
    workers=args.workers, cell_timeout=args.cell_timeout,
    model_cache=not args.no_model_cache)
shutil.rmtree(workdir); print(f"  [clean] removed {workdir}")
