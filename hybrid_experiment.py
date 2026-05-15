"""
Hybrid Generalization Metric — Experiment Runner
"""

import time
import json
import os
import argparse
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np

import pm4py
import hybrid_algorithm as algo

XES_PATH = "data/BPI-Challenge_2017/BPI Challenge 2017.xes.gz"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define discover functions using PM4Py wrappers
MINERS = {
    "Inductive Miner (IM)": lambda log: pm4py.discover_petri_net_inductive(log),
    "Heuristics Miner": lambda log: pm4py.discover_petri_net_heuristics(log),
    "Alpha Miner": lambda log: pm4py.discover_petri_net_alpha(log),
}

MINER_ALIASES = {
    "IM":          "Inductive Miner (IM)",
    "Heuristics":  "Heuristics Miner",
    "Alpha":       "Alpha Miner",
}

def parse_args():
    p = argparse.ArgumentParser(description="Hybrid Generative-Structural Evaluation")
    p.add_argument(
        "-m", "--miner",
        nargs="+",
        choices=list(MINER_ALIASES.keys()) + ["all"],
        default=["all"],
        help="Miner(s) to evaluate. 'all' runs all miners."
    )
    p.add_argument(
        "-w", "--weights",
        nargs="+",
        type=float,
        default=[0.5],
        help="List of fusion weights (w) to sweep. E.g., -w 0.0 0.5 1.0. Default: [0.5]"
    )
    p.add_argument(
        "-r", "--runs",
        type=int,
        default=1,
        help="Number of independent complete runs per configuration (for Box Plots). Default: 1"
    )
    p.add_argument(
        "-s", "--shadow-traces",
        type=int,
        default=1000,
        help="Number of stochastic traces for the shadow log. Default: 1000"
    )
    p.add_argument(
        "-i", "--iterations",
        type=int,
        default=5,
        help="Shadow log iterations per run (local stability check). Default: 5"
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed. Increments automatically per run for independence. Default: 42"
    )
    p.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Custom output JSON path."
    )
    return p.parse_args()

def resolve_miners(requested):
    expanded = list(MINER_ALIASES.keys()) if "all" in requested else requested
    return {MINER_ALIASES[name]: MINERS[MINER_ALIASES[name]] for name in set(expanded)}

#Main Execution
def main():
    args = parse_args()
    t_start = time.time()
    
    active_miners = resolve_miners(args.miner)

    print("=" * 80)
    print(f"  Hybrid Generalization Evaluation (Method 2) - Extensive Logging")
    print(f"     Miners              : {list(active_miners.keys())}")
    print(f"     Weights to Sweep    : {args.weights}")
    print(f"     Independent Runs    : {args.runs} (Useful for Box Plots)")
    print(f"     Shadow Traces       : {args.shadow_traces}")
    print(f"     Base Seed           : {args.seed}")
    print("=" * 80)

    #1. Load Data
    print("\n[1/3] Loading event log...")
    if os.path.exists(XES_PATH):
        event_log = pm4py.read_xes(XES_PATH)
    else:
        print("Log not found, generating dummy dataset...")
        import pandas as pd
        df = pd.DataFrame({'case:concept:name':['1','1','1','2','2','2','3','3'], 
                           'concept:name':['A','B','C','A','X','C','A','B']})
        event_log = pm4py.format_dataframe(df, case_id='case:concept:name', activity_key='concept:name')

    #2. Evaluation Loop
    print(f"\n[2/3] Running Evaluations...")
    all_results = []
    
    for w in args.weights:
        print(f"\n--- Testing Weight: w={w} ---")
        for miner_name, miner_fn in active_miners.items():
            for run_idx in range(1, args.runs + 1):
                
                #Increment seed per run
                current_seed = (args.seed + run_idx) if args.seed is not None else None
                
                print(f"   Run {run_idx}/{args.runs} | Seed: {current_seed}")
                
                res = algo.evaluate_miner(
                    event_log, 
                    miner_name, 
                    miner_fn, 
                    w=w, 
                    num_shadow_traces=args.shadow_traces,
                    iterations=args.iterations,
                    seed=current_seed
                )
                
                #Append experiment metadata for pandas
                res['run_id'] = run_idx
                res['base_seed'] = args.seed
                all_results.append(res)

    #3. Aggregate Summary (Calculate mean across the multiple runs)
    print(f"\n[3/3] Final Aggregated Summary (Averaged across {args.runs} runs)")
    print("-" * 105)
    print(f"{'Miner':<22} | {'Weight':>6} | {'Avg Gen_Struct':>14} | {'Avg Gen_Shadow':>14} | {'Avg Gen_Total':>14} | {'Total StdDev':>12}")
    print("-" * 105)
    
    #Group results for the summary table
    summary = defaultdict(list)
    for r in all_results:
        key = (r['miner'], r['w_weight'])
        summary[key].append(r)
        
    for (miner, w), runs_data in summary.items():
        avg_struct = np.mean([r['gen_struct'] for r in runs_data])
        avg_shadow = np.mean([r['gen_shadow_mean'] for r in runs_data])
        avg_total = np.mean([r['gen_total'] for r in runs_data])
        std_total = np.std([r['gen_total'] for r in runs_data])
        
        print(f"{miner:<22} | {w:>6.2f} | {avg_struct:>14.4f} | {avg_shadow:>14.4f} | {avg_total:>14.4f} | ±{std_total:<11.4f}")

    #4. Export to JSON
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = args.output or os.path.join(OUTPUT_DIR, f"hybrid_extensive_{timestamp}.json")
    
    with open(out_path, "w") as f:
        json.dump({"config": vars(args), "results": all_results}, f, indent=2)
        
    print(f"\n Results saved to {out_path} (Total time: {time.time()-t_start:.1f}s)")

if __name__ == "__main__":
    main()
