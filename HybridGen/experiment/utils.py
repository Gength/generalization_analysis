"""
HybridGen Experiment Utilities — shared across experiment versions.
"""

import os
import json
import argparse
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np
import pm4py


# =====================================================================
# Constants
# =====================================================================

MINERS = {
    "Inductive Miner (IM)": lambda log: pm4py.discover_petri_net_inductive(log),
    "Heuristics Miner":    lambda log: pm4py.discover_petri_net_heuristics(log),
    "Alpha Miner":         lambda log: pm4py.discover_petri_net_alpha(log),
}

MINER_ALIASES = {
    "IM":         "Inductive Miner (IM)",
    "Heuristics": "Heuristics Miner",
    "Alpha":      "Alpha Miner",
}


# =====================================================================
# CLI — Base argument parser (version-specific args added by caller)
# =====================================================================

def base_parse_args(description="Hybrid Generative-Structural Evaluation",
                    output_dir_default="output"):
    """Return an ArgumentParser with common arguments. Caller adds version-specific ones."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("-m", "--miner", nargs="+",
                   choices=list(MINER_ALIASES.keys()) + ["all"], default=["all"],
                   help="Miner(s) to evaluate. 'all' runs all miners.")
    p.add_argument("-w", "--weights", nargs="+", type=float, default=[0.5],
                   help="Fusion weights (w) to sweep. Default: [0.5]")
    p.add_argument("-r", "--runs", type=int, default=1,
                   help="Independent runs per config. Default: 1")
    p.add_argument("-s", "--shadow-traces", type=int, default=1000,
                   help="Stochastic traces for shadow log. Default: 1000")
    p.add_argument("-i", "--iterations", type=int, default=5,
                   help="Shadow log iterations per run. Default: 5")
    p.add_argument("--seed", type=int, default=None,
                   help="Base random seed. Default: 42")
    p.add_argument("-d", "--data-path", type=str,
                   default="data/BPI-Challenge_2017/BPI Challenge 2017.xes.gz",
                   help="Path to input XES event log.")
    p.add_argument("-o", "--output-dir", type=str, default=output_dir_default,
                   help="Output directory for JSON results.")
    p.add_argument("--output-file", type=str, default=None,
                   help="Custom output JSON filename (overrides auto-generated name).")
    return p


# =====================================================================
# Helpers
# =====================================================================

def resolve_miners(requested):
    expanded = list(MINER_ALIASES.keys()) if "all" in requested else requested
    return {MINER_ALIASES[name]: MINERS[MINER_ALIASES[name]] for name in set(expanded)}


def load_event_log(xes_path: str):
    """Load and standardize event log from XES path."""
    print("\n[1/3] Loading event log...")
    if os.path.exists(xes_path):
        event_log = pm4py.read_xes(xes_path)
    else:
        print(f"  Log not found at {xes_path}, generating dummy dataset...")
        import pandas as pd
        df = pd.DataFrame({
            'case:concept:name': ['1','1','1','2','2','2','3','3'],
            'concept:name':      ['A','B','C','A','X','C','A','B'],
        })
        event_log = pm4py.format_dataframe(df, case_id='case:concept:name', activity_key='concept:name')
    print("      Converting DataFrame to standard EventLog object...")
    event_log = pm4py.convert_to_event_log(event_log)
    print(f"      Loaded: {len(event_log)} traces | {sum(len(t) for t in event_log)} events")
    return event_log


def print_header(active_miners: dict, args):
    """Print experiment header."""
    print("=" * 80)
    print(f"  Hybrid Generalization Evaluation — Extensive Logging")
    print(f"     Miners              : {list(active_miners.keys())}")
    print(f"     Weights to Sweep    : {args.weights}")
    print(f"     Independent Runs    : {args.runs}")
    print(f"     Shadow Traces       : {args.shadow_traces}")
    max_n = getattr(args, 'max_n', None)
    if max_n is not None:
        print(f"     N-gram Order (max_n): {max_n}")
    print(f"     Base Seed           : {args.seed}")
    print(f"     Data Path           : {args.data_path}")
    print("=" * 80)


def print_summary_table(all_results: list):
    """Print aggregated summary table across runs."""
    summary = defaultdict(list)
    for r in all_results:
        summary[(r['miner'], r['w_weight'])].append(r)

    n_runs = max(r.get('run_id', 1) for r in all_results)
    print(f"\n[3/3] Final Aggregated Summary (Averaged across {n_runs} runs)")
    print("-" * 105)
    print(f"{'Miner':<22} | {'Weight':>6} | {'Avg Gen_Struct':>14} | {'Avg Gen_Shadow':>14} | {'Avg Gen_Total':>14} | {'Total StdDev':>12}")
    print("-" * 105)

    for (miner, w), runs_data in summary.items():
        avg_struct = np.mean([r['gen_struct'] for r in runs_data])
        avg_shadow = np.mean([r['gen_shadow_mean'] for r in runs_data])
        avg_total  = np.mean([r['gen_total'] for r in runs_data])
        std_total  = np.std([r['gen_total'] for r in runs_data])
        print(f"{miner:<22} | {w:>6.2f} | {avg_struct:>14.4f} | {avg_shadow:>14.4f} | {avg_total:>14.4f} | ±{std_total:<11.4f}")


def print_stratified_table(all_results: list):
    """Print stratified mutation analysis if available."""
    has_stratified = any(r.get('gen_shadow_regular_mean') is not None for r in all_results)
    if not has_stratified:
        return

    summary = defaultdict(list)
    for r in all_results:
        summary[(r['miner'], r['w_weight'])].append(r)

    print(f"\n{'─' * 105}")
    print(f"  Stratified Mutation Analysis: Regular vs. Mutated Shadow Traces")
    print(f"{'─' * 105}")
    print(f"{'Miner':<22} | {'Weight':>6} | {'Mutations':>9} | {'Regular Fit':>12} | {'Mutated Fit':>12} | {'Δ (Reg-Mut)':>12}")
    print(f"{'─' * 105}")

    for (miner, w), runs_data in summary.items():
        avg_reg  = np.mean([r['gen_shadow_regular_mean'] for r in runs_data])
        avg_mut_count = np.mean([r['avg_mutations_per_run'] for r in runs_data])
        avg_mut = np.mean([r['gen_shadow_mutated_mean'] for r in runs_data])
        delta   = avg_reg - avg_mut

        if avg_mut_count < 0.5:
            print(f"{miner:<22} | {w:>6.2f} | {avg_mut_count:>8.1f} | {avg_reg:>12.4f} | {'—':>12} | {'—':>12}  (insufficient mutated traces)")
        else:
            marker = ""
            if delta > 0.05:
                marker = " ← model struggles with mutations"
            elif delta < -0.01:
                marker = " (mutated traces fit better — model is very permissive)"
            print(f"{miner:<22} | {w:>6.2f} | {avg_mut_count:>8.0f} | {avg_reg:>12.4f} | {avg_mut:>12.4f} | {delta:>+11.4f}{marker}")
    print(f"{'─' * 105}")


def export_results(all_results: list, args, version: str = "1"):
    """Export results to JSON using args.output_dir."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.output_file or os.path.join(out_dir, f"hybrid_extensive_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump({"version": version, "config": vars(args), "results": all_results}, f, indent=2)
    return out_path
