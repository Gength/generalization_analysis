"""
Pick-One-Out Generalization Metric — Experiment Runner
=======================================================
CLI front-end and experiment orchestration for the pick-one-out
generalization evaluation.

This module handles:
  - Command-line argument parsing
  - Configuration (paths, miner registry, seed, etc.)
  - Output formatting & JSON serialization
  - End-to-end experiment flow (orchestrates algorithm module)

The core algorithms live in pick_one_out_algorithm.py.

Usage:
    conda activate pm4py
    python pick_one_out_experiment.py -t 2 -m IM -w 8
"""

import time
import json
import os
import sys
import argparse
from math import ceil
from datetime import datetime, timezone
from collections import defaultdict

import pm4py
import pick_one_out_algorithm as algo

from pick_one_out_algorithm import evaluate_miner
from pm4py.algo.evaluation.generalization import algorithm as generalization_eval

# ─── Configuration ───────────────────────────────────────────────────────────

XES_PATH = "data/BPI-Challenge_2017/BPI Challenge 2017.xes.gz"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# All available miners: long_name -> discovery function
MINERS = {
    "Inductive Miner (IM)": lambda log: algo.discover_petri_net_for_miner("Inductive Miner (IM)", log),
    "Inductive Miner (IMf)": lambda log: algo.discover_petri_net_for_miner("Inductive Miner (IMf)", log),
    "Heuristics Miner": lambda log: algo.discover_petri_net_for_miner("Heuristics Miner", log),
    "Alpha Miner": lambda log: algo.discover_petri_net_for_miner("Alpha Miner", log),
}

# CLI short-name -> long-name mapping
MINER_ALIASES = {
    "IM":          "Inductive Miner (IM)",
    "IMf":         "Inductive Miner (IMf)",
    "Heuristics":  "Heuristics Miner",
    "Alpha":       "Alpha Miner",
}

RANDOM_SEED = 42


# ─── Preprocessing ──────────────────────────────────────────────────────────

def load_and_prepare_log(path):
    """Load XES and convert to EventLog (list of traces)."""
    print(f"Loading event log from {path} ...")
    t0 = time.time()
    from pm4py.objects.log.importer.xes import importer as xes_importer
    event_log = xes_importer.apply(path)
    df = pm4py.convert_to_dataframe(event_log)
    # expose original dataframe for efficient filtering in workers
    algo.ORIGINAL_DF = df
    algo.CASE_IDS_ORDERED = df["case:concept:name"].drop_duplicates().tolist()
    print(f"       Loaded {len(event_log)} cases, "
          f"{sum(len(t) for t in event_log)} events in {time.time() - t0:.1f}s")
    return event_log


def compute_variants(event_log):
    """Return dict: variant_tuple -> list of trace indices."""
    # a process variant (or simply variant) is defined as a unique, ordered sequence of activities executed from the beginning to the end of a process case.
    print("Computing variants ...")
    t0 = time.time()
    variants = defaultdict(list)
    for idx, trace in enumerate(event_log):
        variant_key = tuple(event["concept:name"] for event in trace)
        variants[variant_key].append(idx)
    print(f"       Found {len(variants)} unique variants in {time.time() - t0:.1f}s")
    return variants


def build_global_dfg(event_log):
    """Build global Directly-Follows Graph: edge -> frequency."""
    dfg = defaultdict(int)
    for trace in event_log:
        for i in range(len(trace) - 1):
            a = trace[i]["concept:name"]
            b = trace[i + 1]["concept:name"]
            dfg[(a, b)] += 1
    return dfg


# ─── PM4Py Baseline Metrics ─────────────────────────────────────────────────

def compute_baseline_metrics(event_log):
    """PM4Py built-in generalization on the full log using Inductive Miner."""
    print(f"\n{'='*60}")
    print(f"Computing PM4Py baseline metrics")
    print(f"{'='*60}")

    net, im, fm = pm4py.discover_petri_net_inductive(event_log)
    generalization = generalization_eval.apply(event_log, net, im, fm)

    print(f"       Generalization (PM4Py): {generalization:.4f}")

    return {"generalization_pm4py": generalization}


# ─── Output & Summary ───────────────────────────────────────────────────────

def print_summary(all_results, baseline, max_variants, total_variants,
                  summary_path, details_path, run_config):
    """Print final comparison table; save summary + details to separate JSONs."""
    method = run_config.get("method", "method1")

    print(f"\n{'='*60}")
    print(f"[4/4] FINAL SUMMARY")
    print(f"{'='*60}")

    # Method 1 table
    if all_results:
        print(f"\n{'Miner':<30} {'Method1 Pure':>12} {'Method1 Joint':>14}")
        print("-" * 62)
        for r in all_results:
            print(f"{r['miner']:<30} {r['score_pure']:>12.4f} "
                  f"{r['score_joint']:>14.4f}")

    # Baseline result
    if baseline:
        print(f"\n  PM4Py Baseline (IM) generalization: {baseline.get('generalization_pm4py', 'N/A')}")

    # ── Summary JSON (lightweight: config + scores only) ─────────────────
    summary = {
        "run_config": run_config,
        "dataset": "BPI Challenge 2017",
        "total_variants": total_variants,
        "num_variants_sampled": max_variants,
    }
    if all_results:
        summary["miner_results"] = [
            {
                "miner": r["miner"],
                "score_pure_weighting": r["score_pure"],
                "score_joint_weighting": r["score_joint"],
                "num_variants_evaluated": len(r["results"]),
            }
            for r in all_results
        ]
    if baseline:
        summary["pm4py_baseline"] = baseline

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary  → {summary_path}")

    # ── Details JSON (full per-variant data) ─────────────────────────────
    details = {
        "run_config": run_config,
        "detailed_results": all_results,
    }
    if not baseline:
        with open(details_path, "w") as f:
            json.dump(details, f, indent=2, default=str)
        print(f"Details  → {details_path}")


# ─── Argument Parser ─────────────────────────────────────────────────────────

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Process Model Generalization Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Method 1 (pick-one-out): 2%% variants, IM only
  python pick_one_out_experiment.py --method method1 -t 2 -m IM

  # Method 1: 10%% variants, all miners
  python pick_one_out_experiment.py --method method1 -t 10 -m all

  # PM4Py baseline only
  python pick_one_out_experiment.py --method baseline
        """,
    )
    p.add_argument(
        "--method",
        choices=["method1", "baseline"],
        default="method1",
        help="Evaluation method to use: 'method1' = pick-one-out, "
             "'baseline' = PM4Py built-in generalization. "
             "Default: method1",
    )
    p.add_argument(
        "-m", "--miner",
        nargs="+",
        choices=list(MINER_ALIASES.keys()) + ["all"],
        default=["all"],
        help="Miner(s) for Method 1 evaluation. "
             "'all' = all 4 miners. "
             "Ignored when --method is not method1. "
             "Default: all",
    )
    p.add_argument(
        "-t", "--test-variant",
        type=float,
        default=100.0,
        help="Percentage of variants to sample (0–100) for Method 1. "
             "Default: 100",
    )
    p.add_argument(
        "-w", "--workers",
        type=int,
        default=max(os.cpu_count() - 1, 1),
        help="Number of worker processes for parallel evaluation. "
             "Default: cpu_count - 1",
    )
    p.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output JSON path. Default: output/pick_one_out_results.json",
    )
    return p.parse_args()


# ─── Miner Selection ─────────────────────────────────────────────────────────

def resolve_miners(requested):
    """Resolve CLI miner selections into dict: long_name -> discovery_fn."""
    miners_to_run = {}

    expanded = []
    for name in requested:
        if name == "all":
            expanded.extend(MINER_ALIASES.keys())
        else:
            expanded.append(name)

    seen = set()
    for name in expanded:
        if name not in seen:
            seen.add(name)
            long_name = MINER_ALIASES[name]
            miners_to_run[long_name] = MINERS[long_name]

    return miners_to_run


# ─── Output Path ─────────────────────────────────────────────────────────────

def _make_output_paths(method, miners_requested, test_variant_pct, explicit_path):
    """Generate output paths for summary and details JSON files.

    Returns (summary_path, details_path).
    """
    if not miners_requested or miners_requested == ["all"]:
        miner_label = "all"
    else:
        miner_label = "-".join(miners_requested)

    pct_label = f"{test_variant_pct:.0f}pct" if test_variant_pct == int(test_variant_pct) else f"{test_variant_pct:.1f}pct"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{method}_{miner_label}_{pct_label}_{timestamp}"

    if explicit_path:
        # Use user-provided path as prefix, derive summary/details from it
        dirname = os.path.dirname(explicit_path) or OUTPUT_DIR
        stem = os.path.splitext(os.path.basename(explicit_path))[0]
        summary_path = os.path.join(dirname, f"{stem}_summary.json")
        details_path = os.path.join(dirname, f"{stem}_details.json")
    else:
        summary_path = os.path.join(OUTPUT_DIR, f"{base}_summary.json")
        details_path = os.path.join(OUTPUT_DIR, f"{base}_details.json")

    return summary_path, details_path


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    algo.NUM_WORKERS = args.workers
    algo.RANDOM_SEED = RANDOM_SEED

    t_start = time.time()

    try:
        _run_experiment(args, t_start)
    except KeyboardInterrupt:
        print("\n\n⏹ Experiment cancelled by user.")
        sys.exit(130)


def _run_experiment(args, t_start):
    """Core experiment flow."""
    method = args.method
    is_mtd1 = (method == "method1")

    # Resolve miners (only relevant for Method 1)
    active_miners = resolve_miners(args.miner) if is_mtd1 else {}
    if is_mtd1 and not active_miners:
        print("Error: no miner selected for Method 1.")
        sys.exit(1)

    # Validate variant percentage
    if is_mtd1 and (args.test_variant <= 0 or args.test_variant > 100):
        print("Error: --test-variant must be in (0, 100]")
        sys.exit(1)

    print("=" * 60)
    print(f"  Generalization Evaluation")
    print(f"     Method              : {method}")
    if is_mtd1:
        print(f"     Variant sampling    : {args.test_variant:.2f}%")
        print(f"     Miners              : {list(active_miners.keys())}")
    print(f"     Worker processes    : {algo.NUM_WORKERS}")
    print("=" * 60)

    # ── Step 1: Load event log ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("[1/4] Load event log")
    print(f"{'='*60}")
    event_log = load_and_prepare_log(XES_PATH)

    # ── Step 2: Compute variants & global DFG ───────────────────────────────
    print(f"\n{'='*60}")
    print("[2/4] Compute variants & global DFG")
    print(f"{'='*60}")
    variants = compute_variants(event_log)
    global_dfg = build_global_dfg(event_log)
    print(f"       Global DFG has {len(global_dfg)} unique edges")
    freqs = [len(v) for v in variants.values()]
    total_variants = len(variants)
    max_variants = ceil(total_variants * args.test_variant / 100.0) if is_mtd1 else total_variants
    print(f"       Variant stats — min_freq={min(freqs)}, max_freq={max(freqs)}, "
          f"mean_freq={sum(freqs)/len(freqs):.1f}, "
          f"singletons={sum(1 for f in freqs if f == 1)}")

    # ── Step 3: Evaluation ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[3/4] Evaluation ({method})")
    print(f"{'='*60}")

    all_results = []
    baseline = {}

    if is_mtd1:
        # Method 1: Pick-One-Out for each miner
        for miner_name, miner_fn in active_miners.items():
            result = evaluate_miner(event_log, variants, global_dfg,
                                    miner_name, miner_fn, max_variants)
            all_results.append(result)
    else:
        # Baseline: PM4Py built-in generalization
        baseline = compute_baseline_metrics(event_log)

    # ── Step 4: Summary ─────────────────────────────────────────────────────
    summary_path, details_path = _make_output_paths(
        method, args.miner, args.test_variant, args.output
    )
    run_config = {
        "method": method,
        "xes_path": XES_PATH,
        "miners_requested": args.miner if is_mtd1 else [],
        "miners_evaluated": list(active_miners.keys()) if is_mtd1 else [],
        "test_variant_pct": args.test_variant if is_mtd1 else None,
        "max_variants": max_variants if is_mtd1 else None,
        "total_variants": total_variants,
        "num_workers": algo.NUM_WORKERS,
        "random_seed": RANDOM_SEED,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_runtime_s": round(time.time() - t_start, 1),
    }
    print_summary(all_results, baseline, max_variants, total_variants,
                  summary_path, details_path, run_config)

    print(f"\nTotal execution time: {time.time() - t_start:.1f}s")
    if is_mtd1 and args.test_variant < 5:
        print("💡 Results look correct? Increase -t for full evaluation, e.g.:")
        print("   python pick_one_out_experiment.py --method method1 -t 100 -m IM")


if __name__ == "__main__":
    main()
