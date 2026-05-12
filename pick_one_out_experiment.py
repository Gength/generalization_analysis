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

import pick_one_out_algorithm as algo

from pick_one_out_algorithm import (
    load_and_prepare_log,
    compute_variants,
    build_global_dfg,
    evaluate_miner,
    compute_baseline_metrics,
)

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
    "baseline":    None,   # special: runs PM4Py built-in only, no pick-one-out
}

RANDOM_SEED = 42


# ─── Output & Summary ───────────────────────────────────────────────────────

def print_summary(all_results, baseline, max_variants, total_variants,
                  output_path, run_config):
    """Print final comparison table and save full results + config to JSON."""
    print(f"\n{'='*60}")
    print(f"[5/5] FINAL SUMMARY")
    print(f"{'='*60}")
    if all_results:
        print(f"\n{'Miner':<30} {'Method1 Pure':>12} {'Method1 Joint':>14} {'PM4Py Gen':>12}")
        print("-" * 74)
        for r in all_results:
            pm4py_gen = f"{r['pm4py_generalization']:.4f}" if r.get('pm4py_generalization') is not None else "N/A"
            print(f"{r['miner']:<30} {r['score_pure']:>12.4f} "
                  f"{r['score_joint']:>14.4f} {pm4py_gen:>12}")
    if baseline:
        print(f"\n  PM4Py Baseline (IM) generalization: {baseline.get('generalization_pm4py', 'N/A')}")

    # Save results
    output = {
        "run_config": run_config,
        "dataset": "BPI Challenge 2017",
        "total_variants": total_variants,
        "num_variants_sampled": max_variants,
        "miner_results": [
            {
                "miner": r["miner"],
                "score_pure_weighting": r["score_pure"],
                "score_joint_weighting": r["score_joint"],
                "pm4py_generalization": r.get("pm4py_generalization"),
                "num_variants_evaluated": len(r["results"]),
            }
            for r in all_results
        ],
        "pm4py_baseline": baseline,
        "detailed_results": all_results,
    }
    path = output_path or os.path.join(OUTPUT_DIR, "pick_one_out_results.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {path}")


# ─── Argument Parser ─────────────────────────────────────────────────────────

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Pick-One-Out Generalization Metric — BPI Challenge 2017",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test: 1%% variants, single miner
  python pick_one_out_experiment.py -t 1 -m IM

  # Small-scale validation: 10%% variants, all miners (no baseline)
  python pick_one_out_experiment.py -t 10 -m all

  # Full scale on IM only
  python pick_one_out_experiment.py -t 100 -m IM

  # PM4Py baseline only (dispatch to a separate node)
  python pick_one_out_experiment.py -m baseline

  # Multi-node dispatch:
  #   node 1: python pick_one_out_experiment.py -t 100 -m IM IMf
  #   node 2: python pick_one_out_experiment.py -t 100 -m Heuristics Alpha
  #   node 3: python pick_one_out_experiment.py -m baseline
        """,
    )
    p.add_argument(
        "-m", "--miner",
        nargs="+",
        choices=list(MINER_ALIASES.keys()) + ["all"],
        default=["all"],
        help="Miner(s) to evaluate. 'all' = the 4 miners (NOT including baseline). "
             "'baseline' = PM4Py built-in generalization only (no pick-one-out). "
             "Combine freely: -m IM baseline  runs IM + baseline. "
             "Default: all",
    )
    p.add_argument(
        "-t", "--test-variant",
        type=float,
        default=100.0,
        help="Percentage of variants to sample (0–100). "
             "e.g. -t 1 = ~159 variants, -t 100 = all 15930. "
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
    """
    Resolve CLI miner selections into two lists:
      - miners_to_run:  dict long_name -> discovery_fn  (for pick-one-out)
      - run_baseline:   bool  (run PM4Py built-in generalization)

    'all' expands to all 4 miners.  'baseline' is independent — include
    it explicitly if you want the PM4Py built-in metric alongside miners.
    This allows dispatching miners and baseline to separate nodes, e.g.:

        python pick_one_out_experiment.py -t 100 -m IM IMf    # node 1
        python pick_one_out_experiment.py -t 100 -m baseline  # node 2
    """
    miners_to_run = {}
    run_baseline = False

    # Expand "all" to the 4 miners (NOT including baseline)
    expanded = []
    for name in requested:
        if name == "all":
            expanded.extend([k for k in MINER_ALIASES if k != "baseline"])
        else:
            expanded.append(name)

    # Deduplicate while preserving order
    seen = set()
    ordered = []
    for name in expanded:
        if name not in seen:
            seen.add(name)
            ordered.append(name)

    for name in ordered:
        if name == "baseline":
            run_baseline = True
        else:
            long_name = MINER_ALIASES[name]
            miners_to_run[long_name] = MINERS[long_name]

    return miners_to_run, run_baseline


# ─── Output Path ─────────────────────────────────────────────────────────────

def _make_output_path(miners_requested, test_variant_pct, explicit_path):
    """Generate a unique output filename that won't collide across nodes.

    Format: output/pick_one_out_{MINERS}_{PCT}pct_{TIMESTAMP}.json

    Examples:
        output/pick_one_out_IM_100pct_20260512_143022.json
        output/pick_one_out_IM-IMf_100pct_20260512_143030.json
        output/pick_one_out_baseline_20260512_143035.json
    """
    if explicit_path:
        return explicit_path

    # Build a compact label for the miner(s) being run
    if not miners_requested or miners_requested == ["all"]:
        miner_label = "all"
    else:
        miner_label = "-".join(miners_requested)

    pct_label = f"{test_variant_pct:.0f}pct" if test_variant_pct == int(test_variant_pct) else f"{test_variant_pct:.1f}pct"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pick_one_out_{miner_label}_{pct_label}_{timestamp}.json"
    return os.path.join(OUTPUT_DIR, filename)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Set algorithm globals from CLI args
    algo.NUM_WORKERS = args.workers
    algo.RANDOM_SEED = RANDOM_SEED

    t_start = time.time()

    try:
        _run_experiment(args, t_start)
    except KeyboardInterrupt:
        print("\n\n⏹ Experiment cancelled by user.")
        sys.exit(130)


def _run_experiment(args, t_start):
    """Core experiment flow — separated so KeyboardInterrupt is caught cleanly."""
    # Resolve miner selection
    active_miners, run_baseline = resolve_miners(args.miner)
    if not active_miners and not run_baseline:
        print("Error: no miner or baseline selected.")
        sys.exit(1)

    # Validate and compute max_variants from percentage
    if args.test_variant <= 0 or args.test_variant > 100:
        print("Error: --test-variant must be in (0, 100]")
        sys.exit(1)

    print("=" * 60)
    print(f"  Pick-One-Out Generalization Evaluation")
    print(f"     Variant sampling   : {args.test_variant:.2f}%")
    print(f"     Miners to evaluate : {list(active_miners.keys()) if active_miners else '(none)'}")
    print(f"     PM4Py baseline     : {'yes' if run_baseline else 'no'}")
    print(f"     Worker processes   : {algo.NUM_WORKERS}")
    print("=" * 60)

    # 1. Load log
    event_log = load_and_prepare_log(XES_PATH)

    # 2. Compute variants & global DFG
    variants = compute_variants(event_log)
    global_dfg = build_global_dfg(event_log)
    print(f"       Global DFG has {len(global_dfg)} unique edges")

    # Print variant stats
    freqs = [len(v) for v in variants.values()]
    total_variants = len(variants)
    max_variants = ceil(total_variants * args.test_variant / 100.0)
    print(f"       Variant stats — min_freq={min(freqs)}, max_freq={max(freqs)}, "
          f"mean_freq={sum(freqs)/len(freqs):.1f}, "
          f"singletons={sum(1 for f in freqs if f == 1)}")
    print(f"       Sampling {max_variants}/{total_variants} variants "
          f"({args.test_variant:.2f}%)")

    # 3. Evaluate each miner (pick-one-out)
    all_results = []
    for miner_name, miner_fn in active_miners.items():
        result = evaluate_miner(event_log, variants, global_dfg,
                                miner_name, miner_fn, max_variants)
        all_results.append(result)

    # 4. PM4Py baseline (if requested)
    baseline = {}
    if run_baseline:
        baseline = compute_baseline_metrics(event_log)

    # 5. Summary
    output_path = _make_output_path(args.miner, args.test_variant, args.output)
    run_config = {
        "xes_path": XES_PATH,
        "miners_requested": args.miner,
        "miners_evaluated": list(active_miners.keys()) if active_miners else [],
        "baseline_requested": run_baseline,
        "test_variant_pct": args.test_variant,
        "max_variants": max_variants,
        "total_variants": total_variants,
        "num_workers": algo.NUM_WORKERS,
        "random_seed": RANDOM_SEED,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_runtime_s": round(time.time() - t_start, 1),
    }
    print_summary(all_results, baseline, max_variants, total_variants,
                  output_path, run_config)

    print(f"\nTotal execution time: {time.time() - t_start:.1f}s")
    if args.test_variant < 5:
        print("💡 Results look correct? Increase -t for full evaluation, e.g.:")
        print("   python pick_one_out_experiment.py -t 100 -m IM   # full scale on IM only")
        print("   python pick_one_out_experiment.py -t 100          # full scale on all miners")


if __name__ == "__main__":
    main()
