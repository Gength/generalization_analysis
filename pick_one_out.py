"""
Pick-One-Out Generalization Metric — Prototype Implementation
=============================================================
Based on BPI Challenge 2017 event log.

Method 1: For each unique variant v_i, remove its traces from the log,
discover a model M_{∖v_i} on the remaining log, then replay v_i on M_{∖v_i}.
The final score is a weighted average of all replay scores.

Two weighting schemes are implemented:
  - Variant A: Pure variant-frequency weighting  w(v_i) = ln(f(v_i) + 1)
  - Variant B: Joint variant–transition weighting
       w(v_i) = ln(f(v_i) + 1) × ln(avg_global_transition_freq + 1)

Usage:
    conda activate pm4py
    python pick_one_out.py
"""

import pm4py
import time
import json
import os
import sys
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from math import log, ceil
from datetime import datetime
import pandas as pd
import numpy as np

from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.evaluation.generalization import algorithm as generalization_eval
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness

# ─── Configuration ───────────────────────────────────────────────────────────

XES_PATH = "data/BPI-Challenge_2017/BPI Challenge 2017.xes.gz"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# All available miners: long_name -> discovery function
MINERS = {
    "Inductive Miner (IM)": lambda log: pm4py.discover_petri_net_inductive(log),
    "Inductive Miner (IMf)": lambda log: pm4py.discover_petri_net_inductive(
        log, noise_threshold=0.2
    ),
    "Heuristics Miner": lambda log: pm4py.discover_petri_net_heuristics(
        log, dependency_threshold=0.9
    ),
    "Alpha Miner": lambda log: pm4py.discover_petri_net_alpha(log),
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
ORIGINAL_DF = None
CASE_IDS_ORDERED = None

# Module-level globals shared with worker processes (via fork).
# Using multiprocessing (NOT multithreading) because Petri net discovery
# and replay fitness are CPU-bound; threading would hit the GIL.
EVENT_LOG = None
GLOBAL_DFG = None
ACTIVE_MINER_NAME = None
ORIGINAL_DF = None
NUM_WORKERS = None  # set from CLI args in main()

# ─── Helper Functions ────────────────────────────────────────────────────────


def load_and_prepare_log(path):
    """Load XES and convert to EventLog (list of traces)."""
    print(f"[1/5] Loading event log from {path} ...")
    t0 = time.time()
    # Use classic XES importer to avoid the r4pm codepath
    from pm4py.objects.log.importer.xes import importer as xes_importer
    event_log = xes_importer.apply(path)
    df = pm4py.convert_to_dataframe(event_log)
    # expose original dataframe for efficient filtering in workers
    global ORIGINAL_DF, CASE_IDS_ORDERED
    ORIGINAL_DF = df
    CASE_IDS_ORDERED = df["case:concept:name"].drop_duplicates().tolist()
    print(f"       Loaded {len(event_log)} cases, "
          f"{sum(len(t) for t in event_log)} events in {time.time() - t0:.1f}s")
    return event_log


def event_log_to_dataframe(log):
    """Convert an event log (list of traces) into a dataframe."""
    rows = []
    for case_index, trace in enumerate(log):
        case_id = None
        if CASE_IDS_ORDERED is not None and case_index < len(CASE_IDS_ORDERED):
            case_id = CASE_IDS_ORDERED[case_index]
        for event in trace:
            row = dict(event)
            if case_id is not None:
                row["case:concept:name"] = case_id
            rows.append(row)
    return pd.DataFrame(rows)


def compute_variants(event_log):
    """Return dict: variant_tuple -> list of trace indices."""
    print("[2/5] Computing variants ...")
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


def compute_joint_weight(variant_tuple, variant_freq, global_dfg):
    """
    Joint weighting formula (Variant B):
      w(v_i) = ln(f(v_i) + 1) × ln(avg_global_transition_freq + 1)

    Where avg_global_transition_freq is the average global DFG frequency
    of all directly-follows edges within this variant.
    """
    freq_component = log(variant_freq + 1)

    if len(variant_tuple) < 2:
        transition_component = log(1 + 1)  # single-event trace
    else:
        edge_freqs = []
        for i in range(len(variant_tuple) - 1):
            edge = (variant_tuple[i], variant_tuple[i + 1])
            edge_freqs.append(global_dfg.get(edge, 0))
        avg_edge_freq = sum(edge_freqs) / len(edge_freqs)
        transition_component = log(avg_edge_freq + 1)

    return freq_component * transition_component


def compute_pure_freq_weight(variant_freq):
    """Pure frequency weighting (Variant A): w(v_i) = ln(f(v_i) + 1)."""
    return log(variant_freq + 1)


def discover_petri_net_for_miner(miner_name, log):
    """Discover a Petri net using the requested miner."""
    # pm4py discovery functions expect a DataFrame in the current API
    # Convert event-log list to DataFrame if necessary.
    if isinstance(log, list):
        log = event_log_to_dataframe(log)

    if miner_name == "Inductive Miner (IM)":
        return pm4py.discover_petri_net_inductive(log)
    if miner_name == "Inductive Miner (IMf)":
        return pm4py.discover_petri_net_inductive(log, noise_threshold=0.2)
    if miner_name == "Heuristics Miner":
        return pm4py.discover_petri_net_heuristics(log, dependency_threshold=0.9)
    if miner_name == "Alpha Miner":
        return pm4py.discover_petri_net_alpha(log)
    raise ValueError(f"Unsupported miner: {miner_name}")


def replay_trace_on_model(trace, net, im, fm):
    """Replay a single trace on the given Petri net.
    
    Uses token-based replay first (fast); falls back to alignment-based
    only when token-based fails, to keep per-variant cost low.
    """
    # Token-based replay first (orders of magnitude faster than alignment)
    try:
        fitness_result = replay_fitness.apply(
            [trace], net, im, fm,
            variant=replay_fitness.Variants.TOKEN_BASED
        )
        return fitness_result.get("averageFitness",
                                  fitness_result.get("average_trace_fitness", 0.0))
    except Exception:
        pass

    # Fall back to alignment-based replay
    try:
        fitness_result = replay_fitness.apply(
            [trace], net, im, fm,
            variant=replay_fitness.Variants.ALIGNMENT_BASED
        )
        return fitness_result.get("averageFitness",
                                  fitness_result.get("average_trace_fitness", 0.0))
    except Exception:
        return 0.0


def evaluate_variant_task(task):
    """Worker task for one variant in the leave-one-out loop."""
    variant_tuple = task["variant_tuple"]
    trace_indices = task["trace_indices"]
    variant_freq = len(trace_indices)

    # Determine case ids to remove
    removed_case_ids = set()
    if CASE_IDS_ORDERED is not None:
        for idx in trace_indices:
            if idx < len(CASE_IDS_ORDERED):
                removed_case_ids.add(CASE_IDS_ORDERED[idx])

    # Build remaining log (prefer DataFrame path for efficiency)
    if ORIGINAL_DF is not None and len(removed_case_ids) > 0:
        remaining_df = ORIGINAL_DF[~ORIGINAL_DF["case:concept:name"].isin(removed_case_ids)]
        remaining_cases = remaining_df["case:concept:name"].nunique()
        if remaining_cases < 10:
            return {
                "skip": True,
                "reason": "insufficient remaining traces",
                "variant": " → ".join(variant_tuple),
            }
        remaining_log_input = remaining_df
    else:
        trace_index_set = set(trace_indices)
        remaining_log_input = [trace for idx, trace in enumerate(EVENT_LOG)
                               if idx not in trace_index_set]
        if len(remaining_log_input) < 10:
            return {
                "skip": True,
                "reason": "insufficient remaining traces",
                "variant": " → ".join(variant_tuple),
            }

    # Discover model on remaining log and replay the held-out variant
    try:
        net, im, fm = discover_petri_net_for_miner(ACTIVE_MINER_NAME, remaining_log_input)
        representative_trace = EVENT_LOG[trace_indices[0]]
        replay_score = replay_trace_on_model(representative_trace, net, im, fm)
    except Exception as exc:
        return {
            "skip": True,
            "reason": f"discovery/replay failed: {exc}",
            "variant": " → ".join(variant_tuple),
        }

    pure_weight = compute_pure_freq_weight(variant_freq)
    joint_weight = compute_joint_weight(variant_tuple, variant_freq, GLOBAL_DFG)

    return {
        "skip": False,
        "variant": " → ".join(variant_tuple),
        "freq": variant_freq,
        "replay_score": replay_score,
        "pure_weight": pure_weight,
        "joint_weight": joint_weight,
    }


def _report_task_result(task_result, done_count, total):
    """Print one worker result immediately when it finishes.

    Reports a timestamp, completion count, replay score, and weights.
    """
    ts = datetime.now().strftime("%H:%M:%S")
    if task_result.get("skip"):
        print(f"       [{done_count}/{total}] {ts} ⏭ Skipped ({task_result.get('reason')})")
    else:
        print(f"       [{done_count}/{total}] {ts} "
              f"freq={task_result['freq']}, "
              f"replay={task_result['replay_score']:.3f}, "
              f"pure_w={task_result['pure_weight']:.3f}, "
              f"joint_w={task_result['joint_weight']:.3f}")


def evaluate_miner(event_log, variants, global_dfg, miner_name, miner_fn,
                   max_variants):
    """
    Core Pick-One-Out evaluation for a single miner.

    For each sampled variant v_i:
      1. Remove all traces of v_i from the log
      2. Discover model M_{∖v_i}
      3. Replay v_i on M_{∖v_i}
      4. Compute weight w(v_i)
    """
    print(f"\n{'='*60}")
    print(f"[3/5] Evaluating: {miner_name}")
    print(f"{'='*60}")

    # Sort variants by frequency and sample
    sorted_variants = sorted(variants.items(), key=lambda x: len(x[1]), reverse=True)
    sample_count = min(max_variants, len(sorted_variants))
    sampled = sorted_variants[:sample_count]
    print(f"       Sampling top {len(sampled)}/{len(variants)} variants")

    results = []
    total = len(sampled)

    # Compute PM4Py generalization on the full log for this miner (baseline per-miner)
    try:
        if ORIGINAL_DF is not None:
            net_full, im_full, fm_full = discover_petri_net_for_miner(miner_name, ORIGINAL_DF)
        else:
            net_full, im_full, fm_full = discover_petri_net_for_miner(miner_name, event_log)
        # compute PM4Py generalization using event_log (list) or DataFrame
        pm4py_gen = generalization_eval.apply(event_log, net_full, im_full, fm_full)
    except Exception as e:
        pm4py_gen = None

    # Share the heavy objects with worker processes through module-level globals.
    global EVENT_LOG, GLOBAL_DFG, ACTIVE_MINER_NAME
    EVENT_LOG = event_log
    GLOBAL_DFG = global_dfg
    ACTIVE_MINER_NAME = miner_name

    tasks = [
        {"variant_tuple": variant_tuple, "trace_indices": trace_indices}
        for variant_tuple, trace_indices in sampled
    ]

    done_count = 0
    if NUM_WORKERS <= 1:
        # Single-process: sequential, report each as it finishes
        for task in tasks:
            task_result = evaluate_variant_task(task)
            done_count += 1
            _report_task_result(task_result, done_count, total)
            if not task_result.get("skip"):
                results.append(task_result)
    else:
        print(f"       Using {NUM_WORKERS} worker processes")
        ctx = mp.get_context("fork")
        from concurrent.futures import as_completed
        with ProcessPoolExecutor(max_workers=NUM_WORKERS, mp_context=ctx) as executor:
            future_map = {
                executor.submit(evaluate_variant_task, task): task
                for task in tasks
            }
            for future in as_completed(future_map):
                done_count += 1
                task_result = future.result()
                _report_task_result(task_result, done_count, total)
                if not task_result.get("skip"):
                    results.append(task_result)

    # Compute weighted scores
    if not results:
        return {
            "miner": miner_name,
            "score_pure": 0,
            "score_joint": 0,
            "pm4py_generalization": pm4py_gen,
            "results": [],
        }

    sum_pure_w = sum(r["pure_weight"] for r in results)
    sum_joint_w = sum(r["joint_weight"] for r in results)

    score_pure = (sum(r["replay_score"] * r["pure_weight"] for r in results)
                  / sum_pure_w) if sum_pure_w > 0 else 0
    score_joint = (sum(r["replay_score"] * r["joint_weight"] for r in results)
                   / sum_joint_w) if sum_joint_w > 0 else 0

    print(f"\n       >>> Pure Frequency Weighted Score: {score_pure:.4f}")
    print(f"       >>> Joint Variant–Transition Weighted Score: {score_joint:.4f}")

    return {
        "miner": miner_name,
        "score_pure": score_pure,
        "score_joint": score_joint,
        "pm4py_generalization": pm4py_gen,
        "results": results,
    }


def compute_baseline_metrics(event_log):
    """Compute PM4Py built-in metrics on the full log."""
    print(f"\n{'='*60}")
    print(f"[4/5] Computing PM4Py baseline metrics")
    print(f"{'='*60}")

    net, im, fm = pm4py.discover_petri_net_inductive(event_log)

    # Generalization (PM4Py built-in)
    generalization = generalization_eval.apply(
        event_log, net, im, fm
    )

    print(f"       Generalization (PM4Py): {generalization:.4f}")

    return {
        "generalization_pm4py": generalization,
    }


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
  python pick_one_out.py -t 1 -m IM

  # Small-scale validation: 10%% variants, all miners (no baseline)
  python pick_one_out.py -t 10 -m all

  # Full scale on IM only
  python pick_one_out.py -t 100 -m IM

  # PM4Py baseline only (dispatch to a separate node)
  python pick_one_out.py -m baseline

  # Multi-node dispatch:
  #   node 1: python pick_one_out.py -t 100 -m IM IMf
  #   node 2: python pick_one_out.py -t 100 -m Heuristics Alpha
  #   node 3: python pick_one_out.py -m baseline
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


def resolve_miners(requested):
    """
    Resolve CLI miner selections into two lists:
      - miners_to_run:  dict long_name -> discovery_fn  (for pick-one-out)
      - run_baseline:   bool  (run PM4Py built-in generalization)

    'all' expands to all 4 miners.  'baseline' is independent — include
    it explicitly if you want the PM4Py built-in metric alongside miners.
    This allows dispatching miners and baseline to separate nodes, e.g.:

        python pick_one_out.py -t 100 -m IM IMf    # node 1
        python pick_one_out.py -t 100 -m baseline  # node 2
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
    global NUM_WORKERS
    args = parse_args()
    NUM_WORKERS = args.workers
    t_start = time.time()

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
    print(f"     Worker processes   : {NUM_WORKERS}")
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
        "num_workers": NUM_WORKERS,
        "random_seed": RANDOM_SEED,
        "timestamp_utc": datetime.utcnow().isoformat(),
        "total_runtime_s": round(time.time() - t_start, 1),
    }
    print_summary(all_results, baseline, max_variants, total_variants,
                  output_path, run_config)

    print(f"\nTotal execution time: {time.time() - t_start:.1f}s")
    if args.test_variant < 5:
        print("💡 Results look correct? Increase -t for full evaluation, e.g.:")
        print("   python pick_one_out.py -t 100 -m IM   # full scale on IM only")
        print("   python pick_one_out.py -t 100          # full scale on all miners")


if __name__ == "__main__":
    main()
