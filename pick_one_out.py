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
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from math import log
import pandas as pd
import numpy as np

from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.evaluation.generalization import algorithm as generalization_eval
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness

# ─── Configuration ───────────────────────────────────────────────────────────

XES_PATH = "data/BPI-Challenge_2017/BPI Challenge 2017.xes.gz"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Miners to evaluate
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

# Sampling: limit number of variants for feasibility
# BPI 2017 has 15,930 variants — we sample up to MAX_VARIANTS
# Default reduced for quicker test runs; increase for full experiments.
MAX_VARIANTS = 15930  # adjust for speed vs. coverage trade-off
RANDOM_SEED = 42
# Use multiple worker processes for the expensive leave-one-out loop.
# Keep this below the total core count to avoid oversubscription.
NUM_WORKERS = 1
ORIGINAL_DF = None
CASE_IDS_ORDERED = None
EVENT_LOG = None
GLOBAL_DFG = None
ACTIVE_MINER_NAME = None

# Module-level globals shared with worker processes.
EVENT_LOG = None
GLOBAL_DFG = None
ACTIVE_MINER_NAME = None
ORIGINAL_DF = None

# ─── Helper Functions ────────────────────────────────────────────────────────


def load_and_prepare_log(path):
    """Load XES and convert to EventLog (list of traces)."""
    print(f"[1/5] Loading event log from {path} ...")
    t0 = time.time()
    df = pm4py.read_xes(path)
    event_log = log_converter.apply(df, variant=log_converter.Variants.TO_EVENT_LOG)
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
    """Replay a single trace on the given Petri net using alignment."""
    try:
        fitness_result = replay_fitness.apply(
            [trace], net, im, fm,
            variant=replay_fitness.Variants.ALIGNMENT_BASED
        )
        return fitness_result.get("averageFitness",
                                  fitness_result.get("average_trace_fitness", 0.0))
    except Exception:
        # If alignment fails, fall back to token-based replay
        try:
            fitness_result = replay_fitness.apply(
                [trace], net, im, fm,
                variant=replay_fitness.Variants.TOKEN_BASED
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
    # determine case ids to remove (use case:concept:name from representative trace)
    removed_case_ids = set()
    if CASE_IDS_ORDERED is not None:
        for idx in trace_indices:
            if idx < len(CASE_IDS_ORDERED):
                removed_case_ids.add(CASE_IDS_ORDERED[idx])

    # Build remaining dataframe by filtering ORIGINAL_DF by case id
    if ORIGINAL_DF is not None and len(removed_case_ids) > 0:
        remaining_df = ORIGINAL_DF[~ORIGINAL_DF["case:concept:name"].isin(removed_case_ids)]
        # check remaining number of cases (not raw event rows)
        remaining_cases = remaining_df["case:concept:name"].nunique()
        if remaining_cases < 10:
            return {
                "skip": True,
                "reason": "insufficient remaining traces",
                "variant": " → ".join(variant_tuple),
            }
    else:
        # fallback to list-based remaining log
        trace_index_set = set(trace_indices)
        remaining_log = [trace for idx, trace in enumerate(EVENT_LOG)
                         if idx not in trace_index_set]
        if len(remaining_log) < 10:
            return {
                "skip": True,
                "reason": "insufficient remaining traces",
                "variant": " → ".join(variant_tuple),
            }
        net, im, fm = discover_petri_net_for_miner(ACTIVE_MINER_NAME, remaining_log)
        representative_trace = EVENT_LOG[trace_indices[0]]
        replay_score = replay_trace_on_model(representative_trace, net, im, fm)

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
        return {
            "skip": True,
            "reason": "insufficient remaining traces",
            "variant": " → ".join(variant_tuple),
        }

    try:
        # Prefer the DataFrame path for discovery if available
        if 'remaining_df' in locals():
            net, im, fm = discover_petri_net_for_miner(ACTIVE_MINER_NAME, remaining_df)
        else:
            net, im, fm = discover_petri_net_for_miner(ACTIVE_MINER_NAME, remaining_log)
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


def evaluate_miner(event_log, variants, global_dfg, miner_name, miner_fn):
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
    sampled = sorted_variants[:MAX_VARIANTS]
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

    if NUM_WORKERS <= 1:
        task_results = [evaluate_variant_task(task) for task in tasks]
    else:
        print(f"       Using {NUM_WORKERS} worker processes")
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=NUM_WORKERS, mp_context=ctx) as executor:
            task_results = list(executor.map(evaluate_variant_task, tasks, chunksize=1))

    for rank, task_result in enumerate(task_results):
        if task_result.get("skip"):
            if (rank + 1) % 20 == 0 or rank == 0:
                print(f"       [{rank+1}/{total}] Skipped variant ({task_result.get('reason')})")
            continue

        results.append(task_result)

        if (rank + 1) % 20 == 0 or rank == 0:
            print(f"       [{rank+1}/{total}] freq={task_result['freq']}, "
                  f"replay={task_result['replay_score']:.3f}, "
                  f"pure_w={task_result['pure_weight']:.3f}, "
                  f"joint_w={task_result['joint_weight']:.3f}")

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


def print_summary(all_results, baseline):
    """Print final comparison table."""
    print(f"\n{'='*60}")
    print(f"[5/5] FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Miner':<30} {'Method1 Pure':>12} {'Method1 Joint':>14} {'PM4Py Gen':>12}")
    print("-" * 74)
    for r in all_results:
        pm4py_gen = f"{r['pm4py_generalization']:.4f}" if r.get('pm4py_generalization') is not None else "N/A"
        print(f"{r['miner']:<30} {r['score_pure']:>12.4f} "
              f"{r['score_joint']:>14.4f} {pm4py_gen:>12}")

    # Save results
    output = {
        "dataset": "BPI Challenge 2017",
        "num_variants_sampled": MAX_VARIANTS,
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
        "detailed_results": all_results,
    }
    path = os.path.join(OUTPUT_DIR, "pick_one_out_results.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    print(f"Using up to {NUM_WORKERS} worker processes for leave-one-out evaluation.")

    # 1. Load log
    event_log = load_and_prepare_log(XES_PATH)

    # 2. Compute variants & global DFG
    variants = compute_variants(event_log)
    global_dfg = build_global_dfg(event_log)
    print(f"       Global DFG has {len(global_dfg)} unique edges")

    # Print variant stats
    freqs = [len(v) for v in variants.values()]
    print(f"       Variant stats — min_freq={min(freqs)}, max_freq={max(freqs)}, "
          f"mean_freq={sum(freqs)/len(freqs):.1f}, "
          f"singletons={sum(1 for f in freqs if f == 1)}")

    # 3. Evaluate each miner
    all_results = []
    for miner_name, miner_fn in MINERS.items():
        result = evaluate_miner(event_log, variants, global_dfg, miner_name, miner_fn)
        all_results.append(result)

    # 4. Baseline
    baseline = compute_baseline_metrics(event_log)

    # 5. Summary
    print_summary(all_results, baseline)

    print(f"\nTotal execution time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
