"""
Pick-One-Out Generalization Metric — Core Algorithm
====================================================
Based on BPI Challenge 2017 event log.

This module contains the pure algorithmic components:
  - Event log loading & preprocessing
  - Variant computation & global DFG construction
  - Weighting functions (pure frequency & joint variant–transition)
  - Petri net discovery via multiple miners
  - Trace replay (token-based + alignment fallback)
  - Pick-one-out evaluation loop (single-process & multiprocessing)
  - PM4Py baseline metric computation

It does NOT contain CLI argument parsing, output formatting, or
experiment orchestration — those live in pick_one_out_experiment.py.
"""

import pm4py
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from math import log, ceil
from datetime import datetime
import pandas as pd
import numpy as np

from pm4py.algo.evaluation.generalization import algorithm as generalization_eval
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness


# ─── Module-level globals shared with worker processes (via fork) ──────────
# Using multiprocessing (NOT multithreading) because Petri net discovery
# and replay fitness are CPU-bound; threading would hit the GIL.

ORIGINAL_DF = None          # DataFrame of the full event log
CASE_IDS_ORDERED = None     # Ordered list of case IDs matching EVENT_LOG
EVENT_LOG = None            # Full event log (list of traces)
GLOBAL_DFG = None           # Global Directly-Follows Graph: edge -> freq
ACTIVE_MINER_NAME = None    # Currently active miner name (set before workers)
NUM_WORKERS = 1             # Number of worker processes (set from CLI)


# ─── Log Loading & Preprocessing ────────────────────────────────────────────

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


# ─── Variant & DFG Computation ─────────────────────────────────────────────

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


# ─── Weighting Functions ────────────────────────────────────────────────────

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


# ─── Petri Net Discovery ────────────────────────────────────────────────────

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


# ─── Trace Replay ───────────────────────────────────────────────────────────

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


# ─── Pick-One-Out Evaluation ────────────────────────────────────────────────

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

    Parameters
    ----------
    event_log : list of traces
        The full event log.
    variants : dict
        variant_tuple -> list of trace indices.
    global_dfg : dict
        Global DFG: edge -> frequency.
    miner_name : str
        Human-readable miner name (e.g. "Inductive Miner (IM)").
    miner_fn : callable
        Discovery function (kept for API compatibility; discovery is
        actually dispatched via ACTIVE_MINER_NAME + discover_petri_net_for_miner).
    max_variants : int
        Maximum number of top-frequency variants to sample.

    Returns
    -------
    dict with keys: miner, score_pure, score_joint, pm4py_generalization, results
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


# ─── PM4Py Baseline Metrics ─────────────────────────────────────────────────

def compute_baseline_metrics(event_log):
    """Compute PM4Py built-in metrics on the full log using Inductive Miner."""
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
