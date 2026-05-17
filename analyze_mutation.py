"""
Mutation Distribution Analysis — Compare v1 (DFG) vs v2 (N-gram + Katz Backoff)
Tracks mutation behavior during shadow log generation to compare the two versions.
"""

import random
import time
import sys
import os
import json
from collections import defaultdict, Counter
from datetime import datetime, timezone

import numpy as np
import pm4py
from pm4py.objects.log.obj import EventLog, Trace, Event

# =====================================================================
# 0. Configuration
# =====================================================================
XES_PATH = "data/BPI-Challenge_2017/BPI Challenge 2017.xes.gz"
NUM_SHADOW_TRACES = 1000
MAX_TRACE_LENGTH = 100
SAFE_THRESHOLD = 5
SEED = 42
OUTPUT_DIR = "output"

# =====================================================================
# 1. V1 Algorithm (instrumented version, tracks mutation decisions)
# =====================================================================

def generate_shadow_log_v1_instrumented(event_log, num_traces=1000, max_trace_length=100):
    """
    V1: 1-gram DFG + Good-Turing, with instrumentation to track each mutation decision.
    """
    # Statistics
    stats = {
        "version": "v1_DFG",
        "total_decisions": 0,       # Total number of decisions made
        "mutations": 0,             # Number of mutation events (decisions)
        "mutated_traces": 0,        # Number of traces that contain ≥1 mutation
        "exploits": 0,              # Number of times historical paths were followed
        "forced_breaks": 0,         # Dead-end forced terminations
        "p_unseen_values": [],      # All P_unseen values
        "p_unseen_by_activity": defaultdict(list),  # P_unseen grouped by activity
        "trace_lengths": [],        # Lengths of generated traces
        "mutation_positions": [],   # Relative positions of mutations within traces
    }

    dfg, starts, ends = pm4py.discover_dfg(event_log)

    outgoing = defaultdict(dict)
    for (a, b), count in dfg.items():
        outgoing[a][b] = count

    alphabet = list(set([a for a, _ in dfg.keys()] + [b for _, b in dfg.keys()]))

    # Calculate P_unseen for each activity
    p_unseen = {}
    for act in alphabet:
        out_edges = outgoing[act]
        n_total = sum(out_edges.values())
        n_1 = sum(1 for target, count in out_edges.items() if count == 1)
        p_unseen[act] = (n_1 / n_total) if n_total > 0 else 1.0

    shadow_log = EventLog()
    start_choices = list(starts.keys())
    start_weights = list(starts.values())

    for i in range(num_traces):
        current = random.choices(start_choices, weights=start_weights, k=1)[0]
        trace = Trace(attributes={"concept:name": f"shadow_{i}"})
        trace.append(Event({"concept:name": current}))
        decisions_in_trace = 0
        mutations_in_trace = 0
        step = 0

        while True:
            step += 1
            # Termination check
            if current in ends:
                times_as_end = ends[current]
                total_occurrences = times_as_end + sum(outgoing[current].values())
                if random.random() < (times_as_end / total_occurrences):
                    break

            if len(trace) >= max_trace_length:
                break

            # Mutation decision
            stats["total_decisions"] += 1
            decisions_in_trace += 1
            current_p = p_unseen.get(current, 1.0)
            stats["p_unseen_values"].append(current_p)
            stats["p_unseen_by_activity"][current].append(current_p)

            if random.random() < current_p:
                # Mutation triggered!
                stats["mutations"] += 1
                mutations_in_trace += 1
                stats["mutation_positions"].append(step / max_trace_length)  # relative position
                nxt = random.choice(alphabet)
            else:
                # Follow historical path
                stats["exploits"] += 1
                out_edges = outgoing[current]
                if not out_edges:
                    stats["forced_breaks"] += 1
                    break
                nxt = random.choices(list(out_edges.keys()), weights=list(out_edges.values()), k=1)[0]

            trace.append(Event({"concept:name": nxt}))
            current = nxt

        stats["trace_lengths"].append(len(trace))
        shadow_log.append(trace)

    return shadow_log, stats


# =====================================================================
# 2. V2 Algorithm (instrumented version, tracks mutation decisions and backoffs)
# =====================================================================

def _evaluate_state_instrumented(state_tuple, ngram_dict, safe_threshold, stats):
    """V2 state evaluation, with instrumentation."""
    out_edges = ngram_dict.get(state_tuple, {})
    n_total = sum(out_edges.values())

    if n_total < safe_threshold:
        stats["backoffs_due_to_sparsity"] += 1
        return None, None

    n_1 = sum(1 for count in out_edges.values() if count == 1)
    p_unseen = (n_1 / n_total) if n_total > 0 else 1.0

    if p_unseen == 1.0:
        stats["backoffs_due_to_collapse"] += 1
        return None, None

    return p_unseen, out_edges


def generate_trace_dfs_instrumented(current_seq, ngram_outgoings, ends, alphabet, max_length, safe_threshold, max_n, stats, had_mutation=False):
    """V2 DFS generation, with instrumentation. Returns (sequence, had_mutation)."""
    current_local = current_seq[-1]

    # Termination check
    if current_local in ends:
        times_as_end = ends[current_local]
        out_edges_n1 = ngram_outgoings[1].get((current_local,), {})
        total_occurrences = times_as_end + sum(out_edges_n1.values())
        if random.random() < (times_as_end / total_occurrences):
            return current_seq, had_mutation

    if len(current_seq) >= max_length:
        return current_seq, had_mutation

    # Dynamic Katz Backoff: max_n -> max_n-1 -> ... -> 2 -> 1
    p_unseen, valid_out_edges = None, None
    ngram_level_used = 0

    # Try from highest order down to N=2, stopping at the first statistically safe state
    for n in range(max_n, 1, -1):
        if len(current_seq) >= n:
            p_unseen, valid_out_edges = _evaluate_state_instrumented(
                tuple(current_seq[-n:]), ngram_outgoings[n], safe_threshold, stats
            )
            if p_unseen is not None:
                ngram_level_used = n
                stats[f"ngram_{n}_used"] += 1
                break

    if p_unseen is None:
        # Absolute fallback to N=1
        stats["ngram_1_used"] += 1
        stats["forced_backoff_to_n1"] += 1
        out_edges = ngram_outgoings[1].get((current_local,), {})
        n_total = sum(out_edges.values())
        n_1 = sum(1 for count in out_edges.values() if count == 1)
        p_unseen = (n_1 / n_total) if n_total > 0 else 1.0
        valid_out_edges = out_edges
        ngram_level_used = 1

    # Mutation decision
    stats["total_decisions"] += 1
    stats["p_unseen_values"].append(p_unseen)
    stats["ngram_level_distribution"].append(ngram_level_used)

    if random.random() < p_unseen:
        # Mutation triggered!
        stats["mutations"] += 1
        stats["mutation_positions"].append(len(current_seq) / max_length)
        next_node = random.choice(alphabet)
        had_mutation = True
    else:
        # Follow historical path
        stats["exploits"] += 1
        if not valid_out_edges:
            stats["forced_breaks"] += 1
            return current_seq, had_mutation
        next_node = random.choices(
            list(valid_out_edges.keys()), weights=list(valid_out_edges.values()), k=1
        )[0]

    return generate_trace_dfs_instrumented(
        current_seq + [next_node], ngram_outgoings, ends, alphabet, max_length, safe_threshold, max_n, stats, had_mutation
    )


def generate_shadow_log_v2_instrumented(event_log, num_traces=1000, max_trace_length=100, safe_threshold=5, max_n=3):
    """
    V2: N-gram + Katz Backoff, with instrumentation.
    
    Args:
        max_n: Maximum N-gram order. N-gram stats built for N=1..max_n.
    """
    # Initialize dynamic stats for all N-gram levels
    stats = {
        "version": f"v2_Ngram_Katz_N{max_n}",
        "max_n": max_n,
        "total_decisions": 0,
        "mutations": 0,
        "mutated_traces": 0,
        "exploits": 0,
        "forced_breaks": 0,
        "p_unseen_values": [],
        "trace_lengths": [],
        "mutation_positions": [],
        # V2-specific (dynamic keys per N level)
        "backoffs_due_to_sparsity": 0,
        "backoffs_due_to_collapse": 0,
        "forced_backoff_to_n1": 0,
        "ngram_level_distribution": [],
    }
    # Initialize per-N usage counters
    for n in range(1, max_n + 1):
        stats[f"ngram_{n}_used"] = 0

    # Build N-gram statistics for N=1..max_n
    ngram_outgoings = {n: defaultdict(Counter) for n in range(1, max_n + 1)}
    starts = Counter()
    ends = Counter()
    alphabet = set()

    for trace in event_log:
        seq = [event["concept:name"] for event in trace]
        if not seq:
            continue
        starts[seq[0]] += 1
        ends[seq[-1]] += 1
        alphabet.update(seq)
        trace_len = len(seq)
        for i in range(trace_len - 1):
            nxt_act = seq[i + 1]
            for n in range(1, max_n + 1):
                if i >= n - 1:
                    state_tuple = tuple(seq[i - (n - 1) : i + 1])
                    ngram_outgoings[n][state_tuple][nxt_act] += 1

    alphabet = list(alphabet)
    shadow_log = EventLog()
    start_choices = list(starts.keys())
    start_weights = list(starts.values())

    for i in range(num_traces):
        start_node = random.choices(start_choices, weights=start_weights, k=1)[0]
        final_sequence, had_mutation = generate_trace_dfs_instrumented(
            [start_node], ngram_outgoings, ends, alphabet, max_trace_length, safe_threshold, max_n, stats
        )
        trace = Trace(attributes={"concept:name": f"shadow_{i}"})
        for act in final_sequence:
            trace.append(Event({"concept:name": act}))
        stats["trace_lengths"].append(len(trace))
        if had_mutation:
            stats["mutated_traces"] += 1
        shadow_log.append(trace)

    return shadow_log, stats


# =====================================================================
# 3. Analysis & Output
# =====================================================================

def analyze_stats(stats):
    """Analyze collected statistics."""
    analysis = {}

    total = stats["total_decisions"]
    if total > 0:
        analysis["mutation_rate"] = stats["mutations"] / total
        analysis["exploit_rate"] = stats["exploits"] / total
    else:
        analysis["mutation_rate"] = 0.0
        analysis["exploit_rate"] = 0.0

    # Mutated trace count and ratio
    num_traces = len(stats["trace_lengths"])
    analysis["mutated_trace_count"] = stats.get("mutated_traces", 0)
    analysis["mutated_trace_ratio"] = (analysis["mutated_trace_count"] / num_traces) if num_traces > 0 else 0.0
    analysis["total_traces"] = num_traces

    p_vals = stats["p_unseen_values"]
    if p_vals:
        analysis["p_unseen_mean"] = np.mean(p_vals)
        analysis["p_unseen_median"] = np.median(p_vals)
        analysis["p_unseen_std"] = np.std(p_vals)
        analysis["p_unseen_min"] = np.min(p_vals)
        analysis["p_unseen_max"] = np.max(p_vals)
        # Histogram (10 bins)
        hist, bin_edges = np.histogram(p_vals, bins=10, range=(0, 1))
        analysis["p_unseen_histogram"] = {
            "bins": bin_edges.tolist(),
            "counts": hist.tolist()
        }
        # Bracket statistics
        analysis["p_unseen_low"] = sum(1 for p in p_vals if p < 0.2) / len(p_vals)     # Low mutation
        analysis["p_unseen_mid"] = sum(1 for p in p_vals if 0.2 <= p < 0.5) / len(p_vals)  # Mid mutation
        analysis["p_unseen_high"] = sum(1 for p in p_vals if 0.5 <= p < 0.8) / len(p_vals) # High mutation
        analysis["p_unseen_extreme"] = sum(1 for p in p_vals if p >= 0.8) / len(p_vals)    # Extreme mutation

    # Trace lengths
    tls = stats["trace_lengths"]
    if tls:
        analysis["trace_length_mean"] = np.mean(tls)
        analysis["trace_length_median"] = np.median(tls)
        analysis["trace_length_std"] = np.std(tls)
        analysis["trace_length_min"] = np.min(tls)
        analysis["trace_length_max"] = np.max(tls)

    # Mutation position distribution
    mps = stats["mutation_positions"]
    if mps:
        analysis["mutation_position_mean"] = np.mean(mps)

    # V2-specific: dynamic N-gram level usage
    max_n = stats.get("max_n", 3)
    if any(f"ngram_{n}_used" in stats for n in range(1, max_n + 1)):
        total_ngram = sum(stats.get(f"ngram_{n}_used", 0) for n in range(1, max_n + 1))
        if total_ngram > 0:
            analysis["ngram_ratios"] = {
                n: stats.get(f"ngram_{n}_used", 0) / total_ngram for n in range(1, max_n + 1)
            }
        analysis["max_n"] = max_n
        analysis["backoffs_sparsity"] = stats.get("backoffs_due_to_sparsity", 0)
        analysis["backoffs_collapse"] = stats.get("backoffs_due_to_collapse", 0)
        analysis["forced_to_n1"] = stats.get("forced_backoff_to_n1", 0)

    analysis["forced_breaks"] = stats["forced_breaks"]
    analysis["total_decisions"] = total

    return analysis


def print_comparison(analysis_v1, analyses_v2_dict):
    """
    Print comparison results.
    
    Args:
        analysis_v1: V1 analysis dict (single).
        analyses_v2_dict: dict of {max_n: analysis_dict} for V2 at different N-gram orders.
    """
    print("\n" + "=" * 90)
    print("  Mutation Distribution Comparison: V1 (DFG) vs V2 (N-gram + Katz Backoff)")
    print("=" * 90)

    # ── Table 1: Mutation Overview across N ──
    max_ns = sorted(analyses_v2_dict.keys())
    col_width = 14
    header_cols = " | ".join([f"{'V1':>{col_width}}"] + [f"{'V2 N=' + str(n):>{col_width}}" for n in max_ns])
    
    print(f"\n{'─' * (22 + (col_width + 3) * (len(max_ns) + 1))}")
    print(f"  📊 Mutation Rate vs. N-gram Order")
    print(f"{'─' * (22 + (col_width + 3) * (len(max_ns) + 1))}")
    print(f"  {'Metric':<22} | {header_cols}")
    print(f"  {'─' * (20 + (col_width + 3) * (len(max_ns) + 1))}")
    
    rows = [
        ("Total Decisions", "total_decisions"),
        ("Mutation Rate", "mutation_rate"),
        ("Mutated Traces", "mutated_trace_count"),
        ("Mutated Trace %", "mutated_trace_ratio"),
        ("P_unseen Mean", "p_unseen_mean"),
        ("P_unseen Max", "p_unseen_max"),
        ("P_unseen Std", "p_unseen_std"),
        ("Mean Trace Length", "trace_length_mean"),
    ]
    
    for label, key in rows:
        v1_val = analysis_v1.get(key, 0)
        v2_vals = [analyses_v2_dict[n].get(key, 0) for n in max_ns]
        
        if key == "total_decisions":
            v1_str = f"{v1_val:>{col_width}}"
            v2_strs = [f"{v:>{col_width}}" for v in v2_vals]
        elif key == "mutated_trace_count":
            total = analysis_v1.get("total_traces", 1000)
            v1_str = f"{v1_val:>{col_width}}"
            v2_strs = [f"{v:>{col_width}}" for v in v2_vals]
        elif key == "mutated_trace_ratio":
            v1_str = f"{v1_val:>{col_width}.2%}"
            v2_strs = [f"{v:>{col_width}.2%}" for v in v2_vals]
        elif key in ("mutation_rate", "p_unseen_mean", "p_unseen_max", "p_unseen_std"):
            v1_str = f"{v1_val:>{col_width}.6f}"
            v2_strs = [f"{v:>{col_width}.6f}" for v in v2_vals]
        else:
            v1_str = f"{v1_val:>{col_width}.2f}"
            v2_strs = [f"{v:>{col_width}.2f}" for v in v2_vals]
        
        all_vals = " | ".join([v1_str] + v2_strs)
        print(f"  {label:<22} | {all_vals}")

    # ── Table 2: N-gram Level Usage (V2 only) ──
    print(f"\n{'─' * 60}")
    print(f"  🔬 V2 N-gram Level Usage Distribution")
    print(f"{'─' * 60}")
    header = f"  {'max_n':<8}"
    for n in range(1, max(max_ns) + 1):
        header += f" | {'N=' + str(n):>8}"
    header += f" | {'Backoffs':>9} | {'→N1':>6}"
    print(header)
    separator_len = 55 + 12 * max(max_ns)
    print(f"  {'─' * separator_len}")
    
    for mn in max_ns:
        a = analyses_v2_dict[mn]
        ratios = a.get("ngram_ratios", {})
        row = f"  {mn:<8}"
        for n in range(1, max(max_ns) + 1):
            r = ratios.get(n, 0) if n <= mn else None
            if r is not None:
                row += f" | {r:>7.1%}"
            else:
                row += f" | {'—':>8}"
        row += f" | {a.get('backoffs_sparsity', 0):>9} | {a.get('forced_to_n1', 0):>6}"
        print(row)

    # ── Key Insights ──
    print(f"\n{'─' * 60}")
    print(f"  💡 Key Insights")
    print(f"{'─' * 60}")
    
    v1_rate = analysis_v1["mutation_rate"]
    for mn in max_ns:
        v2_rate = analyses_v2_dict[mn]["mutation_rate"]
        ratio = v2_rate / v1_rate if v1_rate > 0 else float('inf')
        v2_traces = analyses_v2_dict[mn]["mutated_trace_count"]
        print(f"  N={mn}: mutation_rate={v2_rate:.6f} | mutated_traces={v2_traces} | {ratio:.0f}× V1")
    
    # Best N recommendation
    best_n = max(max_ns, key=lambda n: analyses_v2_dict[n]["mutated_trace_count"])
    print(f"  ✅ Highest exploration: N={best_n} ({analyses_v2_dict[best_n]['mutated_trace_count']} mutated traces)")


# =====================================================================
# 4. Main
# =====================================================================

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    print("=" * 80)
    print("  Mutation Distribution Analysis — N-gram Order Sweep")
    print(f"  Shadow Traces: {NUM_SHADOW_TRACES} | Safe Threshold: {SAFE_THRESHOLD} | Seed: {SEED}")
    N_SWEEP_LIST = [1, 2, 3, 4, 5, 6, 7, 8]
    print(f"  V2 N-gram sweep: {N_SWEEP_LIST}")
    print("=" * 80)

    # Load log
    print("\n[1/4] Loading event log...")
    if os.path.exists(XES_PATH):
        event_log = pm4py.read_xes(XES_PATH)
    else:
        print("  ⚠️  Log not found, generating dummy data...")
        import pandas as pd
        df = pd.DataFrame({
            'case:concept:name': ['1','1','1','2','2','2','3','3','4','4','4','4'],
            'concept:name': ['A','B','C','A','X','C','A','B','A','B','D','E']
        })
        event_log = pm4py.format_dataframe(df, case_id='case:concept:name', activity_key='concept:name')
    event_log = pm4py.convert_to_event_log(event_log)
    print(f"  Loaded: {len(event_log)} traces | {sum(len(t) for t in event_log)} events")

    # Generate V1 shadow log + stats (once)
    print(f"\n[2/4] Generating V1 (DFG) shadow log + stats...")
    random.seed(SEED)
    np.random.seed(SEED)
    t0 = time.time()
    _, stats_v1 = generate_shadow_log_v1_instrumented(
        event_log, num_traces=NUM_SHADOW_TRACES, max_trace_length=MAX_TRACE_LENGTH
    )
    print(f"  Done in {time.time()-t0:.1f}s | {stats_v1['total_decisions']} decisions | {stats_v1['mutated_traces']} mutated traces")

    # Generate V2 shadow log + stats for multiple max_n values
    all_stats_v2 = {}
    
    print(f"\n[3/4] Generating V2 (N-gram + Katz) shadow logs for N={N_SWEEP_LIST}...")
    for max_n in N_SWEEP_LIST:
        random.seed(SEED)
        np.random.seed(SEED)
        t0 = time.time()
        _, stats_v2 = generate_shadow_log_v2_instrumented(
            event_log, num_traces=NUM_SHADOW_TRACES, max_trace_length=MAX_TRACE_LENGTH,
            safe_threshold=SAFE_THRESHOLD, max_n=max_n
        )
        all_stats_v2[max_n] = stats_v2
        print(f"  N={max_n}: {time.time()-t0:.1f}s | {stats_v2['total_decisions']} decisions | {stats_v2['mutated_traces']} mutated traces")

    # Analyze
    print(f"\n[4/4] Analyzing...")
    analysis_v1 = analyze_stats(stats_v1)
    analyses_v2 = {max_n: analyze_stats(stats) for max_n, stats in all_stats_v2.items()}

    print_comparison(analysis_v1, analyses_v2)

    # Export JSON
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"mutation_analysis_{timestamp}.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "num_shadow_traces": NUM_SHADOW_TRACES,
                "safe_threshold": SAFE_THRESHOLD,
                "seed": SEED,
                "max_trace_length": MAX_TRACE_LENGTH,
                "n_sweep": N_SWEEP_LIST,
            },
            "v1": {"raw_stats": {k: v for k, v in stats_v1.items() if k != "p_unseen_by_activity"}, "analysis": analysis_v1},
            "v2": {str(n): {"raw_stats": {k: v for k, v in s.items() if k not in ("p_unseen_by_activity",)}, "analysis": analyses_v2[n]} for n, s in all_stats_v2.items()},
        }, f, indent=2, default=lambda x: list(x) if isinstance(x, np.ndarray) else str(x))
    print(f"  Full results exported to: {out_path}")


if __name__ == "__main__":
    main()
