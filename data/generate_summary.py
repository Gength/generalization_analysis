#!/usr/bin/env python3
"""
Generate summary statistics for the event log.

Reads 'X.xes.gz' and produces summary.txt containing:
  - Total case count & event count
  - Unique activities and trace variants
  - Trace variant frequency distribution
  - TLRA (Log Representativeness)
  - N-gram Sparsity & Good-Turing Mutation Analysis
  - Top 10 trace variants
"""

import os
import statistics
import time
from collections import Counter, defaultdict

import pm4py


def analyze_ngram_sparsity(variant_counter: Counter, max_n: int = 4, safe_threshold: int = 5) -> str:
    """
    Analyze data sparsity and Good-Turing mutation probability across different N-gram levels.
    
    Args:
        variant_counter: Counter mapping variant tuples to their absolute frequencies.
        max_n: Maximum lookback level (e.g., 3 for Regional Marking, 4 for further analysis).
        safe_threshold: Minimum occurrences to consider a state as having 'sufficient historical data'.
        
    Returns:
        A formatted string containing the diagnostic table to be written to the summary.
    """
    print(f"Analyzing N-gram Sparsity up to N={max_n} ...")
    
    # Data structure: ngram_outgoings[N][state_tuple][next_activity] = frequency
    ngram_outgoings = {n: defaultdict(Counter) for n in range(1, max_n + 1)}
    
    # 1. Iterate through all variants to extract N-gram outgoing distributions
    for variant_tuple, freq in variant_counter.items():
        trace_len = len(variant_tuple)
        
        for i in range(trace_len - 1):
            nxt_act = variant_tuple[i + 1]
            
            # Extract states from 1 to max_n order
            for n in range(1, max_n + 1):
                if i >= n - 1:
                    # Extract the current activity and its previous (n-1) activities to form the state tuple
                    state_tuple = variant_tuple[i - (n - 1) : i + 1]
                    ngram_outgoings[n][state_tuple][nxt_act] += freq

    # 2. Calculate evaluation metrics and format the output table
    lines = []
    lines.append("=" * 85)
    lines.append(f"{'N-gram (Order)':<15} | {'Unique States':<15} | {'Safe States (>'+str(safe_threshold)+')':<20} | {'Avg P_unseen':<12} | {'Collapse (P=1.0)':<15}")
    lines.append("=" * 85)

    for n in range(1, max_n + 1):
        state_dict = ngram_outgoings[n]
        total_unique_states = len(state_dict)
        
        if total_unique_states == 0:
            continue
            
        safe_states_count = 0
        collapse_count = 0
        p_unseen_list = []
        
        for state, out_edges in state_dict.items():
            n_total = sum(out_edges.values())
            
            # N_1: Number of outgoing edges that occurred exactly once (Singletons)
            n_1 = sum(1 for target, count in out_edges.items() if count == 1)
            
            # Calculate Good-Turing mutation probability (P_unseen)
            p_unseen = (n_1 / n_total) if n_total > 0 else 1.0
            p_unseen_list.append(p_unseen)
            
            if n_total >= safe_threshold:
                safe_states_count += 1
                
            if p_unseen == 1.0:
                collapse_count += 1
                
        avg_p_unseen = sum(p_unseen_list) / len(p_unseen_list) if p_unseen_list else 0.0
        safe_ratio = (safe_states_count / total_unique_states * 100) if total_unique_states > 0 else 0.0
        collapse_ratio = (collapse_count / total_unique_states * 100) if total_unique_states > 0 else 0.0
        
        lines.append(f"N={n} (Lookback {n-1}) | {total_unique_states:<15} | {safe_states_count:<7} ({safe_ratio:>5.1f}%) | {avg_p_unseen:>10.2f} | {collapse_count:<5} ({collapse_ratio:>5.1f}%)")
    
    lines.append("=" * 85)
    return "\n".join(lines)


def generate_summary(xes_path: str, output_path: str) -> None:
    # ── 1. Read the event log ──────────────────────────────────────────
    print(f"Reading {xes_path}...")
    t_start = time.time()
    from pm4py.objects.log.importer.xes import importer as xes_importer
    event_log = xes_importer.apply(xes_path)

    num_cases = len(event_log)
    num_events = sum(len(trace) for trace in event_log)
    print(f"  -> Loaded {num_cases} cases, {num_events} events in {time.time() - t_start:.1f}s")

    # ── 2. Extract variants (ordered sequences of activity names) ──────
    variant_counter = Counter()
    activities_set = set()

    trace_lengths = []
    for trace in event_log:
        seq = tuple(event["concept:name"] for event in trace)
        variant_counter[seq] += 1
        activities_set.update(seq)
        trace_lengths.append(len(seq))

    num_activities = len(activities_set)
    num_variants = len(variant_counter)

    # ── 3. Trace length distribution ───────────────────────────────────
    trace_lengths.sort()
    tl_min = trace_lengths[0]
    tl_max = trace_lengths[-1]
    tl_mean = statistics.mean(trace_lengths)
    tl_median = statistics.median(trace_lengths)
    tl_stdev = statistics.stdev(trace_lengths)
    # Percentiles via linear interpolation
    def _percentile(sorted_data, p):
        k = (len(sorted_data) - 1) * p / 100
        f = int(k)
        c = k - f
        if f + 1 < len(sorted_data):
            return sorted_data[f] + c * (sorted_data[f + 1] - sorted_data[f])
        return sorted_data[f]
    tl_p25 = _percentile(trace_lengths, 25)
    tl_p75 = _percentile(trace_lengths, 75)
    tl_p90 = _percentile(trace_lengths, 90)
    tl_p95 = _percentile(trace_lengths, 95)
    tl_p99 = _percentile(trace_lengths, 99)

    # ── 4. Frequency distribution stats & TLRA ─────────────────────────
    freqs = list(variant_counter.values())
    min_freq = min(freqs)
    max_freq = max(freqs)
    mean_freq = sum(freqs) / len(freqs)
    singletons = sum(1 for f in freqs if f == 1)
    freq_le_5 = sum(1 for f in freqs if f <= 5)
    
    # Calculate TLRA: 1 - (|lang(L)| / |L|)
    tlra = 1.0 - (num_variants / num_cases) if num_cases > 0 else 0.0

    # ── 5. N-gram Sparsity Analysis ────────────────────────────────────
    ngram_report_str = analyze_ngram_sparsity(variant_counter, max_n=4, safe_threshold=5)

    # ── 6. Top 10 variants ─────────────────────────────────────────────
    top10 = variant_counter.most_common(10)

    # ── 7. Write summary ───────────────────────────────────────────────
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Cases: {num_cases}\n")
        f.write(f"Events: {num_events:,}\n")
        f.write("\n")
        f.write("A trace variant is defined as a unique, ordered sequence of "
                "activities executed from the beginning to the end of a process case.\n")
        f.write("\n")
        f.write(f"Unique activities: {num_activities}\n")
        f.write("\n")
        f.write(f"Unique trace variants: {num_variants:,}\n")
        f.write("\n")
        
        # TLRA Output Section
        f.write("Log Representativeness:\n")
        f.write(f"  TLRA (Trace-Based Log Representativeness Approx.): {tlra:.4f}\n")
        f.write("  * Evaluates the probability that an additional trace has been seen previously in the log.\n")
        f.write("  * Higher values (closer to 1.0) indicate the log already highly represents the system.\n")
        f.write("\n")
        
        f.write("Trace length distribution (events per case):\n")
        f.write(f"  Min:      {tl_min}\n")
        f.write(f"  Max:      {tl_max}\n")
        f.write(f"  Mean:     {tl_mean:.1f}\n")
        f.write(f"  Median:   {tl_median:.1f}\n")
        f.write(f"  Std:      {tl_stdev:.1f}\n")
        f.write(f"  P25:      {tl_p25:.1f}\n")
        f.write(f"  P75:      {tl_p75:.1f}\n")
        f.write(f"  P90:      {tl_p90:.1f}\n")
        f.write(f"  P95:      {tl_p95:.1f}\n")
        f.write(f"  P99:      {tl_p99:.1f}\n")
        f.write("\n")
        f.write("Trace variant frequency distribution:\n")
        f.write(f"  Min freq: {min_freq}\n")
        f.write(f"  Max freq: {max_freq}\n")
        f.write(f"  Mean freq: {mean_freq:.1f}\n")
        f.write(f"  Singletons (freq=1): {singletons:,}\n")
        f.write(f"  freq<=5: {freq_le_5:,}\n")
        f.write("\n")
        
        # N-gram Sparsity Section
        f.write("N-gram State Sparsity & Good-Turing Mutation Probability:\n")
        f.write(ngram_report_str + "\n")
        f.write("\n[Diagnostic Guide]\n")
        f.write("1. Safe States: % of states with total outgoing frequency > threshold (default 5).\n")
        f.write("   -> If very low at N=3, it means most 3-level states lack sufficient historical support.\n")
        f.write("2. Avg P_unseen: Average Good-Turing mutation probability.\n")
        f.write("   -> If this spikes dramatically at higher N, the Shadow Log will degrade into a random walk.\n")
        f.write("3. Collapse (P=1.0): % of states where the mutation probability is 100%.\n")
        f.write("   -> This indicates total data sparsity: all historical paths from this state only occurred once.\n")
        f.write("\n")
        
        f.write("Top 10 trace variants by frequency:\n")
        for rank, (trace, freq) in enumerate(top10, 1):
            # Truncate long traces with "..." like the original
            trace_str = str(trace)
            if len(trace) > 15:
                trace_str = str(trace[:15]) + "..."
            f.write(f"  Variant {rank}: freq={freq}, trace={trace_str}\n")

    print(f"Summary written to {output_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname("./BPI-Challenge_2020/")
    xes_file = os.path.join(base_dir, "RequestForPayment.xes.gz")
    summary_file = os.path.join(base_dir, "RequestForPayment_summary.txt")
    generate_summary(xes_file, summary_file)