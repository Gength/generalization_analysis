#!/usr/bin/env python3
"""
Generate summary statistics for the BPI Challenge 2017 event log.

Reads 'BPI Challenge 2017.xes.gz' and produces summary.txt containing:
  - Total case count & event count
  - Unique activities and trace variants
  - Trace variant frequency distribution
  - TLRA (Log Representativeness)
  - Top 10 trace variants
"""

import os
from collections import Counter

import pm4py


def generate_summary(xes_path: str, output_path: str) -> None:
    # ── 1. Read the event log ──────────────────────────────────────────
    print(f"Reading {xes_path}...")
    from pm4py.objects.log.importer.xes import importer as xes_importer
    event_log = xes_importer.apply(xes_path)

    num_cases = len(event_log)
    num_events = sum(len(trace) for trace in event_log)

    # ── 2. Extract variants (ordered sequences of activity names) ──────
    variant_counter = Counter()
    activities_set = set()

    for trace in event_log:
        seq = tuple(event["concept:name"] for event in trace)
        variant_counter[seq] += 1
        activities_set.update(seq)

    num_activities = len(activities_set)
    num_variants = len(variant_counter)

    # ── 3. Frequency distribution stats & TLRA ─────────────────────────
    freqs = list(variant_counter.values())
    min_freq = min(freqs)
    max_freq = max(freqs)
    mean_freq = sum(freqs) / len(freqs)
    singletons = sum(1 for f in freqs if f == 1)
    freq_le_5 = sum(1 for f in freqs if f <= 5)
    
    # Calculate TLRA: 1 - (|lang(L)| / |L|)
    tlra = 1.0 - (num_variants / num_cases) if num_cases > 0 else 0.0

    # ── 4. Top 10 variants ─────────────────────────────────────────────
    top10 = variant_counter.most_common(10)

    # ── 5. Write summary ───────────────────────────────────────────────
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
        
        f.write("Trace variant frequency distribution:\n")
        f.write(f"  Min freq: {min_freq}\n")
        f.write(f"  Max freq: {max_freq}\n")
        f.write(f"  Mean freq: {mean_freq:.1f}\n")
        f.write(f"  Singletons (freq=1): {singletons:,}\n")
        f.write(f"  freq<=5: {freq_le_5:,}\n")
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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    xes_file = os.path.join(base_dir, "BPI Challenge 2017.xes.gz")
    summary_file = os.path.join(base_dir, "summary.txt")
    generate_summary(xes_file, summary_file)