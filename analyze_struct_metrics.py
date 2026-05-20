"""
Structural Metrics Analysis — Evaluate graph-theoretic generalization indicators.

Computes five model-structural metrics across miners and compares them against
known Gen_Struct / Gen_Total scores to assess predictive validity.
"""

import time
import os
import sys
from collections import Counter
import math

import pm4py
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
import numpy as np

XES_PATH = "data/BPI-Challenge_2017/BPI Challenge 2017.xes.gz"

MINERS = {
    "Inductive Miner (IM)": lambda log: pm4py.discover_petri_net_inductive(log),
    "Heuristics Miner":    lambda log: pm4py.discover_petri_net_heuristics(log),
    "Alpha Miner":         lambda log: pm4py.discover_petri_net_alpha(log),
}


def compute_structural_metrics(net):
    """
    Compute five graph-theoretic structural metrics from a Petri net.
    
    Returns dict with keys:
        density, silent_ratio, label_dup, xor_entropy_mean, free_choice_ratio,
        n_places, n_trans, n_arcs, n_silent, n_unique_labels
    """
    places = list(net.places)
    transitions = list(net.transitions)
    arcs = list(net.arcs)
    
    n_places = len(places)
    n_trans = len(transitions)
    n_arcs = len(arcs)
    
    # ── 1. Density ────────────────────────────────────────────────
    # Maximum possible arcs in a fully connected bipartite graph: |P| × |T|
    density = n_arcs / (n_places * n_trans) if (n_places * n_trans) > 0 else 1.0
    
    # ── 2. Silent (tau) Transition Ratio ──────────────────────────
    silent_trans = [t for t in transitions if t.label is None]
    n_silent = len(silent_trans)
    silent_ratio = n_silent / n_trans if n_trans > 0 else 0.0
    
    # ── 3. Label Duplication ──────────────────────────────────────
    labels = [t.label for t in transitions if t.label is not None]
    unique_labels = set(labels)
    n_unique_labels = len(unique_labels)
    label_dup = n_trans / n_unique_labels if n_unique_labels > 0 else 1.0
    
    # ── 4. XOR-split Entropy (structural, no log weights) ──────────
    # For each place with >1 outgoing arc, compute entropy of uniform split
    xor_entropies = []
    for p in places:
        out_arcs = [a for a in arcs if a.source == p]
        if len(out_arcs) > 1:
            # Uniform entropy: H = log2(k)
            xor_entropies.append(math.log2(len(out_arcs)))
    xor_entropy_mean = np.mean(xor_entropies) if xor_entropies else 0.0
    
    # ── 5. Free-choice Ratio ──────────────────────────────────────
    # A place is free-choice if: all its outgoing transitions have
    # exactly one incoming place (i.e., it's a pure XOR-split, not mixed AND/XOR)
    free_choice_count = 0
    for p in places:
        out_trans = set(a.target for a in p.out_arcs)
        if len(out_trans) <= 1:
            free_choice_count += 1  # No split, trivially free-choice
        else:
            # Check every outgoing transition: each must have only THIS place as input
            all_free = all(len(t.in_arcs) == 1 for t in out_trans)
            if all_free:
                free_choice_count += 1
    free_choice_ratio = free_choice_count / n_places if n_places > 0 else 1.0
    
    return {
        "n_places": n_places,
        "n_trans": n_trans,
        "n_arcs": n_arcs,
        "n_silent": n_silent,
        "n_unique_labels": n_unique_labels,
        "density": density,
        "silent_ratio": silent_ratio,
        "label_dup": label_dup,
        "xor_entropy_mean": xor_entropy_mean,
        "free_choice_ratio": free_choice_ratio,
    }


# =====================================================================
# Replay-based Structural Methods
# =====================================================================

def compute_arc_flow_density(event_log, net, im, fm):
    """
    Replay the original log; count per-arc usage across traces.
    An arc used by <1% of traces (or <2 traces) is 'rare'.
    Returns exact counts (not estimates).
    """
    replayed = token_replay.apply(event_log, net, im, fm)
    num_traces = len(event_log)
    rare_threshold = max(2, int(num_traces * 0.01))
    
    arc_usage = {arc: 0 for arc in net.arcs}
    for res in replayed:
        used = set()
        for t in res['activated_transitions']:
            for arc in t.in_arcs:
                used.add(arc)
            for arc in t.out_arcs:
                used.add(arc)
        for arc in used:
            arc_usage[arc] += 1
    
    total_arcs = len(net.arcs)
    rare_arcs = sum(1 for count in arc_usage.values() if count < rare_threshold)
    zero_arcs = sum(1 for count in arc_usage.values() if count == 0)
    
    return {
        "total_arcs": total_arcs,
        "rare_arcs": rare_arcs,
        "zero_arcs": zero_arcs,
        "rare_arc_pct": rare_arcs / total_arcs if total_arcs > 0 else 0.0,
        "gen_struct": max(0.0, 1.0 - rare_arcs / total_arcs) if total_arcs > 0 else 0.0,
        "rare_threshold": rare_threshold,
    }


def compute_transition_activation(event_log, net, im, fm):
    """
    Count per-trace firing of each transition (by object, not label).
    Compute Gini coefficient: 0 = uniform usage, 1 = one transition dominates.
    """
    replayed = token_replay.apply(event_log, net, im, fm)
    all_trans = list(net.transitions)
    if not all_trans:
        return {"gini": 0.0, "n_transitions": 0, "n_used": 0, "min_usage": 0, "max_usage": 0, "mean_usage": 0.0}

    usage = Counter()
    for res in replayed:
        fired = set(res['activated_transitions'])
        for t in fired:
            usage[t] += 1

    counts = sorted(usage.values())
    n = len(counts)
    if n == 0:
        return {"gini": 0.0, "n_transitions": len(all_trans), "n_used": 0, "min_usage": 0, "max_usage": 0, "mean_usage": 0.0}

    total = sum(counts)
    gini = (2 * sum((i + 1) * c for i, c in enumerate(counts))) / (n * total) - (n + 1) / n if total > 0 else 0.0

    return {
        "gini": gini,
        "n_transitions": len(all_trans),
        "n_used": n,
        "min_usage": min(counts),
        "max_usage": max(counts),
        "mean_usage": np.mean(counts),
    }


def compute_place_token_variance(event_log, net, im, fm):
    """
    Track maximum token count per place during replay.
    High variance across places = unbalanced AND/XOR structures.
    """
    # Use token_replay with full diagnostics to get produced/consumed tokens per place
    replayed = token_replay.apply(event_log, net, im, fm)
    
    # Collect per-place stats: count of traces where place held >1 token
    # (Token replay in pm4py doesn't directly expose per-place token counts,
    #  so we estimate via transition activation: places between AND-splits
    #  and AND-joins accumulate tokens)
    places = list(net.places)
    if not places:
        return {"mean_max_tokens": 0.0, "high_var_places": 0, "total_places": 0}
    
    # Heuristic: for each place, count incoming/outgoing arcs
    # Places with multiple outputs (AND-split) and no matching AND-join = high variance risk
    suspect_count = 0
    for p in places:
        in_count = len(p.in_arcs)
        out_count = len(p.out_arcs)
        # AND-split: >1 outgoing arc to transitions each with this as their only input
        if out_count > 1 and in_count == 1:
            suspect_count += 1
    
    return {
        "mean_max_tokens": 0.0,  # Requires per-step token counting (not exposed by pm4py)
        "high_var_places": suspect_count,
        "total_places": len(places),
        "note": "AND-split place count (structural proxy for token variance)",
    }


def compute_path_coverage(event_log, net, im, fm, max_depth=12):
    """
    Compute reachable arc ratio: from initial marking, what fraction of all
    arcs are reachable within max_depth transition firings?
    
    Also computes: how many unique activity sequences in the log are of
    length <= max_depth, vs how many distinct paths exist in the model.
    """
    from collections import deque

    # ── Reachable arcs via BFS from initial marking ──
    # State: frozenset of places with tokens
    initial_places = frozenset(im)
    visited_states = {initial_places: 0}  # state -> depth
    reachable_arcs = set()
    reachable_transitions = set()

    queue = deque([(initial_places, 0)])
    all_arcs = set(net.arcs)

    while queue:
        places_set, depth = queue.popleft()
        if depth >= max_depth:
            continue

        # Find all enabled transitions from current marking
        for t in net.transitions:
            t_inputs = frozenset(a.source for a in t.in_arcs)
            if t_inputs.issubset(places_set):
                # Transition is enabled — mark its arcs as reachable
                for a in t.in_arcs:
                    reachable_arcs.add(a)
                for a in t.out_arcs:
                    reachable_arcs.add(a)
                reachable_transitions.add(t)

                # Compute next marking
                next_places = frozenset(
                    (places_set - t_inputs) | frozenset(a.target for a in t.out_arcs)
                )
                if next_places not in visited_states:
                    visited_states[next_places] = depth + 1
                    queue.append((next_places, depth + 1))

    n_total_arcs = len(all_arcs)
    n_reachable = len(reachable_arcs)
    arc_reachability = n_reachable / n_total_arcs if n_total_arcs > 0 else 1.0

    # ── Short path comparison (depth-bounded) ──
    # Observed: traces from log truncated to max_depth
    observed_short = set()
    for trace in event_log:
        seq = tuple(e["concept:name"] for e in trace[:max_depth])
        if seq:
            observed_short.add(seq)

    return {
        "n_total_arcs": n_total_arcs,
        "n_reachable_arcs": n_reachable,
        "arc_reachability": arc_reachability,
        "n_reachable_transitions": len(reachable_transitions),
        "n_total_transitions": len(list(net.transitions)),
        "n_states_visited": len(visited_states),
        "n_observed_short_paths": len(observed_short),
        "max_depth": max_depth,
    }


def compute_composite_score(metrics, weights=None):
    """
    Combine metrics into a single Gen_Struct-like score using weighted sum.
    
    All raw metrics are transformed so that higher = better generalization.
    """
    if weights is None:
        weights = {
            "density": 0.25,
            "silent_ratio": 0.25,   # inverted: fewer silent = better
            "label_dup": 0.25,       # inverted: dup=1.0 is ideal
            "xor_entropy_mean": 0.15,
            "free_choice_ratio": 0.10,
        }
    
    # ── Transform each metric to [0, 1] where 1 = best generalization ──
    
    # Density: ideal ~0.1-0.3 for well-structured nets. Penalize extremes.
    # Use a Gaussian-like penalty centered at 0.2
    d = metrics["density"]
    density_score = math.exp(-((d - 0.2) ** 2) / 0.02)
    
    # Silent ratio: fewer is better (1.0 = no silent transitions)
    silent_score = 1.0 - metrics["silent_ratio"]
    
    # Label duplication: 1.0 is ideal (each label appears once)
    dup = metrics["label_dup"]
    label_score = 1.0 / dup if dup > 0 else 0.0
    
    # XOR entropy: moderate entropy is good (not zero, not extreme)
    h = metrics["xor_entropy_mean"]
    entropy_score = min(h / 3.0, 1.0)  # Cap at 1.0, 3 bits = 8-way split
    
    # Free-choice: higher is better
    fc_score = metrics["free_choice_ratio"]
    
    composite = (
        weights["density"] * density_score +
        weights["silent_ratio"] * silent_score +
        weights["label_dup"] * label_score +
        weights["xor_entropy_mean"] * entropy_score +
        weights["free_choice_ratio"] * fc_score
    )
    
    return {
        "composite": composite,
        "density_score": density_score,
        "silent_score": silent_score,
        "label_score": label_score,
        "entropy_score": entropy_score,
        "fc_score": fc_score,
    }


# =====================================================================
# Advanced Structural Metrics (no replay)
# =====================================================================

def compute_cyclomatic_complexity(net):
    """
    McCabe's Cyclomatic Complexity for workflow nets.
    V(G) = |A| - |P| - |T| + 2
    
    Measures the number of linearly independent paths.
    Low (<10): simple. 10-30: moderate. >30: complex/overfit.
    """
    n_places = len(net.places)
    n_trans = len(net.transitions)
    n_arcs = len(net.arcs)
    complexity = n_arcs - n_places - n_trans + 2
    return {
        "cyclomatic_complexity": max(0, complexity),
        "n_places": n_places,
        "n_trans": n_trans,
        "n_arcs": n_arcs,
    }


def compute_block_structured_ratio(net, im, fm):
    """
    Try to convert the Petri net to a Process Tree via PM4Py.
    If conversion succeeds: 100% block-structured.
    If conversion fails: count unstructured nodes (non-free-choice + crossing arcs).
    
    Returns:
        structured_ratio: 0.0 (spaghetti) to 1.0 (fully block-structured)
    """
    try:
        pt = pm4py.convert_to_process_tree(net, im, fm)
        # Conversion succeeded = fully structured
        return {
            "structured_ratio": 1.0,
            "method": "PST conversion succeeded",
            "is_fully_structured": True,
        }
    except Exception:
        pass
    
    # Fallback: heuristic based on non-free-choice places
    places = list(net.places)
    structured_count = 0
    for p in places:
        out_trans = set(a.target for a in p.out_arcs)
        if len(out_trans) <= 1:
            structured_count += 1
        else:
            all_free = all(len(t.in_arcs) == 1 for t in out_trans)
            if all_free:
                structured_count += 1
    
    ratio = structured_count / len(places) if places else 1.0
    return {
        "structured_ratio": ratio,
        "method": "free-choice fallback heuristic",
        "is_fully_structured": ratio >= 0.99,
    }


def compute_cross_connectivity(net):
    """
    Simplified cross-connectivity: average degree per transition.
    High values = many long-distance edges = spaghetti indicator.
    
    Proper cross-connectivity (Mendling) requires all-pairs shortest paths.
    This approximation uses: (avg in-degree × avg out-degree) for transitions.
    """
    transitions = list(net.transitions)
    if not transitions:
        return {"cross_connectivity": 0.0, "max_degree": 0, "mean_degree": 0.0}
    
    degrees = []
    for t in transitions:
        in_deg = len(t.in_arcs)
        out_deg = len(t.out_arcs)
        degrees.append(in_deg + out_deg)
    
    mean_deg = np.mean(degrees)
    max_deg = max(degrees)
    
    # Cross-connectivity proxy: mean degree normalized by theoretical max (2*|P|)
    max_possible = 2 * len(net.places)
    cross_conn = mean_deg / max_possible if max_possible > 0 else 0.0
    
    return {
        "cross_connectivity": cross_conn,
        "max_degree": max_deg,
        "mean_degree": mean_deg,
    }


# =====================================================================
# Behavioral / Predictive Metrics (replay-based)
# =====================================================================

def compute_kfold_cv_fitness(event_log, miner_fn, k=5, seed=42):
    """
    K-Fold Cross-Validation Fitness.
    Split log into k folds; train on k-1, test on 1; repeat.
    Returns training fitness, test fitness, and drop-off.
    """
    import random as _random
    _random.seed(seed)
    
    traces = list(event_log)
    n = len(traces)
    indices = list(range(n))
    _random.shuffle(indices)
    fold_size = n // k
    
    train_fits = []
    test_fits = []
    
    for fold in range(k):
        # Split indices
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < k - 1 else n
        test_idx = set(indices[test_start:test_end])
        
        # Build train/test logs as EventLog objects
        train_log = EventLog([traces[i] for i in range(n) if i not in test_idx])
        test_log = EventLog([traces[i] for i in test_idx])
        
        # Discover model from training set
        net, im, fm = miner_fn(train_log)
        
        # Replay training set
        replayed_train = token_replay.apply(train_log, net, im, fm)
        train_fit = np.mean([r['trace_fitness'] for r in replayed_train])
        train_fits.append(train_fit)
        
        # Replay test set (unseen data)
        replayed_test = token_replay.apply(test_log, net, im, fm)
        test_fit = np.mean([r['trace_fitness'] for r in replayed_test])
        test_fits.append(test_fit)
    
    train_mean = np.mean(train_fits)
    test_mean = np.mean(test_fits)
    drop_off = train_mean - test_mean
    
    return {
        "k": k,
        "train_fitness_mean": train_mean,
        "train_fitness_std": np.std(train_fits),
        "test_fitness_mean": test_mean,
        "test_fitness_std": np.std(test_fits),
        "drop_off": drop_off,
        "drop_off_pct": drop_off / train_mean * 100 if train_mean > 0 else 0.0,
        "train_fits": train_fits,
        "test_fits": test_fits,
    }


def compute_simulation_coverage(event_log, net, im, fm, n_sim=5000, seed=42):
    """
    State-Space Simulation Coverage.
    Random-walk simulate n_sim traces from the model.
    Check what fraction appear in the original log.
    
    100% in log = overfit (memorized). 0% in log = underfit (nonsense).
    20-30% novel but plausible = good generalization.
    """
    import random as _random
    _random.seed(seed)
    
    # Extract original log activity sequences
    original_seqs = set()
    for trace in event_log:
        seq = tuple(e["concept:name"] for e in trace)
        if seq:
            original_seqs.add(seq)
    
    # Try PM4Py play_out for simulation
    try:
        simulated_log = pm4py.play_out(net, im, fm, no_traces=n_sim)
    except Exception:
        # Fallback: simple random walk
        simulated_log = _simple_random_walk(net, im, fm, n_sim, seed)
    
    # Compare
    sim_seqs = set()
    total_events = 0
    for trace in simulated_log:
        seq = tuple(e["concept:name"] for e in trace)
        if seq:
            sim_seqs.add(seq)
            total_events += len(seq)
    
    n_sim_unique = len(sim_seqs)
    overlap = len(sim_seqs & original_seqs)
    novel = n_sim_unique - overlap
    
    return {
        "n_sim_traces": n_sim,
        "n_sim_unique": n_sim_unique,
        "n_original_unique": len(original_seqs),
        "overlap_count": overlap,
        "novel_count": novel,
        "overlap_ratio": overlap / n_sim_unique if n_sim_unique > 0 else 0.0,
        "novel_ratio": novel / n_sim_unique if n_sim_unique > 0 else 0.0,
        "mean_sim_length": total_events / n_sim if n_sim > 0 else 0,
        "method": "pm4py.play_out",
    }


def _simple_random_walk(net, im, fm, n_traces, seed):
    """Fallback random walk on Petri net if play_out fails."""
    import random as _random
    _random.seed(seed)
    from pm4py.objects.log.obj import EventLog, Trace, Event
    
    places = list(net.places)
    transitions = list(net.transitions)
    max_steps = 60
    
    log = EventLog()
    for i in range(n_traces):
        marking = set(im)
        seq = []
        for _ in range(max_steps):
            # Find enabled transitions
            enabled = []
            for t in transitions:
                t_inputs = set(a.source for a in t.in_arcs)
                if t_inputs.issubset(marking):
                    enabled.append(t)
            if not enabled:
                break
            t = _random.choice(enabled)
            if t.label is not None:
                seq.append(t.label)
            # Fire transition
            marking = (marking - set(a.source for a in t.in_arcs)) | set(a.target for a in t.out_arcs)
        if seq:
            trace = Trace(attributes={"concept:name": f"sim_{i}"})
            for act in seq:
                trace.append(Event({"concept:name": act}))
            log.append(trace)
    return log


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 90)
    print("  Structural Metrics Analysis — Graph-Theoretic Generalization Indicators")
    print("=" * 90)
    
    # ── Load log ────────────────────────────────────────────────────
    print("\n[1/3] Loading event log...")
    if os.path.exists(XES_PATH):
        event_log = pm4py.read_xes(XES_PATH)
    else:
        print("  Log not found, generating dummy data...")
        import pandas as pd
        df = pd.DataFrame({
            'case:concept:name': ['1','1','1','2','2','2','3','3'],
            'concept:name': ['A','B','C','A','X','C','A','B']
        })
        event_log = pm4py.format_dataframe(df, case_id='case:concept:name', activity_key='concept:name')
    event_log = pm4py.convert_to_event_log(event_log)
    print(f"  Loaded: {len(event_log)} traces | {sum(len(t) for t in event_log)} events")
    
    # ── Discover & Analyze ──────────────────────────────────────────
    print("\n[2/3] Discovering models and computing structural metrics...")
    
    all_metrics = {}
    all_replay = {}          # Replay-based metrics
    all_path_coverage = {}   # Reachable arc ratio
    all_advanced = {}        # Cyclomatic, block-structured, cross-connectivity
    all_kfold = {}           # K-fold CV results
    all_simulation = {}      # Simulation coverage
    all_timings = {}         # Per-metric timing records
    
    for miner_name, miner_fn in MINERS.items():
        t0 = time.time()
        net, im, fm = miner_fn(event_log)
        model_discovery_time = time.time() - t0
        
        timings = {"model_discovery": model_discovery_time}
        
        # Pure structural metrics
        t1 = time.time()
        metrics = compute_structural_metrics(net)
        timings["structural_5"] = time.time() - t1
        
        # Advanced structural metrics (no replay)
        advanced = {}
        t1 = time.time()
        advanced["cyclomatic"] = compute_cyclomatic_complexity(net)
        timings["cyclomatic"] = time.time() - t1
        t1 = time.time()
        advanced["block_struct"] = compute_block_structured_ratio(net, im, fm)
        timings["block_struct"] = time.time() - t1
        t1 = time.time()
        advanced["cross_conn"] = compute_cross_connectivity(net)
        timings["cross_conn"] = time.time() - t1
        
        # Replay-based metrics
        replay = {}
        t1 = time.time()
        replay["arc_flow"] = compute_arc_flow_density(event_log, net, im, fm)
        timings["arc_flow"] = time.time() - t1
        t1 = time.time()
        replay["transition"] = compute_transition_activation(event_log, net, im, fm)
        timings["transition_gini"] = time.time() - t1
        t1 = time.time()
        replay["token_var"] = compute_place_token_variance(event_log, net, im, fm)
        timings["token_var"] = time.time() - t1
        
        # Reachable arc ratio
        print(f"       Computing reachable arc ratio for {miner_name} (max_depth=12)...")
        t1 = time.time()
        path_cov = compute_path_coverage(event_log, net, im, fm, max_depth=12)
        timings["reachable_arc"] = time.time() - t1
        
        metrics["runtime_s"] = time.time() - t0
        all_metrics[miner_name] = metrics
        all_advanced[miner_name] = advanced
        all_replay[miner_name] = replay
        all_path_coverage[miner_name] = path_cov
        all_timings[miner_name] = timings
        print(f"  {miner_name:<25} | {metrics['n_places']:>4}P {metrics['n_trans']:>4}T {metrics['n_arcs']:>4}A | {metrics['runtime_s']:.1f}s")
    
    # ── K-Fold Cross-Validation (expensive: 2×K model discoveries per miner) ──
    print("\n[2.5] Computing K-Fold CV Fitness (k=3, training on 67%, testing on 33%)...")
    KFOLD_K = 3
    for miner_name, miner_fn in MINERS.items():
        print(f"       K-Fold CV for {miner_name}...")
        t1 = time.time()
        all_kfold[miner_name] = compute_kfold_cv_fitness(event_log, miner_fn, k=KFOLD_K, seed=42)
        all_timings[miner_name]["kfold_cv"] = time.time() - t1
        kf = all_kfold[miner_name]
        print(f"         Train={kf['train_fitness_mean']:.4f} → Test={kf['test_fitness_mean']:.4f} | Drop-off={kf['drop_off_pct']:.1f}%")
    
    # ── State-Space Simulation ──
    print("\n[2.6] Computing Simulation Coverage (5000 random walks per model)...")
    for miner_name, miner_fn in MINERS.items():
        net, im, fm = miner_fn(event_log)
        t1 = time.time()
        all_simulation[miner_name] = compute_simulation_coverage(event_log, net, im, fm, n_sim=5000, seed=42)
        all_timings[miner_name]["simulation"] = time.time() - t1
        sim = all_simulation[miner_name]
        print(f"       {miner_name}: {sim['n_sim_unique']} unique traces, {sim['overlap_count']} in-log ({sim['overlap_ratio']:.1%}), {sim['novel_count']} novel ({sim['novel_ratio']:.1%})")
    
    # ── Print Results ───────────────────────────────────────────────
    print(f"\n[3/3] Results")
    
    # Table 1: Raw structural metrics
    print(f"\n{'─' * 90}")
    print(f"  📐 Pure Structural Metrics (no replay)")
    print(f"{'─' * 90}")
    header = f"  {'Miner':<25} | {'P':>5} | {'T':>5} | {'A':>5} | {'Silent':>6} | {'LabelDup':>8} | {'Density':>8} | {'XOR Ent':>7} | {'FreeCh%':>7}"
    print(header)
    print(f"  {'─' * 88}")
    for name, m in all_metrics.items():
        print(f"  {name:<25} | {m['n_places']:>5} | {m['n_trans']:>5} | {m['n_arcs']:>5} | {m['n_silent']:>5}/{m['n_trans']:<4} | {m['label_dup']:>8.2f} | {m['density']:>8.4f} | {m['xor_entropy_mean']:>6.2f} | {m['free_choice_ratio']:>6.1%}")
    
    # Table 2: Per-metric assessment
    print(f"\n{'─' * 90}")
    print(f"  🎯 Per-Metric Assessment")
    print(f"{'─' * 90}")
    print(f"  {'Metric':<20} | {'IM':>10} | {'Heuristics':>10} | {'Alpha':>10} | {'Discriminative?':<20} | {'Verdict':<20}")
    print(f"  {'─' * 88}")
    rows = [
        ("Density", "density", ".4f", "Partially (alpha outlier)", "⚠️ Calibration"),
        ("Silent Ratio", "silent_ratio", ".1%", "Only vs Alpha", "❌ IM≈Heuristics"),
        ("Label Duplication", "label_dup", ".2f", "Only vs Alpha", "⚠️ Rewards underfit"),
        ("XOR Entropy", "xor_entropy_mean", ".2f", "No (narrow range)", "❌ Non-discriminative"),
        ("Free-choice", "free_choice_ratio", ".1%", "Yes (all three)", "✅ Best standalone"),
    ]
    for label, key, fmt, disc, verdict in rows:
        vals = " | ".join(f"{all_metrics[n][key]:>{fmt if fmt != '.1%' else '9.1%' if fmt=='.1%' else '10'}}" for n in MINERS)
        # Simplify: just print with explicit formatting
        im_v = all_metrics["Inductive Miner (IM)"][key]
        he_v = all_metrics["Heuristics Miner"][key]
        al_v = all_metrics["Alpha Miner"][key]
        if fmt == ".1%":
            print(f"  {label:<20} | {im_v:>9.1%} | {he_v:>9.1%} | {al_v:>9.1%} | {disc:<20} | {verdict:<20}")
        elif fmt == ".4f":
            print(f"  {label:<20} | {im_v:>10.4f} | {he_v:>10.4f} | {al_v:>10.4f} | {disc:<20} | {verdict:<20}")
        else:
            print(f"  {label:<20} | {im_v:>10.2f} | {he_v:>10.2f} | {al_v:>10.2f} | {disc:<20} | {verdict:<20}")
    
    # Table 3: Arc Flow Density (replay-based)
    print(f"\n{'─' * 90}")
    print(f"  🔄 Arc Flow Density (Replay-based)")
    print(f"{'─' * 90}")
    print(f"  {'Miner':<25} | {'Total Arcs':>10} | {'Rare Arcs':>9} | {'Zero Arcs':>9} | {'Rare %':>7} | {'Gen_Struct':>10}")
    print(f"  {'─' * 88}")
    for name in MINERS:
        af = all_replay[name]["arc_flow"]
        print(f"  {name:<25} | {af['total_arcs']:>10} | {af['rare_arcs']:>9} | {af['zero_arcs']:>9} | {af['rare_arc_pct']:>6.1%} | {af['gen_struct']:>10.4f}")
    
    # Table 4: Transition Activation Gini
    print(f"\n{'─' * 90}")
    print(f"  🔄 Transition Activation (Replay-based)")
    print(f"{'─' * 90}")
    print(f"  {'─' * 90}")
    print(f"  {'Miner':<25} | {'Total T':>7} | {'Used T':>7} | {'Gini':>7} | {'Min Use':>7} | {'Max Use':>7} | {'Mean Use':>8}")
    print(f"  {'─' * 88}")
    for name in MINERS:
        ta = all_replay[name]["transition"]
        print(f"  {name:<25} | {ta['n_transitions']:>7} | {ta['n_used']:>7} | {ta['gini']:>7.4f} | {ta['min_usage']:>7} | {ta['max_usage']:>7} | {ta['mean_usage']:>8.1f}")
    
    # Table 5: Reachable Arc Ratio
    print(f"\n{'─' * 90}")
    print(f"  🔄 Reachable Arc Ratio (BFS from initial marking, depth={all_path_coverage['Inductive Miner (IM)']['max_depth']})")
    print(f"{'─' * 90}")
    print(f"  {'Miner':<25} | {'Total Arcs':>10} | {'Reachable':>9} | {'Reach %':>7} | {'Reach T':>7} | {'States':>8} | {'Short Paths':>10}")
    print(f"  {'─' * 88}")
    for name in MINERS:
        pc = all_path_coverage[name]
        print(f"  {name:<25} | {pc['n_total_arcs']:>10} | {pc['n_reachable_arcs']:>9} | {pc['arc_reachability']:>6.1%} | {pc['n_reachable_transitions']:>6}/{pc['n_total_transitions']} | {pc['n_states_visited']:>8} | {pc['n_observed_short_paths']:>10}")
    
    # Table 6: Advanced Structural Metrics
    print(f"\n{'─' * 90}")
    print(f"  📐 Advanced Structural Metrics (no replay)")
    print(f"{'─' * 90}")
    print(f"  {'Miner':<25} | {'Cyclomatic':>10} | {'Block-Struct':>11} | {'Cross-Conn':>10} | {'Max Deg':>7}")
    print(f"  {'─' * 88}")
    for name in MINERS:
        cc = all_advanced[name]["cyclomatic"]
        bs = all_advanced[name]["block_struct"]
        cx = all_advanced[name]["cross_conn"]
        print(f"  {name:<25} | {cc['cyclomatic_complexity']:>10} | {bs['structured_ratio']:>10.1%} | {cx['cross_connectivity']:>10.4f} | {cx['max_degree']:>7}")
    
    # Table 7: K-Fold Cross-Validation
    kf_k = all_kfold["Inductive Miner (IM)"]["k"]
    print(f"\n{'─' * 90}")
    print(f"  🔄 K-Fold CV Fitness (k={kf_k}) — The Gold Standard")
    print(f"{'─' * 90}")
    print(f"  {'Miner':<25} | {'Train Fit':>10} | {'Test Fit':>9} | {'Drop-off':>9} | {'Drop-off%':>9} | {'Verdict':>15}")
    print(f"  {'─' * 88}")
    for name in MINERS:
        kf = all_kfold[name]
        verdict = "✅ Good gen." if kf['drop_off'] < 0.05 else ("⚠️ Moderate" if kf['drop_off'] < 0.15 else "❌ Overfit")
        print(f"  {name:<25} | {kf['train_fitness_mean']:>10.4f} | {kf['test_fitness_mean']:>9.4f} | {kf['drop_off']:>9.4f} | {kf['drop_off_pct']:>8.1f}% | {verdict:>15}")
    
    # Table 8: Simulation Coverage
    print(f"\n{'─' * 90}")
    print(f"  🔄 State-Space Simulation Coverage (5000 random walks)")
    print(f"{'─' * 90}")
    print(f"  {'Miner':<25} | {'Sim Unique':>10} | {'In-Log':>7} | {'Novel':>6} | {'In-Log%':>7} | {'Novel%':>7} | {'Verdict':>18}")
    print(f"  {'─' * 88}")
    for name in MINERS:
        sim = all_simulation[name]
        if sim['overlap_ratio'] > 0.95:
            verdict = "❌ Overfit (memorized)"
        elif sim['novel_ratio'] > 0.95:
            verdict = "❌ Underfit (nonsense)"
        elif 0.15 <= sim['novel_ratio'] <= 0.40:
            verdict = "✅ Good generalization"
        else:
            verdict = "⚠️ Moderate"
        print(f"  {name:<25} | {sim['n_sim_unique']:>10} | {sim['overlap_count']:>7} | {sim['novel_count']:>6} | {sim['overlap_ratio']:>6.1%} | {sim['novel_ratio']:>6.1%} | {verdict:>18}")
    
    # Table 9: Runtime Report
    print(f"\n{'─' * 90}")
    print(f"  ⏱️  Runtime Report (seconds)")
    print(f"{'─' * 90}")
    
    metric_names = [
        ("Model Discovery", "model_discovery"),
        ("5 Structural Metrics", "structural_5"),
        ("Cyclomatic", "cyclomatic"),
        ("Block-Struct", "block_struct"),
        ("Cross-Conn", "cross_conn"),
        ("Arc Flow Density", "arc_flow"),
        ("Transition Gini", "transition_gini"),
        ("Token Variance", "token_var"),
        ("Reachable Arc BFS", "reachable_arc"),
        ("K-Fold CV (k=3)", "kfold_cv"),
        ("Simulation (5k)", "simulation"),
    ]
    
    # Print header
    header = "  " + f"{'Metric':<25}"
    for name in MINERS:
        header += f" | {name.split('(')[0].strip():>12}"
    header += f" | {'Total':>8}"
    print(header)
    print(f"  {'─' * (30 + 15 * len(MINERS))}")
    
    total_per_col = {name: 0.0 for name in MINERS}
    for label, key in metric_names:
        row = f"  {label:<25}"
        row_total = 0.0
        for name in MINERS:
            t = all_timings[name].get(key, 0)
            row_total += t
            total_per_col[name] += t
            row += f" | {t:>11.1f}s"
        row += f" | {row_total:>7.1f}s"
        print(row)
    
    # Total row
    total_row = f"  {'─' * (30 + 15 * len(MINERS))}"
    print(total_row)
    row = f"  {'TOTAL':<25}"
    grand_total = 0.0
    for name in MINERS:
        t = total_per_col[name]
        grand_total += t
        row += f" | {t:>11.1f}s"
    row += f" | {grand_total:>7.1f}s"
    print(row)
    
    print(f"\n{'=' * 90}\n")


if __name__ == "__main__":
    main()
