"""
HybridGen Algorithm v2.2 — Gen_Struct v2.

Shadow log: N-gram + Katz Backoff (same as v2.1).
Gen_Struct: multi-dimensional (ArcFlow + Gini + Reach + Cyclo).
"""

import random
import time
import sys
import numpy as np
from collections import defaultdict, Counter

import pm4py
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness

# Increase recursion depth slightly just to be safe with long traces
# sys.setrecursionlimit(2000)

# =====================================================================
# 1. Generative Behavioral Analysis (Gen_shadow) with DFS & Katz Backoff
# =====================================================================

def _evaluate_state(state_tuple, ngram_dict, safe_threshold):
    """
    Helper function to evaluate statistical safety and Good-Turing probability.
    Returns (P_unseen, valid_out_edges) or (None, None) if the state is too sparse.
    """
    out_edges = ngram_dict.get(state_tuple, {})
    # N: Total number of times we transitioned out of this specific state historically
    n_total = sum(out_edges.values())
    
    # Check for data sparsity (Curse of Dimensionality)
    # If a state lacks sufficient historical support, we force a backoff
    if n_total < safe_threshold:
        return None, None
        
    # Calculate Good-Turing probability (P_unseen)
    # N_1: How many target activities followed this state exactly one time (Singletons)
    # High number of 'singletons' -> high variance/unpredictability
    n_1 = sum(1 for count in out_edges.values() if count == 1)
    p_unseen = (n_1 / n_total) if n_total > 0 else 1.0
    
    # If P_unseen is 100%, the state collapses (all historical paths were singletons). 
    # Force backoff to find a more stable generalization.
    if p_unseen == 1.0:
        return None, None
        
    return p_unseen, out_edges

def generate_trace_dfs(current_seq, ngram_outgoings, ends, alphabet, max_length, safe_threshold, max_n, had_mutation=False):
    """
    Recursive DFS function to generate a single shadow trace.
    Naturally maintains the state context through the call stack.
    
    Args:
        max_n: Maximum N-gram order (lookback = max_n - 1). Backoff from max_n down to 2,
               with absolute fallback to N=1.
    
    Returns:
        (sequence, had_mutation): The generated activity sequence and a boolean flag
        indicating whether any mutation event occurred during generation.
    """
    current_local = current_seq[-1]
    
    # 1. Termination Checks
    # Decision 1: Should the trace terminate here based on historical end frequency?
    if current_local in ends:
        times_as_end = ends[current_local]
        # Total occurrences = times it was an end + times it transitioned elsewhere
        out_edges_n1 = ngram_outgoings[1].get((current_local,), {})
        total_occurrences = times_as_end + sum(out_edges_n1.values())
        
        # Probabilistic roll to stop the trace
        if random.random() < (times_as_end / total_occurrences):
            return current_seq, had_mutation
            
    # Safety catch: Prevent infinite loops in case of cyclic mutations
    if len(current_seq) >= max_length:
        return current_seq, had_mutation

    # 2. Dynamic Katz Backoff Strategy (max_n -> ... -> 2 -> 1)
    p_unseen, valid_out_edges = None, None
    
    # Try from highest order down to N=2, stopping at the first statistically safe state
    for n in range(max_n, 1, -1):
        if len(current_seq) >= n:
            p_unseen, valid_out_edges = _evaluate_state(
                tuple(current_seq[-n:]), ngram_outgoings[n], safe_threshold
            )
            if p_unseen is not None:
                break
        
    # Absolute Fallback to N=1 (Local Marking, Lookback 0)
    # Lowest context fidelity, but 100% safe from sparsity collapse
    if p_unseen is None:
        # For N=1, we deliberately bypass the safe_threshold to ensure the walker doesn't get stuck
        out_edges = ngram_outgoings[1].get((current_local,), {})
        n_total = sum(out_edges.values())
        n_1 = sum(1 for count in out_edges.values() if count == 1)
        p_unseen = (n_1 / n_total) if n_total > 0 else 1.0
        valid_out_edges = out_edges

    # 3. Mutate vs. Exploit
    # Decision 2: Mutate or follow historical paths?
    if random.random() < p_unseen:
        # Mutation Triggered: The algorithm actively explores a logically valid but historically unseen path.
        # It picks a random activity from the entire known alphabet.
        # Note: This ensures syntactic validity (it uses a real event name) 
        # but purposefully ignores strict business logic to test the model's 
        # robustness against unseen or anomalous behavior (Adversarial Generalization).
        next_node = random.choice(alphabet)
        had_mutation = True
    else:
        # Predictable Path Triggered: Follow historical DFG distribution.
        if not valid_out_edges:
            return current_seq, had_mutation # Dead end reached
        # Weighted random choice based on how many times paths were taken in the past
        # TODO
        # replace frequency counts with ln(count+1) to reduce the dominance of extremely common paths and increase variability in the generated traces, while still respecting historical likelihoods.
        weights = [np.log(count + 1) for count in valid_out_edges.values()]
        next_node = random.choices(list(valid_out_edges.keys()), weights=weights, k=1)[0]
        
    # 4. DFS Recursion
    # Pass a new list (current_seq + [next_node]) to avoid mutation side-effects and maintain immutability
    return generate_trace_dfs(current_seq + [next_node], ngram_outgoings, ends, alphabet, max_length, safe_threshold, max_n, had_mutation)

def generate_shadow_log(event_log, num_traces=1000, max_trace_length=100, safe_threshold=5, max_n=3):
    """
    Generates a synthetic 'shadow' log acting as probabilistic based future behavior 
    using Good-Turing estimation and Katz Backoff.
    
    Args:
        max_n: Maximum N-gram order. N-gram statistics are pre-computed for N=1..max_n.
    
    Returns:
        (shadow_log, mutation_flags): The generated EventLog and a list of booleans
        indicating which traces contain at least one mutation event.
    """
    
    # 1. Discover historical N-gram Directly-Follows Graphs (#howTheProcessExecutedInThePast)
    # Pre-compute dictionaries up to max_n for fast lookups during generation
    ngram_outgoings = {n: defaultdict(Counter) for n in range(1, max_n + 1)}
    starts = Counter()
    ends = Counter()
    alphabet = set()
    
    # Extract every unique activity and path in the log
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
            # Extract states for all N-gram orders from 1 to max_n
            for n in range(1, max_n + 1):
                if i >= n - 1:
                    state_tuple = tuple(seq[i - (n - 1) : i + 1])
                    ngram_outgoings[n][state_tuple][nxt_act] += 1

    alphabet = list(alphabet)
    shadow_log = EventLog()
    mutation_flags = []
    start_choices = list(starts.keys())
    start_weights = list(starts.values())

    # 2. Begin generating the requested number of synthetic traces
    for i in range(num_traces):
        # Pick a starting activity based on its historical likelihood of starting a process
        start_node = random.choices(start_choices, weights=start_weights, k=1)[0]
        
        # Execute the recursive DFS walker to build the sequence
        final_sequence, had_mutation = generate_trace_dfs(
            [start_node], ngram_outgoings, ends, alphabet, max_trace_length, safe_threshold, max_n
        )
        
        # Convert the generated string sequence back into a PM4Py Trace object
        trace = Trace(attributes={"concept:name": f"shadow_{i}"})
        for act in final_sequence:
            trace.append(Event({"concept:name": act}))
            
        # Add the completed synthetic trace to the shadow log
        shadow_log.append(trace)
        mutation_flags.append(had_mutation)
        
    return shadow_log, mutation_flags

def calculate_gen_shadow_stable(event_log, net, im, fm, num_traces, iterations=5, safe_threshold=5, max_n=3):
    """
    Run the shadow log generation K times to ensure mathematical determinism and
    return the mean and standard deviation, plus stratified scores for regular vs.
    mutated traces.
    
    Args:
        max_n: Maximum N-gram order passed through to shadow log generation.
    
    Returns:
        (mean, std, raw_scores, reg_mean, reg_std, mut_mean, mut_std, mutation_counts)
    """
    scores = []
    regular_scores = []    # Per-iteration: mean fitness on non-mutated traces
    mutated_scores = []    # Per-iteration: mean fitness on mutated traces
    mutation_counts = []   # Per-iteration: number of mutated traces

    # Run the stochastic generation and fitness check multiple times
    for i in range(iterations):
        # 1. Generate a new shadow log with mutation flags
        shadow_log, mutation_flags = generate_shadow_log(
            event_log, num_traces=num_traces, safe_threshold=safe_threshold, max_n=max_n
        )
        
        # 2. Replay the shadow log on the discovered model (per-trace for stratification)
        replayed = token_replay.apply(shadow_log, net, im, fm)
        
        # 3. Collect per-trace fitness
        trace_fitnesses = [res['trace_fitness'] for res in replayed]
        
        # Overall fitness (existing metric)
        overall_fitness = sum(trace_fitnesses) / len(trace_fitnesses) if trace_fitnesses else 0.0
        scores.append(overall_fitness)
        
        # 4. Stratified analysis: split by mutation flag
        reg_fits = [f for f, flag in zip(trace_fitnesses, mutation_flags) if not flag]
        mut_fits = [f for f, flag in zip(trace_fitnesses, mutation_flags) if flag]
        
        mutation_counts.append(len(mut_fits))
        regular_scores.append(np.mean(reg_fits) if reg_fits else 0.0)
        mutated_scores.append(np.mean(mut_fits) if mut_fits else 0.0)

    # Aggregate across iterations
    reg_mean = np.mean(regular_scores) if regular_scores else 0.0
    reg_std = np.std(regular_scores) if regular_scores else 0.0
    mut_mean = np.mean(mutated_scores) if mutated_scores else 0.0
    mut_std = np.std(mutated_scores) if mutated_scores else 0.0
    
    return np.mean(scores), np.std(scores), scores, reg_mean, reg_std, mut_mean, mut_std, mutation_counts

# =====================================================================
# 2. Structural Analysis (Gen_struct) 
# =====================================================================

def calculate_gen_struct(event_log, net, initial_marking, final_marking):
    """
    Gen_Struct v2 — Multi-dimensional structural generalization.
    
    Combines four complementary dimensions:
      ArcFlow  (0.35): replay-based arc usage density  (anti-overfit)
      Gini     (0.20): transition activation equality   (anti-overfit)
      Reach    (0.20): reachable arc ratio from BFS     (anti-underfit)
      Cyclo    (0.25): normalized cyclomatic complexity (anti-spaghetti)
    
    Single token_replay shared by ArcFlow + Gini.
    Reachable arc ratio and cyclomatic complexity are structural (no replay).
    """

    # ── 1. Single token replay (shared by ArcFlow + Gini) ──────────
    replayed = token_replay.apply(event_log, net, initial_marking, final_marking)
    num_traces = len(event_log)
    
    # ── 2. Arc Flow Density (existing metric) ──────────────────────
    arc_usage = {arc: 0 for arc in net.arcs}
    for res in replayed:
        used_arcs = set()
        for t in res['activated_transitions']:
            for arc in t.in_arcs:
                used_arcs.add(arc)
            for arc in t.out_arcs:
                used_arcs.add(arc)
        for arc in used_arcs:
            arc_usage[arc] += 1

    total_arcs = len(net.arcs)
    if total_arcs == 0:
        return 0.0

    rare_threshold = max(2, int(num_traces * 0.01))
    rare_arcs = sum(1 for count in arc_usage.values() if count < rare_threshold)
    arc_flow_score = max(0.0, 1.0 - rare_arcs / total_arcs)
    
    # ── 3. Transition Activation Gini (from same replay) ───────────
    from collections import Counter as _Counter
    usage = _Counter()
    for res in replayed:
        fired = set(res['activated_transitions'])
        for t in fired:
            usage[t] += 1
    
    counts = sorted(usage.values())
    n = len(counts)
    gini = 0.0
    if n > 0:
        total = sum(counts)
        if total > 0:
            gini = (2 * sum((i + 1) * c for i, c in enumerate(counts))) / (n * total) - (n + 1) / n
    gini_score = 1.0 - gini  # invert: low gini = uniform usage = better
    
    # ── 4. Reachable Arc Ratio (structural BFS, no replay) ─────────
    from collections import deque as _deque
    im_set = frozenset(initial_marking)
    visited = {im_set: 0}
    queue = _deque([(im_set, 0)])
    reachable_arcs = set()
    max_depth = 12
    
    while queue:
        places, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for t in net.transitions:
            t_inputs = frozenset(a.source for a in t.in_arcs)
            if t_inputs.issubset(places):
                for a in t.in_arcs:
                    reachable_arcs.add(a)
                for a in t.out_arcs:
                    reachable_arcs.add(a)
                nxt = frozenset((places - t_inputs) | frozenset(a.target for a in t.out_arcs))
                if nxt not in visited:
                    visited[nxt] = depth + 1
                    queue.append((nxt, depth + 1))
    
    reach_ratio = len(reachable_arcs) / total_arcs if total_arcs > 0 else 1.0
    reach_score = 1.0 - reach_ratio  # invert: low reachability = more latent structure = better
    
    # ── 5. Cyclomatic Complexity (structural, O(1)) ─────────────────
    n_places = len(net.places)
    n_trans = len(net.transitions)
    n_arcs = len(net.arcs)
    cyclomatic = max(0, n_arcs - n_places - n_trans + 2)
    # Normalize: worst observed is 60 (Heuristics on BPI 2017), cap at 60
    cyclo_norm = min(cyclomatic / 60.0, 1.0)
    cyclo_score = 1.0 - cyclo_norm  # invert: lower complexity = better
    
    # ── 6. Weighted combination ─────────────────────────────────────
    gen_struct = (
        0.35 * arc_flow_score +
        0.20 * gini_score +
        0.20 * reach_score +
        0.25 * cyclo_score
    )
    
    return max(0.0, min(1.0, gen_struct))

# =====================================================================
# 3. Core Evaluation Orchestrator
# =====================================================================

def evaluate_miner(event_log, miner_name, miner_fn, w=0.5, num_shadow_traces=1000, iterations=5, seed=42, max_n=3):
    """
    Executes the hybrid evaluation: Model Discovery -> Gen_struct -> Gen_shadow -> Total Score.
    
    Args:
        max_n: Maximum N-gram order for shadow log generation (default 3).
    """
    #1. Set a seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    print(f"       Evaluating {miner_name}...")
    t0 = time.time()
    
    #2. Execute the process discovery algorithm provided
    net, im, fm = miner_fn(event_log)
    
    #3. Calculate gen_struct
    gen_struct = calculate_gen_struct(event_log, net, im, fm)
    
    #4. Calculate gen_shadow
    # Using safe_threshold=5 optimal for sparse, highly variable logs like BPI 2017
    shadow_mean, shadow_std, raw_scores, reg_mean, reg_std, mut_mean, mut_std, mutation_counts = calculate_gen_shadow_stable(
        event_log, net, im, fm, num_shadow_traces, iterations, safe_threshold=5, max_n=max_n
    )
    
    #5. Combine the scores using the user-defined weight (Default is 0.5)
    gen_total = (w * shadow_mean) + ((1.0 - w) * gen_struct)
    
    #Stop the timer
    runtime = time.time() - t0
    avg_mutations = np.mean(mutation_counts) if mutation_counts else 0
    print(f"         └─ Gen_Total: {gen_total:.4f} | Struct: {gen_struct:.2f} | Shadow Mean: {shadow_mean:.4f} (±{shadow_std:.4f}) | {runtime:.1f}s")
    if avg_mutations > 0:
        print(f"            └─ Stratified: Regular={reg_mean:.4f} (±{reg_std:.4f}) | Mutated({avg_mutations:.0f})={mut_mean:.4f} (±{mut_std:.4f})")
    
    #6. Format the results
    return {
        "miner": miner_name,
        "gen_struct": gen_struct,
        "gen_shadow_mean": shadow_mean,
        "gen_shadow_std": shadow_std,
        "gen_shadow_raw_iterations": raw_scores,
        "gen_total": gen_total,
        "w_weight": w,
        # Stratified mutation analysis
        "gen_shadow_regular_mean": reg_mean,
        "gen_shadow_regular_std": reg_std,
        "gen_shadow_mutated_mean": mut_mean,
        "gen_shadow_mutated_std": mut_std,
        "mutation_counts_per_iteration": mutation_counts,
        "avg_mutations_per_run": avg_mutations,
        #Quick Human-readable version
        "determinism_rating": "High" if shadow_std < 0.02 else "Moderate" if shadow_std < 0.05 else "Low",
        "runtime_s": runtime
    }
# =====================================================================
# HybridGen Registry
# =====================================================================
from . import register_algorithm
register_algorithm("v22")
