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
__all__ = [
    "calculate_gen_shadow_stable",
    "evaluate_miner",
]
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

def generate_trace_dfs(current_seq, ngram_outgoings, ngram_termination_ends, ngram_termination_totals, ends, alphabet, max_length, safe_threshold, max_n, had_mutation=False):
    """
    Recursive DFS function to generate a single shadow trace.
    Naturally maintains the state context through the call stack.
    
    Args:
        max_n: Maximum N-gram order (lookback = max_n - 1). Backoff from max_n down to 2,
               with absolute fallback to N=1.
        ngram_termination_ends, ngram_termination_totals: [v23 FIX] N-gram termination
               statistics for context-aware termination probability (Katz backoff).
    
    Returns:
        (sequence, had_mutation): The generated activity sequence and a boolean flag
        indicating whether any mutation event occurred during generation.
    """
    current_local = current_seq[-1]
    
    # =====================================================================
    # [v23 FIX] Context-aware termination with Katz Backoff
    #
    # OLD BUG: Termination used only current_local (activity name), ignoring
    # context. Activities like "Release A" had P_end=0.59 regardless of whether
    # they appeared at step 3 or step 30. In the original log these activities
    # ALWAYS appear in the last 20% of traces, but the DFS walker generated
    # them early and terminated prematurely — causing shadow traces to be
    # systematically shorter than originals.
    #
    # FIX: Apply the same Katz backoff strategy used for next-activity selection.
    #   Try N=max_n -> ... -> N=2 using the last N activities as state.
    #   If state has sufficient support (>= safe_threshold), compute
    #   P_end = end_count / total_count. Fall back to N-1 if sparse.
    #   Absolute fallback to N=1 (always available, like next-activity backoff).
    #
    # Result: A state (Leucocytes, Release A) vs (CRP, Release A) will have
    # DIFFERENT termination probabilities based on their historical ending
    # behavior, not a single flat probability for "Release A".
    # =====================================================================
    p_end = None
    
    # Try from highest N down to 2
    for n in range(max_n, 1, -1):
        if len(current_seq) >= n:
            state = tuple(current_seq[-n:])
            total = ngram_termination_totals[n].get(state, 0)
            if total >= safe_threshold:
                end_count = ngram_termination_ends[n].get(state, 0)
                p_end = end_count / total
                break
    
    # Absolute Fallback to N=1 (Local Marking, Lookback 0)
    if p_end is None:
        state = (current_local,)
        total = ngram_termination_totals[1].get(state, 0)
        end_count = ngram_termination_ends[1].get(state, 0)
        p_end = end_count / total if total > 0 else 0.0
    
    # Probabilistic roll to stop the trace
    if p_end > 0 and random.random() < p_end:
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
    return generate_trace_dfs(current_seq + [next_node], ngram_outgoings,
                              ngram_termination_ends, ngram_termination_totals,
                              ends, alphabet, max_length, safe_threshold, max_n, had_mutation)

def generate_shadow_log(event_log, num_traces=1000, max_trace_length=100, safe_threshold=5, max_n=3):
    """
    Generates a synthetic 'shadow' log acting as probabilistic based future behavior 
    using Good-Turing estimation and Katz Backoff.
    
    v23 change: Any generated trace that matches an original trace from event_log 
    exactly (same sequence of activities) is discarded and regenerated.
    
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
    
    # v23: Also collect the set of original trace sequences for deduplication
    original_trace_seqs = set()
    
    # =====================================================================
    # [v23 FIX] Pre-compute N-gram termination statistics for context-aware
    # termination. For each N-gram state (N=1..max_n), track:
    #   - ngram_termination_totals[n][state] = total occurrences
    #   - ngram_termination_ends[n][state]  = count as trace end
    # These are used by generate_trace_dfs with Katz backoff to compute
    # context-dependent P_end(state) rather than flat per-activity P_end.
    # =====================================================================
    ngram_termination_ends = {n: Counter() for n in range(1, max_n + 1)}
    ngram_termination_totals = {n: Counter() for n in range(1, max_n + 1)}
    
    # Extract every unique activity and path in the log
    for trace in event_log:
        seq = [event["concept:name"] for event in trace]
        if not seq:
            continue
            
        # v23: Store original trace sequence as tuple for fast lookup
        original_trace_seqs.add(tuple(seq))
        
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
        
        # [v23 FIX] Collect N-gram termination statistics for context-aware P_end
        for i in range(trace_len):
            for n in range(1, max_n + 1):
                if i >= n - 1:
                    term_state = tuple(seq[i - n + 1 : i + 1])
                    ngram_termination_totals[n][term_state] += 1
                    if i == trace_len - 1:  # last activity = trace end
                        ngram_termination_ends[n][term_state] += 1

    alphabet = list(alphabet)
    shadow_log = EventLog()
    mutation_flags = []
    start_choices = list(starts.keys())
    start_weights = list(starts.values())

    # v23: Track deduplication statistics
    MAX_RETRIES_PER_TRACE = 100
    total_retries = 0

    # 2. Begin generating the requested number of synthetic traces
    for i in range(num_traces):
        retries = 0
        
        while True:
            # Pick a starting activity based on its historical likelihood of starting a process
            start_node = random.choices(start_choices, weights=start_weights, k=1)[0]
            
            # Execute the recursive DFS walker to build the sequence
            final_sequence, had_mutation = generate_trace_dfs(
                [start_node], ngram_outgoings,
                ngram_termination_ends, ngram_termination_totals,  # [v23 FIX] context-aware termination
                ends, alphabet, max_trace_length, safe_threshold, max_n
            )
            
            # v23: Check if this trace matches any original trace exactly
            if tuple(final_sequence) not in original_trace_seqs:
                break  # unique trace — accept it
            
            retries += 1
            total_retries += 1
            
            if retries >= MAX_RETRIES_PER_TRACE:
                # Safety cap: accept the trace anyway to avoid infinite loops
                # (extremely unlikely except on tiny logs with very constrained generation)
                print(f"  [v23 warn] trace {i}: reached max retries ({MAX_RETRIES_PER_TRACE}), accepting duplicate")
                break
        
        # Convert the generated string sequence back into a PM4Py Trace object
        trace = Trace(attributes={"concept:name": f"shadow_{i}"})
        for act in final_sequence:
            trace.append(Event({"concept:name": act}))
            
        # Add the completed synthetic trace to the shadow log
        shadow_log.append(trace)
        mutation_flags.append(had_mutation)
    
    if total_retries > 0:
        print(f"  [v23] dedup: {total_retries} traces regenerated (avg {total_retries/num_traces:.1f} per shadow trace)")
        
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
# 2. Core Evaluation Orchestrator  (v23: Gen_struct removed)
# =====================================================================

def evaluate_miner(event_log, miner_name, miner_fn, w=0.5, num_shadow_traces=1000, iterations=5, seed=42, max_n=3):
    """
    Executes the evaluation: Model Discovery -> Gen_shadow -> Total Score.
    
    v23 change: calculate_gen_struct is removed. The score relies solely on 
    Gen_shadow (token replay fitness of the shadow log on the discovered model).
    The w weight parameter is accepted for API compatibility but has no effect.
    
    Args:
        max_n: Maximum N-gram order for shadow log generation (default 3).
    """
    #1. Set a seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    print(f"       Evaluating {miner_name} (v23: Gen_shadow only)...")
    t0 = time.time()
    
    #2. Execute the process discovery algorithm provided
    net, im, fm = miner_fn(event_log)
    
    #3. Calculate gen_shadow
    # Using safe_threshold=5 optimal for sparse, highly variable logs like BPI 2017
    shadow_mean, shadow_std, raw_scores, reg_mean, reg_std, mut_mean, mut_std, mutation_counts = calculate_gen_shadow_stable(
        event_log, net, im, fm, num_shadow_traces, iterations, safe_threshold=5, max_n=max_n
    )
    
    #4. Score (v23: gen_struct removed — total = shadow_mean)
    gen_total = shadow_mean
    gen_struct = 0.0  # placeholder for backwards-compatible result dict
    
    #Stop the timer
    runtime = time.time() - t0
    avg_mutations = np.mean(mutation_counts) if mutation_counts else 0
    print(f"         └─ Gen_Total (v23): {gen_total:.4f} | Shadow Mean: {shadow_mean:.4f} (±{shadow_std:.4f}) | {runtime:.1f}s")
    if avg_mutations > 0:
        print(f"            └─ Stratified: Regular={reg_mean:.4f} (±{reg_std:.4f}) | Mutated({avg_mutations:.0f})={mut_mean:.4f} (±{mut_std:.4f})")
    
    #5. Format the results
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
register_algorithm("v2.3")
