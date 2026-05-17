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

def generate_trace_dfs(current_seq, ngram_outgoings, ends, alphabet, max_length, safe_threshold):
    """
    Recursive DFS function to generate a single shadow trace.
    Naturally maintains the state context through the call stack.
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
            return current_seq
            
    # Safety catch: Prevent infinite loops in case of cyclic mutations
    if len(current_seq) >= max_length:
        return current_seq

    # 2. Dynamic Katz Backoff Strategy (N=3 -> N=2 -> N=1)
    p_unseen, valid_out_edges = None, None
    
    # Attempt N=3 (Regional Marking, Lookback 2)
    # Highest context fidelity, but most vulnerable to data sparsity
    if len(current_seq) >= 3:
        p_unseen, valid_out_edges = _evaluate_state(tuple(current_seq[-3:]), ngram_outgoings[3], safe_threshold)
        
    # Fallback to N=2 (Lookback 1)
    # Medium context fidelity, used when N=3 is too sparse
    if p_unseen is None and len(current_seq) >= 2:
        p_unseen, valid_out_edges = _evaluate_state(tuple(current_seq[-2:]), ngram_outgoings[2], safe_threshold)
        
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
        next_node = random.choice(alphabet)
    else:
        # Predictable Path Triggered: Follow historical DFG distribution.
        if not valid_out_edges:
            return current_seq # Dead end reached
        # Weighted random choice based on how many times paths were taken in the past
        next_node = random.choices(list(valid_out_edges.keys()), weights=list(valid_out_edges.values()), k=1)[0]
        
    # 4. DFS Recursion
    # Pass a new list (current_seq + [next_node]) to avoid mutation side-effects and maintain immutability
    return generate_trace_dfs(current_seq + [next_node], ngram_outgoings, ends, alphabet, max_length, safe_threshold)

def generate_shadow_log(event_log, num_traces=1000, max_trace_length=100, safe_threshold=5):
    """
    Generates a synthetic 'shadow' log acting as probabilistic based future behavior 
    using Good-Turing estimation and Katz Backoff.
    """
    
    # 1. Discover historical N-gram Directly-Follows Graphs (#howTheProcessExecutedInThePast)
    # Pre-compute dictionaries up to N=3 for fast lookups during generation
    ngram_outgoings = {1: defaultdict(Counter), 2: defaultdict(Counter), 3: defaultdict(Counter)}
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
            # Extract states for N=1 (Local), N=2 (Lookback 1), N=3 (Lookback 2)
            for n in range(1, 4):
                if i >= n - 1:
                    state_tuple = tuple(seq[i - (n - 1) : i + 1])
                    ngram_outgoings[n][state_tuple][nxt_act] += 1

    alphabet = list(alphabet)
    shadow_log = EventLog()
    start_choices = list(starts.keys())
    start_weights = list(starts.values())

    # 2. Begin generating the requested number of synthetic traces
    for i in range(num_traces):
        # Pick a starting activity based on its historical likelihood of starting a process
        start_node = random.choices(start_choices, weights=start_weights, k=1)[0]
        
        # Execute the recursive DFS walker to build the sequence
        final_sequence = generate_trace_dfs([start_node], ngram_outgoings, ends, alphabet, max_trace_length, safe_threshold)
        
        # Convert the generated string sequence back into a PM4Py Trace object
        trace = Trace(attributes={"concept:name": f"shadow_{i}"})
        for act in final_sequence:
            trace.append(Event({"concept:name": act}))
            
        # Add the completed synthetic trace to the shadow log
        shadow_log.append(trace)
        
    return shadow_log

def calculate_gen_shadow_stable(event_log, net, im, fm, num_traces, iterations=5, safe_threshold=5):
    """
    Run the shadow log generation K times to ensure mathematical determinism and return the mean and standard deviation
    """
    scores = []

    #Run the stochastic generation and fitness check multiple times
    for i in range(iterations):
        #1. Generate a new shadow log
        shadow_log = generate_shadow_log(event_log, num_traces=num_traces, safe_threshold=safe_threshold)
        
        #2. Replay the shadow log on the discovered model
        #If the model is too restrictive it will fail
        fitness_res = replay_fitness.apply(shadow_log, net, im, fm, variant=replay_fitness.Variants.TOKEN_BASED)

        #Store the fitness score (0.0 to 1.0) of this iteration
        scores.append(fitness_res['log_fitness'])
    
    #Return the mean + variance + iteration list
    return np.mean(scores), np.std(scores), scores

# =====================================================================
# 2. Structural Analysis (Gen_struct) 
# =====================================================================

def calculate_gen_struct(event_log, net, initial_marking, final_marking):
    """
    Evaluates Structure (#OverfittingPenalty) via "Arc Flow Density".
    Penalizes the model if it contains numerous rarely used arcs (Spaghetti Model).
    """

    #1. Replay the original log against the discovered model
    replayed = token_replay.apply(event_log, net, initial_marking, final_marking)

    #2. Initialize a counter for every arc in the Petri Net
    arc_usage = {arc: 0 for arc in net.arcs}
    
    #3. Track how many Traces use each arc
    for res in replayed:
        used_arcs = set()
        
        #Look at every transition that was fired to support this trace
        for t in res['activated_transitions']:
            #Record the incoming and outgoing structural "wires" for that transition
            for arc in t.in_arcs:
                used_arcs.add(arc)
            for arc in t.out_arcs:
                used_arcs.add(arc)
                
        # Count an arc only once per trace
        for arc in used_arcs:
            arc_usage[arc] += 1

    #Calculate the total number of structural wires in the model
    total_arcs = len(net.arcs)
    
    #Safety catch for completely broken or empty models
    if total_arcs == 0:
        return 0.0
        
    #4. Defining Bloat 
    #An arc is "bloated" if it handles a negligible fraction of the log (i.e. 1%)
    num_traces = len(event_log)

    #We use max(2, ...) to ensure that paths created purely for a single outlier trace (count=1) are always penalized, regardless of how large the log is
    rare_threshold = max(2, int(num_traces * 0.01))

    #Count how many arcs in the model fall below this usefulness threshold
    rare_arcs = sum(1 for arc, count in arc_usage.items() if count < rare_threshold)

    #5. Calculate the structural penalty
    #i.e. If 80% of the model's arcs are only used to handle 1% of the exceptions, the overfit_penalty is 0.8
    overfit_penalty = rare_arcs / total_arcs

    #Ensure the score never drops below absolute 0
    return max(0.0, 1.0 - overfit_penalty)

# =====================================================================
# 3. Core Evaluation Orchestrator
# =====================================================================

def evaluate_miner(event_log, miner_name, miner_fn, w=0.5, num_shadow_traces=1000, iterations=5, seed=42):
    """
    Executes the hybrid evaluation: Model Discovery -> Gen_struct -> Gen_shadow -> Total Score.
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
    shadow_mean, shadow_std, raw_scores = calculate_gen_shadow_stable(
        event_log, net, im, fm, num_shadow_traces, iterations, safe_threshold=5
    )
    
    #5. Combine the scores using the user-defined weight (Default is 0.5)
    gen_total = (w * shadow_mean) + ((1.0 - w) * gen_struct)
    
    #Stop the timer
    runtime = time.time() - t0
    print(f"         └─ Gen_Total: {gen_total:.4f} | Struct: {gen_struct:.2f} | Shadow Mean: {shadow_mean:.4f} (±{shadow_std:.4f}) | {runtime:.1f}s")
    
    #6. Format the results
    return {
        "miner": miner_name,
        "gen_struct": gen_struct,
        "gen_shadow_mean": shadow_mean,
        "gen_shadow_std": shadow_std,
        "gen_shadow_raw_iterations": raw_scores,
        "gen_total": gen_total,
        "w_weight": w,
        #Quick Human-readable version
        "determinism_rating": "High" if shadow_std < 0.02 else "Moderate" if shadow_std < 0.05 else "Low",
        "runtime_s": runtime
    }