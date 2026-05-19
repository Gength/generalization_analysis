import random
import time
import sys
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

import pm4py
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness

# Increase recursion depth slightly just to be safe with long traces
sys.setrecursionlimit(2000)

# =====================================================================
# 1. Generative Behavioral Analysis (Gen_shadow) with DFS & Katz Backoff
# =====================================================================

def _evaluate_state(state_tuple, ngram_dict, safe_threshold):
    """Helper function to evaluate statistical safety and Good-Turing probability."""
    out_edges = ngram_dict.get(state_tuple, {})
    n_total = sum(out_edges.values())
    
    # Check for data sparsity (Curse of Dimensionality)
    if n_total < safe_threshold:
        return None, None
        
    n_1 = sum(1 for count in out_edges.values() if count == 1)
    p_unseen = (n_1 / n_total) if n_total > 0 else 1.0
    
    # State collapse check
    if p_unseen == 1.0:
        return None, None
        
    return p_unseen, out_edges

def generate_trace_dfs(current_seq, ngram_outgoings, ends, alphabet, max_length, safe_threshold, max_n, had_mutation=False):
    """Recursive DFS function to generate a single shadow trace."""
    current_local = current_seq[-1]
    
    # 1. Termination Checks
    if current_local in ends:
        times_as_end = ends[current_local]
        out_edges_n1 = ngram_outgoings[1].get((current_local,), {})
        total_occurrences = times_as_end + sum(out_edges_n1.values())
        
        if random.random() < (times_as_end / total_occurrences):
            return current_seq, had_mutation
            
    if len(current_seq) >= max_length:
        return current_seq, had_mutation

    # 2. Dynamic Katz Backoff Strategy (max_n -> ... -> 2 -> 1)
    p_unseen, valid_out_edges = None, None
    
    for n in range(max_n, 1, -1):
        if len(current_seq) >= n:
            p_unseen, valid_out_edges = _evaluate_state(
                tuple(current_seq[-n:]), ngram_outgoings[n], safe_threshold
            )
            if p_unseen is not None:
                break
        
    # Absolute Fallback to N=1
    if p_unseen is None:
        out_edges = ngram_outgoings[1].get((current_local,), {})
        n_total = sum(out_edges.values())
        n_1 = sum(1 for count in out_edges.values() if count == 1)
        p_unseen = (n_1 / n_total) if n_total > 0 else 1.0
        valid_out_edges = out_edges

    # 3. Mutate vs. Exploit
    if random.random() < p_unseen:
        next_node = random.choice(alphabet)
        had_mutation = True
    else:
        if not valid_out_edges:
            return current_seq, had_mutation 
        next_node = random.choices(list(valid_out_edges.keys()), weights=list(valid_out_edges.values()), k=1)[0]
        
    # 4. DFS Recursion
    return generate_trace_dfs(current_seq + [next_node], ngram_outgoings, ends, alphabet, max_length, safe_threshold, max_n, had_mutation)

def generate_shadow_log(event_log, num_traces=1000, max_trace_length=100, safe_threshold=5, max_n=3):
    """Generates a synthetic 'shadow' log using Good-Turing estimation and Katz Backoff."""
    
    ngram_outgoings = {n: defaultdict(Counter) for n in range(1, max_n + 1)}
    starts = Counter()
    ends = Counter()
    alphabet = set()
    
    # --- PANDAS DATAFRAME FIX ---
    if isinstance(event_log, pd.DataFrame):
        grouped = event_log.groupby('case:concept:name')
        trace_list = [group['concept:name'].tolist() for _, group in grouped]
    else:
        trace_list = [[event["concept:name"] for event in trace] for trace in event_log]
    # ----------------------------

    # Extract every unique activity and path in the log
    for seq in trace_list:
        if not seq: continue
            
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
    mutation_flags = []
    start_choices = list(starts.keys())
    start_weights = list(starts.values())

    for i in range(num_traces):
        start_node = random.choices(start_choices, weights=start_weights, k=1)[0]
        
        final_sequence, had_mutation = generate_trace_dfs(
            [start_node], ngram_outgoings, ends, alphabet, max_trace_length, safe_threshold, max_n
        )
        
        trace = Trace(attributes={"concept:name": f"shadow_{i}"})
        for act in final_sequence:
            trace.append(Event({"concept:name": act}))
            
        shadow_log.append(trace)
        mutation_flags.append(had_mutation)
        
    return shadow_log, mutation_flags

def calculate_gen_shadow_stable(event_log, net, im, fm, num_traces, iterations=5, safe_threshold=5, max_n=3):
    """Run the shadow log generation K times to ensure mathematical determinism."""
    scores, regular_scores, mutated_scores, mutation_counts = [], [], [], []

    for i in range(iterations):
        shadow_log, mutation_flags = generate_shadow_log(
            event_log, num_traces=num_traces, safe_threshold=safe_threshold, max_n=max_n
        )
        
        replayed = token_replay.apply(shadow_log, net, im, fm)
        trace_fitnesses = [res['trace_fitness'] for res in replayed]
        
        overall_fitness = sum(trace_fitnesses) / len(trace_fitnesses) if trace_fitnesses else 0.0
        scores.append(overall_fitness)
        
        reg_fits = [f for f, flag in zip(trace_fitnesses, mutation_flags) if not flag]
        mut_fits = [f for f, flag in zip(trace_fitnesses, mutation_flags) if flag]
        
        mutation_counts.append(len(mut_fits))
        regular_scores.append(np.mean(reg_fits) if reg_fits else 0.0)
        mutated_scores.append(np.mean(mut_fits) if mut_fits else 0.0)

    reg_mean = np.mean(regular_scores) if regular_scores else 0.0
    reg_std = np.std(regular_scores) if regular_scores else 0.0
    mut_mean = np.mean(mutated_scores) if mutated_scores else 0.0
    mut_std = np.std(mutated_scores) if mutated_scores else 0.0
    
    return np.mean(scores), np.std(scores), scores, reg_mean, reg_std, mut_mean, mut_std, mutation_counts

# =====================================================================
# 2. Structural Analysis (Gen_struct) 
# =====================================================================

def calculate_gen_struct(event_log, net, initial_marking, final_marking):
    replayed = token_replay.apply(event_log, net, initial_marking, final_marking)
    arc_usage = {arc: 0 for arc in net.arcs}
    
    for res in replayed:
        used_arcs = set()
        for t in res['activated_transitions']:
            for arc in t.in_arcs: used_arcs.add(arc)
            for arc in t.out_arcs: used_arcs.add(arc)
        for arc in used_arcs: arc_usage[arc] += 1

    total_arcs = len(net.arcs)
    if total_arcs == 0: return 0.0
        
    num_traces = len(event_log['case:concept:name'].unique()) if isinstance(event_log, pd.DataFrame) else len(event_log)
    rare_threshold = max(2, int(num_traces * 0.01))
    rare_arcs = sum(1 for arc, count in arc_usage.items() if count < rare_threshold)
    return max(0.0, 1.0 - (rare_arcs / total_arcs))

# =====================================================================
# 3. Core Evaluation Orchestrator
# =====================================================================

def evaluate_miner(event_log, miner_name, miner_fn, w=0.5, num_shadow_traces=1000, iterations=5, seed=42, max_n=3):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    print(f"       Evaluating {miner_name}...")
    t0 = time.time()
    
    net, im, fm = miner_fn(event_log)
    gen_struct = calculate_gen_struct(event_log, net, im, fm)
    
    shadow_mean, shadow_std, raw_scores, reg_mean, reg_std, mut_mean, mut_std, mutation_counts = calculate_gen_shadow_stable(
        event_log, net, im, fm, num_shadow_traces, iterations, safe_threshold=5, max_n=max_n
    )
    
    gen_total = (w * shadow_mean) + ((1.0 - w) * gen_struct)
    runtime = time.time() - t0
    
    return {
        "miner": miner_name,
        "gen_struct": gen_struct,
        "gen_shadow_mean": shadow_mean,
        "gen_shadow_std": shadow_std,
        "gen_shadow_raw_iterations": raw_scores,
        "gen_total": gen_total,
        "w_weight": w,
        "gen_shadow_regular_mean": reg_mean,
        "gen_shadow_regular_std": reg_std,
        "gen_shadow_mutated_mean": mut_mean,
        "gen_shadow_mutated_std": mut_std,
        "mutation_counts_per_iteration": mutation_counts,
        "avg_mutations_per_run": np.mean(mutation_counts) if mutation_counts else 0,
        "determinism_rating": "High" if shadow_std < 0.02 else "Moderate" if shadow_std < 0.05 else "Low",
        "runtime_s": runtime
    }