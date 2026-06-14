import random
import time
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

import pm4py
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness

__all__ = ["calculate_gen_shadow_stable", "evaluate_miner"]

# =====================================================================
# Generative Behavioral Analysis (v24 - Algorithmic Optimization)
# Features: Variant Compression, O(1) Memoization, Flat Iteration
# =====================================================================

def generate_shadow_log(event_log, num_traces=1000, max_trace_length=100, safe_threshold=5, max_n=6):
    """
    Generates a synthetic shadow log.
    v24 Upgrades:
      1. Variant Compression: Processes unique trace variants rather than raw traces.
      2. O(1) Memoization: Caches Good-Turing math and Katz Backoff resolutions.
      3. Flat Iteration: Replaces DFS recursion with a highly optimized while-loop.
    """
    
    # 1. Variant-Based Log Parsing (Input Compression)
    variant_counts = Counter()
    if isinstance(event_log, pd.DataFrame):
        grouped = event_log.groupby('case:concept:name')
        for _, group in grouped:
            variant_counts[tuple(group['concept:name'])] += 1
    else:
        for trace in event_log:
            variant_counts[tuple(e["concept:name"] for e in trace)] += 1

    # Dictionaries for N-gram frequencies
    ngram_outgoings = {n: defaultdict(Counter) for n in range(1, max_n + 1)}
    ngram_term_totals = {n: Counter() for n in range(1, max_n + 1)}
    ngram_term_ends = {n: Counter() for n in range(1, max_n + 1)}
    starts = Counter()
    alphabet = set()
    original_trace_seqs = set()

    # Process by variant, multiplying by frequency count
    for seq, count in variant_counts.items():
        if not seq: continue
        original_trace_seqs.add(seq)
        starts[seq[0]] += count
        alphabet.update(seq)

        trace_len = len(seq)
        for i in range(trace_len - 1):
            nxt_act = seq[i + 1]
            for n in range(1, max_n + 1):
                if i >= n - 1:
                    state_tuple = seq[i - (n - 1) : i + 1]
                    ngram_outgoings[n][state_tuple][nxt_act] += count

        for i in range(trace_len):
            for n in range(1, max_n + 1):
                if i >= n - 1:
                    term_state = seq[i - n + 1 : i + 1]
                    ngram_term_totals[n][term_state] += count
                    if i == trace_len - 1:
                        ngram_term_ends[n][term_state] += count

    alphabet = list(alphabet)
    start_choices = list(starts.keys())
    start_weights = list(starts.values())

    # 2. O(1) Memoization Caches
    memo_eval = {}
    memo_term = {}

    def get_eval(current_seq):
        """Returns cached (p_unseen, choices, log_weights) for a given state."""
        key = tuple(current_seq[-max_n:])
        if key in memo_eval:
            return memo_eval[key]

        p_unseen, valid_out_edges = None, None
        
        # Katz Backoff resolution
        for n in range(max_n, 1, -1):
            if len(current_seq) >= n:
                state = tuple(current_seq[-n:])
                out_edges = ngram_outgoings[n].get(state, {})
                n_total = sum(out_edges.values())
                
                if n_total >= safe_threshold:
                    n_1 = sum(1 for c in out_edges.values() if c == 1)
                    p_unseen = (n_1 / n_total) if n_total > 0 else 1.0
                    if p_unseen < 1.0:
                        valid_out_edges = out_edges
                        break

        # Fallback to N=1
        if p_unseen is None or p_unseen == 1.0:
            state = (current_seq[-1],)
            out_edges = ngram_outgoings[1].get(state, {})
            n_total = sum(out_edges.values())
            n_1 = sum(1 for c in out_edges.values() if c == 1)
            p_unseen = (n_1 / n_total) if n_total > 0 else 1.0
            valid_out_edges = out_edges

        choices = list(valid_out_edges.keys())
        # Cache the natural log transformation mathematically once!
        weights = [np.log(c + 1) for c in valid_out_edges.values()]

        memo_eval[key] = (p_unseen, choices, weights)
        return p_unseen, choices, weights

    def get_term(current_seq):
        """Returns cached termination probability (p_end) for a given state."""
        key = tuple(current_seq[-max_n:])
        if key in memo_term:
            return memo_term[key]

        p_end = None
        for n in range(max_n, 1, -1):
            if len(current_seq) >= n:
                state = tuple(current_seq[-n:])
                total = ngram_term_totals[n].get(state, 0)
                if total >= safe_threshold:
                    p_end = ngram_term_ends[n].get(state, 0) / total
                    break

        if p_end is None:
            state = (current_seq[-1],)
            total = ngram_term_totals[1].get(state, 0)
            p_end = ngram_term_ends[1].get(state, 0) / total if total > 0 else 0.0

        memo_term[key] = p_end
        return p_end

    # 3. Flat Iteration Generation Loop (Replacing DFS)
    shadow_log = EventLog()
    mutation_flags = []
    MAX_RETRIES = 100

    for i in range(num_traces):
        retries = 0
        while True:
            # 3A. Initialize trace
            curr_act = random.choices(start_choices, weights=start_weights, k=1)[0]
            seq = [curr_act]
            had_mutation = False

            # 3B. Walk
            while len(seq) < max_trace_length:
                # Terminate?
                p_end = get_term(seq)
                if p_end > 0 and random.random() < p_end:
                    break

                # Mutate vs Exploit
                p_unseen, choices, weights = get_eval(seq)
                if random.random() < p_unseen:
                    seq.append(random.choice(alphabet))
                    had_mutation = True
                else:
                    if not choices:
                        break # Dead end
                    seq.append(random.choices(choices, weights=weights, k=1)[0])

            # 3C. Deduplication Check (Must be 100% Unseen)
            if tuple(seq) not in original_trace_seqs:
                break
                
            retries += 1
            if retries >= MAX_RETRIES:
                break

        # Convert to EventLog object
        trace = Trace(attributes={"concept:name": f"shadow_{i}"})
        for act in seq:
            trace.append(Event({"concept:name": act}))
            
        shadow_log.append(trace)
        mutation_flags.append(had_mutation)

    return shadow_log, mutation_flags


def calculate_gen_shadow_stable(event_log, net, im, fm, num_traces, iterations=5, safe_threshold=5, max_n=6):
    """Runs generation iteratively and evaluates fitness."""
    scores = []
    regular_scores, mutated_scores, mutation_counts = [], [], []

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

def evaluate_miner(event_log, miner_name, miner_fn, w=0.5, num_shadow_traces=1000, iterations=5, seed=42, max_n=6):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    print(f"       Evaluating {miner_name} (v24)...")
    t0 = time.time()
    
    net, im, fm = miner_fn(event_log)
    shadow_mean, shadow_std, raw_scores, reg_mean, reg_std, mut_mean, mut_std, mutation_counts = calculate_gen_shadow_stable(
        event_log, net, im, fm, num_shadow_traces, iterations, safe_threshold=5, max_n=max_n
    )
    
    runtime = time.time() - t0
    
    return {
        "miner": miner_name,
        "gen_struct": 0.0,
        "gen_shadow_mean": shadow_mean,
        "gen_shadow_std": shadow_std,
        "gen_total": shadow_mean,
        "runtime_s": runtime
    }

# Register to the library
from . import register_algorithm
register_algorithm("v2.4")