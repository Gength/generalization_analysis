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
# Generative Behavioral Analysis (v25 - Katz-Consistent Mutation)
# See Method_GenShadow.md §6 for the design rationale.
#
# Changes over v24:
#   1. Katz-consistent mutation proposal: a mutated activity is drawn from
#      the deepest lower-order successor distribution restricted to
#      activities UNSEEN at the resolved order ("plausible in a related,
#      shorter context, but never observed in this specific one"),
#      falling back to global activity frequencies, instead of uniform
#      over the alphabet. p_unseen (the Good-Turing mass) is unchanged,
#      so the mutation *rate* calibration of the N-sweep is preserved;
#      only the proposal (what gets inserted) changes.
#   2. Duplicate-leak accounting: v24 silently kept a duplicate trace
#      when the dedup retry cap (100) was exhausted. v25 counts these
#      and reports them, so inflated scores on highly repetitive logs
#      (e.g. TLRA >= 0.95) become visible.
#   3. Openness profile in the result: regular/mutated fitness are
#      reported separately alongside the blended score.
# =====================================================================

def generate_shadow_log(event_log, num_traces=1000, max_trace_length=100, safe_threshold=5, max_n=6):
    """
    Generates a synthetic shadow log.
    Returns (shadow_log, mutation_flags, duplicates_kept).
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
    global_counts = Counter()
    alphabet = set()
    original_trace_seqs = set()

    # Process by variant, multiplying by frequency count
    for seq, count in variant_counts.items():
        if not seq: continue
        original_trace_seqs.add(seq)
        starts[seq[0]] += count
        alphabet.update(seq)
        for act in seq:
            global_counts[act] += count

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
        """Returns cached (p_unseen, choices, log_weights, mut_choices, mut_weights)."""
        key = tuple(current_seq[-max_n:])
        if key in memo_eval:
            return memo_eval[key]

        p_unseen, valid_out_edges = None, None

        # Katz Backoff resolution (identical to v24)
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
        weights = [np.log(c + 1) for c in valid_out_edges.values()]

        # ---- v25: Katz-consistent mutation proposal -------------------
        # Successors of a longer suffix are a subset of successors of any
        # shorter suffix, so scanning orders deepest-first and restricting
        # to activities unseen at the resolved order yields the deepest
        # lower-order context that offers genuinely novel continuations.
        seen = set(valid_out_edges.keys())
        mut_choices, mut_weights = [], []
        for m in range(min(len(current_seq), max_n), 0, -1):
            state = tuple(current_seq[-m:])
            cand = {a: c for a, c in ngram_outgoings[m].get(state, {}).items() if a not in seen}
            if cand:
                mut_choices = list(cand.keys())
                mut_weights = [np.log(c + 1) for c in cand.values()]
                break
        if not mut_choices:
            # Global-frequency fallback: any activity the resolved context
            # has not produced, weighted by overall prevalence.
            cand = {a: c for a, c in global_counts.items() if a not in seen}
            mut_choices = list(cand.keys())
            mut_weights = [np.log(c + 1) for c in cand.values()]
        # If mut_choices is still empty, the context has been observed
        # before every activity in the alphabet; mutation is impossible
        # and the walker exploits instead (handled in the walk loop).

        memo_eval[key] = (p_unseen, choices, weights, mut_choices, mut_weights)
        return memo_eval[key]

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

    # 3. Flat Iteration Generation Loop
    shadow_log = EventLog()
    mutation_flags = []
    duplicates_kept = 0
    truncated = 0
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
                p_unseen, choices, weights, mut_choices, mut_weights = get_eval(seq)
                if random.random() < p_unseen and mut_choices:
                    seq.append(random.choices(mut_choices, weights=mut_weights, k=1)[0])
                    had_mutation = True
                else:
                    if not choices:
                        break # Dead end
                    seq.append(random.choices(choices, weights=weights, k=1)[0])

            if len(seq) >= max_trace_length:
                # Trace was cut mid-walk, not ended by p_end: it is an
                # incomplete process instance, not valid future behavior,
                # and token replay will penalize it on any model.
                truncated += 1

            # 3C. Deduplication Check (Must be 100% Unseen)
            if tuple(seq) not in original_trace_seqs:
                break

            retries += 1
            if retries >= MAX_RETRIES:
                duplicates_kept += 1
                break

        # Convert to EventLog object
        trace = Trace(attributes={"concept:name": f"shadow_{i}"})
        for act in seq:
            trace.append(Event({"concept:name": act}))

        shadow_log.append(trace)
        mutation_flags.append(had_mutation)

    return shadow_log, mutation_flags, duplicates_kept, truncated


def calculate_gen_shadow_stable(event_log, net, im, fm, num_traces, iterations=5, safe_threshold=5, max_n=6):
    """Runs generation iteratively and evaluates fitness."""
    scores = []
    regular_scores, mutated_scores, mutation_counts = [], [], []
    duplicates_kept_total = 0
    truncated_total = 0

    for i in range(iterations):
        shadow_log, mutation_flags, duplicates_kept, truncated = generate_shadow_log(
            event_log, num_traces=num_traces, safe_threshold=safe_threshold, max_n=max_n
        )
        duplicates_kept_total += duplicates_kept
        truncated_total += truncated
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

    return (np.mean(scores), np.std(scores), scores, reg_mean, reg_std,
            mut_mean, mut_std, mutation_counts, duplicates_kept_total, truncated_total)

def evaluate_miner(event_log, miner_name, miner_fn, w=0.5, num_shadow_traces=1000, iterations=5, seed=42, max_n=6):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    print(f"       Evaluating {miner_name} (v25)...")
    t0 = time.time()

    net, im, fm = miner_fn(event_log)
    (shadow_mean, shadow_std, raw_scores, reg_mean, reg_std,
     mut_mean, mut_std, mutation_counts, duplicates_kept, truncated) = calculate_gen_shadow_stable(
        event_log, net, im, fm, num_shadow_traces, iterations, safe_threshold=5, max_n=max_n
    )

    runtime = time.time() - t0

    return {
        "miner": miner_name,
        "gen_struct": 0.0,
        "gen_shadow_mean": shadow_mean,
        "gen_shadow_std": shadow_std,
        "gen_shadow_raw_iterations": list(raw_scores),
        "gen_total": shadow_mean,
        # Openness profile (Method_GenShadow.md §6, option 3)
        "gen_shadow_regular": reg_mean,
        "gen_shadow_regular_std": reg_std,
        "gen_shadow_mutated": mut_mean,
        "gen_shadow_mutated_std": mut_std,
        "mutated_traces_per_iteration": mutation_counts,
        # Dedup transparency: > 0 means the retry cap was exhausted and
        # duplicate traces entered the shadow log (score inflation risk).
        "duplicates_kept": duplicates_kept,
        # Truncation transparency: traces cut at max_trace_length are
        # incomplete process instances and depress fitness artifactually.
        "truncated_traces": truncated,
        "runtime_s": runtime
    }

# Register to the library
from . import register_algorithm
register_algorithm("v25")
