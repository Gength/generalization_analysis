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
# Generative Behavioral Analysis (v26 - Acceptance & Probe Integrity)
# See Method_GenShadow.md §6-7 for the design rationale.
#
# Changes over v25 (which introduced the Katz-consistent mutation
# proposal and duplicate/truncation accounting):
#   1. Acceptance rate: alongside mean token-replay fitness (partial
#      credit), v26 reports gen_accept — the fraction of shadow traces
#      the model replays PERFECTLY (pm4py 'trace_is_fit'). This is the
#      direct operationalization of "accepts future valid behavior";
#      mean fitness is kept as gen_total for comparability.
#      Both are reported stratified (regular vs mutated).
#   2. Data-driven trace-length cap: max_trace_length defaults to
#      min(max(100, 2 x longest observed trace), 1000) instead of a
#      hard 100. Walks cut at the cap are incomplete process instances
#      that depress fitness artifactually on strict models (Sepsis has
#      traces up to length 185 > 100). Truncations are still counted.
#   3. successor_weighting parameter: 'log' (legacy ln(f+1) damping,
#      default for comparability) or 'mle' (proportional to raw
#      frequency). 'mle' samples from the estimated future-trace
#      distribution itself, making the score an unbiased acceptance
#      estimate and resolving the inconsistency of computing p_unseen
#      from raw counts while sampling from damped ones. 'log' remains
#      the deliberate rare-behavior stress-test mode.
# =====================================================================

def generate_shadow_log(event_log, num_traces=1000, max_trace_length=None,
                        safe_threshold=5, max_n=6, successor_weighting="log"):
    """
    Generates a synthetic shadow log.
    Returns (shadow_log, mutation_flags, duplicates_kept, truncated, cap_used).
    """

    if successor_weighting == "log":
        weight_fn = lambda c: np.log(c + 1)
    elif successor_weighting == "mle":
        weight_fn = float
    else:
        raise ValueError(f"Unknown successor_weighting: {successor_weighting!r} (use 'log' or 'mle')")

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
    max_observed_len = 0

    # Process by variant, multiplying by frequency count
    for seq, count in variant_counts.items():
        if not seq: continue
        original_trace_seqs.add(seq)
        starts[seq[0]] += count
        alphabet.update(seq)
        max_observed_len = max(max_observed_len, len(seq))
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

    # v26: data-driven cap — generous enough for legitimate long walks,
    # bounded against degenerate loops.
    if max_trace_length is None:
        max_trace_length = min(max(100, 2 * max_observed_len), 1000)

    alphabet = list(alphabet)
    start_choices = list(starts.keys())
    start_weights = list(starts.values())

    # 2. O(1) Memoization Caches
    memo_eval = {}
    memo_term = {}

    def get_eval(current_seq):
        """Returns cached (p_unseen, choices, weights, mut_choices, mut_weights)."""
        key = tuple(current_seq[-max_n:])
        if key in memo_eval:
            return memo_eval[key]

        p_unseen, valid_out_edges = None, None

        # Katz Backoff resolution (identical to v24/v25)
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
        weights = [weight_fn(c) for c in valid_out_edges.values()]

        # Katz-consistent mutation proposal (v25): deepest lower-order
        # context offering successors unseen at the resolved order.
        seen = set(valid_out_edges.keys())
        mut_choices, mut_weights = [], []
        for m in range(min(len(current_seq), max_n), 0, -1):
            state = tuple(current_seq[-m:])
            cand = {a: c for a, c in ngram_outgoings[m].get(state, {}).items() if a not in seen}
            if cand:
                mut_choices = list(cand.keys())
                mut_weights = [weight_fn(c) for c in cand.values()]
                break
        if not mut_choices:
            cand = {a: c for a, c in global_counts.items() if a not in seen}
            mut_choices = list(cand.keys())
            mut_weights = [weight_fn(c) for c in cand.values()]

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

    return shadow_log, mutation_flags, duplicates_kept, truncated, max_trace_length


def calculate_gen_shadow_stable(event_log, net, im, fm, num_traces, iterations=5,
                                safe_threshold=5, max_n=6, successor_weighting="log"):
    """Runs generation iteratively, evaluates fitness AND acceptance."""
    scores, accept_scores = [], []
    regular_scores, mutated_scores, mutation_counts = [], [], []
    accept_regular, accept_mutated = [], []
    duplicates_kept_total = 0
    truncated_total = 0
    cap_used = None

    for i in range(iterations):
        shadow_log, mutation_flags, duplicates_kept, truncated, cap_used = generate_shadow_log(
            event_log, num_traces=num_traces, safe_threshold=safe_threshold,
            max_n=max_n, successor_weighting=successor_weighting
        )
        duplicates_kept_total += duplicates_kept
        truncated_total += truncated
        replayed = token_replay.apply(shadow_log, net, im, fm)
        trace_fitnesses = [res['trace_fitness'] for res in replayed]
        trace_accepts = [1.0 if res.get('trace_is_fit', res['trace_fitness'] >= 1.0) else 0.0
                         for res in replayed]

        overall_fitness = sum(trace_fitnesses) / len(trace_fitnesses) if trace_fitnesses else 0.0
        overall_accept = sum(trace_accepts) / len(trace_accepts) if trace_accepts else 0.0
        scores.append(overall_fitness)
        accept_scores.append(overall_accept)

        reg_fits = [f for f, flag in zip(trace_fitnesses, mutation_flags) if not flag]
        mut_fits = [f for f, flag in zip(trace_fitnesses, mutation_flags) if flag]
        reg_acc = [a for a, flag in zip(trace_accepts, mutation_flags) if not flag]
        mut_acc = [a for a, flag in zip(trace_accepts, mutation_flags) if flag]

        mutation_counts.append(len(mut_fits))
        regular_scores.append(np.mean(reg_fits) if reg_fits else 0.0)
        mutated_scores.append(np.mean(mut_fits) if mut_fits else 0.0)
        accept_regular.append(np.mean(reg_acc) if reg_acc else 0.0)
        accept_mutated.append(np.mean(mut_acc) if mut_acc else 0.0)

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "raw_scores": scores,
        "accept_mean": float(np.mean(accept_scores)),
        "accept_std": float(np.std(accept_scores)),
        "raw_accepts": accept_scores,
        "reg_mean": float(np.mean(regular_scores)),
        "reg_std": float(np.std(regular_scores)),
        "mut_mean": float(np.mean(mutated_scores)),
        "mut_std": float(np.std(mutated_scores)),
        "accept_reg_mean": float(np.mean(accept_regular)),
        "accept_mut_mean": float(np.mean(accept_mutated)),
        "mutation_counts": mutation_counts,
        "duplicates_kept": duplicates_kept_total,
        "truncated": truncated_total,
        "max_trace_length_used": cap_used,
    }

def evaluate_miner(event_log, miner_name, miner_fn, w=0.5, num_shadow_traces=1000,
                   iterations=5, seed=42, max_n=6, successor_weighting="log"):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    print(f"       Evaluating {miner_name} (v26, weighting={successor_weighting})...")
    t0 = time.time()

    net, im, fm = miner_fn(event_log)
    r = calculate_gen_shadow_stable(
        event_log, net, im, fm, num_shadow_traces, iterations,
        safe_threshold=5, max_n=max_n, successor_weighting=successor_weighting
    )

    runtime = time.time() - t0

    return {
        "miner": miner_name,
        "gen_struct": 0.0,
        "gen_shadow_mean": r["mean"],
        "gen_shadow_std": r["std"],
        "gen_shadow_raw_iterations": list(r["raw_scores"]),
        "gen_total": r["mean"],
        # Acceptance: direct operationalization of "accepts future valid
        # behavior" (fraction of perfectly replayed shadow traces).
        "gen_accept": r["accept_mean"],
        "gen_accept_std": r["accept_std"],
        # Openness profile (fitness and acceptance, regular vs mutated)
        "gen_shadow_regular": r["reg_mean"],
        "gen_shadow_regular_std": r["reg_std"],
        "gen_shadow_mutated": r["mut_mean"],
        "gen_shadow_mutated_std": r["mut_std"],
        "gen_accept_regular": r["accept_reg_mean"],
        "gen_accept_mutated": r["accept_mut_mean"],
        "mutated_traces_per_iteration": r["mutation_counts"],
        # Probe integrity counters
        "duplicates_kept": r["duplicates_kept"],
        "truncated_traces": r["truncated"],
        "max_trace_length_used": r["max_trace_length_used"],
        "successor_weighting": successor_weighting,
        "runtime_s": runtime
    }

# Register to the library
from . import register_algorithm
register_algorithm("v26")
