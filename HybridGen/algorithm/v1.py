import random
import time
import numpy as np
from collections import defaultdict, Counter

import pm4py
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness

__all__ = [
    "calculate_gen_shadow_stable", 
    "calculate_gen_struct",
    "evaluate_miner",
    ]
#1. Generative Behavioral Analysis (Gen_shadow)

def generate_shadow_log(event_log, num_traces=1000, max_trace_length=100):
    '''
    Generates a synthetic 'shadow' log acting as probabilistic based future behavior
    using Good-Turing estimation.
    
    Returns:
        (shadow_log, mutation_flags): The generated EventLog and a list of booleans
        indicating which traces contain at least one mutation event.
    '''
    
    #1. Discover historical Directly-Follows Graph (DFG) (#howTheProcessExecutedInThePast)
    #dfg: dict of (Activity A, Activity B) -> Frequency count
    #starts: dict of Starting Activity -> Frequency count
    #ends: dict of Ending Activity -> Frequency count
    dfg, starts, ends = pm4py.discover_dfg(event_log)
    
    #2. Reorganize the DFG into a nested dictionary for fast lookups
    #Format: outgoing[Current Activity][Next Activity] = Frequency Count
    outgoing = defaultdict(dict)
    for (a, b), count in dfg.items():
        outgoing[a][b] = count
        
    #3. Extract every unique activity in the log
    #Needed for mutation generation
    alphabet = list(set([a for a, _ in dfg.keys()] + [b for _, b in dfg.keys()]))
    
    #4. Calculate the Good-Turing Probability (P_unseen) for every local state
    #Good-Turing: probability that the next event will be something new
    p_unseen = {}
    for act in alphabet:
        out_edges = outgoing[act]
        
        #N: Total number of times we transitioned out of this activity
        n_total = sum(out_edges.values())
        
        #N_1: How many target activities followed this one exactly one time
        #High number of 'singletons' -> high variance/unpredictability
        n_1 = sum(1 for target, count in out_edges.items() if count == 1)
        
        #Formula: P_unseen = N_1 / N
        #If the state never had an outgoing edge (endpoint), we assign a 100% (1.0) chance to mutate if forced to continue
        p_unseen[act] = (n_1 / n_total) if n_total > 0 else 1.0

    #5. Initialize the (empty) shadow log and prepare start distributions
    shadow_log = EventLog()
    mutation_flags = []
    start_choices = list(starts.keys())
    start_weights = list(starts.values())

    #6. Begin generating the requested number of synthetic traces
    for i in range(num_traces):
        
        #Pick a starting activity based on its historical likelihood of starting a process
        current = random.choices(start_choices, weights=start_weights, k=1)[0]
        
        #Create a new Trace object and append the first event.
        trace = Trace(attributes={"concept:name": f"shadow_{i}"})
        trace.append(Event({"concept:name": current}))
        had_mutation = False
        
        #Walk through the graph to build the rest of the trace
        while True:
            
            #Decision 1: Should the trace terminate here?
            if current in ends:
                times_as_end = ends[current]
                #Total times this activity happened = times it was an end + times it transitioned to something else
                total_occurrences = times_as_end + sum(outgoing[current].values())
                
                #Probabilistic roll to stop the trace, based on historical end frequency.
                if random.random() < (times_as_end / total_occurrences):
                    break
                    
            #Safety catch: Prevent infinite loops in case of cyclic mutations
            if len(trace) >= max_trace_length:
                break
                
            #Decision 2: Mutate or follow historical paths?
            #We roll a random number against our calculated Good-Turing probability.
            if random.random() < p_unseen[current]:
                #Mutation Triggered. The algorithm actively explores a logically valid but historically unseen path, it picks a random activity from the entire known alphabet
                nxt = random.choice(alphabet)
                had_mutation = True
            else:
                #Predictable Path Triggered. The algorithm follows the historical DFG distribution. It picks the next activity based on how many times it historically followed this path.
                out_edges = outgoing[current]
                
                #If there are no outgoing edges (historical dead end) but it didn't terminate above, force break
                if not out_edges:
                    break
                    
                #Weighted random choice based on how many times paths were taken in the past
                nxt = random.choices(list(out_edges.keys()), weights=list(out_edges.values()), k=1)[0]
                
            #Append the chosen next event and advance the current state pointer
            trace.append(Event({"concept:name": nxt}))
            current = nxt
            
        #Add the completed synthetic trace to the shadow log
        shadow_log.append(trace)
        mutation_flags.append(had_mutation)
        
    return shadow_log, mutation_flags


def calculate_gen_shadow_stable(event_log, net, im, fm, num_traces, iterations=5):
    """
    Run the shadow log generation K times to ensure mathematical determinism and
    return the mean and standard deviation, plus stratified scores for regular vs.
    mutated traces.
    
    Returns:
        (mean, std, raw_scores, reg_mean, reg_std, mut_mean, mut_std, mutation_counts)
    """
    scores = []
    regular_scores = []    # Per-iteration: mean fitness on non-mutated traces
    mutated_scores = []    # Per-iteration: mean fitness on mutated traces
    mutation_counts = []   # Per-iteration: number of mutated traces
    
    #Run the stochastic generation and fitness check multiple times
    for i in range(iterations):
        #1. Generate a new shadow log with mutation flags
        shadow_log, mutation_flags = generate_shadow_log(event_log, num_traces=num_traces)
        
        #2. Replay the shadow log on the discovered model (per-trace for stratification)
        replayed = token_replay.apply(shadow_log, net, im, fm)
        
        #3. Collect per-trace fitness
        trace_fitnesses = [res['trace_fitness'] for res in replayed]
        
        #Overall fitness (existing metric)
        overall_fitness = sum(trace_fitnesses) / len(trace_fitnesses) if trace_fitnesses else 0.0
        scores.append(overall_fitness)
        
        #4. Stratified analysis: split by mutation flag
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


#3. Structural Analysis (Gen_struct)

def calculate_gen_struct(event_log, net, initial_marking, final_marking):
    """
    Evaluates Structure (#OverfittingPenalty) via "Arc Flow Density", penalizing them if they are rarely used
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
                
        #Only count an arc once per trace
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


#3. Core Evaluation

def evaluate_miner(event_log, miner_name, miner_fn, w=0.5, num_shadow_traces=1000, iterations=5, seed=42):
    """
    Executes the hybrid evaluation with model discovery, structural penalty calculation and stochastic shadow log generation.
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
    shadow_mean, shadow_std, raw_scores, reg_mean, reg_std, mut_mean, mut_std, mutation_counts = calculate_gen_shadow_stable(
        event_log, net, im, fm, num_shadow_traces, iterations
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
register_algorithm("v1")
