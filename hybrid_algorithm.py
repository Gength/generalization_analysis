import random
import time
import numpy as np
from collections import defaultdict, Counter

import pm4py
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness


#1. Generative Behavioral Analysis (Gen_shadow)

def generate_shadow_log(event_log, num_traces=1000, max_trace_length=100):
    #Generates a synthetic 'shadow' log acting as probabilistic based future behaviorusing Good-Turing estimation
    
    
    #1. Discover the historical Directly-Follows Graph (DFG). Tells us how the process executed in the past
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
    #Needed to generate mutations
    alphabet = list(set([a for a, _ in dfg.keys()] + [b for _, b in dfg.keys()]))
    
    #4. Calculate the Good-Turing Probability (P_unseen) for every local state
    #Good-Turing estimates the probability that the next event will be something we have never seen follow this activity before
    p_unseen = {}
    for act in alphabet:
        out_edges = outgoing[act]
        
        #N: Total number of times we transitioned out of this activity
        n_total = sum(out_edges.values())
        
        #N_1: How many target activities followed this one exactly one time?
        #A high number of 'singletons' -> high variance/unpredictability
        n_1 = sum(1 for target, count in out_edges.items() if count == 1)
        
        #Formula: P_unseen = N_1 / N
        #If the state never had an outgoing edge (it was strictly an endpoint), we assign a 100% (1.0) chance to mutate if forced to continue
        p_unseen[act] = (n_1 / n_total) if n_total > 0 else 1.0

    #5. Initialize the empty shadow log and prepare start distributions
    shadow_log = EventLog()
    start_choices = list(starts.keys())
    start_weights = list(starts.values())

    #6. Begin generating the requested number of synthetic traces
    for i in range(num_traces):
        
        #Pick a starting activity based on its historical likelihood of starting a process
        current = random.choices(start_choices, weights=start_weights, k=1)[0]
        
        #Create a new Trace object and append the first event.
        trace = Trace(attributes={"concept:name": f"shadow_{i}"})
        trace.append(Event({"concept:name": current}))
        
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
        
    return shadow_log


def calculate_gen_shadow_stable(event_log, net, im, fm, num_traces, iterations=5):
    """
    Runs the shadow log generation K times to ensure mathematical determinism.
    Because shadow log generation is stochastic (random), a single run might yield outliers.
    We run it multiple times and return the mean and standard deviation.
    """
    scores = []
    
    #Run the stochastic generation and fitness check multiple times
    for i in range(iterations):
        #1. Generate a brand new shadow log
        shadow_log = generate_shadow_log(event_log, num_traces=num_traces)
        
        #2. Replay the shadow log on the discovered model.
        #If the model is too restrictive (underfitting/Alpha miner), this will fail heavily.
        fitness_res = replay_fitness.apply(shadow_log, net, im, fm, 
                                           variant=replay_fitness.Variants.TOKEN_BASED)
        
        #Store the fitness score (0.0 to 1.0) of this iteration
        scores.append(fitness_res['log_fitness'])
    
    #Return the Expected Value (mean), Variance (std), and the raw iteration list
    return np.mean(scores), np.std(scores), scores


#3. Structural Rigor Analysis (Gen_struct)

def calculate_gen_struct(event_log, net, initial_marking, final_marking):
    """
    Evaluates Structural Parsimony (Overfitting Penalty) via Arc Flow Density.
    Spaghetti models (Alpha) and Trace models create an explosion of arcs (lines) to handle rare, coincidental noise. 
    We penalize models where a large percentage of arcs are rarely or never used.
    """
    
    #1. Replay the original historical log against the discovered model
    #This tells us which parts of the model's structure are actually used
    replayed = token_replay.apply(event_log, net, initial_marking, final_marking)
    
    #2. Initialize a counter for every single arc in the Petri Net
    #This includes both visible and invisible arcs
    arc_usage = {arc: 0 for arc in net.arcs}
    
    #3. Track how many Traces use each arc
    for res in replayed:
        used_arcs = set()
        
        #Look at every transition that was fired to support this trace
        for t in res['activated_transitions']:
            #Record the incoming and outgoing structural wires for that transition
            for arc in t.in_arcs:
                used_arcs.add(arc)
            for arc in t.out_arcs:
                used_arcs.add(arc)
                
        #We only count an arc ONCE per trace. Even if a loop uses an arc 50 times in one trace, we want to know if this structural path is relevant to this trace as a whole
        for arc in used_arcs:
            arc_usage[arc] += 1
            
    #Calculate the total number of structural wires in the model
    total_arcs = len(net.arcs)
    
    #Safety catch for completely broken or empty models
    if total_arcs == 0:
        return 0.0
        
    #4. Defining Bloat 
    #An arc is considered "overfit/bloated" if it handles a negligible fraction of the log
    #We define negligible as less than 1% of the total traces
    num_traces = len(event_log)
    
    #We use max(2, ...) to ensure that paths created purely for a single outlier trace (count=1) are always penalized, regardless of how large the log is
    rare_threshold = max(2, int(num_traces * 0.01))
    
    #Count how many arcs in the model fall below this usefulness threshold
    rare_arcs = sum(1 for arc, count in arc_usage.items() if count < rare_threshold)
    
    #5. Calculate the structural penalty
    #i.e. If 80% of the model's arcs are only used to handle 1% of the exceptions, the overfit_penalty is 0.8, dropping the score to 0.2.
    overfit_penalty = rare_arcs / total_arcs
    
    #Ensure the score never drops below absolute 0
    return max(0.0, 1.0 - overfit_penalty)


#3. Core Evaluation

def evaluate_miner(event_log, miner_name, miner_fn, w=0.5, num_shadow_traces=1000, iterations=5, seed=42):
    """
    Executes the hybrid evaluation with model discovery, structural penalty calculation and stochastic shadow log generation.
    """
    
    #1. Set a seed for scientific benchmarking, ensuring the same scores the next run
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    print(f"       Evaluating {miner_name}...")
    t0 = time.time() #Start the timer
    
    #2. Execute the process discovery algorithm provided by the runner
    net, im, fm = miner_fn(event_log)
    
    #3. Calculate Structural Penalty (Reality-based mathematical constraint)
    #Punishes models with too many spaghetti arcs
    gen_struct = calculate_gen_struct(event_log, net, im, fm)
    
    #4. Calculate Generative Behavioral Score (Probabilistic stress test)
    #Rewards models that are flexible enough to handle unseen mutations
    #Returns the average score, standard deviation, and raw list
    shadow_mean, shadow_std, raw_scores = calculate_gen_shadow_stable(
        event_log, net, im, fm, num_shadow_traces, iterations
    )
    
    #5. Hybrid Synthesis: Combine the two scores using the user-defined weight (w)
    #Default is 0.5 (equal balance between parsimony and flexibility)
    gen_total = (w * shadow_mean) + ((1.0 - w) * gen_struct)
    
    #Stop the timer
    runtime = time.time() - t0
    print(f"         └─ Gen_Total: {gen_total:.4f} | Struct: {gen_struct:.2f} | Shadow Mean: {shadow_mean:.4f} (±{shadow_std:.4f}) | {runtime:.1f}s")
    
    #6. Format the results for JSON serialization in the CLI runner
    return {
        "miner": miner_name,
        "gen_struct": gen_struct,
        "gen_shadow_mean": shadow_mean,
        "gen_shadow_std": shadow_std, #The determinism/stability check
        "gen_shadow_raw_iterations": raw_scores,
        "gen_total": gen_total,
        "w_weight": w,
        #Human-readable label indicating if the metric is statistically reliable
        "determinism_rating": "High" if shadow_std < 0.02 else "Moderate" if shadow_std < 0.05 else "Low",
        "runtime_s": runtime
    }