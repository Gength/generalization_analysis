from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
import pandas as pd
import random
import numpy as np
from pm4py.objects.log.obj import EventLog
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
# ─── Model Morphology Generators ────────────────────────────────────────────
def discover_flower_model(log):
    net = PetriNet("Flower Model")
    p_mid = PetriNet.Place("mid")
    net.places.add(p_mid)
    
    activities = log["concept:name"].unique() if isinstance(log, pd.DataFrame) else set(e["concept:name"] for t in log for e in t)
    for act in activities:
        t = PetriNet.Transition(f"t_{act}", act)
        net.transitions.add(t)
        petri_utils.add_arc_from_to(p_mid, t, net)
        petri_utils.add_arc_from_to(t, p_mid, net)
        
    im, fm = Marking(), Marking()
    im[p_mid] = 1; fm[p_mid] = 1 
    return net, im, fm

def discover_trace_model(log):
    net = PetriNet("Trace Model")
    p_start = PetriNet.Place("start")
    p_end = PetriNet.Place("end")
    net.places.update([p_start, p_end])
    
    if isinstance(log, pd.DataFrame):
        variants = log.groupby('case:concept:name')['concept:name'].apply(tuple).unique()
    else:
        variants = set(tuple(e["concept:name"] for e in t) for t in log)
        
    for i, variant in enumerate(variants):
        prev = p_start
        for j, act in enumerate(variant):
            t = PetriNet.Transition(f"t_{i}_{j}", act)
            net.transitions.add(t)
            petri_utils.add_arc_from_to(prev, t, net)
            if j == len(variant) - 1:
                petri_utils.add_arc_from_to(t, p_end, net)
            else:
                p_next = PetriNet.Place(f"p_{i}_{j}")
                net.places.add(p_next)
                petri_utils.add_arc_from_to(t, p_next, net)
                prev = p_next

    im, fm = Marking(), Marking()
    im[p_start] = 1; fm[p_end] = 1
    return net, im, fm

# ─── K-Fold Cross Validation Function ───────────────────────────────────────
def compute_kfold_fitness(log, miner_fn, k=3, pick_one_out=False):
    """
    Compute cross-validated fitness.
    
    Args:
        log: Event log
        miner_fn: Process discovery function
        k: Number of folds (used only when pick_one_out=False)
        pick_one_out: If True, use leave-one-variant-out (each unique trace variant
                      becomes one fold, held out as test while all other variants train).
                      If False (default), use standard random-split K-fold.
    """
    try:
        if pick_one_out:
            # ── Variant-based: leave-one-variant-out ──
            # Group traces by their activity sequence (variant)
            from collections import defaultdict
            variant_map = defaultdict(list)
            for trace in log:
                seq = tuple(e["concept:name"] for e in trace)
                variant_map[seq].append(trace)
            
            variants = list(variant_map.keys())
            n_variants = len(variants)
            if n_variants <= 1:
                print(f"       ⚠️ pick_one_out: only {n_variants} variant(s), skipping")
                return 0.0
            
            fitnesses = []
            for idx, variant in enumerate(variants):
                test_traces = variant_map[variant]
                train_traces = [t for v in variants if v != variant for t in variant_map[v]]
                
                train_log = EventLog(train_traces)
                test_log = EventLog(test_traces)
                
                net, im, fm = miner_fn(train_log)
                fit = replay_fitness.apply(test_log, net, im, fm,
                                           variant=replay_fitness.Variants.TOKEN_BASED)['log_fitness']
                fitnesses.append(fit)
            
            return np.mean(fitnesses)
        
        else:
            # ── Standard random-split K-fold ──
            traces = list(log)
            random.shuffle(traces)
            fold_size = len(traces) // k
            fitnesses = []
            
            for i in range(k):
                start = i * fold_size
                end = (i + 1) * fold_size if i < k - 1 else len(traces)
                test_traces = traces[start:end]
                train_traces = traces[:start] + traces[end:]
                
                train_log = EventLog(train_traces)
                test_log = EventLog(test_traces)
                
                net, im, fm = miner_fn(train_log)
                fit = replay_fitness.apply(test_log, net, im, fm,
                                           variant=replay_fitness.Variants.TOKEN_BASED)['log_fitness']
                fitnesses.append(fit)
            return np.mean(fitnesses)
    except Exception as e:
        print(f"       ⚠️ K-Fold Error: {e}")
        return 0.0