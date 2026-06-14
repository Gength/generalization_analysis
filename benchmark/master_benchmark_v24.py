"""
Ultimate Master Benchmark (All Generations, Rivals, & Baselines)
================================================================
Evaluates:
1. K-Fold CV (Standard vs. Variant-Based)
2. HybridGen Evolution (v21 -> v23 -> v24)
3. Rival Metrics (PM4Py, Negative Events, Anti-Alignments, Entropy)
"""

import sys
sys.setrecursionlimit(10000)

import time
import os
import random
import multiprocessing
import numpy as np
import pandas as pd
import pm4py
from collections import defaultdict
import Levenshtein # Ensure 'pip install Levenshtein' is run

from pm4py.algo.evaluation.generalization import algorithm as generalization_eval
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.objects.log.obj import EventLog, Trace, Event

# --- IMPORT ALGORITHMS FROM HYBRIDGEN REGISTRY ---
from HybridGen.algorithm import load_algorithm
algo_v21 = load_algorithm("v2.1")
algo_v23 = load_algorithm("v2.3")
algo_v24 = load_algorithm("v2.4")

# ─── CONFIGURATION ──────────────────────────────────────────────────────────
DATASETS = {
    "BPI_2017": "data/BPI-Challenge_2017/BPI Challenge 2017.xes.gz",
    "BPI_2019": "data/BPI-Challenge_2019/BPI_Challenge_2019.xes.gz",
    "Sepsis": "data/Sepsis Cases - Event Log_1_all/Sepsis Cases - Event Log.xes.gz",
    "Hospital_Billing": "data/Hospital Billing - Event Log_1_all/Hospital Billing - Event Log.xes.gz",
    "Road_Traffic_Fine": "data/Road Traffic Fine Management Process_1_all/Road_Traffic_Fine_Management_Process.xes.gz"
}

# ─── MODEL GENERATORS ───────────────────────────────────────────────────────
def discover_filtered_trace_model(log, top_k=50):
    """Builds a Trace Model for only the Top K variants to establish 0.0 bound without crashing."""
    net = PetriNet("Filtered Trace Model")
    p_start, p_end = PetriNet.Place("start"), PetriNet.Place("end")
    net.places.update([p_start, p_end])
    
    variant_counts = defaultdict(int)
    for t in log:
        variant_counts[tuple(e["concept:name"] for e in t)] += 1
    top_variants = [v for v, c in sorted(variant_counts.items(), key=lambda i: i[1], reverse=True)[:top_k]]
        
    for i, variant in enumerate(top_variants):
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

def discover_flower_model(log):
    """Builds a Flower Model to establish the 1.0 bound."""
    net = PetriNet("Flower Model")
    p_mid = PetriNet.Place("mid")
    net.places.add(p_mid)
    activities = set(e["concept:name"] for t in log for e in t)
    for act in activities:
        t = PetriNet.Transition(f"t_{act}", act)
        net.transitions.add(t)
        petri_utils.add_arc_from_to(p_mid, t, net)
        petri_utils.add_arc_from_to(t, p_mid, net)
    im, fm = Marking(), Marking()
    im[p_mid] = 1; fm[p_mid] = 1 
    return net, im, fm

MINERS = {
    "01_Filtered Trace Model": discover_filtered_trace_model,
    "02_Alpha Miner": pm4py.discover_petri_net_alpha,
    "03_Alpha+ Miner": pm4py.discover_petri_net_alpha_plus,
    "04_Heuristics (Default)": pm4py.discover_petri_net_heuristics,
    "05_Heuristics (Strict)": lambda log: pm4py.discover_petri_net_heuristics(log, dependency_threshold=0.99),
    "06_Inductive (Strict)": lambda log: pm4py.discover_petri_net_inductive(log, noise_threshold=0.0),
    "07_Inductive (Infrequent)": lambda log: pm4py.discover_petri_net_inductive(log, noise_threshold=0.2),
    "09_Flower Model (Max)": discover_flower_model
}

# ─── METRIC EVALUATORS ──────────────────────────────────────────────────────
def compute_kfold_fitness(log, miner_fn, k=3, variant_based=False):
    try:
        if variant_based:
            variant_map = defaultdict(list)
            for trace in log:
                variant_map[tuple(e["concept:name"] for e in trace)].append(trace)
            unique_variants = list(variant_map.keys())
            random.shuffle(unique_variants)
            fold_size = max(1, len(unique_variants) // k)
            fitnesses = []
            for i in range(k):
                start = i * fold_size
                end = (i + 1) * fold_size if i < k - 1 else len(unique_variants)
                test_vars = set(unique_variants[start:end])
                
                train_traces, test_traces = [], []
                for var, traces in variant_map.items():
                    if var in test_vars: test_traces.extend(traces)
                    else: train_traces.extend(traces)
                        
                if not test_traces or not train_traces: continue
                train_log, test_log = EventLog(train_traces), EventLog(test_traces)
                net, im, fm = miner_fn(train_log)
                fit = replay_fitness.apply(test_log, net, im, fm, variant=replay_fitness.Variants.TOKEN_BASED)['log_fitness']
                fitnesses.append(fit)
            return np.mean(fitnesses) if fitnesses else 0.0
        else:
            traces = list(log)
            random.shuffle(traces)
            fold_size = max(1, len(traces) // k)
            fitnesses = []
            for i in range(k):
                start = i * fold_size
                end = (i + 1) * fold_size if i < k - 1 else len(traces)
                test_log = EventLog(traces[start:end])
                train_log = EventLog(traces[:start] + traces[end:])
                if not test_log or not train_log: continue
                
                net, im, fm = miner_fn(train_log)
                fit = replay_fitness.apply(test_log, net, im, fm, variant=replay_fitness.Variants.TOKEN_BASED)['log_fitness']
                fitnesses.append(fit)
            return np.mean(fitnesses) if fitnesses else 0.0
    except Exception: return 0.0

def metric_negative_events(log, net, im, fm, num_traces=100):
    try:
        sample = random.sample(list(log), min(num_traces, len(log)))
        negative_log = EventLog()
        for trace in sample:
            if len(trace) < 2: continue
            broken_trace = Trace(attributes=trace.attributes)
            events = [e for e in trace]
            idx = random.randint(0, len(events) - 2)
            events[idx], events[idx+1] = events[idx+1], events[idx]
            for e in events: broken_trace.append(e)
            negative_log.append(broken_trace)
            
        if len(negative_log) == 0: return 0.0
        return replay_fitness.apply(negative_log, net, im, fm, variant=replay_fitness.Variants.TOKEN_BASED)['log_fitness']
    except Exception: return 0.0

def metric_anti_alignments(log, net, im, fm, num_traces=100):
    try:
        simulated_log = pm4py.play_out(net, im, fm, max_traces=num_traces)
        real_seqs = set(tuple(e["concept:name"] for e in t) for t in log)
        unseen_seqs = [tuple(e["concept:name"] for e in t) for t in simulated_log if tuple(e["concept:name"] for e in t) not in real_seqs]
        
        if not unseen_seqs: return 0.0 
        distances = []
        for unseen in unseen_seqs:
            str_unseen = "".join([act[:2] for act in unseen]) 
            min_dist = min(Levenshtein.distance(str_unseen, "".join([act[:2] for act in real])) for real in real_seqs)
            distances.append(min_dist)
        return np.mean(distances) 
    except Exception: return -1.0 

def _compute_entropy(net, im, return_dict):
    from pm4py.objects.petri_net.utils import reachability_graph
    try:
        ts = reachability_graph.construct_reachability_graph(net, im)
        return_dict['states'] = len(ts.states)
    except Exception:
        return_dict['states'] = -1

def metric_entropy_state_space(net, im):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    p = multiprocessing.Process(target=_compute_entropy, args=(net, im, return_dict))
    p.start()
    p.join(15) # 15 second strict timeout
    if p.is_alive():
        p.terminate()
        p.join()
        return -1 # Timeout (State Space Explosion)
    return return_dict.get('states', -1)

# ─── MAIN EXECUTION ─────────────────────────────────────────────────────────
def run_master_benchmark():
    results = []
    print("=" * 160)
    print(" 🏆 ULTIMATE BENCHMARK (K-Fold Variants | Algorithm Evolution | Rival Metrics)")
    print("=" * 160)
    
    for ds_name, path in DATASETS.items():
        print(f"\n📁 Dataset: {ds_name}")
        if not os.path.exists(path): continue
            
        log = pm4py.read_xes(path)
        if isinstance(log, pd.DataFrame):
            log = pm4py.convert_to_event_log(log)
        
        for miner_name, miner_fn in MINERS.items():
            clean_miner_name = miner_name.split('_')[-1]
            print(f"   ⚙️ Evaluating: {clean_miner_name}...")
            
            net, im, fm = miner_fn(log)
            
            # --- 1. K-Fold Cross Validation ---
            t_kstd_start = time.time()
            kfold_std = compute_kfold_fitness(log, miner_fn, k=3, variant_based=False)
            t_kstd_end = time.time()
            
            t_kvar_start = time.time()
            kfold_var = compute_kfold_fitness(log, miner_fn, k=3, variant_based=True)
            t_kvar_end = time.time()
            
            # --- 2. Rival Metrics ---
            t_rival_start = time.time()
            try: pm4py_gen = generalization_eval.apply(log, net, im, fm)
            except Exception: pm4py_gen = 0.0
            neg_event_score = metric_negative_events(log, net, im, fm)
            anti_align_dist = metric_anti_alignments(log, net, im, fm)
            entropy_states = metric_entropy_state_space(net, im)
            t_rival_end = time.time()
                
            # --- 3. Shadow Log Evolution (6G for all) ---
            t_v21_start = time.time()
            v21_shadow = algo_v21.calculate_gen_shadow_stable(log, net, im, fm, num_traces=500, iterations=3, max_n=6)[0]
            t_v21_end = time.time()

            t_v23_start = time.time()
            v23_shadow = algo_v23.calculate_gen_shadow_stable(log, net, im, fm, num_traces=500, iterations=3, max_n=6)[0]
            t_v23_end = time.time()

            t_v24_start = time.time()
            v24_shadow = algo_v24.calculate_gen_shadow_stable(log, net, im, fm, num_traces=500, iterations=3, max_n=6)[0]
            t_v24_end = time.time()
            
            results.append({
                "Dataset": ds_name, 
                "Miner": clean_miner_name, 
                # Empirical Ground Truths
                "Fit_3Fold_Std": round(kfold_std, 4),
                "Fit_3Fold_Var": round(kfold_var, 4),
                # Rival Metrics
                "PM4Py_Gen": round(pm4py_gen, 4),
                "Negative_Events": round(neg_event_score, 4),
                "Anti_Align_Dist": round(anti_align_dist, 2),
                "Entropy_States": entropy_states,
                # HybridGen Evolution
                "6G_v21": round(v21_shadow, 4),
                "6G_v23": round(v23_shadow, 4),
                "6G_v24": round(v24_shadow, 4),
                # Runtimes
                "Time_KFold_Std_s": round(t_kstd_end - t_kstd_start, 1),
                "Time_KFold_Var_s": round(t_kvar_end - t_kvar_start, 1),
                "Time_Rivals_s": round(t_rival_end - t_rival_start, 1),
                "Time_v21_s": round(t_v21_end - t_v21_start, 1),
                "Time_v23_s": round(t_v23_end - t_v23_start, 1),
                "Time_v24_s": round(t_v24_end - t_v24_start, 1)
            })

    df = pd.DataFrame(results)
    df.to_csv("master_benchmark_ultimate.csv", index=False)
    print("\n✅ Saved to 'master_benchmark_ultimate.csv'")

if __name__ == "__main__":
    run_master_benchmark()