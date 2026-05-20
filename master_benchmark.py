"""
Pure Generalization Benchmark (Extended)
========================================
Evaluates Gen_Shadow V1, Gen_Shadow V2, and PM4Py Baseline across 
multiple datasets, detailed model morphologies, and profiles runtime.
"""

import time
import os
import pandas as pd
import pm4py
from pm4py.algo.evaluation.generalization import algorithm as generalization_eval
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness

import shadow_alg_both as algo

# ─── Configuration & Dataset Mapping ────────────────────────────────────────
DATASETS = {
    "BPI_2017": "data/BPI-Challenge_2017/BPI Challenge 2017.xes.gz",
    "BPI_2019": "data/BPI-Challenge_2019/BPI_Challenge_2019.xes.gz",
    "Sepsis": "data/Sepsis Cases - Event Log_1_all/Sepsis Cases - Event Log.xes.gz",
    "Hospital_Billing": "data/Hospital Billing - Event Log_1_all/Hospital Billing - Event Log.xes.gz",
    "Road_Traffic_Fine": "data/Road Traffic Fine Management Process_1_all/Road_Traffic_Fine_Management_Process.xes.gz"
}

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

# ─── Expanded Miner Dictionary ──────────────────────────────────────────────
MINERS = {
    #"01_Trace Model (Min)": discover_trace_model,
    "02_Alpha Miner": pm4py.discover_petri_net_alpha,
    "03_Alpha+ Miner": pm4py.discover_petri_net_alpha_plus,
    "04_Heuristics (Default)": pm4py.discover_petri_net_heuristics,
    "05_Heuristics (Strict)": lambda log: pm4py.discover_petri_net_heuristics(log, dependency_threshold=0.99),
    "06_Inductive (Strict)": lambda log: pm4py.discover_petri_net_inductive(log, noise_threshold=0.0),
    "07_Inductive (Infrequent)": lambda log: pm4py.discover_petri_net_inductive(log, noise_threshold=0.2),
    #"08_ILP Miner": pm4py.discover_petri_net_ilp, # Warning: Can be slow on massive logs
    "09_Flower Model (Max)": discover_flower_model
}

# ─── Main Execution ─────────────────────────────────────────────────────────
def run_master_benchmark():
    results = []
    print("=" * 125)
    print(" 🏆 EXTENDED BENCHMARK (Fitness vs V1 vs V2 vs PM4Py Gen)")
    print("=" * 125)
    
    for ds_name, path in DATASETS.items():
        print(f"\n📁 Dataset: {ds_name}")
        if not os.path.exists(path):
            print(f"   ⚠️ Skipping {ds_name} - File not found")
            continue
            
        log = pm4py.read_xes(path)
        
        for miner_name, miner_fn in MINERS.items():
            clean_miner_name = miner_name.split('_')[-1]
            print(f"   ⚙️ Evaluating: {clean_miner_name}...")
            
            # 1. Discover Model
            net, im, fm = miner_fn(log)
            
            # 2. PM4Py Original Log Fitness (How well does it replay the PAST?)
            try:
                fit_res = replay_fitness.apply(log, net, im, fm, variant=replay_fitness.Variants.TOKEN_BASED)
                pm4py_fitness = fit_res['log_fitness']
            except Exception:
                pm4py_fitness = 0.0
            
            # 3. PM4Py Baseline Generalization
            t_pm4py_start = time.time()
            try:
                pm4py_gen = generalization_eval.apply(log, net, im, fm)
            except Exception:
                pm4py_gen = 0.0 
            t_pm4py_end = time.time()
                
            # 4. Gen_Shadow V1 (1-gram)
            t_v1_start = time.time()
            v1_shadow = algo.calculate_gen_shadow_stable(log, net, im, fm, num_traces=500, iterations=3, safe_threshold=5, max_n=1)[0]
            t_v1_end = time.time()
            
            # 5. Gen_Shadow V2 (6-gram Katz Backoff)
            t_v2_start = time.time()
            v2_shadow = algo.calculate_gen_shadow_stable(log, net, im, fm, num_traces=500, iterations=3, safe_threshold=5, max_n=6)[0]
            t_v2_end = time.time()
            
            results.append({
                "Dataset": ds_name,
                "Miner": clean_miner_name, 
                "PM4Py_Fitness": round(pm4py_fitness, 4),    # <--- ADDED!
                "PM4Py_Baseline": round(pm4py_gen, 4),
                "Gen_Shadow_V1": round(v1_shadow, 4),
                "Gen_Shadow_V2": round(v2_shadow, 4),
                "PM4Py_Time_s": round(t_pm4py_end - t_pm4py_start, 2),
                "V1_Time_s": round(t_v1_end - t_v1_start, 2),
                "V2_Time_s": round(t_v2_end - t_v2_start, 2)
            })

    # Output Results
    df = pd.DataFrame(results)
    print("\n" + "=" * 125)
    print(" 📊 FINAL BENCHMARK RESULTS")
    print("=" * 125)
    print(df.to_string(index=False))
    print("=" * 125)
    
    df.to_csv("master_benchmark_results_with_fitness.csv", index=False)
    print("\n✅ Saved to 'master_benchmark_results_with_fitness.csv'")

if __name__ == "__main__":
    run_master_benchmark()