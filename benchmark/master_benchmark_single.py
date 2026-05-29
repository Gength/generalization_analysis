"""
Pure Generalization Benchmark
=======================================================
Evaluates Gen_Shadow (1G, 3G, 6G) using the HybridGen library, 
PM4Py Baseline, and K-Fold Cross-Validation Fitness.
"""

import time
import os
import random
import numpy as np
import pandas as pd
import pm4py
from pm4py.algo.evaluation.generalization import algorithm as generalization_eval
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.objects.log.obj import EventLog

# --- IMPORT THE PARTNER'S NEW LIBRARY ---
from HybridGen.algorithm import load_algorithm

# Load the algorithms dynamically from the registry
algo_v1 = load_algorithm("v1")
algo_v21 = load_algorithm("v21")  # Using V2.1 to get the logarithmic ln(x+1) scaling!

# ─── Configuration & Dataset Mapping ────────────────────────────────────────
DATASETS = {
    "BPI_2017": "data/BPI-Challenge_2017/BPI Challenge 2017.xes.gz",
    "BPI_2019": "data/BPI-Challenge_2019/BPI_Challenge_2019.xes.gz",
    "Sepsis": "data/Sepsis Cases - Event Log_1_all/Sepsis Cases - Event Log.xes.gz",
    "Hospital_Billing": "data/Hospital Billing - Event Log_1_all/Hospital Billing - Event Log.xes.gz",
    "Road_Traffic_Fine": "data/Road Traffic Fine Management Process_1_all/Road_Traffic_Fine_Management_Process.xes.gz"
}

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

# ─── Model Morphology Generators ────────────────────────────────────────────
def discover_flower_model(log):
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

# ─── Expanded Miner Dictionary ──────────────────────────────────────────────
MINERS = {
    "02_Alpha Miner": pm4py.discover_petri_net_alpha,
    "03_Alpha+ Miner": pm4py.discover_petri_net_alpha_plus,
    "04_Heuristics (Default)": pm4py.discover_petri_net_heuristics,
    "05_Heuristics (Strict)": lambda log: pm4py.discover_petri_net_heuristics(log, dependency_threshold=0.99),
    "06_Inductive (Strict)": lambda log: pm4py.discover_petri_net_inductive(log, noise_threshold=0.0),
    "07_Inductive (Infrequent)": lambda log: pm4py.discover_petri_net_inductive(log, noise_threshold=0.2),
    "09_Flower Model (Max)": discover_flower_model
}

# ─── Main Execution ─────────────────────────────────────────────────────────
def run_master_benchmark():
    results = []
    print("=" * 145)
    print(" 🏆 EXTENDED BENCHMARK (Fitness | 3-Fold CV | PM4Py Gen | 1G/3G/6G Shadow)")
    print("=" * 145)
    
    for ds_name, path in DATASETS.items():
        print(f"\n📁 Dataset: {ds_name}")
        if not os.path.exists(path):
            print(f"   ⚠️ Skipping {ds_name} - File not found")
            continue
            
        log = pm4py.read_xes(path)
        # Fix Pandas Bug by globally converting to EventLog up front
        if isinstance(log, pd.DataFrame):
            log = pm4py.convert_to_event_log(log)
        
        for miner_name, miner_fn in MINERS.items():
            clean_miner_name = miner_name.split('_')[-1]
            print(f"   ⚙️ Evaluating: {clean_miner_name}...")
            
            # 1. Discover Model on FULL log
            net, im, fm = miner_fn(log)
            
            # 2. Base Fitness
            try:
                fit_res = replay_fitness.apply(log, net, im, fm, variant=replay_fitness.Variants.TOKEN_BASED)
                pm4py_fitness = fit_res['log_fitness']
            except Exception:
                pm4py_fitness = 0.0
                
            # 3. K-Fold Cross-Validation (3 folds + variant-based)
            t_kfold_start = time.time()
            kfold_fit = compute_kfold_fitness(log, miner_fn, k=3, pick_one_out=False)
            t_kfold_end = time.time()
            
            t_varcv_start = time.time()
            variant_cv_fit = compute_kfold_fitness(log, miner_fn, pick_one_out=True)
            t_varcv_end = time.time()
            
            # 4. PM4Py Baseline Generalization
            try:
                pm4py_gen = generalization_eval.apply(log, net, im, fm)
            except Exception:
                pm4py_gen = 0.0 
                
            # 5. Gen_Shadow V1 (1-gram)
            t_1g_start = time.time()
            v1_shadow = algo_v1.calculate_gen_shadow_stable(log, net, im, fm, num_traces=500, iterations=3)[0]
            t_1g_end = time.time()
            
            # 6. Gen_Shadow V2.1 (3-gram + Katz Backoff + Log Weights)
            t_3g_start = time.time()
            v3_shadow = algo_v21.calculate_gen_shadow_stable(log, net, im, fm, num_traces=500, iterations=3, safe_threshold=5, max_n=3)[0]
            t_3g_end = time.time()
            
            # 7. Gen_Shadow V2.1 (6-gram + Katz Backoff + Log Weights)
            t_6g_start = time.time()
            v6_shadow = algo_v21.calculate_gen_shadow_stable(log, net, im, fm, num_traces=500, iterations=3, safe_threshold=5, max_n=6)[0]
            t_6g_end = time.time()
            
            results.append({
                "Dataset": ds_name,
                "Miner": clean_miner_name, 
                "Fit_Full": round(pm4py_fitness, 4),
                "Fit_3Fold": round(kfold_fit, 4),
                "Fit_VariantCV": round(variant_cv_fit, 4),
                "PM4Py_Gen": round(pm4py_gen, 4),
                "1G_Shadow": round(v1_shadow, 4),
                "3G_Shadow": round(v3_shadow, 4),
                "6G_Shadow": round(v6_shadow, 4),
                "KFold_Time_s": round(t_kfold_end - t_kfold_start, 1),
                "VariantCV_Time_s": round(t_varcv_end - t_varcv_start, 1),
                "1G_Time_s": round(t_1g_end - t_1g_start, 1),
                "3G_Time_s": round(t_3g_end - t_3g_start, 1),
                "6G_Time_s": round(t_6g_end - t_6g_start, 1)
            })

    # Output Results
    df = pd.DataFrame(results)
    print("\n" + "=" * 145)
    print(" 📊 FINAL BENCHMARK RESULTS")
    print("=" * 145)
    print(df.to_string(index=False))
    print("=" * 145)
    
    df.to_csv("master_benchmark_results_v21.csv", index=False)
    print("\n✅ Saved to 'master_benchmark_results_v21.csv'")

if __name__ == "__main__":
    run_master_benchmark()