"""
Generalization Leaderboard Benchmark
====================================
Compares the "Pure Gen_shadow" metric against the PM4Py Baseline metric
across multiple datasets and miners, explicitly including the Flower Model.
"""

import time
import os
import pandas as pd
import pm4py
from pm4py.algo.evaluation.generalization import algorithm as generalization_eval
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils

# Import ONLY the shadow log part from your existing algorithm file
from hybrid_algorithm import calculate_gen_shadow_stable

# ─── 1. Flower Model Generator (To prove the 1.0 Generalization theory) ─────
def discover_flower_model(log):
    """Creates a Petri net that allows ANY sequence of the log's activities."""
    net = PetriNet("Flower Model")
    p_start = PetriNet.Place("start")
    p_mid = PetriNet.Place("mid")
    p_end = PetriNet.Place("end")
    net.places.update([p_start, p_mid, p_end])
    
    t_start = PetriNet.Transition("t_start", None)
    t_end = PetriNet.Transition("t_end", None)
    net.transitions.update([t_start, t_end])
    
    petri_utils.add_arc_from_to(p_start, t_start, net)
    petri_utils.add_arc_from_to(t_start, p_mid, net)
    petri_utils.add_arc_from_to(p_mid, t_end, net)
    petri_utils.add_arc_from_to(t_end, p_end, net)

    if isinstance(log, pd.DataFrame):
        activities = log["concept:name"].unique()
    else:
        activities = set(event["concept:name"] for trace in log for event in trace)
        
    for act in activities:
        t = PetriNet.Transition(f"t_{act}", act)
        net.transitions.add(t)
        petri_utils.add_arc_from_to(p_mid, t, net)
        petri_utils.add_arc_from_to(t, p_mid, net)
        
    im = Marking(); im[p_start] = 1
    fm = Marking(); fm[p_end] = 1
    return net, im, fm

# ─── 2. Configuration ───────────────────────────────────────────────────────

MINERS = {
    "Alpha Miner": pm4py.discover_petri_net_alpha,
    "Heuristics Miner": pm4py.discover_petri_net_heuristics,
    "Inductive Miner": pm4py.discover_petri_net_inductive,
    "Flower Model (Max Gen)": discover_flower_model
}

# Add your local XES paths here. If they don't exist, it falls back to dummies.
DATASETS = {
    "BPI_2017": "data/BPI-Challenge_2017/BPI Challenge 2017.xes.gz",
    #"Sepsis": "data/Sepsis_Cases.xes.gz"
}

def generate_dummy_log(noise_level="clean"):
    """Generates a dummy log if the real XES file is missing."""
    if noise_level == "clean":
        data = {'case:concept:name': ['1','1','1','2','2','2'], 
                'concept:name': ['A','B','C','A','B','C']}
    else:
        data = {'case:concept:name': ['1','1','1','2','2','2','3','3'], 
                'concept:name': ['A','B','C','A','X','C','A','B']}
    return pm4py.format_dataframe(pd.DataFrame(data), case_id='case:concept:name', activity_key='concept:name')

# ─── 3. Main Benchmark Execution ────────────────────────────────────────────

def run_leaderboard():
    results = []
    
    print("=" * 80)
    print(" 🏆 PURE GENERALIZATION LEADERBOARD (Gen_shadow vs. PM4Py Baseline)")
    print("=" * 80)
    
    for ds_name, path in DATASETS.items():
        print(f"\n📁 Loading Dataset: {ds_name}...")
        if os.path.exists(path):
            log = pm4py.read_xes(path)
        else:
            print(f"   ⚠️ File not found. Using fallback synthetic log for {ds_name}.")
            log = generate_dummy_log("noisy" if ds_name == "Sepsis" else "clean")
            
        for miner_name, miner_fn in MINERS.items():
            print(f"   ⚙️ Mining & Evaluating: {miner_name}...")
            
            # 1. Discover
            t0 = time.time()
            net, im, fm = miner_fn(log)
            
            # 2. PM4Py Baseline Metric
            try:
                pm4py_gen = generalization_eval.apply(log, net, im, fm)
            except Exception:
                pm4py_gen = 0.0 # Alpha miner sometimes completely breaks PM4Py's evaluator
            
            # 3. Pure Gen_shadow (Behavioral Flexibility)
            # Using fewer traces (500) and iterations (3) here so the batch runs faster
            shadow_mean, shadow_std, _ = calculate_gen_shadow_stable(
                log, net, im, fm, num_traces=500, iterations=3
            )
            
            runtime = time.time() - t0
            
            results.append({
                "Dataset": ds_name,
                "Miner": miner_name,
                "PM4Py_Baseline": round(pm4py_gen, 4),
                "Gen_Shadow (Pure)": round(shadow_mean, 4),
                "Runtime (s)": round(runtime, 1)
            })

    # ─── 4. Print the Leaderboard Table ──────────────────────────────────────
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print(" 📊 FINAL BENCHMARK RESULTS")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    
    # Save to CSV for your report
    df.to_csv("generalization_leaderboard.csv", index=False)
    print("\n✅ Saved to 'generalization_leaderboard.csv'")

if __name__ == "__main__":
    run_leaderboard()