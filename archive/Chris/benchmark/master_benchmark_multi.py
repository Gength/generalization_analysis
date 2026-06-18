"""
Generalization Benchmark — Multi-Algorithm, Multi-Dataset.

Benchmarks v1 / v2 / v2.1 / v2.2 across datasets, recording:
  - PM4Py Fitness (original log replay)
  - PM4Py Baseline Generalization
  - Gen_Shadow and Gen_Struct for each algorithm version
"""

import time
import os
import pandas as pd
import pm4py
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.algo.evaluation.generalization import algorithm as generalization_eval
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness

from HybridGen.algorithm import load_algorithm
from miners import MINERS  # defined in benchmark/miners.py, imported here to avoid circular imports

# ─── Configuration & Dataset Mapping ────────────────────────────────────────
DATASETS = {
    "BPI_2017": "data/BPI-Challenge_2017/BPI Challenge 2017.xes.gz",
    "BPI_2019": "data/BPI-Challenge_2019/BPI_Challenge_2019.xes.gz",
    "Sepsis": "data/Sepsis Cases - Event Log_1_all/Sepsis Cases - Event Log.xes.gz",
    "Hospital_Billing": "data/Hospital Billing - Event Log_1_all/Hospital Billing - Event Log.xes.gz",
    "Road_Traffic_Fine": "data/Road Traffic Fine Management Process_1_all/Road_Traffic_Fine_Management_Process.xes.gz"
}
# v1: DFG-based, no max_n. v2/v2.1/v2.2: N-gram, tested at N=3 and N=6.
ALGORITHMS = {
    "v1":    {"name": "v1",  "max_n": None, "label": "DFG + Good-Turing"},
    "v2_N=3":    {"name": "v2",  "max_n": 3,    "label": "N-gram v2 + Katz"},
    "v2_N=6":    {"name": "v2",  "max_n": 6,    "label": "N-gram v2 + Katz"},
    "v2.1_N=3":  {"name": "v21", "max_n": 3,    "label": "N-gram v2.1 + Katz"},
    "v2.1_N=6":  {"name": "v21", "max_n": 6,    "label": "N-gram v2.1 + Katz"},
    "v2.2_N=3":  {"name": "v22_eval", "max_n": 3,    "label": "N-gram v2.2 + Gen_Struct"},
    "v2.2_N=6":  {"name": "v22_eval", "max_n": 6,    "label": "N-gram v2.2 + Gen_Struct"},
}

# v2.2 intermediate dimensions (populated when algo returns dict)
STRUCT_DIMS = ["arc_flow_score", "reach_score", "cyclo_score"]

def is_event_log(obj):
    return hasattr(obj, '__iter__') and not isinstance(obj, pd.DataFrame)


def run_master_benchmark():
    results = []
    print("=" * 130)
    print("  BENCHMARK: Gen_Shadow + Gen_Struct across v1/v2/v2.1/v2.2")
    print("=" * 130)

    for ds_name, path in DATASETS.items():
        print(f"\nDataset: {ds_name}")
        if not os.path.exists(path):
            print(f"  Skipping - file not found: {path}")
            continue

        log = pm4py.read_xes(path)
        log = pm4py.convert_to_event_log(log)
        print(f"  Loaded: {len(log)} traces, {sum(len(t) for t in log)} events")

        for miner_label, miner_fn in MINERS.items():
            print(f"  Miner: {miner_label}...")

            t0 = time.time()
            net, im, fm = miner_fn(log)
            discovery_time = time.time() - t0


            fit_res = replay_fitness.apply(log, net, im, fm, variant=replay_fitness.Variants.TOKEN_BASED)
            pm4py_fitness = fit_res['log_fitness']


            t1 = time.time()

            pm4py_gen = generalization_eval.apply(log, net, im, fm)

            pm4py_time = time.time() - t1

            row = {
                "Dataset": ds_name, "Miner": miner_label,
                "PM4Py_Fitness": round(pm4py_fitness, 4),
                "PM4Py_Baseline_Gen": round(pm4py_gen, 4),
                "PM4Py_Time_s": round(pm4py_time, 2),
                "Discovery_Time_s": round(discovery_time, 1),
            }

            for algo_key, algo_cfg in ALGORITHMS.items():
                algo_mod = load_algorithm(algo_cfg["name"])
                mn = algo_cfg["max_n"]  # None for v1, 3/6 for v2+

                t1 = time.time()

                gs_result = algo_mod.calculate_gen_struct(log, net, im, fm)

                struct_time = time.time() - t1

                # Handle dict return (v22_eval) vs float return (v1/v2/v21)
                if isinstance(gs_result, dict):
                    gs = gs_result["gen_struct"]
                    # Store all intermediate dimensions
                    for dim in STRUCT_DIMS:
                        row[f"{algo_key}_Struct_{dim}"] = round(gs_result.get(dim, 0), 4)
                    # Also store raw values for potential reweighting
                    for raw_key in ["arc_flow_raw", "reach_raw", "cyclo_raw"]:
                        if raw_key in gs_result:
                            for rk, rv in gs_result[raw_key].items():
                                row[f"{algo_key}_Raw_{rk}"] = rv
                else:
                    gs = gs_result

                t1 = time.time()

                kwargs = dict(event_log=log, net=net, im=im, fm=fm,
                                num_traces=500, iterations=3)
                if mn is not None:
                    kwargs["max_n"] = mn
                shadow_mean, shadow_std, _ = algo_mod.calculate_gen_shadow_stable(**kwargs)[:3]

                shadow_time = time.time() - t1

                row[f"{algo_key}_Gen_Struct"] = round(gs, 4)
                row[f"{algo_key}_Gen_Shadow"] = round(shadow_mean, 4)
                row[f"{algo_key}_Struct_Time"] = round(struct_time, 1)
                row[f"{algo_key}_Shadow_Time"] = round(shadow_time, 1)
            results.append(row)

    df = pd.DataFrame(results)
    print("\n" + "=" * 130)
    print("  RESULTS")
    print("=" * 130)

    key_cols = ["Dataset", "Miner", "PM4Py_Fitness"]
    for algo_key in ALGORITHMS:
        key_cols += [f"{algo_key}_Gen_Struct", f"{algo_key}_Gen_Shadow"]
    print(df[key_cols].to_string(index=False))

    out_path = "master_benchmark_results_multi.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    return df


if __name__ == "__main__":
    run_master_benchmark()
