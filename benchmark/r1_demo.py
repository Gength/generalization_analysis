"""Compute R1 (K-Fold CV, k=5) for D1 Sepsis."""
import os, sys, json, time, random
from datetime import datetime, timezone
import numpy as np
import pm4py
from pm4py.objects.log.obj import EventLog
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness

# =============================================================================
# CONFIGURATION — Change these for your experiment
# =============================================================================
DATASET_KEY = "D1"            # Which dataset (D1-D5)
MINER_LIST = None              # None = all miners, or ["Alpha", ...]
SEED = 42
K_FOLDS = 5
SHUFFLES = 3

DATASETS = {
    "D1": {
        "name": "Sepsis",
        "log_path": "data/Sepsis Cases - Event Log_1_all/Sepsis Cases - Event Log.xes.gz",
        "config_dir": "benchmark/results/configs",
    },
}
random.seed(SEED)
np.random.seed(SEED)

# =============================================================================
# EXECUTION — Do not edit below this line
# =============================================================================

info = DATASETS.get(DATASET_KEY, list(DATASETS.values())[0])
DATASET = info["name"]
LOG_PATH = info["log_path"]
CONFIG_DIR = info["config_dir"]
K = K_FOLDS

log = pm4py.read_xes(LOG_PATH)
log = pm4py.convert_to_event_log(log)
print(f"Loaded {len(log)} traces")

MINERS = {
    "Alpha": pm4py.discover_petri_net_alpha,
    "Alpha+": pm4py.discover_petri_net_alpha_plus,
    "Heuristics": pm4py.discover_petri_net_heuristics,
    "Heuristics_Strict": lambda l: pm4py.discover_petri_net_heuristics(l, dependency_threshold=0.99),
    "Inductive_Strict": lambda l: pm4py.discover_petri_net_inductive(l, noise_threshold=0.0),
    "Inductive_Infrequent": lambda l: pm4py.discover_petri_net_inductive(l, noise_threshold=0.2),
}

def discover_flower_model(log):
    from pm4py.objects.petri_net.obj import PetriNet, Marking
    from pm4py.objects.petri_net.utils import petri_utils
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
    im[p_mid] = 1
    fm[p_mid] = 1
    return net, im, fm
MINERS["Flower"] = discover_flower_model

K = 5
SHUFFLES = 3

from collections import defaultdict

target_miners = dict(MINERS)
if MINER_LIST is not None:
    target_miners = {k: v for k, v in MINERS.items() if k in MINER_LIST}

for miner_name, miner_fn in target_miners.items():
    print(f"\n{miner_name}:")
    
    # Pre-group by variant
    variant_map = defaultdict(list)
    for trace in log:
        seq = tuple(e["concept:name"] for e in trace)
        variant_map[seq].append(trace)
    variants = list(variant_map.keys())
    n_variants = len(variants)
    
    all_fitnesses = []
    for shuffle in range(SHUFFLES):
        random.shuffle(variants)
        fold_size = max(1, n_variants // K)
        fold_fits = []
        for i in range(K):
            start = i * fold_size
            end = (i + 1) * fold_size if i < K - 1 else n_variants
            test_variants = variants[start:end]
            test_traces = [t for v in test_variants for t in variant_map[v]]
            train_traces = [t for v in variants if v not in test_variants for t in variant_map[v]]
            train_log = EventLog(train_traces)
            test_log = EventLog(test_traces)
            try:
                net, im, fm = miner_fn(train_log)
                fit = replay_fitness.apply(test_log, net, im, fm,
                                           variant=replay_fitness.Variants.TOKEN_BASED)["log_fitness"]
                fold_fits.append(fit)
            except:
                fold_fits.append(0.0)
        all_fitnesses.append(np.mean(fold_fits))
    
    mean = float(np.mean(all_fitnesses))
    std = float(np.std(all_fitnesses))
    print(f"  R1 (k={K}): {mean:.4f} ± {std:.4f}")
    
    config = {
        "dataset": DATASET,
        "miner": miner_name,
        "method": "R1",
        "method_label": f"K-Fold CV (k={K})",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": "local",
        "seed": SEED,
        "parameters": {"k": K, "shuffles": SHUFFLES, "variant_based": True},
        "results": {"mean": mean, "std": std, "raw_shuffles": all_fitnesses, "runtime_s": 0},
        "notes": ""
    }
    safe_miner = miner_name.replace(" ", "_")
    fname = f"{CONFIG_DIR}/{DATASET}__{safe_miner}__R1.json"
    with open(fname, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ {fname}")

print("\nDone! R1 configs written.")
