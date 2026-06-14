"""R1-acceptance: fraction of HELD-OUT traces perfectly replayed (variant-based 5-fold x 3)."""
import sys, random, json
from collections import defaultdict
import numpy as np
import pm4py
from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay

DATASET = sys.argv[1] if len(sys.argv) > 1 else "D1"
PATHS = {"D1": ("Sepsis", "data/Sepsis Cases - Event Log_1_all/Sepsis Cases - Event Log.xes.gz"),
         "D2": ("BPI2013_Incidents", "data/BPI-Challenge_2013/Incident_Management_Log.xes.gz")}
NAME, LOG_PATH = PATHS[DATASET]
SEED = 42

from miners import MINERS

log = pm4py.read_xes(LOG_PATH)
log = pm4py.convert_to_event_log(log)

results = {}
for miner_name, miner_fn in MINERS.items():
    random.seed(SEED); np.random.seed(SEED)
    variant_map = defaultdict(list)
    for trace in log:
        variant_map[tuple(e["concept:name"] for e in trace)].append(trace)
    variants = list(variant_map.keys())
    n_variants, K, SHUFFLES = len(variants), 5, 3
    shuffle_accepts = []
    for _ in range(SHUFFLES):
        random.shuffle(variants)
        fold_size = max(1, n_variants // K)
        fold_accepts = []
        for i in range(K):
            start = i * fold_size
            end = (i + 1) * fold_size if i < K - 1 else n_variants
            test_variants = set(variants[start:end])
            train_log = EventLog([t for v in variants if v not in test_variants for t in variant_map[v]])
            test_log = EventLog([t for v in test_variants for t in variant_map[v]])
            try:
                net, im, fm = miner_fn(train_log)
                replayed = token_replay.apply(test_log, net, im, fm)
                fold_accepts.append(np.mean([1.0 if r["trace_is_fit"] else 0.0 for r in replayed]))
            except Exception:
                fold_accepts.append(0.0)
        shuffle_accepts.append(float(np.mean(fold_accepts)))
    results[miner_name] = (float(np.mean(shuffle_accepts)), float(np.std(shuffle_accepts)))
    print(f"R1_accept[{miner_name}] = {results[miner_name][0]:.4f} +- {results[miner_name][1]:.4f}", flush=True)

# Correlate with M1g gen_accept from configs_v2
def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0: return float("nan")
    return float(np.corrcoef(x, y)[0, 1])
def spearman(x, y):
    return pearson(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))

real = ["Alpha", "Alpha+", "Heuristics", "Heuristics_Strict", "Inductive_Infrequent", "Inductive_Strict"]
for method in ("M1f", "M1g"):
    gen_acc = {}
    for m in MINERS:
        with open(f"benchmark/results/configs_v2/{NAME}__{m}__{method}.json", encoding="utf-8") as f:
            gen_acc[m] = json.load(f)["results"]["gen_accept"]
    x = [gen_acc[m] for m in real]
    y = [results[m][0] for m in real]
    mae = float(np.mean(np.abs(np.array(x) - np.array(y))))
    print(f"\n{method} gen_accept vs R1_accept (real miners): "
          f"Pearson={pearson(x, y):.3f} Spearman={spearman(x, y):.3f} MAE={mae:.3f}")
    print("  miner                  gen_accept  R1_accept")
    for m in real + ["Trace_Filtered", "Flower"]:
        print(f"  {m:22s} {gen_acc[m]:9.4f}  {results[m][0]:9.4f}")
