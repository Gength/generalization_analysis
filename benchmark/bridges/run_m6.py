"""
M6 — Bootstrap Generalization (adapted from src/bsgen/bsgen_eval.py)
====================================================================
Uses the BSGen bootstrap-sampling-with-breeding approach, but replaces
broken Entropia -emp/-emr with PM4Py token replay fitness.

Reference: Polyvyanyy, Moffat, García-Bañuelos (CAiSE 2022)
  "Bootstrapping Generalization of Process Models Discovered from Event Data"

Reads pre-exported models from benchmark/models/manifest.json
"""
import os, sys, json, copy, random, math, time
from datetime import datetime, timezone
from collections import defaultdict
from functools import reduce

import numpy as np
import pm4py
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.algo.evaluation.replay_fitness import algorithm as rf_eval
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.objects.log.exporter.xes import exporter as xes_exporter

sys.path.insert(0, "src/bsgen")
from bsgen_eval import log_sample_with_breeding, dedup

# =============================================================================
# CONFIGURATION — Change these for your experiment
# =============================================================================
DATASET_KEY = "D1"            # Which dataset (D1-D5)
MINER_LIST = None              # None = all miners, or ["Alpha", ...]
SEED = 42
N_BOOTSTRAP = 10               # bootstrap replicates
N_GENERATIONS = 10             # breeding generations
K = 2                          # subtrace length for crossover
P = 1.0                        # breeding probability
N_SAMPLE = 200                 # sample size per replicate

DATASETS = {
    "D1": {
        "name": "Sepsis",
        "log_path": "data/Sepsis Cases - Event Log_1_all/Sepsis Cases - Event Log.xes.gz",
        "manifest": "benchmark/models/manifest.json",
        "model_dir": "benchmark/models",
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
MODEL_DIR = info["model_dir"]
CONFIG_DIR = "benchmark/results/configs"
os.makedirs(CONFIG_DIR, exist_ok=True)

# ─── Load log ───────────────────────────────────────────────────────────────
log = pm4py.read_xes(LOG_PATH)
log = pm4py.convert_to_event_log(log)
print(f"M6 — Bootstrap Generalization ({len(log)} traces)")

with open(f"{MODEL_DIR}/manifest.json") as f:
    manifest = json.load(f)

def write_config(miner, results, notes=""):
    config = {
        "dataset": DATASET, "miner": miner, "method": "M6",
        "method_label": "Bootstrap Generalization (adapted)",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": "local", "seed": SEED,
        "parameters": {"generations": N_GENERATIONS, "k": K, "p": P,
                       "replicates": N_BOOTSTRAP, "sample_size": N_SAMPLE,
                       "source": "bsgen (breeding + token replay)"},
        "results": results, "notes": notes,
    }
    with open(f"{CONFIG_DIR}/{DATASET}__{miner}__M6.json", "w") as f:
        json.dump(config, f, indent=2)

target_miners = list(manifest["miners"].keys()) if MINER_LIST is None else MINER_LIST

for miner_name in target_miners:
    info = manifest["miners"][miner_name]
    print(f"\n  [{miner_name}]")
    t0 = time.time()

    net, im, fm = pm4py.read_pnml(info["pnml"])

    # Bootstrap with breeding: N_BOOTSTRAP replicates
    fitnesses = []
    for i in range(N_BOOTSTRAP):
        try:
            # Generate bred sample
            bred = log_sample_with_breeding(
                log, N_GENERATIONS, N_SAMPLE, K, P)

            # Deduplicate
            uniq = dedup(bred)

            # Token replay fitness
            fit = rf_eval.apply(uniq, net, im, fm,
                                variant=rf_eval.Variants.TOKEN_BASED)
            fitnesses.append(fit["log_fitness"])
        except Exception as e:
            print(f"    replicate {i}: {e}")
            continue

    elapsed = time.time() - t0

    if not fitnesses:
        write_config(miner_name, {"gen_score": -1, "runtime_s": elapsed},
                     "All bootstrap replicates failed")
        print(f"    All {N_BOOTSTRAP} replicates failed")
        continue

    mean = float(np.mean(fitnesses))
    std = float(np.std(fitnesses))
    ci = 1.96 * std / math.sqrt(len(fitnesses))

    write_config(miner_name, {
        "gen_score": mean,
        "std": std,
        "ci_95": ci,
        "n_replicates": len(fitnesses),
        "raw_fitnesses": fitnesses,
        "runtime_s": elapsed,
    })
    print(f"    gen={mean:.4f} ± {std:.4f} (95%CI: ±{ci:.4f}) [{elapsed:.0f}s]")

print(f"\nDone → {CONFIG_DIR}/")
