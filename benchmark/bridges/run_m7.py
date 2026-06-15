"""
M7 — SpeciAL4PM (Species-based Generalization)
===============================================
Uses the special4pm library directly (the tool's own API).
The tool depends on pm4py internally — that's fine, the bridge
does NOT import pm4py.

Reads manifest.json for model list.
"""
import os, sys, json, time, argparse
from datetime import datetime, timezone
from functools import partial

import numpy as np

# Import SpeciAL4PM (the external tool, not our project)
sys.path.insert(0, "src/SpeciAL-core")
from special4pm.estimation import SpeciesEstimator
from special4pm.species import retrieve_species_n_gram, retrieve_species_trace_variant
from special4pm.simulation.simulation import simulate_model

import pm4py  # SpeciAL4PM's dependency

# =============================================================================
# CONFIGURATION — Change these for your experiment
# =============================================================================
DATASET_KEY = "D1"            # Which dataset (D1-D5)
MINER_LIST = None              # None = all miners, or ["Alpha", ...]
OUTPUT_DIR = "benchmark/results/configs_v2"
SEED = 42

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from datasets import DATASETS, get_info, CONFIG_DIR_V2

# ── CLI override ────────────────────────────────────────────────────────────
_cli = argparse.ArgumentParser(description="M7 SpeciAL4PM")
_cli.add_argument("--dataset", default=None, choices=list(DATASETS.keys()),
                  help="Override DATASET_KEY (default: D1)")
_cli.add_argument("--miners", nargs="*", default=None,
                  help="Restrict to specific miners (default: all)")
_cli_args, _ = _cli.parse_known_args()
if _cli_args.dataset:
    DATASET_KEY = _cli_args.dataset
if _cli_args.miners is not None:
    MINER_LIST = _cli_args.miners

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# EXECUTION — Do not edit below this line
# =============================================================================

info = get_info(DATASET_KEY)
MODEL_DIR = info["model_dir"]
CONFIG_DIR = CONFIG_DIR_V2

with open(f"{MODEL_DIR}/manifest.json") as f:
    manifest = json.load(f)

DATASET = manifest["dataset"]
LOG_PATH = manifest["log_path"]

def write_config(miner, results, notes=""):
    config = {
        "dataset": DATASET, "miner": miner, "method": "M7",
        "method_label": "SpeciAL4PM",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": "local", "seed": 42,
        "parameters": {"n_grams": ["1-gram","2-gram","3-gram","tv"]},
        "results": results, "notes": notes,
    }
    with open(f"{CONFIG_DIR}/{DATASET}__{miner}__M7.json", "w") as f:
        json.dump(config, f, indent=2)

# Load log using pm4py (SpeciAL4PM's dependency — needed by SpeciesEstimator)
log = pm4py.read_xes(LOG_PATH)
log = pm4py.convert_to_event_log(log)
print(f"M7 — SpeciAL4PM ({len(log)} traces)")

# Original log profile (one-time)
estimator = SpeciesEstimator(step_size=None, d0=False, d1=False, d2=False, c0=True, c1=True)
estimator.register("1-gram", partial(retrieve_species_n_gram, n=1))
estimator.register("2-gram", partial(retrieve_species_n_gram, n=2))
estimator.register("3-gram", partial(retrieve_species_n_gram, n=3))
estimator.register("tv", retrieve_species_trace_variant)
estimator.apply(log, verbose=False)

orig_c1 = {}
for sp in ["1-gram", "2-gram", "3-gram", "tv"]:
    if "incidence_c1" in estimator.metrics[sp]:
        orig_c1[sp] = estimator.metrics[sp]["incidence_c1"][-1]
print(f"  Original C1: {orig_c1}")

# Per-miner simulation & comparison
target_miners = list(manifest["miners"].keys()) if MINER_LIST is None else MINER_LIST

for miner_name in target_miners:
    info = manifest["miners"][miner_name]
    print(f"  [{miner_name}]", end=" ")
    t0 = time.time()

    from pm4py.objects.petri_net.importer import importer as pnml_importer
    net, im, fm = pnml_importer.apply(info["pnml"])

    try:
        sim_log = simulate_model(net, im, fm, size=len(log))
        sim_estimator = SpeciesEstimator(step_size=None, d0=False, d1=False, d2=False, c0=True, c1=True)
        sim_estimator.register("1-gram", partial(retrieve_species_n_gram, n=1))
        sim_estimator.register("2-gram", partial(retrieve_species_n_gram, n=2))
        sim_estimator.register("3-gram", partial(retrieve_species_n_gram, n=3))
        sim_estimator.register("tv", retrieve_species_trace_variant)
        sim_estimator.apply(sim_log, verbose=False)

        sim_c1 = {}
        for sp in ["1-gram", "2-gram", "3-gram", "tv"]:
            if "incidence_c1" in sim_estimator.metrics[sp]:
                sim_c1[sp] = sim_estimator.metrics[sp]["incidence_c1"][-1]

        ratios = []
        for sp in orig_c1:
            if sp in sim_c1 and orig_c1[sp] > 0:
                ratios.append(min(sim_c1[sp] / orig_c1[sp], 1.0))
        score = float(np.mean(ratios)) if ratios else 0.0
        elapsed = time.time() - t0

        write_config(miner_name, {
            "c1_original": orig_c1, "c1_simulated": sim_c1,
            "gen_score": score, "runtime_s": elapsed,
        })
        print(f"C1_ratio={score:.4f} ({elapsed:.1f}s)")
    except Exception as e:
        write_config(miner_name, {"gen_score": -1, "runtime_s": time.time() - t0},
                     f"SpeciAL4PM error: {e}")
        print(f"ERROR: {e}")

print(f"\nDone → {CONFIG_DIR}/")
