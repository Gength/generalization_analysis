"""
Step 1: Model Discovery — Export PNML + DFG JSON + XES for all miners.
This is the ONLY script that depends on pm4py (our project's tooling).
All bridge scripts read pre-exported files independently.
"""
import os, sys, json, shutil, argparse
from collections import Counter
from miners import MINERS

import pm4py

SEED = 42

# ── CLI ─────────────────────────────────────────────────────────────────────
_cli = argparse.ArgumentParser(description="Model discovery — export PNML + DFG")
_cli.add_argument("--dataset", default="D1", help="Dataset key (D1–D5)")
_args = _cli.parse_args()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import get_info

info = get_info(_args.dataset)
DATASET = info["name"]
LOG_PATH = info["log_path"]
MODEL_DIR = "benchmark/models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("Step 1: Model Discovery — Exporting PNML + DFG JSON")
print("=" * 60)

# ─── Load log ───────────────────────────────────────────────────────────────
log = pm4py.read_xes(LOG_PATH)
log = pm4py.convert_to_event_log(log)
print(f"Loaded: {len(log)} traces, {sum(len(t) for t in log)} events")


# ─── Export DFG JSON (log-level) ────────────────────────────────────────────
def export_dfg_json(log, path):
    dfg, sa, ea = pm4py.discover_dfg(log)
    act_freq = Counter()
    for t in log:
        for e in t:
            act_freq[e["concept:name"]] += 1
    acts = sorted(act_freq.keys())
    act_to_id = {a: i+1 for i, a in enumerate(acts)}
    nodes = [{"id": act_to_id[a], "label": a, "freq": act_freq[a]} for a in acts]
    nodes.append({"id": len(acts)+1, "label": "INPUT", "freq": len(log)})
    nodes.append({"id": len(acts)+2, "label": "OUTPUT", "freq": len(log)})
    arcs = [{"from": act_to_id[a], "to": act_to_id[b], "freq": f} for (a,b), f in dfg.items()]
    first_acts, last_acts = Counter(), Counter()
    for t in log:
        seq = [e["concept:name"] for e in t]
        if seq:
            first_acts[seq[0]] += 1
            last_acts[seq[-1]] += 1
    for a, freq in first_acts.items():
        arcs.append({"from": len(acts)+1, "to": act_to_id[a], "freq": freq})
    for a, freq in last_acts.items():
        arcs.append({"from": act_to_id[a], "to": len(acts)+2, "freq": freq})
    with open(path, "w") as f:
        json.dump({"nodes": nodes, "arcs": arcs}, f, indent=2)

slug = DATASET.lower().replace(" ", "_")
dfg_path = f"{MODEL_DIR}/{slug}_dfg.json"
export_dfg_json(log, dfg_path)
print(f"  DFG JSON → {dfg_path}")

# ─── Copy XES to model dir (so JAR tools find all files under same path) ────
xes_target = f"{MODEL_DIR}/{slug}.xes.gz"
if os.path.abspath(LOG_PATH) != os.path.abspath(xes_target):
    shutil.copy2(LOG_PATH, xes_target)
print(f"  XES → {xes_target}")

# ─── Discover & export per-miner ────────────────────────────────────────────
manifest = {
    "dataset": DATASET,
    "log_path": LOG_PATH,
    "model_dir": MODEL_DIR,
    "xes_file": xes_target,
    "dfg_json": dfg_path,
    "miners": {}
}

from pm4py.algo.evaluation.replay_fitness import algorithm as rf_eval

for miner_name, miner_fn in MINERS.items():
    net, im, fm = miner_fn(log)
    pnml_path = f"{MODEL_DIR}/{miner_name}.pnml"
    pm4py.write_pnml(net, im, fm, pnml_path)
    
    try:
        fitness = rf_eval.apply(log, net, im, fm, variant=rf_eval.Variants.TOKEN_BASED)["log_fitness"]
    except:
        fitness = -1

    print(f"  [{miner_name}]  {len(net.transitions)}t/{len(net.places)}p  fitness={fitness:.4f}  → {pnml_path}")

    manifest["miners"][miner_name] = {
        "pnml": pnml_path,
        "n_transitions": len(net.transitions),
        "n_places": len(net.places),
        "fitness": fitness,
    }

with open(f"{MODEL_DIR}/manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print(f"\nManifest → {MODEL_DIR}/manifest.json")
print("Done!")
