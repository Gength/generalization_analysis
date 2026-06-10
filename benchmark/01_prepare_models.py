"""
Step 1: Model Discovery — Export PNML + DFG JSON + XES for all miners.
This is the ONLY script that depends on pm4py (our project's tooling).
All bridge scripts read pre-exported files independently.
"""
import os, json, shutil
from collections import Counter

import pm4py
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils

SEED = 42
DATASET = "Sepsis"
LOG_PATH = "data/Sepsis Cases - Event Log_1_all/Sepsis Cases - Event Log.xes.gz"
MODEL_DIR = "benchmark/models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("Step 1: Model Discovery — Exporting PNML + DFG JSON")
print("=" * 60)

# ─── Load log ───────────────────────────────────────────────────────────────
log = pm4py.read_xes(LOG_PATH)
log = pm4py.convert_to_event_log(log)
print(f"Loaded: {len(log)} traces, {sum(len(t) for t in log)} events")

# ─── Miners ─────────────────────────────────────────────────────────────────
def flower_miner(log):
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
    "Alpha":                lambda l: pm4py.discover_petri_net_alpha(l),
    "Alpha+":               lambda l: pm4py.discover_petri_net_alpha_plus(l),
    "Heuristics":           lambda l: pm4py.discover_petri_net_heuristics(l),
    "Heuristics_Strict":    lambda l: pm4py.discover_petri_net_heuristics(l, dependency_threshold=0.99),
    "Inductive_Strict":     lambda l: pm4py.discover_petri_net_inductive(l, noise_threshold=0.0),
    "Inductive_Infrequent": lambda l: pm4py.discover_petri_net_inductive(l, noise_threshold=0.2),
    "Flower":               flower_miner,
}

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

dfg_path = f"{MODEL_DIR}/sepsis_dfg.json"
export_dfg_json(log, dfg_path)
print(f"  DFG JSON → {dfg_path}")

# ─── Copy XES to model dir (so JAR tools find all files under same path) ────
xes_target = f"{MODEL_DIR}/sepsis.xes.gz"
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
