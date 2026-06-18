"""
Step 2: Per-Miner DFG JSON Generation
=======================================
Reads the manifest from Step 1, simulates each PNML model, and exports
per-miner DFG JSONs to benchmark/models/dfg_models/.

These DFG JSONs are required by M6 (Bootstrap Generalization via Entropia -bgen)
and optionally by M3 (Entropic Relevance, which uses the log-level DFG).

Usage:
    uv run python benchmark/02_gen_per_miner_dfgs.py --dataset D1
    uv run python benchmark/02_gen_per_miner_dfgs.py --dataset D2

Pipeline position: after 01_prepare_models.py, before any M method.
"""

import os, sys, json, argparse
from collections import Counter
import pm4py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import get_info

# ── CLI ──────────────────────────────────────────────────────────────────────
_cli = argparse.ArgumentParser(description="Step 2: Per-miner DFG JSON generation")
_cli.add_argument("--dataset", default="D1", choices=["D1", "D2", "D3", "D4", "D5"],
                  help="Dataset key (default: D1)")
_cli.add_argument("--no-traces", type=int, default=5000,
                  help="Number of traces to simulate per miner (default: 5000)")
_args = _cli.parse_args()

DATASET_KEY = _args.dataset
N_SIM = _args.no_traces

info = get_info(DATASET_KEY)
manifest_path = info["manifest"]
model_dir = info["model_dir"]
dfg_dir = os.path.join(model_dir, "dfg_models")
os.makedirs(dfg_dir, exist_ok=True)

with open(manifest_path) as f:
    manifest = json.load(f)

print(f"Step 2: Per-miner DFG JSONs — {manifest['dataset']} ({DATASET_KEY})")
print(f"  Model dir: {model_dir}")
print(f"  DFG dir: {dfg_dir}")
print()

for miner_name, miner_info in manifest["miners"].items():
    pnml = miner_info["pnml"]
    if not os.path.exists(pnml):
        print(f"  [{miner_name}] SKIP — PNML not found: {pnml}")
        continue

    net, im, fm = pm4py.read_pnml(pnml)

    # Simulate model to collect behavior
    sim_log = pm4py.play_out(net, im, fm, no_traces=N_SIM)

    # Discover directly-follows graph from simulated log
    dfg, sa, ea = pm4py.discover_dfg(sim_log)

    # Build node frequency counts
    act_freq = Counter()
    for t in sim_log:
        for e in t:
            act_freq[e["concept:name"]] += 1

    acts = sorted(act_freq.keys())
    act_to_id = {a: i+1 for i, a in enumerate(acts)}

    nodes = [{"id": act_to_id[a], "label": a, "freq": act_freq[a]} for a in acts]
    nodes.append({"id": len(acts) + 1, "label": "INPUT", "freq": len(sim_log)})
    nodes.append({"id": len(acts) + 2, "label": "OUTPUT", "freq": len(sim_log)})

    arcs = [{"from": act_to_id[a], "to": act_to_id[b], "freq": f} for (a, b), f in dfg.items()]

    # Add start/end arcs (INPUT → first activities, last activities → OUTPUT)
    first_acts, last_acts = Counter(), Counter()
    for t in sim_log:
        seq = [e["concept:name"] for e in t]
        if seq:
            first_acts[seq[0]] += 1
            last_acts[seq[-1]] += 1

    for a, freq in first_acts.items():
        arcs.append({"from": len(acts) + 1, "to": act_to_id[a], "freq": freq})
    for a, freq in last_acts.items():
        arcs.append({"from": act_to_id[a], "to": len(acts) + 2, "freq": freq})

    out_path = os.path.join(dfg_dir, f"{miner_name}_dfg.json")
    with open(out_path, "w") as f:
        json.dump({"nodes": nodes, "arcs": arcs}, f, indent=2)

    print(f"  [{miner_name}] {len(acts)} activities → {os.path.relpath(out_path)}")

print(f"\nDone — {len(manifest['miners'])} miners → {os.path.relpath(dfg_dir)}/")
