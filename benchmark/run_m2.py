"""
M2 — PM4Py Built-in Generalization
====================================
Single pm4py API call per miner. Fast (~0.3s/miner).
"""
import os, sys, json, time, argparse
from datetime import datetime, timezone

import pm4py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import DATASETS, get_info, CONFIG_DIR_V2
from miners import MINERS

# ── CLI ──
_cli = argparse.ArgumentParser(description="M2 PM4Py Built-in Gen")
_cli.add_argument("--dataset", default="D1", choices=list(DATASETS.keys()))
_cli.add_argument("--miners", nargs="*", default=None)
_args = _cli.parse_args()

info = get_info(_args.dataset)
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "results", "configs_v2")
os.makedirs(CONFIG_DIR, exist_ok=True)

# Load log
log = pm4py.read_xes(info["log_path"])
log = pm4py.convert_to_event_log(log)
print(f"Dataset: {info['name']}  Traces: {len(log)}")

target_miners = {k: v for k, v in MINERS.items() if _args.miners is None or k in _args.miners}

for name, miner_fn in target_miners.items():
    t0 = time.time()
    try:
        net, im, fm = miner_fn(log)
        score = pm4py.algo.evaluation.generalization.algorithm.apply(log, net, im, fm)
        runtime = time.time() - t0
        print(f"  [{name}] score={score:.4f} ({runtime:.2f}s)")
    except Exception as e:
        score = -1
        runtime = time.time() - t0
        print(f"  [{name}] ERROR: {e}")

    config = {
        "dataset": info["name"], "miner": name, "method": "M2",
        "method_label": "PM4Py Built-in Gen",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": "local", "seed": 42,
        "parameters": {},
        "results": {"score": score, "runtime_s": runtime},
        "notes": "",
    }
    path = os.path.join(CONFIG_DIR, f"{info['name']}__{name}__M2.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)

print(f"\nDone! {len(target_miners)} configs → {CONFIG_DIR}")
