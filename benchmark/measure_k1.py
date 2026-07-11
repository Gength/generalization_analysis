"""Measure the K=1 (single-draw, shipped default) runtime of ShadowGen directly,
rather than deriving it as K=5/5. For each (log, miner) cell it discovers the
model (excluded from the timing, as in the benchmark), then times
evaluate_miner at K=1 and K=5 on the same machine, so the ratio and the absolute
K=1 are both measured. Re-measuring K=5 also sanity-checks against the recorded
K=5 runtime. Writes results/k1_timing.json (append/merge per dataset).

Usage: python benchmark/measure_k1.py D1 [D2 ...]
"""
import os
import sys
import json
import time
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTHONHASHSEED", "0")
import numpy as np
import pm4py
import datasets as D
import run_m1_family as R
from HybridGen.algorithm import load_algorithm

SEED = 42
NUM = 1000
OUT = "benchmark/results/k1_timing.json"


def measure(dkey):
    entry = D.DATASETS[dkey]
    name, path = entry["name"], entry["log_path"]
    print(f"\n=== {dkey} {name} ===")
    log = pm4py.convert_to_event_log(pm4py.read_xes(path))
    algo = load_algorithm("v2.6")
    kw = {"successor_weighting": "mle"}
    rows = []
    for mn, fn in R.MINERS.items():
        model = fn(log)  # discovery, not timed
        def cached(_log, _m=model):
            return _m
        rec = {"dataset": name, "miner": mn}
        ks = (1,) if os.environ.get("K1ONLY") else (1, 5)
        for K in ks:
            random.seed(SEED); np.random.seed(SEED)
            r = algo.evaluate_miner(log, mn, cached, num_shadow_traces=NUM,
                                    iterations=K, seed=None, **kw)
            rec[f"k{K}_s"] = round(float(r["runtime_s"]), 3)
            rec[f"k{K}_mean"] = round(float(r["gen_shadow_mean"]), 4)
        rec["ratio"] = round(rec["k5_s"] / rec["k1_s"], 2) if rec.get("k5_s") and rec["k1_s"] else None
        rows.append(rec)
        k5s = f"{rec['k5_s']:8.2f}s" if "k5_s" in rec else "   (skipped)"
        print(f"  {mn:22} K1={rec['k1_s']:8.2f}s  K5={k5s}  ratio={rec['ratio']}")
    return rows


def main():
    keys = sys.argv[1:] or ["D1"]
    data = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    for k in keys:
        rows = measure(k)
        data[k] = rows
        json.dump(data, open(OUT, "w", encoding="utf-8"), indent=1)
    # summary over everything gathered so far
    allrows = [r for rs in data.values() for r in rs]
    k1 = sorted(r["k1_s"] for r in allrows if "k1_s" in r)
    k5 = sorted(r["k5_s"] for r in allrows if "k5_s" in r)
    ratios = sorted(r["ratio"] for r in allrows if r.get("ratio"))
    med = lambda xs: xs[len(xs) // 2] if xs else float("nan")
    print(f"\n--- {len(allrows)} cells gathered ({len(k5)} with measured K=5) ---")
    print(f"measured K=1 median {med(k1):.2f}s over {len(k1)} cells; "
          f"measured ratio median {med(ratios):.2f} over {len(ratios)} cells")
    print(f"written -> {OUT}")


if __name__ == "__main__":
    main()
