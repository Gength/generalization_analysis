"""N/tau sensitivity sweep: is the metric's calibration a broad plateau, or a
sharp optimum at the single (N=6, tau=5) point calibrated on one log?

For each (max_n, safe_threshold) we score the six real miners with the exact
metric scorer (calculate_gen_shadow_stable, MLE weighting, K=5, seed 42) and
report MAE + Pearson vs the R1 ground truth. A plateau defuses the
"calibrated on one log via a proxy" threat.

Usage (repo root): PYTHONHASHSEED=0 python benchmark/nt_sweep.py D1 [--out ...]
"""
import os, sys, json, argparse, random, time
import numpy as np
import pm4py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import DATASETS
from miners import MINERS
from HybridGen.algorithm.v26 import calculate_gen_shadow_stable

SEED = 42
REAL = ["Alpha", "Alpha+", "Heuristics", "Heuristics_Strict",
        "Inductive_Infrequent", "Inductive_Strict"]
CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "configs")


def r1_of(dsname, miner):
    with open(os.path.join(CFG, f"{dsname}__{miner}__R1.json")) as f:
        return json.load(f)["results"]["mean"]


def run(dskey, Ns, taus, num_traces=1000, iters=5):
    ds = DATASETS[dskey]
    dsname = ds["name"]
    log = pm4py.convert_to_event_log(pm4py.read_xes(ds["log_path"]))
    print(f"[{dskey} {dsname}] discovering {len(REAL)} models...", flush=True)
    models = {m: MINERS[m](log) for m in REAL}
    r1 = np.array([r1_of(dsname, m) for m in REAL])
    grid = {}
    for N in Ns:
        for tau in taus:
            gs = []
            for m in REAL:
                random.seed(SEED); np.random.seed(SEED)
                net, im, fm = models[m]
                r = calculate_gen_shadow_stable(log, net, im, fm, num_traces, iters,
                                                safe_threshold=tau, max_n=N,
                                                successor_weighting="mle")
                gs.append(r["mean"])
            a = np.array(gs)
            mae = float(np.mean(np.abs(a - r1)))
            pear = float(np.corrcoef(a, r1)[0, 1]) if np.std(a) > 0 else float("nan")
            grid[f"N{N}_t{tau}"] = {"mae": round(mae, 4), "pearson": round(pear, 4),
                                    "scores": [round(x, 4) for x in gs]}
            print(f"  N={N} tau={tau:>2}: MAE={mae:.4f}  Pearson={pear:.4f}", flush=True)
    return {"dataset": dsname, "Ns": Ns, "taus": taus, "r1": [round(x, 4) for x in r1],
            "grid": grid}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=None)
    ap.add_argument("--Ns", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7, 8])
    ap.add_argument("--taus", type=int, nargs="+", default=[2, 5, 10])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    keys = args.datasets or ["D1"]
    out = {}
    for dk in keys:
        try:
            out[dk] = run(dk, args.Ns, args.taus)
        except Exception as e:
            out[dk] = {"error": repr(e)}
            print(f"{dk} ERROR: {e!r}", flush=True)
    path = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "results", "nt_sweep.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"-> {path}", flush=True)
