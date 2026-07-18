"""Exp1 -- break the shared-ruler threat.

ShadowGen and the R1 ground truth both score via token replay, so their
agreement could be a shared-engine artifact. This recomputes R1 on the six real
miners using cost-zero ALIGNMENTS (a different conformance engine) on the exact
same variant folds, then checks whether ShadowGen (still token-replay scored)
still tracks it. If it does, the agreement is not a ruler artifact.

Test variants are aligned once each (case-count weighted) to match R1's
case-weighted mean; a per-trace time cap guards against unsound-net hangs.

Usage (repo root): PYTHONHASHSEED=0 python benchmark/r1_alignment.py D1
"""
import os, sys, json, random, time, argparse
from collections import defaultdict
import numpy as np
import pm4py
from pm4py.objects.log.obj import EventLog
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import DATASETS
from miners import MINERS

SEED, K = 42, 5
SHUFFLES = int(os.environ.get("R1A_SHUFFLES", 3))   # fewer folds -> cheaper on big logs
REAL = ["Alpha", "Alpha+", "Heuristics", "Heuristics_Strict",
        "Inductive_Infrequent", "Inductive_Strict"]
CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "configs")
_ALIGN_CAP = int(os.environ.get("R1A_ALIGN_CAP", 30))   # per-trace align time cap (s)
ALIGN_PARAMS = {alignments.Parameters.PARAM_MAX_ALIGN_TIME_TRACE: _ALIGN_CAP}


def cfg_val(dsname, miner, method, key="mean"):
    with open(os.path.join(CFG, f"{dsname}__{miner}__{method}.json")) as f:
        return json.load(f)["results"][key]


def fold_partition(vorder, s_idx, fold_idx):
    rng = random.Random(SEED)
    variants = list(vorder)
    for _ in range(s_idx + 1):
        rng.shuffle(variants)
    n = len(variants)
    fs = max(1, n // K)
    start = fold_idx * fs
    end = (fold_idx + 1) * fs if fold_idx < K - 1 else n
    return variants, set(variants[start:end])


def r1_align_miner(miner, vmap, vorder):
    fold_means = []
    for s in range(SHUFFLES):
        for f in range(K):
            random.seed(SEED); np.random.seed(SEED)
            variants, test = fold_partition(vorder, s, f)
            train_log = EventLog([t for v in variants if v not in test for t in vmap[v]])
            test_vs = [v for v in variants if v in test]
            try:
                net, im, fm = MINERS[miner](train_log)
                uniq = EventLog([vmap[v][0] for v in test_vs])
                counts = np.array([len(vmap[v]) for v in test_vs], dtype=float)
                aligned = alignments.apply(uniq, net, im, fm, parameters=ALIGN_PARAMS)
                fits = np.array([a["fitness"] if a and a.get("fitness") is not None else np.nan
                                 for a in aligned])
                ok = ~np.isnan(fits)
                fold_means.append(float(np.average(fits[ok], weights=counts[ok])) if ok.any() else np.nan)
            except Exception as e:
                print(f"    {miner} s{s}f{f} FAIL {e!r}", flush=True)
                fold_means.append(np.nan)
    return float(np.nanmean(fold_means))


def run(dskey):
    ds = DATASETS[dskey]; dsname = ds["name"]
    log = pm4py.convert_to_event_log(pm4py.read_xes(ds["log_path"]))
    vmap = defaultdict(list)
    for t in log:
        vmap[tuple(e["concept:name"] for e in t)].append(t)
    vorder = list(vmap.keys())
    print(f"[{dskey} {dsname}] {len(vorder)} variants; alignment-R1 over {len(REAL)} miners", flush=True)
    r1a, r1t, m1g = {}, {}, {}
    for m in REAL:
        t0 = time.time()
        r1a[m] = r1_align_miner(m, vmap, vorder)
        r1t[m] = cfg_val(dsname, m, "R1")
        m1g[m] = cfg_val(dsname, m, "M1g")
        print(f"  {m:22s} R1-align={r1a[m]:.4f}  R1-replay={r1t[m]:.4f}  ShadowGen={m1g[m]:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    a = np.array([r1a[m] for m in REAL]); t = np.array([r1t[m] for m in REAL]); g = np.array([m1g[m] for m in REAL])
    def stats(x, y):
        return {"pearson": round(float(np.corrcoef(x, y)[0, 1]), 4),
                "mae": round(float(np.mean(np.abs(x - y))), 4)}
    out = {"dataset": dsname, "miners": REAL,
           "R1_align": {m: round(r1a[m], 4) for m in REAL},
           "R1_replay": {m: round(r1t[m], 4) for m in REAL},
           "ShadowGen": {m: round(m1g[m], 4) for m in REAL},
           "ShadowGen_vs_R1align": stats(g, a),
           "ShadowGen_vs_R1replay": stats(g, t),
           "R1align_vs_R1replay": stats(a, t)}
    print("\n== ShadowGen vs alignment-R1:", out["ShadowGen_vs_R1align"],
          "\n== ShadowGen vs replay-R1:   ", out["ShadowGen_vs_R1replay"],
          "\n== R1align vs R1replay:      ", out["R1align_vs_R1replay"], flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    keys = args.datasets or ["D1"]
    path = args.out or os.path.join(CFG, "..", "r1_alignment.json")
    for dk in keys:                    # write after EACH dataset so a later hang can't lose earlier work
        try:
            rec = run(dk)
        except Exception as e:
            rec = {"error": repr(e)}; print(f"{dk} ERROR {e!r}", flush=True)
        merged = {}
        if os.path.exists(path):
            try:
                merged = json.load(open(path))
            except Exception:
                merged = {}
        merged[dk] = rec               # preserve previously computed datasets (D1/D2)
        with open(path, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"-> {path}  (datasets now: {sorted(merged)})", flush=True)
