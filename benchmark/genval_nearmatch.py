"""Near-match generator premise: how CLOSE are generated traces to held-out
real variants, beyond exact match?

Exact match (generator_validation.py) is a conservative lower bound that
collapses on deep logs (D4: 0.09%). This reports the share of generated traces
within k edits (Levenshtein on activity sequences) of SOME held-out real
variant -- i.e. how "close-to-real" the shadow log is where exact reproduction
is impossible. Same R1-accept fold partitions, same MLE generator; the 1-gram
ablation and uniform-random traces are the baselines.

Usage (repo root): PYTHONHASHSEED=0 python benchmark/genval_nearmatch.py [D1 ...]
Writes benchmark/results/genval_nearmatch.json (analysis artifact).
"""
import os, sys, json, time, random, argparse
from collections import defaultdict
import numpy as np
import pm4py
from pm4py.objects.log.obj import EventLog

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import DATASETS
from HybridGen.algorithm.v26 import generate_shadow_log
from generator_validation import names, partitions, random_traces, SEED, K, SHUFFLES, N_GEN

MAXK = 3
BUCKET_CAP = 400  # cap held variants per length bucket (bounds cost on huge logs)


def edit_capped(a, b, maxd):
    """Levenshtein(a,b) with early exit: returns min(dist, maxd+1)."""
    la, lb = len(a), len(b)
    if abs(la - lb) > maxd:
        return maxd + 1
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        rowmin = i
        ai = a[i - 1]
        for j in range(1, lb + 1):
            v = prev[j] + 1
            if cur[j - 1] + 1 < v:
                v = cur[j - 1] + 1
            d = prev[j - 1] + (0 if ai == b[j - 1] else 1)
            if d < v:
                v = d
            cur[j] = v
            if v < rowmin:
                rowmin = v
        if rowmin > maxd:
            return maxd + 1
        prev = cur
    return prev[lb] if prev[lb] <= maxd else maxd + 1


def near_stats(seqs, held, maxk=MAXK):
    """Share of seqs within 1..maxk edits of some held variant (length-banded)."""
    by_len = defaultdict(list)
    for v in held:
        by_len[len(v)].append(v)
    rng = random.Random(SEED)
    for l in list(by_len):
        if len(by_len[l]) > BUCKET_CAP:
            by_len[l] = rng.sample(by_len[l], BUCKET_CAP)
    within = [0] * (maxk + 1)
    for s in seqs:
        L = len(s)
        best = maxk + 1
        for dl in range(0, maxk + 1):
            if dl >= best:
                break
            for cl in ({L} if dl == 0 else {L - dl, L + dl}):
                for v in by_len.get(cl, ()):
                    d = edit_capped(s, v, best - 1)
                    if d < best:
                        best = d
                        if best == 0:
                            break
                if best == 0:
                    break
            if best == 0:
                break
        for k in range(1, maxk + 1):
            if best <= k:
                within[k] += 1
    n = max(len(seqs), 1)
    return {f"within_{k}": within[k] / n for k in range(1, maxk + 1)}


def run_near(key, k=K):
    ds = DATASETS[key]
    log = pm4py.convert_to_event_log(pm4py.read_xes(ds["log_path"]))
    vmap = defaultdict(list)
    for t in log:
        vmap[names(t)].append(t)
    variants = list(vmap.keys())
    alphabet = sorted({a for v in variants for a in v})
    gens = ("shadow", "n1", "random")
    acc = {g: {f"within_{j}": [] for j in range(1, MAXK + 1)} for g in gens}
    t0 = time.time()
    for order, held in partitions(variants, k):
        train_vs = [v for v in order if v not in held]
        train_log = EventLog([t for v in train_vs for t in vmap[v]])
        for gname, kwargs in (("shadow", dict(max_n=6)), ("n1", dict(max_n=1))):
            random.seed(SEED); np.random.seed(SEED)
            shadow, *_ = generate_shadow_log(train_log, num_traces=N_GEN,
                                             successor_weighting="mle", **kwargs)
            st = near_stats([names(t) for t in shadow], held)
            for j in range(1, MAXK + 1):
                acc[gname][f"within_{j}"].append(st[f"within_{j}"])
        random.seed(SEED)
        st = near_stats(random_traces(train_vs, alphabet, N_GEN), held)
        for j in range(1, MAXK + 1):
            acc["random"][f"within_{j}"].append(st[f"within_{j}"])
    out = {"dataset": ds["name"], "n_variants": len(variants),
           "folds": SHUFFLES * k, "runtime_s": time.time() - t0}
    for g in gens:
        out[g] = {kk: float(np.mean(vv)) for kk, vv in acc[g].items()}
    s = out["shadow"]
    print(f"{key} {ds['name']}: shadow within-1/2/3 = "
          f"{s['within_1']:.3f}/{s['within_2']:.3f}/{s['within_3']:.3f} "
          f"| n1 w3={out['n1']['within_3']:.3f} | random w3={out['random']['within_3']:.4f} "
          f"({out['runtime_s']:.0f}s)", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    keys = args.datasets or ["D1", "D2", "D3", "D4", "D5"]
    results = {}
    for dk in keys:
        try:
            results[dk] = run_near(dk)
        except Exception as e:
            results[dk] = {"error": str(e)}
            print(f"{dk} ERROR: {e}", flush=True)
    path = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "results", "genval_nearmatch.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"-> {path}")
