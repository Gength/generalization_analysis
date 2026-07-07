"""Exp4 -- enlarge the agreement sample + bootstrap CIs.

The four-criteria agreement rests on only six miners. This adds six more
discovery configs (ILP + Inductive/Heuristics threshold variants) on D1, scores
every miner uniformly (ShadowGen graded fitness vs R1 = variant k-fold CV
token-replay fitness), and reports the metric-vs-R1 Pearson/Spearman/MAE over
the enlarged set plus a bootstrap 95% CI over the miner set -- so the agreement
is shown robust, not an artifact of a lucky 6-point sample.

Usage (repo root): PYTHONHASHSEED=0 python benchmark/exp4_miners.py D1
"""
import os, sys, json, random, time, argparse
from collections import defaultdict
import numpy as np
import pm4py
from pm4py.objects.log.obj import EventLog
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import DATASETS
from miners import MINERS
from HybridGen.algorithm.v26 import calculate_gen_shadow_stable

SEED, K, SHUFFLES = 42, 5, 3
BASE = ["Alpha", "Alpha+", "Heuristics", "Heuristics_Strict",
        "Inductive_Infrequent", "Inductive_Strict"]
NEW = {
    # ILP miner dropped: ~38 min per discovery x 15 R1 folds is impractical for CV.
    "Inductive_n0.1":  lambda l: pm4py.discover_petri_net_inductive(l, noise_threshold=0.1),
    "Inductive_n0.3":  lambda l: pm4py.discover_petri_net_inductive(l, noise_threshold=0.3),
    "Inductive_n0.4":  lambda l: pm4py.discover_petri_net_inductive(l, noise_threshold=0.4),
    "Heuristics_d0.7": lambda l: pm4py.discover_petri_net_heuristics(l, dependency_threshold=0.7),
    "Heuristics_d0.9": lambda l: pm4py.discover_petri_net_heuristics(l, dependency_threshold=0.9),
}


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


def r1_of(miner_fn, vmap, vorder):
    fits = []
    for s in range(SHUFFLES):
        for f in range(K):
            random.seed(SEED); np.random.seed(SEED)
            variants, test = fold_partition(vorder, s, f)
            train = EventLog([t for v in variants if v not in test for t in vmap[v]])
            test_log = EventLog([t for v in variants if v in test for t in vmap[v]])
            try:
                net, im, fm = miner_fn(train)
                rep = token_replay.apply(test_log, net, im, fm)
                fits.append(float(np.mean([r["trace_fitness"] for r in rep])))
            except Exception:
                fits.append(np.nan)
    return float(np.nanmean(fits))


def m1g_of(miner_fn, log):
    random.seed(SEED); np.random.seed(SEED)
    net, im, fm = miner_fn(log)
    r = calculate_gen_shadow_stable(log, net, im, fm, 1000, 5,
                                    safe_threshold=5, max_n=6, successor_weighting="mle")
    return r["mean"]


def boot_ci(x, y, B=5000):
    idx = np.arange(len(x))
    pears, maes = [], []
    rng = random.Random(SEED)
    for _ in range(B):
        s = [rng.randrange(len(x)) for _ in idx]
        xs, ys = x[s], y[s]
        maes.append(float(np.mean(np.abs(xs - ys))))
        if np.std(xs) > 1e-9 and np.std(ys) > 1e-9:
            pears.append(float(np.corrcoef(xs, ys)[0, 1]))
    q = lambda a, p: float(np.percentile(a, p))
    return {"pearson_ci95": [round(q(pears, 2.5), 3), round(q(pears, 97.5), 3)],
            "mae_ci95": [round(q(maes, 2.5), 4), round(q(maes, 97.5), 4)]}


def run(dskey):
    ds = DATASETS[dskey]
    log = pm4py.convert_to_event_log(pm4py.read_xes(ds["log_path"]))
    vmap = defaultdict(list)
    for t in log:
        vmap[tuple(e["concept:name"] for e in t)].append(t)
    vorder = list(vmap.keys())
    fns = {m: MINERS[m] for m in BASE}
    fns.update(NEW)
    print(f"[{dskey} {ds['name']}] {len(fns)} miners x (M1g + {SHUFFLES*K}-fold R1)", flush=True)
    m1g, r1 = {}, {}
    for name, fn in fns.items():
        t0 = time.time()
        try:
            m1g[name] = m1g_of(fn, log)
            r1[name] = r1_of(fn, vmap, vorder)
            print(f"  {name:18s} ShadowGen={m1g[name]:.3f}  R1={r1[name]:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"  {name:18s} SKIP {e!r}", flush=True)
    names = [n for n in fns if n in m1g and not np.isnan(r1[n])]
    g = np.array([m1g[n] for n in names]); t = np.array([r1[n] for n in names])
    from scipy import stats as st
    out = {"dataset": ds["name"], "n_miners": len(names), "miners": names,
           "ShadowGen": {n: round(m1g[n], 4) for n in names},
           "R1": {n: round(r1[n], 4) for n in names},
           "pearson": round(float(st.pearsonr(g, t)[0]), 4),
           "spearman": round(float(st.spearmanr(g, t)[0]), 4),
           "mae": round(float(np.mean(np.abs(g - t))), 4),
           "bootstrap": boot_ci(g, t)}
    print(f"\n== enlarged agreement ({len(names)} miners): Pearson={out['pearson']} "
          f"Spearman={out['spearman']} MAE={out['mae']}", flush=True)
    print(f"== bootstrap 95% CI: Pearson {out['bootstrap']['pearson_ci95']}  "
          f"MAE {out['bootstrap']['mae_ci95']}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = {}
    for dk in (args.datasets or ["D1"]):
        try:
            res[dk] = run(dk)
        except Exception as e:
            res[dk] = {"error": repr(e)}; print(f"{dk} ERROR {e!r}", flush=True)
    path = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "results", "exp4_miners.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"-> {path}", flush=True)
