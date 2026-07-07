"""Generator-premise test: does the shadow log contain unseen-but-REAL behavior?

The report's central premise ("the shadow log resembles valid future behavior")
rests on construction. This experiment tests it directly: train the generator
on the R1 train fold only, generate a shadow log, and count how many shadow
traces are EXACT matches of held-out real variants (which the generator never
saw, and which its dedup cannot have copied from the training fold).

Baselines: the same measure for the 1-gram ablation (max_n=1) and for uniform
random traces (train alphabet, train length distribution). Folds replicate the
R1-accept partitions exactly (seed 42, cumulative shuffles), 3 shuffles x 5
folds per dataset.

Usage (repo root): PYTHONHASHSEED=0 python benchmark/generator_validation.py [D1 D2 ...]
Writes benchmark/results/generator_validation.json (analysis artifact, not a
benchmark cell).
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

SEED, K, SHUFFLES, N_GEN = 42, 5, 3, 1000


def names(trace):
    return tuple(e["concept:name"] for e in trace)


def partitions(variants, k=K):
    """R1-accept's exact fold partitions: seed once, cumulative shuffles."""
    rng = random.Random(SEED)
    order = list(variants)
    out = []
    for s in range(SHUFFLES):
        rng.shuffle(order)
        n = len(order)
        fold = max(1, n // k)
        for f in range(k):
            start, end = f * fold, ((f + 1) * fold if f < k - 1 else n)
            out.append((list(order), set(order[start:end])))
    return out


def random_traces(train_names, alphabet, n):
    lengths = [len(t) for t in train_names]
    out = []
    for _ in range(n):
        L = random.choice(lengths)
        out.append(tuple(random.choice(alphabet) for _ in range(L)))
    return out


def run_dataset(key, k=K):
    ds = DATASETS[key]
    log = pm4py.convert_to_event_log(pm4py.read_xes(ds["log_path"]))
    vmap = defaultdict(list)
    for t in log:
        vmap[names(t)].append(t)
    variants = list(vmap.keys())
    alphabet = sorted({a for v in variants for a in v})

    res = {g: {"hit_rate": [], "var_coverage": []} for g in ("v26_mle_N6", "N1_ablation", "random")}
    t0 = time.time()
    for order, held in partitions(variants, k):
        train_vs = [v for v in order if v not in held]
        train_log = EventLog([t for v in train_vs for t in vmap[v]])

        for gname, kwargs in (("v26_mle_N6", dict(max_n=6)), ("N1_ablation", dict(max_n=1))):
            random.seed(SEED); np.random.seed(SEED)
            shadow, *_ = generate_shadow_log(train_log, num_traces=N_GEN,
                                             successor_weighting="mle", **kwargs)
            seqs = [names(t) for t in shadow]
            hits = [s for s in seqs if s in held]
            res[gname]["hit_rate"].append(len(hits) / len(seqs))
            res[gname]["var_coverage"].append(len(set(hits)) / len(held))

        random.seed(SEED)
        seqs = random_traces(train_vs, alphabet, N_GEN)
        hits = [s for s in seqs if s in held]
        res["random"]["hit_rate"].append(len(hits) / len(seqs))
        res["random"]["var_coverage"].append(len(set(hits)) / len(held))

    out = {"dataset": ds["name"], "n_variants": len(variants),
           "k_folds": k, "folds": SHUFFLES * k, "n_generated_per_fold": N_GEN,
           "runtime_s": time.time() - t0}
    for g, m in res.items():
        out[g] = {"hit_rate_mean": float(np.mean(m["hit_rate"])),
                  "hit_rate_std": float(np.std(m["hit_rate"])),
                  "var_coverage_mean": float(np.mean(m["var_coverage"]))}
    print(f"{key} {ds['name']}: shadow hit-rate {out['v26_mle_N6']['hit_rate_mean']:.4f} "
          f"| N=1 {out['N1_ablation']['hit_rate_mean']:.4f} "
          f"| random {out['random']['hit_rate_mean']:.6f} "
          f"({out['runtime_s']:.0f}s)", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*")
    ap.add_argument("--folds", type=int, default=K)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    keys = args.datasets or ["D1", "D2", "D3", "D4", "D5"]
    results = {}
    for dk in keys:
        try:
            results[dk] = run_dataset(dk, args.folds)
        except Exception as e:
            results[dk] = {"error": str(e)}
            print(f"{dk} ERROR: {e}", flush=True)
    path = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", "generator_validation.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"-> {path}")
