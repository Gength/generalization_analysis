"""Relaxed generator-premise reading: hit = X% similar, not exact.

The genval table counts a shadow trace as a hit only on an EXACT activity-
sequence match with a held-out real variant. This script relaxes the standard:
a trace counts as a hit at threshold t if its best normalized Levenshtein
similarity to any held-out variant is >= t, with

    sim(s, v) = 1 - d(s, v) / max(|s|, |v|).

Reported for t in {exact, 0.9, 0.8, 0.7}, next to the uniformly random floor
(same alphabet, same length distribution), on the same R1-accept fold
partitions as the exact reading (seed 42, 1 shuffle x 5 folds, theta=1000).

Note the threshold is RELATIVE, so it means different things per log: on L5
(mean length 6.3) 90% requires an exact match for most traces, while on L3
(38.2) it allows ~4 edits. That is the point: it answers "how close is the
shadow log to real future behavior" in units that scale with trace depth.

L4 is skipped: with 28k variants and 57-event traces the banded search is
hours of CPU for one cell, and the near-match reading already covers it.

Usage: python benchmark/genval_relaxed.py D1 D2 D3 D5
"""
import os, sys, json, random
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

import pm4py
from pm4py.objects.log.obj import EventLog
from collections import defaultdict
from datasets import DATASETS
from HybridGen.algorithm.v26 import generate_shadow_log   # FROZEN generator

SEED, K, THETA = 42, 5, 1000
THRESHOLDS = (0.9, 0.8, 0.7)
TMIN = min(THRESHOLDS)
BUCKET_CAP = 400   # cap held variants per length bucket (same as genval_nearmatch)


def names(t):
    return tuple(e["concept:name"] for e in t)


def partitions(variants, k=K):
    rng = random.Random(SEED)
    order = list(variants)
    n = len(order)
    rng.shuffle(order)
    fold = max(1, n // k)
    for f in range(k):
        s, e = f * fold, ((f + 1) * fold if f < k - 1 else n)
        yield order, set(order[s:e])


def edit_capped(a, b, maxd):
    """Levenshtein with early abort once the distance must exceed maxd."""
    if abs(len(a) - len(b)) > maxd:
        return maxd + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        lo = maxd + 1
        for j, cb in enumerate(b, 1):
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            cur.append(v)
            lo = min(lo, v)
        if lo > maxd:
            return maxd + 1
        prev = cur
    return prev[-1]


def best_sim(s, by_len, lens_sorted):
    """max over held variants v of 1 - d/max(|s|,|v|), pruned to sim >= TMIN."""
    L = len(s)
    best = 0.0
    exact_seen = False
    # closest lengths first, so `best` tightens early
    for cl in sorted(lens_sorted, key=lambda x: abs(x - L)):
        m = max(L, cl)
        # even a perfect alignment cannot beat this bucket's ceiling
        ceiling = 1.0 - abs(L - cl) / m
        if ceiling < max(best, TMIN):
            continue
        # d needed to beat current best (and stay above TMIN)
        maxd = int(m * (1.0 - max(best, TMIN)))
        for v in by_len[cl]:
            d = edit_capped(s, v, maxd)
            if d <= maxd:
                sim = 1.0 - d / m
                if sim > best:
                    best = sim
                    if d == 0:
                        exact_seen = True
                    maxd = int(m * (1.0 - max(best, TMIN)))
        if exact_seen:
            break
    return best


def random_traces(train_names, alphabet, n, rng):
    lens = [len(t) for t in train_names]
    return [tuple(rng.choice(alphabet) for _ in range(rng.choice(lens)))
            for _ in range(n)]


def run(dk):
    log = pm4py.convert_to_event_log(pm4py.read_xes(DATASETS[dk]["log_path"]))
    vmap = defaultdict(list)
    for t in log:
        vmap[names(t)].append(t)
    alphabet = sorted({a for v in vmap for a in v})

    acc = {g: {("exact",): [], **{(t,): [] for t in THRESHOLDS}}
           for g in ("shadow", "random")}
    for order, held in partitions(list(vmap.keys())):
        train_vs = [v for v in order if v not in held]
        train_log = EventLog([t for v in train_vs for t in vmap[v]])

        by_len = defaultdict(list)
        rng_b = random.Random(SEED)
        for v in held:
            by_len[len(v)].append(v)
        for l in list(by_len):
            if len(by_len[l]) > BUCKET_CAP:
                by_len[l] = rng_b.sample(by_len[l], BUCKET_CAP)
        lens_sorted = list(by_len.keys())

        random.seed(SEED); np.random.seed(SEED)
        shadow, *_ = generate_shadow_log(train_log, num_traces=THETA,
                                         safe_threshold=5, max_n=6,
                                         successor_weighting="mle")
        gens = {"shadow": [names(t) for t in shadow],
                "random": random_traces(train_vs, alphabet, THETA,
                                        random.Random(SEED))}
        for g, seqs in gens.items():
            sims = [best_sim(s, by_len, lens_sorted) for s in seqs]
            exact = [s for s in seqs if s in held]
            acc[g][("exact",)].append(len(exact) / len(seqs))
            for t in THRESHOLDS:
                acc[g][(t,)].append(sum(1 for x in sims if x >= t) / len(seqs))

    out = {"dataset": DATASETS[dk]["name"]}
    for g in ("shadow", "random"):
        out[g] = {"exact": float(np.mean(acc[g][("exact",)])) * 100,
                  **{f"sim{int(t*100)}": float(np.mean(acc[g][(t,)])) * 100
                     for t in THRESHOLDS}}
    return out


if __name__ == "__main__":
    keys = sys.argv[1:] or ["D1", "D2", "D3", "D5"]
    res = {}
    hdr = f"{'log':5} {'gen':7} {'exact':>7} {'>=90%':>7} {'>=80%':>7} {'>=70%':>7}"
    print(hdr, flush=True)
    for dk in keys:
        r = run(dk)
        res[dk] = r
        for g in ("shadow", "random"):
            v = r[g]
            print(f"{dk:5} {g:7} {v['exact']:>7.2f} {v['sim90']:>7.2f} "
                  f"{v['sim80']:>7.2f} {v['sim70']:>7.2f}", flush=True)
        json.dump(res, open(os.path.join(HERE, "results", "genval_relaxed.json"),
                            "w", encoding="utf-8"), indent=1)
    print("written: benchmark/results/genval_relaxed.json")
