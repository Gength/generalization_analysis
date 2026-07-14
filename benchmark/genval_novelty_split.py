"""Does the NOVELTY actually land on real future behavior?

genval asks what share of shadow traces exactly match a held-out real variant.
Comparing that share across alpha conflates two things: whether a mutated trace
is realistic, and how much of the 1000-trace budget mutation consumes. So do not
compare alphas: split the SHIPPED shadow log (alpha=1) by its own mutation flag
and ask each half separately.

  hit_regular  share of NON-mutated (pure recombination) shadow traces that are
               exactly a held-out real variant
  hit_mutated  share of MUTATED shadow traces that are exactly a held-out real
               variant, i.e. traces carrying an event that was never seen in its
               resolved context, which nonetheless turn out to be real

If hit_mutated is ~0, the Katz-consistent mutation is producing behavior that is
novel but not real, and it is only diluting the shadow log. If hit_mutated is
comparable to hit_regular, novelty is reaching future behavior that recombination
provably cannot reach (a pure recombiner cannot emit a context-unseen step), and
that is precisely the axis on which ShadowGen differs from bootstrap resampling.

Usage: python benchmark/genval_novelty_split.py D1 D2 D3 D5
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
from HybridGen.algorithm.v26 import generate_shadow_log   # the FROZEN generator

SEED, K, THETA = 42, 5, 1000


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


def run(dk):
    log = pm4py.convert_to_event_log(pm4py.read_xes(DATASETS[dk]["log_path"]))
    vmap = defaultdict(list)
    for t in log:
        vmap[names(t)].append(t)

    reg_h, mut_h, mut_share, reg_new, mut_new = [], [], [], [], []
    for order, held in partitions(list(vmap.keys())):
        train_vs = [v for v in order if v not in held]
        train_log = EventLog([t for v in train_vs for t in vmap[v]])
        train_set = set(train_vs)

        random.seed(SEED); np.random.seed(SEED)
        shadow, flags, _d, _t, _c = generate_shadow_log(
            train_log, num_traces=THETA, safe_threshold=5, max_n=6,
            successor_weighting="mle")

        seqs = [names(t) for t in shadow]
        reg = [s for s, f in zip(seqs, flags) if not f]
        mut = [s for s, f in zip(seqs, flags) if f]
        mut_share.append(len(mut) / len(seqs))
        # hit = exactly a real variant the generator never saw
        if reg:
            reg_h.append(sum(1 for s in reg if s in held) / len(reg))
            reg_new.append(sum(1 for s in reg if s not in train_set) / len(reg))
        if mut:
            mut_h.append(sum(1 for s in mut if s in held) / len(mut))
            mut_new.append(sum(1 for s in mut if s not in train_set) / len(mut))

    f = lambda a: (float(np.mean(a)) * 100) if a else float("nan")
    return {"dataset": DATASETS[dk]["name"], "mutated_share_pct": f(mut_share),
            "hit_regular_pct": f(reg_h), "hit_mutated_pct": f(mut_h),
            "unseen_regular_pct": f(reg_new), "unseen_mutated_pct": f(mut_new)}


if __name__ == "__main__":
    keys = sys.argv[1:] or ["D1", "D2", "D3", "D5"]
    out = {}
    print(f"{'log':6} {'mutated%':>9} {'hit_regular%':>13} {'hit_mutated%':>13}")
    for dk in keys:
        r = run(dk)
        out[dk] = r
        print(f"{dk:6} {r['mutated_share_pct']:>9.1f} {r['hit_regular_pct']:>13.2f} "
              f"{r['hit_mutated_pct']:>13.2f}", flush=True)
    p = os.path.join(HERE, "results", "genval_novelty_split.json")
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1)
    print("written:", p)
