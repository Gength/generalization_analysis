"""Are the NOVEL traces plausible, even when they are not exactly real?

genval_novelty_split.py showed that mutated shadow traces exactly match a
held-out real variant far less often than pure recombinations. But exact match
is a conservative lower bound (the held-out fifth is a small sample of the true
future, and a valid novel trace may simply not occur in this log at all). The
fair test of the novelty machinery is the near-match reading the report already
uses: is a mutated trace within k edits of SOME real held-out variant?

  near_regular[k]  share of NON-mutated shadow traces within k edits of a
                   held-out real variant
  near_mutated[k]  same, for traces carrying a context-novel event

If near_mutated stays close to near_regular, the mutations land in the
neighbourhood of real future behavior: novel, but plausible, which is what the
construct asks for. If it collapses, the mutations are noise, and a model is
being penalised for rejecting behavior that is not valid.

Reuses the report's own edit-distance machinery (genval_nearmatch.near_stats),
so the numbers are on the same yardstick as the published <=3-edit figures.

Usage: python benchmark/nearmatch_novelty_split.py D1 D2 D3 D5
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
from HybridGen.algorithm.v26 import generate_shadow_log      # FROZEN generator
from genval_nearmatch import near_stats, MAXK                # report's yardstick
from generator_validation import names, SEED, K

THETA = 1000


def partitions_1shuffle(variants, k=K):
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

    reg_acc = {k: [] for k in range(1, MAXK + 1)}
    mut_acc = {k: [] for k in range(1, MAXK + 1)}
    shares = []
    for order, held in partitions_1shuffle(list(vmap.keys())):
        train_vs = [v for v in order if v not in held]
        train_log = EventLog([t for v in train_vs for t in vmap[v]])
        random.seed(SEED); np.random.seed(SEED)
        shadow, flags, *_ = generate_shadow_log(
            train_log, num_traces=THETA, safe_threshold=5, max_n=6,
            successor_weighting="mle")
        seqs = [names(t) for t in shadow]
        reg = [s for s, f in zip(seqs, flags) if not f]
        mut = [s for s, f in zip(seqs, flags) if f]
        shares.append(len(mut) / len(seqs) if seqs else 0.0)
        held_l = list(held)
        # near_stats returns {"within_k": share_of_seqs}, already normalised
        if reg:
            w = near_stats(reg, held_l)
            for k in range(1, MAXK + 1):
                reg_acc[k].append(w[f"within_{k}"])
        if mut:
            w = near_stats(mut, held_l)
            for k in range(1, MAXK + 1):
                mut_acc[k].append(w[f"within_{k}"])

    pct = lambda a: (float(np.mean(a)) * 100) if a else float("nan")
    return {
        "dataset": DATASETS[dk]["name"],
        "mutated_share_pct": pct(shares),
        "near_regular_pct": {k: pct(reg_acc[k]) for k in range(1, MAXK + 1)},
        "near_mutated_pct": {k: pct(mut_acc[k]) for k in range(1, MAXK + 1)},
    }


if __name__ == "__main__":
    keys = sys.argv[1:] or ["D1", "D2", "D3", "D5"]
    out = {}
    print(f"{'log':5} {'mut%':>6} | "
          + " ".join(f"{'reg<='+str(k):>9}" for k in range(1, MAXK + 1))
          + " | " + " ".join(f"{'mut<='+str(k):>9}" for k in range(1, MAXK + 1)))
    for dk in keys:
        r = run(dk)
        out[dk] = r
        rg = " ".join(f"{r['near_regular_pct'][k]:>9.2f}" for k in range(1, MAXK + 1))
        mu = " ".join(f"{r['near_mutated_pct'][k]:>9.2f}" for k in range(1, MAXK + 1))
        print(f"{dk:5} {r['mutated_share_pct']:>6.1f} | {rg} | {mu}", flush=True)
        json.dump(out, open(os.path.join(HERE, "results",
                  "nearmatch_novelty_split.json"), "w", encoding="utf-8"), indent=1)
    print("written: benchmark/results/nearmatch_novelty_split.json")
