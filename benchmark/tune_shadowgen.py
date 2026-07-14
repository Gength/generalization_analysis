"""Hyperparameter / algorithm-tweak search for ShadowGen.

The shipped metric lives in HybridGen/algorithm/v26.py, which is FROZEN: every
number in the report regenerates from it. This harness therefore never edits it.
It loads the source, textually injects new knobs at asserted anchors, and execs
the result (same pattern as nt_perstep_sweep.py). If v26.py ever changes, an
anchor assertion fails loudly rather than silently tuning a different algorithm.

Knobs
-----
Already in v26 (exposed as call arguments):
  max_n          N, the context bound                     (shipped 6)
  safe_threshold tau, min successor count to trust        (shipped 5)
  num_traces     theta, shadow-log size                   (shipped 1000)
  weighting      'mle' | 'log'                            (shipped 'mle')

NOT in v26; injected here:
  alpha       scales the Good-Turing novelty rate: fire a mutation with
              probability min(clamp, alpha * p_unseen). alpha=1 is shipped.
              alpha=0 disables novelty entirely -> pure recombination, i.e.
              the bootstrap regime. This is the novelty ablation the report
              does not have (it has a *context* ablation, the 1-gram floor).
  pu_clamp    hard ceiling on the per-step mutation probability (shipped 1.0)
  temp        weighting='temp' -> w(c) = c ** (1/temp). temp=1 IS mle, so this
              knob strictly generalizes the shipped weighting; temp>1 flattens
              toward uniform, temp<1 sharpens toward the mode.
  cap_mult    trace-length cap = min(max(100, cap_mult*max_len), 1000).
              v26 hardcodes cap_mult=2.
  mut_uniform propose the novel activity uniformly instead of by frequency.

Objectives
----------
  mae      mean absolute error to the R1 cross-validation ground truth over the
           six non-degenerate miners. The real objective, but note it optimises
           against the very reference we validate against: leave-one-log-out is
           reported so overfitting is visible.
  genval   share of generated traces that exactly match a HELD-OUT real variant
           (generator realism; no Petri net, no R1 scores involved).
  mutrate  realised mutation rate. REPORTED, NEVER OPTIMISED: it rises
           monotonically with N and has no optimum (fig:nsweep), so treating it
           as a target would reinstate the retired proxy criterion.

Usage:
  python benchmark/tune_shadowgen.py selftest
  python benchmark/tune_shadowgen.py mae     --grid quick   [--logs D1,D2,D5]
  python benchmark/tune_shadowgen.py genval  --grid full
"""
import os, sys, json, time, random, argparse, itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import pm4py
from datasets import DATASETS
from miners import MINERS

SEED = 42
REAL = ["Alpha", "Alpha+", "Heuristics", "Heuristics_Strict",
        "Inductive_Infrequent", "Inductive_Strict"]
CFG = os.path.join(HERE, "results", "configs")
OUT = os.path.join(HERE, "results")
V26 = os.path.join(ROOT, "HybridGen", "algorithm", "v26.py")

# --------------------------------------------------------------------------- #
# inject the new knobs into the frozen generator                              #
# --------------------------------------------------------------------------- #
HEADER = '''
_TUNE = {"alpha": 1.0, "pu_clamp": 1.0, "temp": 1.0,
         "cap_mult": 2.0, "mut_uniform": False}

def _tune_pfire(p):
    return min(_TUNE["pu_clamp"], _TUNE["alpha"] * p)
'''

PATCHES = [
    # 1. novelty scale + clamp on the Good-Turing fire probability
    ("                if random.random() < p_unseen and mut_choices:",
     "                if random.random() < _tune_pfire(p_unseen) and mut_choices:"),
    # 2. trace-length cap multiplier (v26 hardcodes 2)
    ("        max_trace_length = min(max(100, 2 * max_observed_len), 1000)",
     "        max_trace_length = min(max(100, int(_TUNE['cap_mult'] * max_observed_len)), 1000)"),
    # 3. temperature weighting (temp=1 reproduces 'mle' exactly)
    ('''    elif successor_weighting == "mle":
        weight_fn = float
    else:''',
     '''    elif successor_weighting == "mle":
        weight_fn = float
    elif successor_weighting == "temp":
        _T = float(_TUNE["temp"])
        weight_fn = (lambda c: 1.0) if _T >= 999 else (lambda c: float(c) ** (1.0 / _T))
    else:'''),
    # 4. uniform mutation proposal (both construction sites)
    ("                mut_weights = [weight_fn(c) for c in cand.values()]",
     "                mut_weights = ([1.0] * len(cand)) if _TUNE['mut_uniform'] else [weight_fn(c) for c in cand.values()]"),
    ("            mut_weights = [weight_fn(c) for c in cand.values()]",
     "            mut_weights = ([1.0] * len(cand)) if _TUNE['mut_uniform'] else [weight_fn(c) for c in cand.values()]"),
    # 5. do not register into the package from an exec'd copy
    ("from . import register_algorithm\nregister_algorithm(\"v2.6\")",
     "# registration disabled in the tuning exec copy"),
]


def load_tunable():
    src = open(V26, encoding="utf-8").read()
    for old, new in PATCHES:
        assert old in src, f"v26.py changed; tuning anchor not found: {old[:70]!r}"
        src = src.replace(old, new, 1)
    src = src.replace("def generate_shadow_log(", HEADER + "\ndef generate_shadow_log(", 1)
    ns = {}
    exec(compile(src, V26 + " [tunable]", "exec"), ns)
    return ns


NS = load_tunable()
TUNE = NS["_TUNE"]


def set_knobs(cfg):
    TUNE["alpha"] = cfg.get("alpha", 1.0)
    TUNE["pu_clamp"] = cfg.get("pu_clamp", 1.0)
    TUNE["temp"] = cfg.get("temp", 1.0)
    TUNE["cap_mult"] = cfg.get("cap_mult", 2.0)
    TUNE["mut_uniform"] = cfg.get("mut_uniform", False)


DEFAULT = dict(max_n=6, safe_threshold=5, num_traces=1000, weighting="mle",
               alpha=1.0, pu_clamp=1.0, temp=1.0, cap_mult=2.0, mut_uniform=False)


def label(cfg):
    d = {**DEFAULT, **cfg}
    parts = [f"N{d['max_n']}", f"t{d['safe_threshold']}", d["weighting"]]
    if d["weighting"] == "temp":
        parts[-1] = f"temp{d['temp']}"
    if d["alpha"] != 1.0:
        parts.append(f"a{d['alpha']}")
    if d["pu_clamp"] != 1.0:
        parts.append(f"clamp{d['pu_clamp']}")
    if d["cap_mult"] != 2.0:
        parts.append(f"cap{d['cap_mult']}")
    if d["mut_uniform"]:
        parts.append("mutU")
    if d["num_traces"] != 1000:
        parts.append(f"th{d['num_traces']}")
    return "_".join(str(p) for p in parts)


# --------------------------------------------------------------------------- #
# scoring                                                                     #
# --------------------------------------------------------------------------- #
def r1_of(dsname, miner):
    with open(os.path.join(CFG, f"{dsname}__{miner}__R1.json"), encoding="utf-8") as f:
        return json.load(f)["results"]["mean"]


def score_cell(log, net, im, fm, cfg):
    """One (log, miner) cell at K=1, via the SHIPPED scoring path, so that the
    default knob setting reproduces the committed number bit-exactly."""
    d = {**DEFAULT, **cfg}
    set_knobs(d)
    random.seed(SEED)
    np.random.seed(SEED)
    r = NS["calculate_gen_shadow_stable"](
        log, net, im, fm, d["num_traces"], iterations=1,
        safe_threshold=d["safe_threshold"], max_n=d["max_n"],
        successor_weighting=d["weighting"])
    mutrate = r["mutation_counts"][0] / float(d["num_traces"])
    return float(r["mean"]), float(mutrate)


def eval_mae(cfg, dslist, cache):
    """MAE + Pearson to R1 over the six non-degenerate miners, per log."""
    per_log = {}
    for dk in dslist:
        ds = DATASETS[dk]["name"]
        log, nets = cache[dk]
        ys, r1s, mrs = [], [], []
        for m in REAL:
            net, im, fm = nets[m]
            s, mr = score_cell(log, net, im, fm, cfg)
            ys.append(s); r1s.append(r1_of(ds, m)); mrs.append(mr)
        ys, r1s = np.array(ys), np.array(r1s)
        per_log[dk] = {
            "mae": float(np.mean(np.abs(ys - r1s))),
            "pearson": float(np.corrcoef(ys, r1s)[0, 1]) if np.std(ys) > 0 else float("nan"),
            "spread": float(ys.max() - ys.min()),
            "mutrate": float(np.mean(mrs)),
            "scores": [round(float(v), 4) for v in ys],
        }
    return per_log


# --------------------------------------------------------------------------- #
# genval: does the shadow log hit held-out real variants?                     #
# --------------------------------------------------------------------------- #
from pm4py.objects.log.obj import EventLog
from collections import Counter, defaultdict


def _names(trace):
    return tuple(e["concept:name"] for e in trace)


def _partitions(variants, k, shuffles):
    """R1-accept's fold partitions: seed once, cumulative shuffles."""
    rng = random.Random(SEED)
    order = list(variants)
    n = len(order)
    out = []
    for _ in range(shuffles):
        rng.shuffle(order)
        fold = max(1, n // k)
        for f in range(k):
            start, end = f * fold, ((f + 1) * fold if f < k - 1 else n)
            out.append((order, set(order[start:end])))
    return out


def eval_genval(cfg, dslist, logcache, k=5, shuffles=1):
    """Share of generated traces that exactly match a HELD-OUT real variant.
    No Petri net, no R1: this measures the generator, not the agreement."""
    d = {**DEFAULT, **cfg}
    per_log = {}
    for dk in dslist:
        log = logcache[dk]
        vmap = defaultdict(list)
        for t in log:
            vmap[_names(t)].append(t)
        variants = list(vmap.keys())
        hits, covs, mrs = [], [], []
        for order, held in _partitions(variants, k, shuffles):
            train_vs = [v for v in order if v not in held]
            train_log = EventLog([t for v in train_vs for t in vmap[v]])
            set_knobs(d)
            random.seed(SEED); np.random.seed(SEED)
            shadow, flags, _dp, _tr, _cp = NS["generate_shadow_log"](
                train_log, num_traces=d["num_traces"],
                safe_threshold=d["safe_threshold"], max_n=d["max_n"],
                successor_weighting=d["weighting"])
            seqs = [_names(t) for t in shadow]
            hit = [s for s in seqs if s in held]
            hits.append(len(hit) / len(seqs) if seqs else 0.0)
            covs.append(len(set(hit)) / len(held) if held else 0.0)
            mrs.append(sum(1 for f in flags if f) / len(flags) if flags else 0.0)
        per_log[dk] = {
            "hit_rate": float(np.mean(hits)) * 100.0,
            "coverage": float(np.mean(covs)) * 100.0,
            "mutrate": float(np.mean(mrs)),
        }
    return per_log


# --------------------------------------------------------------------------- #
# caches, grids, entry point                                                   #
# --------------------------------------------------------------------------- #
def load_log(dk):
    return pm4py.convert_to_event_log(pm4py.read_xes(DATASETS[dk]["log_path"]))


def build_cache(dslist, need_nets=True):
    cache = {}
    for dk in dslist:
        t0 = time.time()
        log = load_log(dk)
        if need_nets:
            nets = {m: MINERS[m](log) for m in REAL}
            cache[dk] = (log, nets)
        else:
            cache[dk] = log
        print(f"[cache] {dk} {DATASETS[dk]['name']} ready in {time.time()-t0:.0f}s", flush=True)
    return cache


def grid(name):
    """Config lists. 'alpha' is the headline: alpha=0 disables novelty entirely."""
    base = []
    if name == "alpha":
        for a in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
            base.append(dict(alpha=a))
    elif name == "N":
        for n in [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]:
            base.append(dict(max_n=n))
    elif name == "tau":
        for t in [1, 2, 3, 5, 8, 10, 15, 20]:
            base.append(dict(safe_threshold=t))
    elif name == "temp":
        base.append(dict(weighting="mle"))
        base.append(dict(weighting="log"))
        for T in [0.5, 0.75, 1.5, 2.0, 3.0, 5.0, 999]:
            base.append(dict(weighting="temp", temp=T))
    elif name == "cap":
        for c in [1.0, 1.5, 2.0, 3.0, 5.0]:
            base.append(dict(cap_mult=c))
    elif name == "theta":
        for th in [250, 500, 1000, 2000, 4000]:
            base.append(dict(num_traces=th))
    elif name == "misc":
        base.append(dict(mut_uniform=True))
        for c in [0.1, 0.25, 0.5, 0.75]:
            base.append(dict(pu_clamp=c))
    elif name == "coord":      # everything one-factor-at-a-time around the default
        seen = set()
        for g in ("alpha", "N", "tau", "temp", "cap", "theta", "misc"):
            for c in grid(g):
                key = label(c)
                if key not in seen:
                    seen.add(key); base.append(c)
    elif name == "joint":      # interactions around the promising region
        for n in [4, 6, 8, 10]:
            for a in [0.0, 0.5, 1.0, 2.0]:
                for t in [2, 5, 10]:
                    base.append(dict(max_n=n, alpha=a, safe_threshold=t))
    else:
        raise SystemExit(f"unknown grid: {name}")
    if not any(label(c) == label({}) for c in base):
        base.insert(0, {})     # always carry the shipped default as the baseline
    return base


def selftest():
    """At default knobs the injected copy must BE the frozen generator.

    We compare against the real module (not against the committed matrix): the
    benchmark scored PNMLs from a model cache that no longer exists locally, so
    an end-to-end number also carries discovery/replay tie-breaking drift (the
    threat the report documents). That drift is a property of the nets, not of
    the generator, and it cancels in this study because every config is scored
    on the same re-discovered nets. What must hold exactly is that the
    injection is behaviour-preserving at default knobs.
    """
    from HybridGen.algorithm.v26 import calculate_gen_shadow_stable as frozen
    dk = "D1"
    log, nets = build_cache([dk], need_nets=True)[dk]
    ok = True
    for m in ("Inductive_Strict", "Heuristics", "Alpha"):
        net, im, fm = nets[m]
        got, _mr = score_cell(log, net, im, fm, {})           # injected, default knobs
        random.seed(SEED); np.random.seed(SEED)
        want = frozen(log, net, im, fm, 1000, iterations=1,
                      safe_threshold=5, max_n=6, successor_weighting="mle")["mean"]
        same = abs(got - want) < 1e-12
        ok &= same
        print(f"  {m:22} injected={got:.9f}  frozen={want:.9f}  "
              f"{'IDENTICAL' if same else 'DIVERGED'}")
    print("SELFTEST", "PASS: injection is behaviour-preserving at default knobs"
          if ok else "FAIL: the injected copy is not the frozen algorithm")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("objective", choices=["selftest", "mae", "genval"])
    ap.add_argument("--grid", default="coord")
    ap.add_argument("--logs", default="D1,D2,D3,D4,D5")
    ap.add_argument("--shuffles", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.objective == "selftest":
        raise SystemExit(selftest())

    dslist = a.logs.split(",")
    cfgs = grid(a.grid)
    print(f"[{a.objective}] grid={a.grid}  {len(cfgs)} configs x {len(dslist)} logs", flush=True)
    cache = build_cache(dslist, need_nets=(a.objective == "mae"))

    results = []
    for i, cfg in enumerate(cfgs, 1):
        t0 = time.time()
        if a.objective == "mae":
            per = eval_mae(cfg, dslist, cache)
            agg = float(np.mean([v["mae"] for v in per.values()]))
            extra = f"meanMAE={agg:.4f}"
        else:
            per = eval_genval(cfg, dslist, cache, shuffles=a.shuffles)
            agg = float(np.mean([v["hit_rate"] for v in per.values()]))
            extra = f"meanHit={agg:.2f}%"
        results.append({"cfg": {**DEFAULT, **cfg}, "label": label(cfg),
                        "per_log": per, "agg": agg})
        print(f"[{i}/{len(cfgs)}] {label(cfg):28} {extra}  ({time.time()-t0:.0f}s)", flush=True)
        out = a.out or os.path.join(OUT, f"tune_{a.objective}_{a.grid}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"objective": a.objective, "grid": a.grid, "logs": dslist,
                       "results": results}, f, indent=1)
    print("written:", out)


if __name__ == "__main__":
    main()
