"""Exp5 -- proper M3: entropic relevance from a Petri-net SDFA.

Replaces the report's DFG approximation with the net's ACTUAL reachable
behavior. Pipeline per miner:
  1. reachability graph over markings (manual firing, silent transitions kept)
  2. eps-close the silent (label=None) transitions + determinize over visible
     labels (subset construction) -> a DFA whose states are sets of markings
  3. transition probabilities from log-replay MLE (per-state firing frequency;
     the residual 1-sum is that state's termination probability)
  4. emit SDFA JSON {"initialState":0,"transitions":[{from,to,label,prob}]}
then Entropia `-r -rel <log.xes> -ret <sdfa> -s` computes entropic relevance.

Writes SDFAs to results/sdfa/ + the plain log; the -r calls run where Java is.
Usage (repo root): PYTHONHASHSEED=0 python benchmark/pn_to_sdfa.py D1
"""
import os, sys, json, argparse
from collections import defaultdict, Counter
import pm4py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import DATASETS
from miners import MINERS, filtered_trace_miner, flower_miner

from pm4py.objects.petri_net import semantics as _sem


def _enabled(net, m):
    return _sem.enabled_transitions(net, m)


def _fire(t, net, m):
    return _sem.execute(t, net, m)


def freeze(m):
    return frozenset((p, c) for p, c in m.items() if c > 0)


def reachability(net, im, cap=20000):
    """Manual reachability: marking -> [(label, next_marking)]; label None = silent."""
    start = freeze(im)
    edges = defaultdict(list)
    seen = {start}
    stack = [im]
    while stack:
        m = stack.pop()
        fm = freeze(m)
        if fm in edges:
            continue
        for t in _enabled(net, m):
            m2 = _fire(t, net, m)
            if m2 is None:
                continue
            f2 = freeze(m2)
            edges[fm].append((t.label, f2))
            if f2 not in seen:
                seen.add(f2)
                stack.append(m2)
                if len(seen) > cap:
                    raise ValueError(f"reachability exceeds {cap} markings (unbounded?)")
    return start, edges


def eps_closure(markings, edges):
    clo = set(markings)
    stack = list(markings)
    while stack:
        m = stack.pop()
        for lbl, m2 in edges.get(m, ()):
            if lbl is None and m2 not in clo:
                clo.add(m2)
                stack.append(m2)
    return frozenset(clo)


def determinize(start, edges):
    s0 = eps_closure({start}, edges)
    ids = {s0: 0}
    order = [s0]
    dtrans = {}
    i = 0
    while i < len(order):
        S = order[i]
        sid = ids[S]
        by_lbl = defaultdict(set)
        for m in S:
            for lbl, m2 in edges.get(m, ()):
                if lbl is not None:
                    by_lbl[lbl].add(m2)
        for lbl, tgts in by_lbl.items():
            T = eps_closure(tgts, edges)
            if T not in ids:
                ids[T] = len(order)
                order.append(T)
            dtrans[(sid, lbl)] = ids[T]
        i += 1
    return dtrans, len(order)


def build_sdfa(net, im, fm, log, cap=20000):
    start, edges = reachability(net, im, cap)
    dtrans, n_states = determinize(start, edges)
    cnt = defaultdict(Counter)
    fits = 0
    for trace in log:
        s = 0
        ok = True
        for e in trace:
            a = e["concept:name"]
            nxt = dtrans.get((s, a))
            if nxt is None:
                ok = False
                break
            cnt[s][a] += 1
            s = nxt
        if ok:
            cnt[s]["__STOP__"] += 1
            fits += 1
    trans = []
    for s in sorted(cnt):
        tot = sum(cnt[s].values())
        if tot == 0:
            continue
        for a, c in cnt[s].items():
            if a == "__STOP__":
                continue
            trans.append({"from": s, "to": dtrans[(s, a)], "label": a, "prob": c / tot})
    stats = {"markings": len(edges), "dfa_states": n_states,
             "sdfa_states_used": len(cnt), "transitions": len(trans),
             "log_fit_frac": round(fits / max(len(log), 1), 4)}
    return {"initialState": 0, "transitions": trans}, stats


def miner_fns():
    fns = {m: MINERS[m] for m in MINERS if m in
           ("Alpha", "Alpha+", "Heuristics", "Heuristics_Strict",
            "Inductive_Infrequent", "Inductive_Strict")}
    fns["Trace_Filtered"] = filtered_trace_miner
    fns["Flower"] = flower_miner
    return fns


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", default="D1")
    ap.add_argument("--cap", type=int, default=20000)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    ds = DATASETS[args.dataset]
    outdir = args.outdir or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "results", "sdfa")
    os.makedirs(outdir, exist_ok=True)
    log = pm4py.convert_to_event_log(pm4py.read_xes(ds["log_path"]))
    logpath = os.path.join(outdir, f"{args.dataset}.xes")
    pm4py.write_xes(log, logpath)
    print(f"[{args.dataset} {ds['name']}] log -> {logpath} ({len(log)} cases)", flush=True)
    manifest = {}
    for name, fn in miner_fns().items():
        try:
            net, im, fm = fn(log)
            sdfa, stats = build_sdfa(net, im, fm, log, args.cap)
            p = os.path.join(outdir, f"{args.dataset}__{name}.sdfa")
            with open(p, "w") as f:
                json.dump(sdfa, f)
            manifest[name] = {"sdfa": os.path.basename(p), **stats}
            print(f"  {name:22s} states={stats['sdfa_states_used']:>4} trans={stats['transitions']:>4} "
                  f"fit={stats['log_fit_frac']}", flush=True)
        except Exception as e:
            manifest[name] = {"error": repr(e)}
            print(f"  {name:22s} FAIL {e!r}", flush=True)
    with open(os.path.join(outdir, f"{args.dataset}_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"-> {outdir}/{args.dataset}_manifest.json", flush=True)
