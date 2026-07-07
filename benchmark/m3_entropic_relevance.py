"""Independent M3: entropic relevance in pure Python, no Java toolchain.

Replaces the original two-jar pipeline (JDFG2Aut + Relevance, both since lost)
with a self-contained implementation, validated in two directions:
  * the ER computation reproduces the closed-source Entropia tool (-r flag) to
    3 decimals on every Exp5 D1 SDFA (Flower 52.268, Inductive_Strict 21.909,
    Alpha+ 35.149, Inductive_Infrequent 33.349, Heuristics_Strict 61.761,
    Trace_Filtered 57.068);
  * the DFG branch reproduces the committed M3 configs where the automaton
    construction coincides (e.g. D1 Alpha 63.313 = committed 63.31285); exact
    agreement on all 40 cells is not expected, since the committed numbers
    used the lost JDFG2Aut converter, a different DFG-to-automaton mapping.

Two model-to-SDFA conversions, mirroring the report (Sect. 5.3):
  * proper  -- reachability graph, eps-closure over silents, determinize,
               probabilities from log replay. Exact but bounded: builds only
               where the net's reachability is finite (23 of 40 cells;
               D1 6/8, D2 8/8, D3 4/8, D4 3/8, D5 2/8 at a 150k-marking cap).
  * dfg     -- first-order directly-follows approximation from play_out
               simulation. Lossy (no concurrency, one-step memory) but always
               builds; this is the at-scale fallback the report discloses.

ER formula (Alkhammash et al.): fitting trace: -log2(rho) - log2(P(trace)),
with P including per-state termination residual; non-fitting trace:
-log2(1-rho) + (|sigma|+1) * log2(|A|+1); weighted mean over the log.

Usage (repo root):
  PYTHONHASHSEED=0 python benchmark/m3_entropic_relevance.py D1 --mode both
Results: benchmark/results/m3_rebuild.json (all five logs, 2026-07-07, cibox).
"""
import os, sys, json, time, math, argparse
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pm4py
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.petri_net import semantics as _sem
from datasets import DATASETS


def _freeze(m):
    return frozenset((p, c) for p, c in m.items() if c > 0)


def reachability(net, im, cap=150000):
    """Marking graph via manual firing; raises if the state space exceeds cap
    (unbounded nets always do)."""
    start = _freeze(im)
    edges = defaultdict(list)
    seen = {start}
    stack = [im]
    while stack:
        m = stack.pop()
        fm = _freeze(m)
        if fm in edges:
            continue
        for t in _sem.enabled_transitions(net, m):
            m2 = _sem.execute(t, net, m)
            if m2 is None:
                continue
            f2 = _freeze(m2)
            edges[fm].append((t.label, f2))
            if f2 not in seen:
                seen.add(f2)
                stack.append(m2)
                if len(seen) > cap:
                    raise ValueError(f"exceeds {cap} markings (unbounded or state-exploding)")
    return start, edges


def _eps_closure(ms, edges):
    clo = set(ms)
    stack = list(ms)
    while stack:
        m = stack.pop()
        for lbl, m2 in edges.get(m, ()):
            if lbl is None and m2 not in clo:
                clo.add(m2)
                stack.append(m2)
    return frozenset(clo)


def determinize(start, edges, dcap=150000):
    s0 = _eps_closure({start}, edges)
    ids = {s0: 0}
    order = [s0]
    dtrans = {}
    i = 0
    while i < len(order):
        S = order[i]
        sid = ids[S]
        by = defaultdict(set)
        for m in S:
            for lbl, m2 in edges.get(m, ()):
                if lbl is not None:
                    by[lbl].add(m2)
        for lbl, tgts in by.items():
            T = _eps_closure(tgts, edges)
            if T not in ids:
                ids[T] = len(order)
                order.append(T)
                if len(order) > dcap:
                    raise ValueError(f"determinize exceeds {dcap} states")
            dtrans[(sid, lbl)] = ids[T]
        i += 1
    return dtrans


def sdfa_proper(net, im, fm, variants, cap=150000):
    """Reachability SDFA with log-replay MLE probabilities (variant-weighted)."""
    start, edges = reachability(net, im, cap)
    dtrans = determinize(start, edges)
    cnt = defaultdict(Counter)
    for v, c in variants.items():
        s, ok = 0, True
        for a in v:
            nxt = dtrans.get((s, a))
            if nxt is None:
                ok = False
                break
            cnt[s][a] += c
            s = nxt
        if ok:
            cnt[s]["__STOP__"] += c
    trans = []
    for s in cnt:
        tot = sum(cnt[s].values())
        for a, c in cnt[s].items():
            if a != "__STOP__":
                trans.append({"from": s, "to": dtrans[(s, a)], "label": a, "prob": c / tot})
    return {"initialState": 0, "transitions": trans}


def sdfa_dfg(net, im, fm, n=5000):
    """First-order directly-follows approximation from play_out simulation."""
    sim = pm4py.play_out(net, im, fm, no_traces=n)
    df, starts, ends = Counter(), Counter(), Counter()
    for t in sim:
        acts = [e["concept:name"] for e in t]
        if not acts:
            continue
        starts[acts[0]] += 1
        ends[acts[-1]] += 1
        for a, b in zip(acts, acts[1:]):
            df[(a, b)] += 1
    allacts = sorted(set([a for a, _ in df] + [b for _, b in df] + list(starts) + list(ends)))
    sid = {a: i + 1 for i, a in enumerate(allacts)}
    trans = []
    ts = sum(starts.values()) or 1
    for a, c in starts.items():
        trans.append({"from": 0, "to": sid[a], "label": a, "prob": c / ts})
    for a in allacts:
        out = sum(df.get((a, b), 0) for b in allacts) + ends.get(a, 0)
        if out == 0:
            continue
        for b in allacts:
            c = df.get((a, b), 0)
            if c > 0:
                trans.append({"from": sid[a], "to": sid[b], "label": b, "prob": c / out})
    return {"initialState": 0, "transitions": trans}


def entropic_relevance(sdfa, variants, total, n_activities):
    """Bits per trace to encode the log under the SDFA (Alkhammash et al.)."""
    tm, outs = {}, {}
    for tr in sdfa["transitions"]:
        tm[(tr["from"], tr["label"])] = (tr["to"], tr["prob"])
        outs[tr["from"]] = outs.get(tr["from"], 0.0) + tr["prob"]
    s0 = sdfa["initialState"]
    fit = 0
    parsed = []
    for v, c in variants.items():
        s, ok, lp = s0, True, 0.0
        for a in v:
            k = (s, a)
            if k not in tm:
                ok = False
                break
            nxt, p = tm[k]
            lp += math.log2(p)
            s = nxt
        if ok:
            lp += math.log2(max(1.0 - outs.get(s, 0.0), 1e-12))  # termination residual
            fit += c
        parsed.append((c, ok, lp, len(v)))
    rho = fit / total
    E = 0.0
    for c, ok, lp, n in parsed:
        w = c / total
        if ok:
            E += w * ((-math.log2(rho) if rho > 0 else 0.0) - lp)
        else:
            E += w * ((-math.log2(1 - rho) if rho < 1 else 0.0) + (n + 1) * math.log2(n_activities + 1))
    return E, rho


MINERS8 = ["Alpha", "Alpha+", "Heuristics", "Heuristics_Strict",
           "Inductive_Infrequent", "Inductive_Strict", "Trace_Filtered", "Flower"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", default="D1")
    ap.add_argument("--mode", choices=["proper", "dfg", "both"], default="both")
    ap.add_argument("--cap", type=int, default=150000)
    ap.add_argument("--models", default=None, help="PNML dir (default benchmark/models/<ds>)")
    args = ap.parse_args()
    ds = DATASETS[args.dataset]
    mdir = args.models or os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", args.dataset)
    log = pm4py.convert_to_event_log(pm4py.read_xes(ds["log_path"]))
    variants = Counter(tuple(e["concept:name"] for e in t) for t in log)
    total = sum(variants.values())
    A = len(set(a for v in variants for a in v))
    print(f"[{args.dataset} {ds['name']}] {total} traces, {len(variants)} variants, A={A}", flush=True)
    for mn in MINERS8:
        p = os.path.join(mdir, f"{mn}.pnml")
        if not os.path.exists(p):
            print(f"  {mn:22s} no PNML at {p}")
            continue
        net, im, fm = pnml_importer.apply(p)
        out = []
        if args.mode in ("proper", "both"):
            t = time.time()
            try:
                E, rho = entropic_relevance(sdfa_proper(net, im, fm, variants, args.cap), variants, total, A)
                out.append(f"proper={E:.3f} (rho={rho:.3f}, {time.time()-t:.1f}s)")
            except ValueError as e:
                out.append(f"proper=intractable ({e})")
        if args.mode in ("dfg", "both"):
            t = time.time()
            E, rho = entropic_relevance(sdfa_dfg(net, im, fm), variants, total, A)
            out.append(f"dfg={E:.3f} (rho={rho:.3f}, {time.time()-t:.1f}s)")
        print(f"  {mn:22s} " + "  ".join(out), flush=True)


if __name__ == "__main__":
    main()
