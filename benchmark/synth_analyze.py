"""
Rigorous re-analysis of Exp1 (synthetic-system ground truth).

The raw pooled Pearson mixes systems with different baseline generalisability
(coverage spans 0.17..1.0), which deflates a correlation that is strong WITHIN
each system. This computes the defensible statistics:

  1. per-system Pearson(ShadowGen, true recall) and (M2, true recall), reported
     over "discriminating" systems (recall spread >= SPREAD_MIN over the six
     miners); systems where every miner generalises identically carry no ranking
     signal and only inject noise into a 6-point correlation.
  2. within-system-centred pooled Pearson (fixed-effects style): each system's
     sg/recall demeaned, then pooled. This is the correct global "does the metric
     track recall within systems" number, free of the baseline-mixing artifact.
  3. pole behaviour across systems (the flower/ trace litmus against a KNOWN
     system) and the generalisation-vs-precision separation.

Reads  benchmark/results/synth_ground_truth.json
Writes benchmark/results/synth_analysis.json
"""
import os, sys, json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
REAL = ["Alpha", "Alpha+", "Heuristics", "Heuristics_Strict",
        "Inductive_Strict", "Inductive_Infrequent"]
SPREAD_MIN = 0.05


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def cells(sysrec, metric):
    """(metric, recall) pairs over REAL miners with both present."""
    xs, ys = [], []
    for m in REAL:
        d = sysrec["models"][m]
        if d.get("error") is None and d.get(metric) is not None and d.get("true_recall") is not None:
            xs.append(d[metric]); ys.append(d["true_recall"])
    return xs, ys


def analyze():
    d = json.load(open(os.path.join(_HERE, "results", "synth_ground_truth.json")))
    systems = d["systems"]

    per_sys = {"shadowgen": [], "m2": []}          # per-system pearson (discriminating)
    per_sys_all = {"shadowgen": [], "m2": []}       # per-system pearson (any spread)
    cen = {"shadowgen": ([], []), "m2": ([], [])}   # within-system-centred pools
    pooled = {"shadowgen": ([], []), "m2": ([], [])}
    spreads, n_disc = [], 0
    win = 0; win_den = 0

    for s in systems:
        xs_sg, ys = cells(s, "shadowgen")
        if len(ys) < 3:
            continue
        spread = float(np.std(ys)); spreads.append(spread)
        disc = spread >= SPREAD_MIN
        if disc:
            n_disc += 1
        for metric in ("shadowgen", "m2"):
            xs, yy = cells(s, metric)
            if len(yy) < 3:
                continue
            r = pearson(xs, yy)
            if r is not None:
                per_sys_all[metric].append(r)
                if disc:
                    per_sys[metric].append(r)
            xs = np.asarray(xs, float); yy = np.asarray(yy, float)
            cen[metric][0].extend(list(xs - xs.mean()))
            cen[metric][1].extend(list(yy - yy.mean()))
            pooled[metric][0].extend(list(xs)); pooled[metric][1].extend(list(yy))
        # paired win on discriminating systems
        rs = pearson(*cells(s, "shadowgen")); rm = pearson(*cells(s, "m2"))
        if disc and rs is not None and rm is not None:
            win_den += 1; win += (rs > rm)

    def summ(v):
        v = np.asarray(v, float)
        return None if not len(v) else {
            "median": float(np.median(v)), "mean": float(np.mean(v)),
            "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75)),
            "min": float(np.min(v)), "frac_positive": float(np.mean(v > 0)), "n": int(len(v))}

    # poles across systems (litmus against a known system)
    fl_r = [s["models"]["Flower"]["true_recall"] for s in systems if s["models"]["Flower"].get("error") is None]
    fl_sg = [s["models"]["Flower"]["shadowgen"] for s in systems if s["models"]["Flower"].get("error") is None]
    fl_pr = [s["models"]["Flower"]["true_precision"] for s in systems
             if s["models"]["Flower"].get("error") is None and s["models"]["Flower"].get("true_precision") is not None]
    tr_r = [s["models"]["Trace_Filtered"]["true_recall"] for s in systems if s["models"]["Trace_Filtered"].get("error") is None]
    tr_sg = [s["models"]["Trace_Filtered"]["shadowgen"] for s in systems if s["models"]["Trace_Filtered"].get("error") is None]

    out = {
        "n_systems": len(systems), "n_discriminating": n_disc, "spread_min": SPREAD_MIN,
        "coverage_range": [float(np.min([s["coverage"] for s in systems])),
                           float(np.max([s["coverage"] for s in systems]))],
        "per_system_pearson_discriminating": {k: summ(per_sys[k]) for k in per_sys},
        "per_system_pearson_all": {k: summ(per_sys_all[k]) for k in per_sys_all},
        "within_system_centred_pooled_pearson": {
            "shadowgen": pearson(*cen["shadowgen"]), "m2": pearson(*cen["m2"])},
        "raw_pooled_pearson": {
            "shadowgen": pearson(*pooled["shadowgen"]), "m2": pearson(*pooled["m2"])},
        "paired_shadowgen_beats_m2_frac": (win / win_den) if win_den else None,
        "paired_n": win_den,
        "poles": {
            "flower_recall_mean": float(np.mean(fl_r)), "flower_shadowgen_mean": float(np.mean(fl_sg)),
            "flower_precision_mean": float(np.mean(fl_pr)) if fl_pr else None,
            "trace_recall_mean": float(np.mean(tr_r)), "trace_shadowgen_mean": float(np.mean(tr_sg)),
            "flower_shadowgen_min": float(np.min(fl_sg))},
    }
    json.dump(out, open(os.path.join(_HERE, "results", "synth_analysis.json"), "w"), indent=2)

    ps = out["per_system_pearson_discriminating"]
    print(f"systems={out['n_systems']}  discriminating(spread>={SPREAD_MIN})={n_disc}  "
          f"coverage {out['coverage_range'][0]:.2f}..{out['coverage_range'][1]:.2f}")
    print("\n-- per-system Pearson vs true recall, DISCRIMINATING systems --")
    for k in ("shadowgen", "m2"):
        s = ps[k]
        print(f"  {k:10s} median={s['median']:.3f}  mean={s['mean']:.3f}  "
              f"IQR[{s['q25']:.3f},{s['q75']:.3f}]  frac_pos={s['frac_positive']:.2f}  n={s['n']}")
    print(f"  paired: ShadowGen beats M2 on {out['paired_shadowgen_beats_m2_frac']*100:.0f}% of "
          f"{out['paired_n']} discriminating systems")
    print("\n-- within-system-centred pooled Pearson (baseline-mixing removed) --")
    print(f"  shadowgen={out['within_system_centred_pooled_pearson']['shadowgen']:.3f}  "
          f"m2={out['within_system_centred_pooled_pearson']['m2']:.3f}")
    print(f"  (raw pooled: shadowgen={out['raw_pooled_pearson']['shadowgen']:.3f}  "
          f"m2={out['raw_pooled_pearson']['m2']:.3f})")
    p = out["poles"]
    print("\n-- litmus against known systems (mean over systems) --")
    print(f"  Flower: true_recall={p['flower_recall_mean']:.3f}  ShadowGen={p['flower_shadowgen_mean']:.3f}"
          f"  (min {p['flower_shadowgen_min']:.3f})  true_precision={p['flower_precision_mean']}")
    print(f"  Trace : true_recall={p['trace_recall_mean']:.3f}  ShadowGen={p['trace_shadowgen_mean']:.3f}")
    print("\n-> benchmark/results/synth_analysis.json")


if __name__ == "__main__":
    analyze()
