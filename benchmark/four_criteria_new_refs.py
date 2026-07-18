"""Score the two new references (temporal hold-out, synthetic known systems) under
the benchmark's own four-criterion protocol: Pearson (calibration), Spearman +
Kendall tau_b (ranking), MAE (absolute error), spread (resolution of the six
miners, metric vs ground truth), with the construct poles reported separately as
litmus values. Same six non-degenerate miners as the matrix.

Reads results/temporal_holdout.json and results/synth_ground_truth.json.
Writes results/four_criteria_new_refs.json and prints the tables.
"""
import os, json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
REAL = ["Alpha", "Alpha+", "Heuristics", "Heuristics_Strict",
        "Inductive_Strict", "Inductive_Infrequent"]
LOGS = ["D1", "D2", "D3", "D4", "D5"]
LAB = {"D1": "L1", "D2": "L2", "D3": "L3", "D4": "L4", "D5": "L5"}
SPREAD_MIN = 0.05


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    rank = lambda v: np.argsort(np.argsort(np.asarray(v, float))).astype(float)
    return pearson(rank(x), rank(y))


def kendall_tau_b(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    C = D = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif dx * dy > 0:
                C += 1
            else:
                D += 1
    den = np.sqrt((C + D + tx) * (C + D + ty))
    return None if den == 0 else float((C - D) / den)


def mae(x, y):
    return float(np.mean(np.abs(np.asarray(x, float) - np.asarray(y, float))))


def spread(v):
    return float(max(v) - min(v))


def crit(metric_vals, gt_vals):
    return {"pearson": pearson(metric_vals, gt_vals),
            "spearman": spearman(metric_vals, gt_vals),
            "kendall_tau_b": kendall_tau_b(metric_vals, gt_vals),
            "mae": mae(metric_vals, gt_vals),
            "spread_metric": spread(metric_vals),
            "spread_gt": spread(gt_vals),
            "n": len(metric_vals)}


out = {"protocol": "four criteria over the six non-degenerate miners; poles separate"}

# ---------- temporal ----------
t = json.load(open(os.path.join(_HERE, "results", "temporal_holdout.json")))
out["temporal"] = {}
print("=== TEMPORAL HOLD-OUT, four criteria per log (six miners) ===")
print(f"{'log':4s} {'metric':9s} {'Pearson':>8s} {'Spearman':>9s} {'tau_b':>7s} "
      f"{'MAE':>7s} {'spread':>7s} {'GT-spread':>9s}")
for dk in LOGS:
    m = t[dk]["models"]
    gt = [m[n]["holdout_fit"] for n in REAL]
    rec = {}
    for key, name in [("shadowgen", "ShadowGen"), ("m2", "M2")]:
        vals, gts = [], []
        for n in REAL:
            if m[n].get("error") is None and m[n].get(key) is not None:
                vals.append(m[n][key]); gts.append(m[n]["holdout_fit"])
        c = crit(vals, gts)
        rec[key] = c
        print(f"{LAB[dk]:4s} {name:9s} {c['pearson']:8.3f} {c['spearman']:9.3f} "
              f"{c['kendall_tau_b']:7.3f} {c['mae']:7.3f} {c['spread_metric']:7.3f} "
              f"{c['spread_gt']:9.3f}")
    rec["poles"] = {p: {"metric_sg": t[dk]["models"][p].get("shadowgen"),
                        "gt": t[dk]["models"][p].get("holdout_fit")}
                    for p in ("Trace_Filtered", "Flower")}
    out["temporal"][dk] = rec

# ---------- synthetic ----------
s = json.load(open(os.path.join(_HERE, "results", "synth_ground_truth.json")))
per = {"shadowgen": {k: [] for k in ("pearson", "spearman", "kendall_tau_b",
                                     "mae", "spread_metric", "spread_gt")},
       "m2": {k: [] for k in ("pearson", "spearman", "kendall_tau_b",
                              "mae", "spread_metric", "spread_gt")}}
fl_recall, fl_sg, tr_recall, tr_sg = [], [], [], []
n_disc = 0
for sys_ in s["systems"]:
    m = sys_["models"]
    gt6 = [m[n]["true_recall"] for n in REAL
           if m[n].get("error") is None and m[n].get("true_recall") is not None]
    if len(gt6) < 3 or np.std(gt6) < SPREAD_MIN:
        continue
    n_disc += 1
    for key in ("shadowgen", "m2"):
        vals, gts = [], []
        for n in REAL:
            if m[n].get("error") is None and m[n].get(key) is not None \
                    and m[n].get("true_recall") is not None:
                vals.append(m[n][key]); gts.append(m[n]["true_recall"])
        if len(vals) < 3:
            continue
        c = crit(vals, gts)
        for k in per[key]:
            if c[k] is not None:
                per[key][k].append(c[k])
    if m["Flower"].get("error") is None:
        fl_recall.append(m["Flower"]["true_recall"]); fl_sg.append(m["Flower"]["shadowgen"])
    if m["Trace_Filtered"].get("error") is None:
        tr_recall.append(m["Trace_Filtered"]["true_recall"]); tr_sg.append(m["Trace_Filtered"]["shadowgen"])

out["synthetic"] = {"n_discriminating": n_disc, "spread_min": SPREAD_MIN, "median_per_system": {}}
print(f"\n=== SYNTHETIC KNOWN SYSTEMS, per-system medians over {n_disc} discriminating systems ===")
print(f"{'metric':9s} {'Pearson':>8s} {'Spearman':>9s} {'tau_b':>7s} "
      f"{'MAE':>7s} {'spread':>7s} {'GT-spread':>9s}")
for key, name in [("shadowgen", "ShadowGen"), ("m2", "M2")]:
    med = {k: float(np.median(v)) for k, v in per[key].items() if v}
    out["synthetic"]["median_per_system"][key] = med
    print(f"{name:9s} {med['pearson']:8.3f} {med['spearman']:9.3f} "
          f"{med['kendall_tau_b']:7.3f} {med['mae']:7.3f} {med['spread_metric']:7.3f} "
          f"{med['spread_gt']:9.3f}")
poles = {"flower": {"true_recall_mean": float(np.mean(fl_recall)),
                    "true_recall_min": float(np.min(fl_recall)),
                    "shadowgen_mean": float(np.mean(fl_sg)),
                    "shadowgen_min": float(np.min(fl_sg))},
         "trace": {"true_recall_mean": float(np.mean(tr_recall)),
                   "shadowgen_mean": float(np.mean(tr_sg))}}
out["synthetic"]["poles"] = poles
print(f"\npoles (all {len(fl_recall)} systems): "
      f"flower true-recall mean={poles['flower']['true_recall_mean']:.4f} "
      f"min={poles['flower']['true_recall_min']:.4f}, ShadowGen mean="
      f"{poles['flower']['shadowgen_mean']:.4f} min={poles['flower']['shadowgen_min']:.4f}")
print(f"                       trace  true-recall mean={poles['trace']['true_recall_mean']:.3f}, "
      f"ShadowGen mean={poles['trace']['shadowgen_mean']:.3f}")

outp = os.path.join(_HERE, "results", "four_criteria_new_refs.json")
json.dump(out, open(outp, "w"), indent=2)
print(f"\n-> {outp}")
