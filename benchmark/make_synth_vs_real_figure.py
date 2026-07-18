"""Supplementary figure: ShadowGen vs a generalization ground truth, on real logs
(left column) next to known synthetic systems (right column, non-circular), with
PM4Py M2 underneath. Every panel annotates the correlation and the MAE.

LEFT column source, via env LEFT:
  LEFT=temporal (default) -> real ground truth = temporal future-fitness (Exp2)
  LEFT=r1                 -> real ground truth = cross-validation R1

Reads configs/* (R1), results/temporal_holdout.json, results/synth_ground_truth.json.
Writes report/figures/fig_synth_vs_real[_temporal].pdf and a PNG preview.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
from datasets import DATASETS

CFG = os.path.join(_HERE, "results", "configs")
REAL = ["Alpha", "Alpha+", "Heuristics", "Heuristics_Strict",
        "Inductive_Strict", "Inductive_Infrequent"]
LOGS = ["D1", "D2", "D3", "D4", "D5"]
POLES = ["Flower", "Trace_Filtered"]
SPREAD_MIN = 0.05
BLUE, RED, GREEN, PURPLE = "#2166ac", "#b2182b", "#1a9850", "#762a83"
LEFT = os.environ.get("LEFT", "temporal")


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def mae(x, y):
    return float(np.mean(np.abs(np.asarray(x, float) - np.asarray(y, float))))


def cfg_val(dsname, miner, method):
    p = os.path.join(CFG, f"{dsname}__{miner}__{method}.json")
    if not os.path.exists(p):
        return None
    res = json.load(open(p))["results"]
    v = res.get("mean")
    if v is None:
        v = res.get("score")
    if v is None or (isinstance(v, (int, float)) and v <= -1):
        return None
    return float(v)


def real_r1(metric):
    x, y = [], []
    poles = {p: ([], []) for p in POLES}
    for dk in LOGS:
        name = DATASETS[dk]["name"]
        for m in REAL:
            mv, gt = cfg_val(name, m, metric), cfg_val(name, m, "R1")
            if mv is not None and gt is not None:
                x.append(gt); y.append(mv)
        for p in POLES:
            mv, gt = cfg_val(name, p, metric), cfg_val(name, p, "R1")
            if mv is not None and gt is not None:
                poles[p][0].append(gt); poles[p][1].append(mv)
    return np.array(x), np.array(y), poles


def real_temporal(metric):
    d = json.load(open(os.path.join(_HERE, "results", "temporal_holdout.json")))
    key = "shadowgen" if metric == "M1g" else "m2"
    x, y = [], []
    poles = {p: ([], []) for p in POLES}
    for dk in LOGS:
        mm = d[dk]["models"]
        for m in REAL:
            if mm[m].get("error") is None and mm[m].get(key) is not None:
                x.append(mm[m]["holdout_fit"]); y.append(mm[m][key])
        for p in POLES:
            if mm[p].get("error") is None and mm[p].get(key) is not None:
                poles[p][0].append(mm[p]["holdout_fit"]); poles[p][1].append(mm[p][key])
    return np.array(x), np.array(y), poles


def synth(metric):
    d = json.load(open(os.path.join(_HERE, "results", "synth_ground_truth.json")))
    key = "shadowgen" if metric == "M1g" else "m2"
    x, y, per_sys = [], [], []
    poles = {p: ([], []) for p in POLES}
    for s in d["systems"]:
        rr = [s["models"][m]["true_recall"] for m in REAL
              if s["models"][m].get("error") is None and s["models"][m].get("true_recall") is not None]
        if len(rr) < 3 or np.std(rr) < SPREAD_MIN:
            continue
        sx, sy = [], []
        for m in REAL:
            mm = s["models"][m]
            if mm.get("error") is None and mm.get(key) is not None and mm.get("true_recall") is not None:
                x.append(mm["true_recall"]); y.append(mm[key])
                sx.append(mm[key]); sy.append(mm["true_recall"])
        r = pearson(sx, sy)
        if r is not None:
            per_sys.append(r)
        for p in POLES:
            mm = s["models"][p]
            if mm.get("error") is None and mm.get(key) is not None:
                poles[p][0].append(mm["true_recall"]); poles[p][1].append(mm[key])
    return np.array(x), np.array(y), poles, float(np.median(per_sys))


def draw(a, x, y, poles, color, note):
    a.scatter(x, y, s=16, alpha=0.4, color=color, edgecolors="none", zorder=2)
    a.scatter(poles["Flower"][0], poles["Flower"][1], s=55, marker="*", color=GREEN,
              edgecolors="k", linewidths=0.4, zorder=4)
    a.scatter(poles["Trace_Filtered"][0], poles["Trace_Filtered"][1], s=24, marker="s",
              color=PURPLE, edgecolors="k", linewidths=0.4, zorder=4)
    a.plot([0, 1], [0, 1], "--", color="0.45", lw=1, zorder=1)
    a.set_xlim(-0.02, 1.02); a.set_ylim(-0.02, 1.02)
    a.text(0.05, 0.95, note, transform=a.transAxes, fontsize=9.5, va="top",
           bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))


real_fn = real_temporal if LEFT == "temporal" else real_r1
left_gt = "temporal future fitness (real logs)" if LEFT == "temporal" else "cross-validation R1 (real logs)"
left_gt_short = "future-case fitness" if LEFT == "temporal" else "cross-validation R1"

fig, ax = plt.subplots(2, 2, figsize=(9.2, 8.8), sharex=True, sharey=True)
stats = {}

# top-left: ShadowGen real
x, y, p = real_fn("M1g"); r, m = pearson(y, x), mae(x, y); stats["SG-real"] = (r, m)
draw(ax[0, 0], x, y, p, BLUE, f"pooled $r$ = {r:.2f}\nMAE = {m:.03f}")
# top-right: ShadowGen synthetic
x, y, p, med = synth("M1g"); m = mae(x, y); stats["SG-synth"] = (med, m)
draw(ax[0, 1], x, y, p, BLUE, f"median $r$ = {med:.2f}\nMAE = {m:.03f}")
# bottom-left: M2 real
x, y, p = real_fn("M2"); r, m = pearson(y, x), mae(x, y); stats["M2-real"] = (r, m)
draw(ax[1, 0], x, y, p, RED, f"pooled $r$ = {r:.2f}\nMAE = {m:.03f}")
# bottom-right: M2 synthetic
x, y, p, med = synth("M2"); m = mae(x, y); stats["M2-synth"] = (med, m)
draw(ax[1, 1], x, y, p, RED, f"median $r$ = {med:.2f}\nMAE = {m:.03f}")

ax[0, 0].set_title(f"Real logs\nground truth: {left_gt_short} (log-derived)", fontsize=10.5)
ax[0, 1].set_title("Known synthetic systems\nground truth: true system recall (non-circular)", fontsize=10.5)
ax[0, 0].set_ylabel("ShadowGen score", fontsize=11)
ax[1, 0].set_ylabel("PM4Py M2 score", fontsize=11)
for a in ax[1]:
    a.set_xlabel("generalization ground truth", fontsize=10.5)
handles = [Line2D([0], [0], marker="*", color="w", markerfacecolor=GREEN, markeredgecolor="k", ms=12, label="flower pole"),
           Line2D([0], [0], marker="s", color="w", markerfacecolor=PURPLE, markeredgecolor="k", ms=8, label="trace pole")]
ax[0, 1].legend(handles=handles, loc="lower right", fontsize=8.5, framealpha=0.9)
fig.suptitle("ShadowGen tracks the ground truth on real logs and on known systems; PM4Py does neither\n"
             "(diagonal = perfect tracking; right column removes the log-circularity)", fontsize=11, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.96])
suffix = "_temporal" if LEFT == "temporal" else ""
outdir = os.path.join(_REPO, "report", "figures")
fig.savefig(os.path.join(outdir, f"fig_synth_vs_real{suffix}.pdf"), bbox_inches="tight")
png = os.path.join(_HERE, "results", f"fig_synth_vs_real{suffix}.png")
fig.savefig(png, dpi=130, bbox_inches="tight")
print("LEFT =", LEFT)
for k, (c, m) in stats.items():
    print(f"  {k:10s} corr={c:.3f}  MAE={m:.4f}")
print("wrote", png)
