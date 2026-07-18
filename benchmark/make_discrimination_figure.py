"""Resolving-power figure (2x2).

Columns: ShadowGen | PM4Py M2.
Row 1 (marginal): frequency distribution of the scores each metric assigns to all
  discovered models. ShadowGen uses the whole range; PM4Py piles almost everything
  near the top and crashes only for the degenerate trace model (effectively yes/no).
Row 2 (conditional): the same scores over the six real miners, split by whether the
  model is actually a poor or good generalizer (true recall below/above median). A
  discriminative metric separates the two; a saturated one overlaps them (AUC).

This visualizes the report's "spread" criterion tied to the ground truth. Uses the
synthetic systems (known recall, many cells); prints real-log AUC/spread as a check.

Reads results/synth_ground_truth.json (+ configs/*). Writes
report/figures/fig_discrimination.pdf and a PNG preview.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
from datasets import DATASETS

REAL = ["Alpha", "Alpha+", "Heuristics", "Heuristics_Strict",
        "Inductive_Strict", "Inductive_Infrequent"]
ALL8 = REAL + ["Trace_Filtered", "Flower"]
CFG = os.path.join(_HERE, "results", "configs")
LOGS = ["D1", "D2", "D3", "D4", "D5"]
POOR, GOOD, SGC, M2C = "#d6604d", "#4393c3", "#2166ac", "#b2182b"


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if not len(pos) or not len(neg):
        return None
    allv = np.concatenate([pos, neg]); order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


d = json.load(open(os.path.join(_HERE, "results", "synth_ground_truth.json")))
# marginal: all 8 models
mg_sg = [s["models"][m]["shadowgen"] for s in d["systems"] for m in ALL8
         if s["models"][m].get("shadowgen") is not None]
mg_m2 = [s["models"][m]["m2"] for s in d["systems"] for m in ALL8
         if s["models"][m].get("m2") is not None]
# conditional: 6 miners with true recall
sg, m2, tr = [], [], []
for s in d["systems"]:
    for m in REAL:
        c = s["models"][m]
        if c.get("error") is None and c.get("true_recall") is not None and c.get("shadowgen") is not None:
            sg.append(c["shadowgen"]); tr.append(c["true_recall"])
            m2.append(c["m2"] if c.get("m2") is not None else np.nan)
sg, m2, tr = np.array(sg), np.array(m2), np.array(tr)
med = np.median(tr); poor, good = tr <= med, tr > med
bins = np.linspace(0, 1, 22)

fig, ax = plt.subplots(2, 2, figsize=(11, 8.4), sharex=True)
# Row 1: marginal
ax[0, 0].hist(mg_sg, bins=bins, density=True, color=SGC, alpha=0.85)
ax[0, 0].set_title("ShadowGen", fontsize=12)
ax[0, 1].hist(mg_m2, bins=bins, density=True, color=M2C, alpha=0.85)
ax[0, 1].set_title("PM4Py M2", fontsize=12)
ax[0, 0].set_ylabel("all models\nfrequency density", fontsize=10.5)
# Row 2: conditional
for col, scores, name in [(0, sg, "ShadowGen"), (1, m2, "PM4Py M2")]:
    a = ax[1, col]; ok = ~np.isnan(scores)
    a.hist(scores[poor & ok], bins=bins, density=True, alpha=0.6, color=POOR,
           label="poor generalizers")
    a.hist(scores[good & ok], bins=bins, density=True, alpha=0.6, color=GOOD,
           label="good generalizers")
    au = auc(scores[good & ok], scores[poor & ok])
    a.text(0.04, 0.95, f"separation\nAUC = {au:.2f}", transform=a.transAxes, va="top",
           fontsize=10, bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    a.set_xlabel("metric score")
ax[1, 0].set_ylabel("six miners, split by\ntrue generalization", fontsize=10.5)
ax[1, 0].legend(loc="upper center", fontsize=8.5, framealpha=0.9)
for a in ax.flat:
    a.set_xlim(0, 1)
fig.suptitle("Resolving power: ShadowGen spreads scores across the range and separates "
             "good from poor\ngeneralizers; PM4Py saturates near the top and overlaps them",
             fontsize=11.5, y=1.0)
fig.tight_layout()
outdir = os.path.join(_REPO, "report", "figures")
fig.savefig(os.path.join(outdir, "fig_discrimination.pdf"), bbox_inches="tight")
png = os.path.join(_HERE, "results", "fig_discrimination.png")
fig.savefig(png, dpi=130, bbox_inches="tight")


def cfg_val(name, m, meth):
    p = os.path.join(CFG, f"{name}__{m}__{meth}.json")
    if not os.path.exists(p):
        return None
    r = json.load(open(p))["results"]; v = r.get("mean")
    if v is None:
        v = r.get("score")
    return None if (v is None or (isinstance(v, (int, float)) and v <= -1)) else float(v)


rsg, rm2, rgt = [], [], []
for dk in LOGS:
    name = DATASETS[dk]["name"]
    for m in REAL:
        a, b, g = cfg_val(name, m, "M1g"), cfg_val(name, m, "M2"), cfg_val(name, m, "R1")
        if a is not None and g is not None:
            rsg.append(a); rgt.append(g); rm2.append(b if b is not None else np.nan)
rsg, rm2, rgt = np.array(rsg), np.array(rm2), np.array(rgt)
rmed = np.median(rgt); rp, rg = rgt <= rmed, rgt > rmed
print("synthetic AUC  ShadowGen=%.3f  M2=%.3f" % (auc(sg[good], sg[poor]), auc(m2[good], m2[poor])))
print("real-log  AUC  ShadowGen=%.3f  M2=%.3f" % (auc(rsg[rg], rsg[rp]), auc(rm2[rg], rm2[rp])))
print("wrote", png)
