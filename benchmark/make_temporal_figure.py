"""Figure for Exp2: ShadowGen vs temporal (future) hold-out fitness, vs PM4Py M2.

Reads benchmark/results/temporal_holdout.json, writes
report/figures/fig_temporal_holdout.pdf and a PNG preview.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
REAL = ["Alpha", "Alpha+", "Heuristics", "Heuristics_Strict",
        "Inductive_Strict", "Inductive_Infrequent"]
LABEL = {"D1": "L1 Sepsis", "D2": "L2 BPI2013", "D3": "L3 BPI2017",
         "D4": "L4 BPI2018", "D5": "L5 BPI2019"}
ORDER = ["D1", "D2", "D3", "D4", "D5"]
d = json.load(open(os.path.join(_HERE, "results", "temporal_holdout.json")))
cmap = plt.get_cmap("viridis")
COL = {k: cmap(i / 4) for i, k in enumerate(ORDER)}

fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.3))
GREEN, RED = "#1a9850", "#b2182b"

for a, metric, title in [(ax[0], "shadowgen", "ShadowGen (from past only)"),
                         (ax[1], "m2", "PM4Py M2 (from past only)")]:
    for k in ORDER:
        m = d[k]["models"]
        x = [m[n]["holdout_fit"] for n in REAL if m[n].get("error") is None and m[n].get(metric) is not None]
        y = [m[n][metric] for n in REAL if m[n].get("error") is None and m[n].get(metric) is not None]
        a.scatter(x, y, s=34, color=COL[k], alpha=0.85, edgecolors="k",
                  linewidths=0.3, label=LABEL[k], zorder=3)
    # poles pooled across logs
    for pole, mk, c, lab in [("Flower", "*", GREEN, "flower pole"),
                             ("Trace_Filtered", "s", RED, "trace pole")]:
        px = [d[k]["models"][pole]["holdout_fit"] for k in ORDER
              if d[k]["models"][pole].get(metric) is not None]
        py = [d[k]["models"][pole][metric] for k in ORDER
              if d[k]["models"][pole].get(metric) is not None]
        a.scatter(px, py, s=70 if mk == "*" else 34, marker=mk, color=c,
                  edgecolors="k", linewidths=0.4, zorder=4, label=lab)
    a.plot([0, 1], [0, 1], "--", color="0.4", lw=1, zorder=1)
    a.set_xlim(-0.02, 1.02); a.set_ylim(-0.02, 1.02)
    a.set_xlabel("token-replay fitness of future cases")
    a.set_title(title, fontsize=11)
ax[0].set_ylabel("metric score (from past)")
ax[0].text(0.04, 0.95, "Pearson 0.96-1.00\nSpearman 1.0 (4/5)", transform=ax[0].transAxes,
           fontsize=9.5, va="top", bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
ax[1].text(0.04, 0.30, "Pearson -0.66 to 0.26", transform=ax[1].transAxes,
           fontsize=9.5, va="top", bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
ax[0].legend(loc="lower right", fontsize=7.2, framealpha=0.9, ncol=1)

# per-log Pearson bars
a = ax[2]
xs = np.arange(len(ORDER)); w = 0.38
sg = [d[k]["shadowgen_vs_holdout"]["pearson"] for k in ORDER]
m2 = [d[k]["m2_vs_holdout"]["pearson"] for k in ORDER]
a.bar(xs - w/2, sg, w, color="#2166ac", label="ShadowGen")
a.bar(xs + w/2, m2, w, color=RED, label="PM4Py M2")
a.axhline(0, color="0.3", lw=0.8)
a.set_xticks(xs); a.set_xticklabels([LABEL[k].split()[0] for k in ORDER])
a.set_ylim(-0.8, 1.05); a.set_ylabel("Pearson $r$ vs future fitness")
a.set_title("Per-log agreement", fontsize=11)
a.legend(loc="lower left", fontsize=8.5)

fig.suptitle("Temporal hold-out: train on the past, validate on the literal future "
             "(ShadowGen tracks it; M2 is negative on the diverse logs)", fontsize=11.5, y=1.02)
fig.tight_layout()
outdir = os.path.join(_REPO, "report", "figures")
fig.savefig(os.path.join(outdir, "fig_temporal_holdout.pdf"), bbox_inches="tight")
png = os.path.join(_HERE, "results", "fig_temporal_holdout.png")
fig.savefig(png, dpi=130, bbox_inches="tight")
print("wrote", os.path.join(outdir, "fig_temporal_holdout.pdf"))
print("wrote", png)
