"""Figure for Exp4: bootstrap 95% CIs on the ShadowGen-vs-R1 agreement, per log
and pooled, against PM4Py M2.

Reads benchmark/results/exp4_bootstrap.json, writes
report/figures/fig_bootstrap_ci.pdf and a PNG preview.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
d = json.load(open(os.path.join(_HERE, "results", "exp4_bootstrap.json")))
LOGS = ["D1", "D2", "D3", "D4", "D5"]
LAB = {"D1": "L1", "D2": "L2", "D3": "L3", "D4": "L4", "D5": "L5"}
BLUE, RED = "#2166ac", "#b2182b"


def series(method_key):
    pts, los, his = [], [], []
    for k in LOGS:
        c = d["per_log"][method_key][k]["pearson"]
        pts.append(c["point"]); los.append(c["point"] - c["lo"]); his.append(c["hi"] - c["point"])
    pc = d["pooled"][method_key]["pearson"]
    pts.append(pc["point"]); los.append(pc["point"] - pc["lo"]); his.append(pc["hi"] - pc["point"])
    return np.array(pts), np.array([los, his])


fig, ax = plt.subplots(figsize=(8.2, 4.4))
xt = np.arange(len(LOGS) + 1)
labels = [LAB[k] for k in LOGS] + ["Pooled\n(30 cells)"]
sg, sgerr = series("shadowgen_vs_R1")
m2, m2err = series("m2_vs_R1")

ax.errorbar(xt - 0.11, sg, yerr=sgerr, fmt="o", color=BLUE, capsize=4, ms=7,
            lw=1.5, label="ShadowGen vs R1")
ax.errorbar(xt + 0.11, m2, yerr=m2err, fmt="s", color=RED, capsize=4, ms=6,
            lw=1.5, label="PM4Py M2 vs R1")
ax.axhline(0, color="0.3", lw=0.8)
ax.axhline(1, color="0.85", lw=0.8, ls=":")

# eleven-miner L1 point, if present
e = d.get("eleven_miner_L1", {})
if "pearson" in e and e["pearson"].get("point") is not None:
    p = e["pearson"]
    ax.errorbar([0 + 0.28], [p["point"]], yerr=[[p["point"] - p["lo"]], [p["hi"] - p["point"]]],
                fmt="^", color="#1a9850", capsize=4, ms=7, lw=1.5,
                label=f"ShadowGen, L1 eleven miners")

ax.set_xticks(xt); ax.set_xticklabels(labels)
ax.set_ylim(-0.75, 1.08); ax.set_ylabel("Pearson $r$ vs cross-validation R1")
ax.set_title("Agreement with the ground truth: bootstrap 95% CIs\n"
             "(ShadowGen tight and high; PM4Py straddles zero)", fontsize=11)
ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
outdir = os.path.join(_REPO, "report", "figures")
fig.savefig(os.path.join(outdir, "fig_bootstrap_ci.pdf"), bbox_inches="tight")
png = os.path.join(_HERE, "results", "fig_bootstrap_ci.png")
fig.savefig(png, dpi=130, bbox_inches="tight")
print("wrote", os.path.join(outdir, "fig_bootstrap_ci.pdf"))
print("wrote", png)
