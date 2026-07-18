"""Figure for Exp1: ShadowGen vs known-system recall, against PM4Py M2.

Reads benchmark/results/synth_ground_truth.json (+ synth_analysis.json) and writes
report/figures/fig_synth_groundtruth.pdf and a PNG preview.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
REAL = ["Alpha", "Alpha+", "Heuristics", "Heuristics_Strict",
        "Inductive_Strict", "Inductive_Infrequent"]
SPREAD_MIN = 0.05

d = json.load(open(os.path.join(_HERE, "results", "synth_ground_truth.json")))
an = json.load(open(os.path.join(_HERE, "results", "synth_analysis.json")))
systems = d["systems"]


def points(metric):
    xs, ys = [], []
    for s in systems:
        recs = [s["models"][m] for m in REAL]
        rr = [r["true_recall"] for r in recs if r.get("error") is None and r.get("true_recall") is not None]
        if len(rr) < 3 or np.std(rr) < SPREAD_MIN:
            continue
        for m in REAL:
            r = s["models"][m]
            if r.get("error") is None and r.get(metric) is not None and r.get("true_recall") is not None:
                xs.append(r["true_recall"]); ys.append(r[metric])
    return np.array(xs), np.array(ys)


def poles(model, metric):
    xs, ys = [], []
    for s in systems:
        r = s["models"][model]
        if r.get("error") is None and r.get(metric) is not None:
            xs.append(r["true_recall"]); ys.append(r[metric])
    return np.array(xs), np.array(ys)


psg = an["per_system_pearson_discriminating"]["shadowgen"]
pm2 = an["per_system_pearson_discriminating"]["m2"]
per_sg = d["aggregate"]["shadowgen_vs_recall_6miners"]["per_system_pearson_values"]
per_m2 = d["aggregate"]["m2_vs_recall_6miners"]["per_system_pearson_values"]

fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.3))
BLUE, RED, GREEN = "#2166ac", "#b2182b", "#1a9850"

for a, metric, title, med in [
        (ax[0], "shadowgen", "ShadowGen (from log only)", psg["median"]),
        (ax[1], "m2", "PM4Py M2 (from log only)", pm2["median"])]:
    x, y = points(metric)
    a.scatter(x, y, s=14, alpha=0.35, color=BLUE, edgecolors="none", zorder=2)
    fx, fy = poles("Flower", metric); tx, ty = poles("Trace_Filtered", metric)
    a.scatter(fx, fy, s=55, marker="*", color=GREEN, edgecolors="k", linewidths=0.4,
              zorder=4, label="flower pole")
    a.scatter(tx, ty, s=26, marker="s", color=RED, edgecolors="k", linewidths=0.4,
              zorder=4, label="trace pole")
    a.plot([0, 1], [0, 1], "--", color="0.4", lw=1, zorder=1)
    a.set_xlim(-0.02, 1.02); a.set_ylim(-0.02, 1.02)
    a.set_xlabel("true recall of the known system")
    a.set_title(title, fontsize=11)
    a.text(0.04, 0.93, f"per-system median $r={med:.2f}$", transform=a.transAxes,
           fontsize=10, va="top", bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    a.legend(loc="lower right", fontsize=8, framealpha=0.9)
ax[0].set_ylabel("metric score")

# per-system correlation distributions
a = ax[2]
bins = np.linspace(-1, 1, 21)
a.hist(per_sg, bins=bins, alpha=0.7, color=BLUE, label=f"ShadowGen (med {psg['median']:.2f})")
a.hist(per_m2, bins=bins, alpha=0.6, color=RED, label=f"PM4Py M2 (med {pm2['median']:.2f})")
a.axvline(0, color="0.3", lw=0.8)
a.set_xlabel("per-system Pearson $r$ vs true recall")
a.set_ylabel("systems")
a.set_title("Per-system correlation", fontsize=11)
a.legend(loc="upper left", fontsize=8)

n_disc = an["n_discriminating"]
fig.suptitle(f"Synthetic known-system validation: {n_disc} discriminating systems "
             f"(ShadowGen tracks true recall; M2 does not)", fontsize=11.5, y=1.02)
fig.tight_layout()
outdir = os.path.join(_REPO, "report", "figures")
os.makedirs(outdir, exist_ok=True)
fig.savefig(os.path.join(outdir, "fig_synth_groundtruth.pdf"), bbox_inches="tight")
png = os.path.join(_HERE, "results", "fig_synth_groundtruth.png")
fig.savefig(png, dpi=130, bbox_inches="tight")
print("wrote", os.path.join(outdir, "fig_synth_groundtruth.pdf"))
print("wrote", png)
