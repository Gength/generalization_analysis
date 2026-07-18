"""Deck figure: resolving power (score distributions split by true generalization).

Deck conventions (see make_deck_figures.py): PROPER NAMES, no M-ids, since the talk
never defines them. Wide aspect (2.12) to fill slide 9's picture box (8.83 x 4.16 in)
and slide-legible fonts.

Writes presentation/figures/deck_discrimination.png
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
POOR, GOOD = "#d6604d", "#4393c3"


def within_auc(groups):
    """P(a good generalizer outscores a poor one), counting ONLY same-system pairs.

    Good/poor are labelled inside each system by that system's own median true
    recall. This is the practitioner's question (compare models discovered from the
    same log) and matches how the report computes Spearman/Kendall: per log, over
    the six miners. A global-median split would count cross-system pairs, which
    flatters a flat metric via baseline differences between systems.
    """
    corr = tot = 0.0
    for scores, tr in groups:
        scores, tr = np.asarray(scores, float), np.asarray(tr, float)
        ok = ~np.isnan(scores); scores, tr = scores[ok], tr[ok]
        if len(tr) < 3:
            continue
        med = np.median(tr)
        gi, pi = np.where(tr > med)[0], np.where(tr <= med)[0]
        for i in gi:
            for j in pi:
                tot += 1
                if scores[i] > scores[j]:
                    corr += 1
                elif scores[i] == scores[j]:
                    corr += 0.5
    return corr / tot, int(tot)


d = json.load(open(os.path.join(_HERE, "results", "synth_ground_truth.json")))
# All 78 systems, six real miners (poles excluded). Labels are assigned WITHIN each
# system by its own median true recall, so the six miners of a system split 3/3.
sg, m2, lab = [], [], []
grp = {"shadowgen": [], "m2": []}
for s in d["systems"]:
    cells = [(n, s["models"][n]) for n in REAL
             if s["models"][n].get("error") is None
             and s["models"][n].get("true_recall") is not None
             and s["models"][n].get("shadowgen") is not None]
    if len(cells) < 3:
        continue
    trs = np.array([c["true_recall"] for _, c in cells])
    med_s = np.median(trs)
    for (n, c), t in zip(cells, trs):
        sg.append(c["shadowgen"])
        m2.append(c["m2"] if c.get("m2") is not None else np.nan)
        lab.append(t > med_s)                       # True = good generalizer, in-system
    for key in ("shadowgen", "m2"):
        grp[key].append(([c.get(key) if c.get(key) is not None else np.nan for _, c in cells],
                         list(trs)))
sg, m2, lab = np.array(sg), np.array(m2), np.array(lab)
good, poor = lab, ~lab
bins = np.linspace(0, 1, 22)

fig, ax = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
for a, scores, key, name in [(ax[0], sg, "shadowgen", "ShadowGen"),
                             (ax[1], m2, "m2", "PM4Py")]:
    ok = ~np.isnan(scores)
    a.hist(scores[poor & ok], bins=bins, density=True, alpha=0.65, color=POOR,
           label="poor generalizers")
    a.hist(scores[good & ok], bins=bins, density=True, alpha=0.65, color=GOOD,
           label="good generalizers")
    au, npairs = within_auc(grp[key])
    a.set_title(f"{name}", fontsize=19, pad=10)
    a.text(0.03, 0.96, f"ranks good above poor:\n{au*100:.0f}% of pairs",
           transform=a.transAxes, va="top", fontsize=14,
           bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.95))
    print("%-10s within-system AUC = %.3f  (%d pairs)" % (name, au, npairs))
    a.set_xlabel("metric score assigned to the model", fontsize=14)
    a.set_xlim(0, 1)
    a.tick_params(labelsize=12)
ax[0].set_ylabel("frequency density", fontsize=14)
# legend goes in the right panel's empty left region: the left panel's upper area
# is occupied by the AUC box and the tall good-generalizer bar
ax[1].legend(loc="center left", fontsize=13, framealpha=0.95)
fig.tight_layout()
outdir = os.path.join(_REPO, "presentation", "figures")
os.makedirs(outdir, exist_ok=True)
out = os.path.join(outdir, "deck_discrimination.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
print("wrote", out)
