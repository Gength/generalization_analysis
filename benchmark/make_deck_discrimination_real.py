"""Deck figure: resolving power on the REAL logs, against R1 (the main ground truth).

Within each log, the six miners are ranked by R1 and split at that log's median:
top three = good generalizers, bottom three = poor. The label comes only from the
ground truth, never from a metric score. We then count, over same-log (good, poor)
pairs, how often the metric gives the good model the higher score.

30 cells (5 logs x 6 miners) is too few for a histogram, so every model is drawn as
its own dot. Separation vs overlap is then directly visible.

Deck conventions: proper names (no M-ids), wide aspect for slide 9's picture box.
Writes presentation/figures/deck_discrimination.png
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

CFG = os.path.join(_HERE, "results", "configs")
REAL = ["Alpha", "Alpha+", "Heuristics", "Heuristics_Strict",
        "Inductive_Strict", "Inductive_Infrequent"]
LOGS = ["D1", "D2", "D3", "D4", "D5"]
POOR, GOOD = "#d6604d", "#2166ac"


def cfg_val(nm, m, meth):
    p = os.path.join(CFG, f"{nm}__{m}__{meth}.json")
    if not os.path.exists(p):
        return None
    r = json.load(open(p))["results"]
    v = r.get("mean")
    if v is None:
        v = r.get("score")
    return None if (v is None or (isinstance(v, (int, float)) and v <= -1)) else float(v)


# gather per log, label within log by R1 median. Rows stay per-log on the plot:
# the statistic counts within-log pairs, so pooling the dots across logs would show
# a cross-log "overlap" the statistic never counts.
groups = {"M1g": [], "M2": []}
rows = {"M1g": [], "M2": []}          # per log: (scores, is_good)
for dk in LOGS:
    nm = DATASETS[dk]["name"]
    cells = []
    for m in REAL:
        sg, m2, r1 = cfg_val(nm, m, "M1g"), cfg_val(nm, m, "M2"), cfg_val(nm, m, "R1")
        if sg is not None and r1 is not None:
            cells.append({"sg": sg, "m2": m2, "r1": r1})
    if len(cells) < 3:
        continue
    r1s = np.array([c["r1"] for c in cells])
    med = np.median(r1s)
    isgood = r1s > med
    for key in ("M1g", "M2"):
        k = "sg" if key == "M1g" else "m2"
        sc = [c[k] for c in cells]
        groups[key].append((sc, list(r1s)))
        rows[key].append((sc, isgood))


def pair_stat(groups_list, key_idx=None):
    corr = tot = 0.0
    for sc, tr in groups_list:
        sc, tr = np.array(sc, float), np.array(tr, float)
        ok = ~np.isnan(sc)
        sc, tr = sc[ok], tr[ok]
        med = np.median(tr)
        for i in np.where(tr > med)[0]:
            for j in np.where(tr <= med)[0]:
                tot += 1
                if sc[i] > sc[j]:
                    corr += 1
                elif sc[i] == sc[j]:
                    corr += 0.5
    return corr, int(tot)


LABELS = ["L1", "L2", "L3", "L4", "L5"]
fig, ax = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True, sharey=True)
for a, key, name in [(ax[0], "M1g", "ShadowGen"), (ax[1], "M2", "PM4Py")]:
    for ri, (sc, isgood) in enumerate(rows[key]):
        y = len(rows[key]) - 1 - ri                      # L1 at top
        a.axhline(y, color="0.88", lw=8, zorder=0)       # the log's lane
        for v, g in zip(sc, isgood):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            a.scatter([v], [y], s=110, color=GOOD if g else POOR, alpha=0.9,
                      edgecolors="k", linewidths=0.6, zorder=3,
                      label=("good generalizers" if g else "poor generalizers")
                      if (ri == 0 and key == "M1g") else None)
    corr, tot = pair_stat(groups[key])
    a.set_title(name, fontsize=19, pad=10)
    a.text(0.03, 0.97, "ranks good above poor:\n%g of %d pairs (%.0f%%)"
           % (corr, tot, 100 * corr / tot), transform=a.transAxes, va="top",
           fontsize=13.5, bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.95))
    a.set_xlabel("metric score assigned to the model", fontsize=14)
    a.set_xlim(0, 1.04); a.set_ylim(-0.8, len(rows[key]) - 0.2)
    a.set_yticks(range(len(rows[key])))
    a.set_yticklabels(LABELS[:len(rows[key])][::-1], fontsize=14)
    a.grid(axis="x", alpha=0.25)
    a.tick_params(labelsize=12)
    print("%-10s ranks good above poor in %g of %d pairs (%.3f)" % (name, corr, tot, corr / tot))
ax[0].set_ylabel("compared within each log", fontsize=14)
h, l = ax[0].get_legend_handles_labels()
seen = dict(zip(l, h))
ax[0].legend(seen.values(), seen.keys(), loc="lower left", fontsize=11.5, framealpha=0.95)
fig.tight_layout()
outdir = os.path.join(_REPO, "presentation", "figures")
os.makedirs(outdir, exist_ok=True)
out = os.path.join(outdir, "deck_discrimination_real_lanes.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
print("wrote", out)

# ---------------------------------------------------------------------------
# Frequency version, to match the synthetic figure's form.
# Counts, not density: 15 models per group is far too few for a density curve,
# and density would inflate a single bin into a misleading spike.
# ---------------------------------------------------------------------------
bins = np.linspace(0, 1, 15)
fig2, ax2 = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
for a, key, name in [(ax2[0], "M1g", "ShadowGen"), (ax2[1], "M2", "PM4Py")]:
    g = np.array([v for sc, ig in rows[key] for v, k in zip(sc, ig)
                  if k and v is not None and not np.isnan(v)], float)
    p = np.array([v for sc, ig in rows[key] for v, k in zip(sc, ig)
                  if not k and v is not None and not np.isnan(v)], float)
    a.hist(p, bins=bins, alpha=0.65, color=POOR, label="poor generalizers")
    a.hist(g, bins=bins, alpha=0.65, color=GOOD, label="good generalizers")
    a.plot(p, np.full(len(p), -0.35), "|", color=POOR, ms=11, mew=2)   # rug: every model
    a.plot(g, np.full(len(g), -0.35), "|", color=GOOD, ms=11, mew=2)
    corr, tot = pair_stat(groups[key])
    a.set_title(name, fontsize=19, pad=10)
    a.text(0.03, 0.97, "ranks good above poor:\n%g of %d pairs (%.0f%%)"
           % (corr, tot, 100 * corr / tot), transform=a.transAxes, va="top",
           fontsize=13.5, bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.95))
    a.set_xlabel("metric score assigned to the model", fontsize=14)
    a.set_xlim(0, 1.02); a.set_ylim(-0.9, None)
    a.tick_params(labelsize=12)
ax2[0].set_ylabel("number of models", fontsize=14)
ax2[1].legend(loc="center left", fontsize=12.5, framealpha=0.95)
fig2.tight_layout()
out2 = os.path.join(outdir, "deck_discrimination_real.png")
fig2.savefig(out2, dpi=200, bbox_inches="tight")
print("wrote", out2, "(15 good / 15 poor models; ticks under the axis are the individual models)")
