"""Presentation-only figure variants, generated from the same config matrix.

The report labels methods by their benchmark ids (M1..M9), which Table 2 of the
report defines. The talk never introduces that numbering, so the slides carry
the proper method names instead. Same data, same design, different labels:
nothing here recomputes a number that the report does not already report.

Outputs PNGs to presentation/figures/ (PowerPoint embeds PNG reliably):
  deck_calibration.png   ShadowGen vs R1, five logs        (slide "tracks the ground truth")
  deck_bad_calibration.png  PM4Py + SpeciAL vs R1          (slide "what not tracking looks like")
  deck_pareto.png        speed vs accuracy, no AVATAR      (slide "accurate and affordable")

Usage: python benchmark/make_deck_figures.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CFG = "benchmark/results/configs"
OUT = "presentation/figures"
os.makedirs(OUT, exist_ok=True)

REAL = ["Alpha", "Alpha+", "Heuristics", "Heuristics_Strict",
        "Inductive_Infrequent", "Inductive_Strict"]
MINERS8 = ["Trace_Filtered"] + REAL + ["Flower"]
DS5 = [("D1", "Sepsis"), ("D2", "BPI2013_Incidents"), ("D3", "BPI2017"),
       ("D4", "BPI2018"), ("D5", "BPI2019")]
MARKS = {"D1": ("o", "#378ADD"), "D2": ("s", "#1D9E75"), "D3": ("^", "#9673a6"),
         "D4": ("D", "#E8943A"), "D5": ("v", "#b8403e")}
GREEN, BLUE, RED = "#1D9E75", "#378ADD", "#b8403e"


def results_of(p):
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8")).get("results", {})


def score_of(p):
    r = results_of(p)
    if not r:
        return np.nan
    for k in ("mean", "score", "gen_score"):
        if r.get(k) is not None:
            v = float(r[k])
            return np.nan if v < 0 else v  # -1 = over-budget sentinel
    return np.nan


def col(ds, meth):
    return np.array([score_of(f"{CFG}/{ds}__{m}__{meth}.json") for m in REAL])


def time_stats(meth):
    ts = []
    for _, ds in DS5:
        for m in MINERS8:
            r = results_of(f"{CFG}/{ds}__{m}__{meth}.json")
            if not r:
                continue
            t = r.get("runtime_s")
            s = next((r.get(k) for k in ("mean", "score", "gen_score")
                      if r.get(k) is not None), None)
            if t is not None and float(t) > 0 and (s is None or float(s) >= 0):
                ts.append(float(t))
    return (np.nan,) * 3 if not ts else (float(np.median(ts)), min(ts), max(ts))


def mae_stats(meth):
    maes = []
    for _, ds in DS5:
        y = col(ds, meth); r1 = col(ds, "R1")
        mask = ~(np.isnan(y) | np.isnan(r1))
        if mask.sum() >= 3:
            maes.append(float(np.mean(np.abs(y[mask] - r1[mask]))))
    return (np.nan,) * 3 if not maes else (float(np.mean(maes)), min(maes), max(maes))


def covnote(meth):
    dd = [d for d, ds in DS5
          if (~(np.isnan(col(ds, meth)) | np.isnan(col(ds, "R1")))).sum() >= 3]
    return "" if len(dd) == 5 else " (" + "+".join("L" + d[1] for d in dd) + ")"


def fmt_dur(s):
    if s >= 3600:
        return f"{s/3600:.1f} h"
    if s >= 90:
        return f"{s/60:.0f} min"
    return f"{s:.0f} s"


def _calib_panel(ax, meth, name, rfmt="+.2f"):
    """One metric against the R1 ground truth, one marker per log.

    rfmt: the well-calibrated panel needs 3 decimals (0.996 would round to 1.00
    at 2 and read as a perfect correlation); the failing panels need the sign.
    """
    ax.plot([0, 1], [0, 1], ls="--", c="0.6", lw=1, zorder=1)
    for d, ds in DS5:
        r1 = col(ds, "R1"); y = col(ds, meth)
        mask = ~(np.isnan(r1) | np.isnan(y))
        if mask.sum() < 3:
            continue
        pear = np.corrcoef(r1[mask], y[mask])[0, 1]
        mae = float(np.mean(np.abs(y[mask] - r1[mask])))
        mk, c = MARKS[d]
        ax.scatter(r1[mask], y[mask], marker=mk, s=52, color=c, edgecolor="white",
                   linewidth=0.7, zorder=3,
                   label=f"L{d[1]}  (r={pear:{rfmt}}, MAE {mae:.3f})")
    ax.set_xlabel("R1 cross-validation fitness (ground truth)", fontsize=9)
    ax.set_ylabel(name, fontsize=9)
    ax.set_xlim(0.1, 1.03); ax.set_ylim(0.1, 1.03); ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(ls=":", alpha=.4)


def deck_calibration():
    """ShadowGen on the diagonal, all five logs."""
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    _calib_panel(ax, "M1g", "ShadowGen", rfmt=".3f")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{OUT}/deck_calibration.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  deck_calibration.png")


def deck_bad_calibration():
    """The two failure modes on the same axes: a flat band, and a cloud."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    for ax, meth, name in [(axes[0], "M2", "PM4Py"), (axes[1], "M7", "SpeciAL")]:
        _calib_panel(ax, meth, name)
        ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(f"{OUT}/deck_bad_calibration.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  deck_bad_calibration.png")


def deck_pareto():
    """Speed vs accuracy, linear time axis. AVATAR is omitted (its 9-12 h GPU
    training would force a log axis); ShadowGen sits at its shipped K=1 point."""
    pts = [("PM4Py", "M2", BLUE, 8, 8, "left"),
           ("SpeciAL", "M7", BLUE, 8, -13, "left"),
           ("Bootstrap (adapted)", "M6adapted", BLUE, 8, 6, "left"),
           ("Negative events", "M9", RED, 9, 3, "left"),
           ("Bootstrap (published F1)", "M6original", RED, -8, 6, "right")]
    XMAX = 150
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for name, meth, c, dx, dy, ha in pts:
        t, tlo, thi = time_stats(meth)
        mae, mlo, mhi = mae_stats(meth)
        if np.isnan(t) or np.isnan(mae):
            continue
        # worst cells run to hours, off this linear axis: clip with an arrow and
        # label the true maximum rather than silently dropping the spread
        if not np.isnan(tlo):
            if thi > XMAX:
                ax.annotate("", xy=(XMAX - 1, mae), xytext=(tlo, mae),
                            arrowprops=dict(arrowstyle="->", color=c, lw=0.9, alpha=0.5),
                            zorder=2)
                ax.annotate(fmt_dur(thi), (XMAX - 2, mae), textcoords="offset points",
                            xytext=(0, 4), fontsize=7, ha="right", color=c, alpha=0.9)
            else:
                ax.plot([tlo, thi], [mae, mae], color=c, lw=0.9, alpha=0.45, zorder=2)
        if not np.isnan(mlo) and mhi > mlo:
            ax.plot([t, t], [mlo, mhi], color=c, lw=0.9, alpha=0.45, zorder=2)
        ax.scatter(t, mae, s=80, color=c, edgecolor="white", linewidth=0.8, zorder=3)
        ax.annotate(name + covnote(meth), (t, mae), textcoords="offset points",
                    xytext=(dx, dy), fontsize=8.5, ha=ha)
        print(f"  deck_pareto {name:26} t={t:>6.1f}s (max {thi:.0f})  MAE={mae:.3f}")

    # ShadowGen at its shipped single-draw point: K=5 median / 5 (measured ratio
    # 5.0), accuracy from each cell's first draw against R1.
    t1 = time_stats("M1g")[0] / 5.0
    pl = []
    for _, ds in DS5:
        d = []
        for mn in REAL:
            rm = results_of(f"{CFG}/{ds}__{mn}__M1g.json")
            rr = results_of(f"{CFG}/{ds}__{mn}__R1.json")
            if not (rm and rr):
                continue
            ri = rm.get("raw_iterations")
            r1 = next((rr.get(k) for k in ("mean", "score", "gen_score")
                       if rr.get(k) is not None), None)
            if ri and r1 is not None:
                d.append(abs(float(ri[0]) - float(r1)))
        if len(d) >= 3:
            pl.append(float(np.mean(d)))
    m1, lo, hi = float(np.mean(pl)), float(min(pl)), float(max(pl))
    ax.plot([t1, t1], [lo, hi], color=GREEN, lw=0.9, alpha=0.45, zorder=2)
    ax.scatter(t1, m1, s=220, marker="*", color=GREEN, edgecolor="white",
               linewidth=0.8, zorder=4)
    ax.annotate("ShadowGen", (t1, m1), textcoords="offset points",
                xytext=(8, 6), fontsize=9.5, ha="left")
    print(f"  deck_pareto {'ShadowGen (K=1)':26} t={t1:>6.1f}s            MAE={m1:.3f}")

    ax.set_xlabel("median time per model over all logs (s)", fontsize=9.5)
    ax.set_ylabel("mean MAE vs R1 over covered logs  (lower = better)", fontsize=9.5)
    ax.set_xlim(0, XMAX); ax.set_ylim(-0.02, 0.72)
    ax.annotate("better", xy=(4, 0.02), xytext=(22, 0.14), fontsize=9, color="0.3",
                arrowprops=dict(arrowstyle="->", color="0.45"))
    ax.text(0.98, 0.97, "not shown: AVATAR (9–12 h GPU training per log);\n"
                        "anti-alignments and pattern-based: infeasible on every log",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=RED)
    ax.text(0.63, 0.115, "bars: min–max over cells (horizontal, arrows run off axis,\n"
                         "label = worst cell) and over per-log error (vertical)",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=7, color="0.4")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(ls=":", alpha=.5)
    fig.tight_layout(); fig.savefig(f"{OUT}/deck_pareto.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    deck_calibration()
    deck_bad_calibration()
    deck_pareto()
    print(f"Deck figures written to {OUT}/")
