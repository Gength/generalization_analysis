"""Generate all report figures from the config JSONs (single source of truth).

This merges the former make_report_figures.py and make_extra_figures.py into one
entry point. All figures are derived from the same config files the tables are
built from, so figures and tables cannot drift. Outputs vector PDFs to report/figures/.

Figures currently used by the report:
  fig_calibration_<tag>.pdf  metric score vs R1, small-multiples (on the line = calibrated)
  fig_accept_<tag>.pdf       acceptance vs graded fitness, per miner (v2.6-mle)
  fig_nsweep.pdf             realized mutation rate vs N-gram order (BPI 2017 sweep)
  fig_pareto.pdf             speed vs accuracy (time vs MAE-to-R1; lower-left = better)

Also generated but no longer referenced by the report (kept for slides / reuse):
  fig_landscape_<tag>.pdf, fig_mae_<tag>.pdf, fig_metric_corr_<tag>.pdf, fig_ladder.pdf,
  fig_runtime.pdf

Usage: python benchmark/make_figures.py                       # D1 Sepsis (default)
       python benchmark/make_figures.py --dataset BPI2013_Incidents
"""
import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Single source of truth: the consolidated matrix (M6adapted = token-replay
# bootstrap adaptation, M6original = published Entropia -bgen, R1accept = acceptance GT).
CFG_V1 = "benchmark/results/configs"
CFG_V2 = "benchmark/results/configs"
OUT = "report/figures"
os.makedirs(OUT, exist_ok=True)

# AVATAR cost anchor: full published config (100 pre-epochs, 5000 adversarial
# steps) projected from the measured D1 CPU anchor (45.11 s/adv-step, 88 min
# for the reduced config; see results/avatar_rebuild/quick_anchor_timing.json).
# 302400 s = 3.5 days wall per log on benchmark-class CPU. Supersedes the
# unbacked 14400 s (4 h, GPU, partner-provenance) anchor.
AVATAR_ANCHOR_S = 302400.0

MINERS7 = ["Alpha", "Alpha+", "Heuristics", "Heuristics_Strict",
           "Inductive_Infrequent", "Inductive_Strict", "Flower"]
MINERS8 = ["Trace_Filtered"] + MINERS7
REAL = MINERS7[:-1]  # six real miners (no flower pole)
MLABEL = {"Trace_Filtered": "Trace*", "Alpha": "Alpha", "Alpha+": "Alpha+",
          "Heuristics": "Heuristics", "Heuristics_Strict": "Heur-strict",
          "Inductive_Infrequent": "Ind-infreq", "Inductive_Strict": "Ind-strict",
          "Flower": "Flower*"}

# --------------------------------------------------------------------------- #
# config readers                                                              #
# --------------------------------------------------------------------------- #
def cfg(d, dataset, miner, method):
    return f"{d}/{dataset}__{miner}__{method}.json"

def score_of(path):
    if not os.path.exists(path):
        return np.nan
    r = json.load(open(path, encoding="utf-8")).get("results", {})
    for k in ("mean", "score", "gen_score"):
        if r.get(k) is not None:
            v = float(r[k])
            # -1 is the "exceeds compute budget" sentinel; render as a timeout
            # cell (NaN -> distinct gray "--"), not a low score.
            return np.nan if v < 0 else v
    return np.nan

def _score(d, ds, m, meth):
    return score_of(cfg(d, ds, m, meth))

def col(ds, meth):
    """Score vector over the six real miners (all cells live in configs/)."""
    d = CFG_V1 if meth in ("M2", "M5", "M6adapted", "M7") else CFG_V2
    vals = np.array([_score(d, ds, m, meth) for m in REAL])
    if np.all(np.isnan(vals)):
        other = CFG_V2 if d == CFG_V1 else CFG_V1
        vals = np.array([_score(other, ds, m, meth) for m in REAL])
    return vals

# --------------------------------------------------------------------------- #
# shared plot helpers                                                         #
# --------------------------------------------------------------------------- #
def plot_heatmap(M, rows, cols, cmap, vmin, vmax, fmt, out, cbar_label):
    fig, ax = plt.subplots(figsize=(0.85 * len(cols) + 2.0, 0.5 * len(rows) + 1.6))
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=8)
    ax.set_ylabel("miner", fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                ax.text(j, i, "--", ha="center", va="center", fontsize=7, color="0.5")
            else:
                norm = (M[i, j] - vmin) / (vmax - vmin)
                dark = norm < 0.22 or norm > 0.80
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center",
                        fontsize=7, color="white" if dark else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label(cbar_label, fontsize=8); cb.ax.tick_params(labelsize=7)
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2); ax.tick_params(which="minor", length=0)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)

def plot_grouped_bars(rows, fit, acc, out):
    x = np.arange(len(rows)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    b1 = ax.bar(x - w/2, fit, w, label="Fitness (graded)", color="#378ADD")
    b2 = ax.bar(x + w/2, acc, w, label="Acceptance (strict)", color="#1D9E75")
    ax.set_xticks(x); ax.set_xticklabels(rows, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("score", fontsize=11); ax.set_ylim(0, 1.12)
    ax.tick_params(axis="y", labelsize=10)
    ax.legend(fontsize=10, frameon=False, ncol=2, loc="upper left")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.015, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)

# --------------------------------------------------------------------------- #
# per-dataset figures                                                         #
# --------------------------------------------------------------------------- #
def _r1_per_miner(ds):
    r1 = {}
    for m in MINERS8:
        v = score_of(cfg(CFG_V2, ds, m, "R1"))
        if np.isnan(v):
            v = score_of(cfg(CFG_V1, ds, m, "R1"))
        r1[m] = v
    return r1

def fig_landscape(ds, tag):
    """Landscape score heatmap (7 miners x methods+refs). No longer in the report."""
    versions = [("v1", CFG_V2, "M1a"), ("v2.1", CFG_V2, "M1c"), ("v2.6-mle", CFG_V2, "M1g")]
    externals = [("PM4Py", CFG_V1, "M2"), ("AVATAR", CFG_V1, "M5"),
                 ("Bootstrap", CFG_V1, "M6adapted"), ("SpeciAL", CFG_V1, "M7")]
    infeasible = [("AntiAlign", CFG_V1, "M4"), ("Pattern", CFG_V1, "M8")]
    refs = [("R1 CV", CFG_V1, "R1"), ("R2 LOVO", CFG_V1, "R2"), ("R3 Rand", CFG_V1, "R3")]
    cols = versions + externals + infeasible + refs
    M = np.full((len(MINERS7), len(cols)), np.nan)
    for i, miner in enumerate(MINERS7):
        for j, (_, d, meth) in enumerate(cols):
            M[i, j] = score_of(cfg(d, ds, miner, meth))
    plot_heatmap(M, [MLABEL[m] for m in MINERS7], [c[0] for c in cols],
                 cmap="RdYlGn", vmin=0, vmax=1, fmt="{:.2f}",
                 out=f"{OUT}/fig_landscape_{tag}.pdf", cbar_label="generalization score")

def fig_calibration_v2(ds, tag):
    """Cross-paradigm calibration across ALL FIVE logs, one panel per metric.
    Each panel plots the metric vs R1 over the six non-degenerate miners of every
    log (up to 30 points), marked by log. This subsumes the former single-log
    panel and the M1-only scale scatter (fig_calibration_scale). AVATAR shows only
    its off-protocol L1/L2 points, which is itself the feasibility story: metrics
    that could not run on every log appear with fewer points."""
    from matplotlib.lines import Line2D
    MARKS = {"D1": ("o", "#378ADD"), "D2": ("s", "#1D9E75"), "D3": ("^", "#9673a6"),
             "D4": ("D", "#E8943A"), "D5": ("v", "#b8403e")}
    panels = [("M1 ShadowGen (ours)", "M1g"), ("M2 PM4Py", "M2"), ("M5 AVATAR", "M5"),
              ("M6adapted", "M6adapted"), ("M6original -bgen", "M6original"), ("M7 SpeciAL", "M7")]
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.2), sharex=True, sharey=True)
    for ax, (name, meth) in zip(axes.ravel(), panels):
        ax.plot([0, 1], [0, 1], ls="--", c="0.6", lw=1, zorder=1)
        xs, ys = [], []
        for d, dsname in DS5:
            r1 = col(dsname, "R1"); y = col(dsname, meth)
            mk, c = MARKS[d]
            ax.scatter(r1, y, marker=mk, s=22, color=c, edgecolor="white",
                       linewidth=0.35, zorder=3, alpha=0.9)
            m = ~(np.isnan(r1) | np.isnan(y))
            xs.append(np.asarray(r1)[m]); ys.append(np.asarray(y)[m])
        xs = np.concatenate(xs) if xs else np.array([])
        ys = np.concatenate(ys) if ys else np.array([])
        mae = float(np.mean(np.abs(ys - xs))) if xs.size else float("nan")
        ax.set_title(f"{name}  (MAE {mae:.3f})", fontsize=10.5)
        ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05); ax.set_aspect("equal")
        ax.tick_params(labelsize=8.5)
    for ax in axes[-1]:
        ax.set_xlabel("R1 cross-validation fitness", fontsize=9.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("metric score", fontsize=9.5)
    handles = [Line2D([0], [0], marker=MARKS[d][0], color="w", markerfacecolor=MARKS[d][1],
                      markeredgecolor="white", markersize=7, label=f"L{d[1]}") for d, _ in DS5]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.945))
    fig.suptitle("Calibration against R1 across all five logs (six non-degenerate miners; on the dashed line = perfect)",
                 fontsize=10, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(f"{OUT}/fig_calibration_{tag}v2.pdf", bbox_inches="tight"); plt.close(fig)

def fig_mae(ds, tag):
    """Key-methods MAE-to-R1 heatmap. No longer in the report."""
    r1 = _r1_per_miner(ds)
    cols2 = [("v1", CFG_V2, "M1a"), ("v2.1", CFG_V2, "M1c"), ("v2.6-mle", CFG_V2, "M1g"),
             ("PM4Py", CFG_V1, "M2"), ("AVATAR", CFG_V1, "M5"),
             ("Bootstrap", CFG_V1, "M6adapted"), ("SpeciAL", CFG_V1, "M7")]
    A = np.full((len(MINERS7), len(cols2)), np.nan)
    for i, miner in enumerate(MINERS7):
        for j, (_, d, meth) in enumerate(cols2):
            s = score_of(cfg(d, ds, miner, meth))
            A[i, j] = abs(s - r1[miner]) if not (np.isnan(s) or np.isnan(r1[miner])) else np.nan
    plot_heatmap(A, [MLABEL[m] for m in MINERS7], [c[0] for c in cols2],
                 cmap="RdYlGn_r", vmin=0, vmax=0.6, fmt="{:.3f}",
                 out=f"{OUT}/fig_mae_{tag}.pdf", cbar_label="|score - R1|  (lower = better)")

def fig_accept(ds, tag):
    """Acceptance vs graded fitness (M1g = v2.6 mle, 8 miners)."""
    fit, acc = [], []
    for m in MINERS8:
        r = json.load(open(cfg(CFG_V2, ds, m, "M1g"), encoding="utf-8"))["results"]
        fit.append(r["mean"]); acc.append(r.get("gen_accept", np.nan))
    plot_grouped_bars([MLABEL[m] for m in MINERS8], fit, acc, out=f"{OUT}/fig_accept_{tag}.pdf")

def fig_calibration(ds, tag):
    """Calibration small-multiples: each metric vs R1 over the six real miners."""
    r1 = col(ds, "R1")
    panels = [("v1", "M1a"), ("v2.6-mle", "M1g"), ("PM4Py", "M2"),
              ("AVATAR", "M5"), ("Bootstrap", "M6adapted"), ("SpeciAL", "M7")]
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.0), sharex=True, sharey=True)
    for ax, (name, meth) in zip(axes.ravel(), panels):
        y = col(ds, meth)
        ax.plot([0, 1], [0, 1], ls="--", c="0.6", lw=1, zorder=1)
        ax.scatter(r1, y, c="#378ADD", s=30, zorder=3, edgecolor="white", linewidth=0.6)
        mae = np.nanmean(np.abs(y - r1))
        ax.set_title(f"{name}  (MAE {mae:.3f})", fontsize=9)
        ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05); ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("R1 cross-validation fitness", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("metric score", fontsize=8)
    fig.suptitle("Calibration against held-out ground truth (six real miners; on the dashed line = perfect)",
                 fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{OUT}/fig_calibration_{tag}.pdf", bbox_inches="tight"); plt.close(fig)

def fig_metric_corr(ds, tag):
    """Inter-metric Pearson matrix over the six real miners. No longer in the report."""
    methods = [("ours v2.6-mle", "M1g"), ("PM4Py", "M2"), ("AVATAR", "M5"),
               ("Bootstrap", "M6adapted"), ("SpeciAL", "M7"),
               ("R1 CV", "R1"), ("R2 LOVO", "R2"), ("R3 rand", "R3")]
    cols = [col(ds, m) for _, m in methods]
    n = len(methods)
    C = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            a, b = cols[i], cols[j]
            mask = ~(np.isnan(a) | np.isnan(b))
            if mask.sum() >= 3 and a[mask].std() > 0 and b[mask].std() > 0:
                C[i, j] = np.corrcoef(a[mask], b[mask])[0, 1]
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    im = ax.imshow(np.ma.masked_invalid(C), cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    labels = [nm for nm, _ in methods]
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=8)
    for i in range(n):
        for j in range(n):
            if not np.isnan(C[i, j]):
                ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(C[i, j]) > 0.6 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("Pearson r over six real miners", fontsize=8); cb.ax.tick_params(labelsize=7)
    ax.set_title("Inter-metric agreement (six real miners)", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_metric_corr_{tag}.pdf", bbox_inches="tight"); plt.close(fig)

# --------------------------------------------------------------------------- #
# dataset-independent figures                                                 #
# --------------------------------------------------------------------------- #
def _mae_vs_r1(ds, meth):
    r1 = np.array([_score(CFG_V2, ds, m, "R1") for m in REAL])
    x = np.array([_score(CFG_V2, ds, m, meth) for m in REAL])
    return float(np.mean(np.abs(x - r1))) if not np.any(np.isnan(np.r_[x, r1])) else np.nan

def fig_ladder():
    """MAE-per-version across all five datasets."""
    versions = [("v1", "M1a"), ("v2.1", "M1c"), ("v2.4", "M1d"),
                ("v2.5", "M1e"), ("v2.6-log", "M1f"), ("v2.6-mle", "M1g")]
    STYLE = {"D1": ("o-", "#378ADD"), "D2": ("s--", "#1D9E75"), "D3": ("^-", "#9673a6"),
             "D4": ("D--", "#E8943A"), "D5": ("v-", "#b8403e")}
    xs = list(range(len(versions)))
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    allv = []
    for d, ds in DS5:
        ys = [_mae_vs_r1(ds, me) for _, me in versions]
        allv += [v for v in ys if not np.isnan(v)]
        mk, c = STYLE[d]
        ax.plot(xs, ys, mk, color=c, label=f"{d}", ms=4.5, lw=1.4)
    ax.set_xticks(xs); ax.set_xticklabels([v for v, _ in versions], rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("MAE vs R1  (lower = better)", fontsize=9)
    ax.set_ylim(0, max(allv) * 1.15)
    ax.legend(fontsize=8, frameon=False, ncol=5, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", ls=":", alpha=.5)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_ladder.pdf", bbox_inches="tight"); plt.close(fig)

def fig_calibration_scale():
    """v2.6-mle vs R1 on the six real miners of every dataset (30 points).
    The graded-fitness analogue of fig_accept_scale."""
    MARKS = {"D1": ("o", "#378ADD"), "D2": ("s", "#1D9E75"), "D3": ("^", "#9673a6"),
             "D4": ("D", "#E8943A"), "D5": ("v", "#b8403e")}
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.plot([0, 1], [0, 1], ls="--", c="0.6", lw=1, zorder=1)
    for d, ds in DS5:
        r1 = col(ds, "R1"); mg = col(ds, "M1g")
        mask = ~(np.isnan(r1) | np.isnan(mg))
        pear = np.corrcoef(r1[mask], mg[mask])[0, 1]
        mae = float(np.nanmean(np.abs(mg - r1)))
        mk, c = MARKS[d]
        ax.scatter(r1[mask], mg[mask], marker=mk, s=52, color=c, edgecolor="white",
                   linewidth=0.7, zorder=3,
                   label=f"L{d[1]}  (r={pear:.3f}, MAE {mae:.3f})")
    ax.set_xlabel("R1 cross-validation fitness (ground truth)", fontsize=9)
    ax.set_ylabel("M1 ShadowGen", fontsize=9)
    ax.set_xlim(0.1, 1.03); ax.set_ylim(0.1, 1.03); ax.set_aspect("equal")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(ls=":", alpha=.4)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_calibration_scale.pdf", bbox_inches="tight")
    plt.close(fig)

def fig_nsweep():
    """Realized mutation rate vs N (BPI 2017 sweep)."""
    N = list(range(1, 9))
    rate = [0.027, 0.211, 0.376, 0.664, 0.882, 1.333, 1.255, 1.208]  # x1e-3
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.plot(N, rate, "o-", color="#378ADD")
    ax.scatter([6], [1.333], s=90, facecolor="none", edgecolor="#b8403e", linewidth=1.6, zorder=5)
    ax.annotate("peak at $N{=}6$", xy=(6, 1.333), xytext=(6.15, 0.85), fontsize=11,
                arrowprops=dict(arrowstyle="->", color="0.45"))
    ax.set_xlabel("N-gram order $N$", fontsize=12)
    ax.set_ylabel(r"realized mutation rate ($\times10^{-3}$)", fontsize=12)
    ax.set_xticks(N); ax.tick_params(labelsize=11)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(ls=":", alpha=.5)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_nsweep.pdf", bbox_inches="tight"); plt.close(fig)

def fig_runtime():
    """Per-model time on D1, log scale (bar). Superseded by fig_pareto in the report."""
    data = [("PM4Py (M2)", 0.4, "work"), ("SpeciAL (M7)", 1.0, "work"),
            ("R3 random", 4.4, "work"), ("ShadowGen", 5.4, "ours"),
            ("Bootstrap (M6adapted)", 11.1, "work"), ("R1 5-fold CV", 120, "gt"),
            ("M8 pattern", 600, "infeasible"), ("AVATAR (M5)", AVATAR_ANCHOR_S, "slow"),
            ("M4 anti-align.", 48414, "infeasible")]
    data.sort(key=lambda t: t[1])
    labels = [d[0] for d in data]; vals = [d[1] for d in data]
    cmap = {"ours": "#1D9E75", "work": "#378ADD", "gt": "#9673a6",
            "slow": "#E8943A", "infeasible": "#b8403e"}
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.barh(labels, vals, color=[cmap[d[2]] for d in data])
    ax.set_xscale("log"); ax.set_xlabel("time per model (s, log scale)", fontsize=9)
    for i, v in enumerate(vals):
        lab = f"{v:.0f}s" if v < 90 else (f"{v/60:.0f} min" if v < 5400 else f"{v/3600:.1f} h")
        ax.text(v * 1.18, i, lab, va="center", fontsize=7)
    ax.set_xlim(0.2, 1.5e6); ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_runtime.pdf", bbox_inches="tight"); plt.close(fig)

def fig_pareto():
    """Speed vs accuracy on D1: time per model (log x) against MAE-to-R1 (y).
    Lower-left = better. Runtimes match fig_runtime; MAE via col() (= Table 4)."""
    ds = "Sepsis"
    r1 = col(ds, "R1")
    # (label, runtime_s, method_id, kind, label_dx, label_dy, ha)
    pts = [("ShadowGen (ours)", 5.4, "M1g", "ours", 8, 4, "left"),
           ("Bootstrap (M6adapted)", 11.1, "M6adapted", "work", 8, -12, "left"),
           ("PM4Py (M2)", 0.4, "M2", "work", 8, 0, "left"),
           ("SpeciAL (M7)", 1.0, "M7", "work", 8, 2, "left"),
           ("AVATAR (M5)", AVATAR_ANCHOR_S, "M5", "slow", -8, 4, "right"),
           ("R3 random floor", 4.4, "R3", "floor", 0, 9, "center")]
    cmap = {"ours": "#1D9E75", "work": "#378ADD", "slow": "#E8943A", "floor": "0.55"}
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for label, t, meth, kind, dx, dy, ha in pts:
        y = col(ds, meth)
        mae = float(np.nanmean(np.abs(y - r1)))
        ax.scatter(t, mae, s=80, color=cmap[kind], edgecolor="white", linewidth=0.8, zorder=3)
        ax.annotate(label, (t, mae), textcoords="offset points", xytext=(dx, dy),
                    fontsize=8, ha=ha)
    ax.set_xscale("log")
    ax.set_xlabel("time per model (s, log scale)", fontsize=9)
    ax.set_ylabel("MAE vs R1  (lower = better)", fontsize=9)
    ax.set_ylim(-0.02, 0.33); ax.set_xlim(0.2, 1e6)
    ax.annotate("better", xy=(0.55, 0.005), xytext=(4.0, 0.075), fontsize=9, color="0.3",
                arrowprops=dict(arrowstyle="->", color="0.45"))
    ax.text(0.98, 0.97, "M4, M8: no score (infeasible)", transform=ax.transAxes,
            ha="right", va="top", fontsize=7.5, color="#b8403e")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(ls=":", alpha=.5)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_pareto.pdf", bbox_inches="tight"); plt.close(fig)

def fig_scale():
    """MAE to R1 per dataset (D1-D5) for v1, v2.6-mle, PM4Py. One bar group per log."""
    DS = [("D1\nSepsis", "Sepsis"), ("D2\nBPI 2013", "BPI2013_Incidents"),
          ("D3\nBPI 2017", "BPI2017"), ("D4\nBPI 2018", "BPI2018"),
          ("D5\nBPI 2019", "BPI2019")]
    METHS = [("v1", "M1a", "#9BC4E8"), ("v2.6-mle (ours)", "M1g", "#1D9E75"),
             ("PM4Py", "M2", "#E8943A")]
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    x = np.arange(len(DS)); w = 0.26
    for k, (name, meth, color) in enumerate(METHS):
        maes = []
        for _, ds in DS:
            y = col(ds, meth); r1 = col(ds, "R1")
            maes.append(float(np.nanmean(np.abs(y - r1))))
        bars = ax.bar(x + (k - 1) * w, maes, w, label=name, color=color)
        for bar, v in zip(bars, maes):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.004, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=6.3)
        print(f"  scale {meth:4}", [round(m, 3) for m in maes])
    ax.set_xticks(x); ax.set_xticklabels([d[0] for d in DS], fontsize=8)
    ax.set_ylabel("MAE vs R1  (lower = better)", fontsize=9)
    ax.set_ylim(0, 0.26)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", ls=":", alpha=.5)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_scale.pdf", bbox_inches="tight"); plt.close(fig)

DS5 = [("D1", "Sepsis"), ("D2", "BPI2013_Incidents"), ("D3", "BPI2017"),
       ("D4", "BPI2018"), ("D5", "BPI2019")]

def _accept_pair(ds, miner):
    """(gen_accept of M1g, R1accept mean) for one cell, or (nan, nan)."""
    import json as _json
    ga = ra = np.nan
    p1 = f"{CFG_V1}/{ds}__{miner}__M1g.json"
    p2 = f"{CFG_V1}/{ds}__{miner}__R1accept.json"
    if os.path.exists(p1):
        v = _json.load(open(p1, encoding="utf-8"))["results"].get("gen_accept")
        ga = np.nan if v is None else float(v)
    if os.path.exists(p2):
        ra = float(_json.load(open(p2, encoding="utf-8"))["results"]["accept_mean"])
    return ga, ra

def fig_accept_scale():
    """Acceptance validation across all five logs: gen_accept (v2.6-mle) vs the
    acceptance ground truth R1-accept; y=x = perfect. Shows the exact poles and
    the D4 saturation outlier (Inductive-strict)."""
    MARKS = {"D1": ("o", "#378ADD"), "D2": ("s", "#1D9E75"), "D3": ("^", "#9673a6"),
             "D4": ("D", "#E8943A"), "D5": ("v", "#b8403e")}
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.plot([0, 1], [0, 1], ls="--", c="0.6", lw=1, zorder=1)
    for d, ds in DS5:
        xs, ys = [], []
        for m in ["Trace_Filtered"] + REAL + ["Flower"]:
            ga, ra = _accept_pair(ds, m)
            if not (np.isnan(ga) or np.isnan(ra)):
                xs.append(ra); ys.append(ga)
        mk, c = MARKS[d]
        ax.scatter(xs, ys, marker=mk, s=52, color=c, edgecolor="white",
                   linewidth=0.7, label=d, zorder=3)
    ax.set_xlabel("R1-accept (held-out real traces replayed perfectly)", fontsize=9)
    ax.set_ylabel("gen_accept (shadow traces replayed perfectly)", fontsize=9)
    ax.set_xlim(-0.03, 1.05); ax.set_ylim(-0.03, 1.05); ax.set_aspect("equal")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(ls=":", alpha=.4)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_accept_scale.pdf", bbox_inches="tight")
    plt.close(fig)

def fig_accept_landscape():
    """The acceptance ground truth itself: R1-accept per (dataset, miner).
    Only the Inductive family strictly accepts unseen behavior, and trace
    depth (D4, avg 57 events) defeats strict acceptance everywhere."""
    miners = ["Trace_Filtered"] + REAL + ["Flower"]
    M = np.full((len(DS5), len(miners)), np.nan)
    for i, (d, ds) in enumerate(DS5):
        for j, m in enumerate(miners):
            _, ra = _accept_pair(ds, m)
            M[i, j] = ra
    labels = ["Trace*", "Alpha", "Alpha+", "Heuristics", "Heur-strict",
              "Ind-infreq", "Ind-strict", "Flower*"]
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    im = ax.imshow(np.ma.masked_invalid(M), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(miners))); ax.set_xticklabels(labels, rotation=30,
                                                          ha="right", fontsize=8)
    ynames = {"D1": "D1 Sepsis", "D2": "D2 BPI 2013", "D3": "D3 BPI 2017",
              "D4": "D4 BPI 2018 (avg 57 ev)", "D5": "D5 BPI 2019"}
    ax.set_yticks(range(len(DS5)))
    ax.set_yticklabels([ynames[d] for d, _ in DS5], fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                dark = M[i, j] < 0.22 or M[i, j] > 0.80
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if dark else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("R1-accept", fontsize=8); cb.ax.tick_params(labelsize=7)
    ax.set_xticks(np.arange(-.5, len(miners), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(DS5), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_accept_landscape.pdf", bbox_inches="tight")
    plt.close(fig)

DS_MARKS = {"D1": ("o", "#378ADD"), "D2": ("s", "#1D9E75"), "D3": ("^", "#9673a6"),
            "D4": ("D", "#E8943A"), "D5": ("v", "#b8403e")}

def _method_cover(meth):
    """Datasets on which `meth` has valid scores over the six real miners."""
    out = []
    for d, ds in DS5:
        vals = col(ds, meth)
        if not np.all(np.isnan(vals)):
            out.append((d, ds))
    return out

def fig_calibration_grid_scale():
    """Per-method calibration against R1 with EVERY dataset the method returned
    on (points marked per dataset). Partial coverage is stated in the panel title."""
    panels = [("v1", "M1a"), ("v2.6-mle", "M1g"), ("PM4Py", "M2"),
              ("AVATAR", "M5"), ("Bootstrap adapted", "M6adapted"), ("SpeciAL", "M7")]
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 5.4), sharex=True, sharey=True)
    for ax, (name, meth) in zip(axes.ravel(), panels):
        ax.plot([0, 1], [0, 1], ls="--", c="0.6", lw=1, zorder=1)
        cover = _method_cover(meth)
        errs = []
        for d, ds in cover:
            y = col(ds, meth); r1 = col(ds, "R1")
            mask = ~(np.isnan(y) | np.isnan(r1))
            mk, c = DS_MARKS[d]
            ax.scatter(r1[mask], y[mask], marker=mk, s=26, color=c,
                       edgecolor="white", linewidth=0.5, zorder=3)
            errs += list(np.abs(y[mask] - r1[mask]))
        mae = float(np.mean(errs)) if errs else np.nan
        dd = [d for d, _ in cover]
        note = "all logs" if len(dd) == 5 else "+".join(dd)
        ax.set_title(f"{name}  (MAE {mae:.3f}, {note})", fontsize=8.5)
        ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05); ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("R1 cross-validation fitness", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("metric score", fontsize=8)
    handles = [plt.Line2D([], [], marker=DS_MARKS[d][0], color=DS_MARKS[d][1],
                          ls="", ms=6, label=d) for d, _ in DS5]
    fig.legend(handles=handles, ncol=5, fontsize=8, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{OUT}/fig_calibration_grid_scale.pdf", bbox_inches="tight")
    plt.close(fig)

def fig_pareto_scale():
    """Speed vs accuracy aggregated over the five logs: median cell runtime (x)
    against the mean of per-dataset MAEs to R1 (y). Methods that only ran on
    some logs carry their coverage in the label."""
    import json as _json
    MINERS8 = ["Trace_Filtered"] + REAL + ["Flower"]

    def time_stats(meth):
        """(median, min, max) cell runtime over all logs x miners."""
        ts = []
        for _, ds in DS5:
            for m in MINERS8:
                p = f"{CFG_V1}/{ds}__{m}__{meth}.json"
                if not os.path.exists(p):
                    continue
                r = _json.load(open(p, encoding="utf-8")).get("results", {})
                t = r.get("runtime_s")
                s = None
                for k in ("mean", "score", "gen_score"):
                    if r.get(k) is not None:
                        s = float(r[k]); break
                if t is not None and float(t) > 0 and (s is None or s >= 0):
                    ts.append(float(t))
        if not ts:
            return (np.nan, np.nan, np.nan)
        return (float(np.median(ts)), float(min(ts)), float(max(ts)))

    def mae_stats(meth):
        """(mean, min, max) of the per-dataset MAEs to R1."""
        maes = []
        for _, ds in DS5:
            y = col(ds, meth); r1 = col(ds, "R1")
            mask = ~(np.isnan(y) | np.isnan(r1))
            if mask.sum() >= 3:
                maes.append(float(np.mean(np.abs(y[mask] - r1[mask]))))
        if not maes:
            return (np.nan, np.nan, np.nan)
        return (float(np.mean(maes)), float(min(maes)), float(max(maes)))

    def covnote(meth):
        dd = [d for d, _ in _method_cover(meth)]
        return "" if len(dd) == 5 else " (" + "+".join("L" + d[1] for d in dd) + ")"

    # (label base, method, kind, time override, dx, dy, ha)
    pts = [("M1 ShadowGen (ours)", "M1g", "ours", None, 8, 4, "left"),
           ("M2 PM4Py", "M2", "work", None, 8, 8, "left"),
           ("M7 SpeciAL", "M7", "work", None, 8, -11, "left"),
           ("M6adapted", "M6adapted", "work", None, 8, -14, "left"),
           ("M6original -bgen", "M6original", "fail", None, -8, 6, "right"),
           ("M5 AVATAR", "M5", "slow", AVATAR_ANCHOR_S, -8, 4, "right")]
    cmap = {"ours": "#1D9E75", "work": "#378ADD", "slow": "#E8943A",
            "fail": "#b8403e", "floor": "0.55"}
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for base, meth, kind, t_over, dx, dy, ha in pts:
        tmed, tlo, thi = time_stats(meth)
        t = t_over if t_over is not None else tmed
        mae, mlo, mhi = mae_stats(meth)
        if np.isnan(t) or np.isnan(mae):
            continue
        c = cmap[kind]
        # whiskers so the point does not hide the spread: horizontal = min-max
        # cell runtime, vertical = min-max per-log MAE. AVATAR's x is the
        # projected full-training cost (AVATAR_ANCHOR_S), not a measured
        # median, so it gets no time whisker.
        if t_over is None and not np.isnan(tlo):
            ax.plot([tlo, thi], [mae, mae], color=c, lw=0.9, alpha=0.45, zorder=2)
        if not np.isnan(mlo) and mhi > mlo:
            ax.plot([t, t], [mlo, mhi], color=c, lw=0.9, alpha=0.45, zorder=2)
        ax.scatter(t, mae, s=80, color=c, edgecolor="white",
                   linewidth=0.8, zorder=3)
        ax.annotate(base + covnote(meth), (t, mae), textcoords="offset points",
                    xytext=(dx, dy), fontsize=8, ha=ha)
        print(f"  pareto_scale {meth:7} t={t:>8.1f}s (min {tlo:.1f}, max {thi:.1f})  "
              f"MAE={mae:.3f} (min {mlo:.3f}, max {mhi:.3f})")

    # ShadowGen 1-iteration "fast mode": ~same accuracy, 1/5 the time.
    # calculate_gen_shadow_stable loops N identical generate+replay steps with
    # no shared setup, so single-iteration time is the 5-iter median / 5
    # (verified on BPI2017: measured 1-iter/5-iter ratio 5.3). Accuracy uses
    # raw_iterations[0], the metric's own first-iteration score per cell.
    import json as _j2
    t5m = time_stats("M1g")[0]
    t1 = t5m / 5.0
    m1_pl = []
    for _, ds in DS5:
        d = []
        for mn in REAL:
            pm = f"{CFG_V1}/{ds}__{mn}__M1g.json"
            pr = f"{CFG_V1}/{ds}__{mn}__R1.json"
            if not (os.path.exists(pm) and os.path.exists(pr)):
                continue
            rm = _j2.load(open(pm, encoding="utf-8")).get("results", {})
            rr = _j2.load(open(pr, encoding="utf-8")).get("results", {})
            ri = rm.get("raw_iterations")
            r1 = next((rr.get(k) for k in ("mean", "score", "gen_score")
                       if rr.get(k) is not None), None)
            if ri and r1 is not None:
                d.append(abs(float(ri[0]) - float(r1)))
        if len(d) >= 3:
            m1_pl.append(np.mean(d))
    if m1_pl and not np.isnan(t1):
        m1 = float(np.mean(m1_pl)); c1 = cmap["ours"]
        ax.annotate("", xy=(t1, m1), xytext=(t5m, mae_stats("M1g")[0]),
                    arrowprops=dict(arrowstyle="->", color=c1, lw=0.8, alpha=0.55, zorder=2))
        ax.scatter(t1, m1, s=80, facecolor="white", edgecolor=c1, linewidth=1.6, zorder=4)
        ax.annotate("M1 1-iter (fast)", (t1, m1), textcoords="offset points",
                    xytext=(-6, -14), fontsize=8, ha="left", color=c1)
        print(f"  pareto_scale M1g-1it t={t1:.1f}s MAE={m1:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("median time per model over all logs (s, log scale)", fontsize=9)
    ax.set_ylabel("mean MAE vs R1 over covered logs  (lower = better)", fontsize=9)
    ax.set_ylim(-0.03, 0.72); ax.set_xlim(0.2, 1e6)
    ax.annotate("better", xy=(0.55, 0.01), xytext=(4.0, 0.13), fontsize=9, color="0.3",
                arrowprops=dict(arrowstyle="->", color="0.45"))
    ax.text(0.98, 0.97, "M4, M8: no score on any log (infeasible)",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color="#b8403e")
    ax.text(0.02, 0.97, "bars: min-max over cells (horizontal)\nand over per-log error (vertical)",
            transform=ax.transAxes, ha="left", va="top", fontsize=7, color="0.4")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(ls=":", alpha=.5)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_pareto_scale.pdf", bbox_inches="tight")
    plt.close(fig)

def fig_genval():
    """Generator-premise validation: fraction of generated traces that are
    EXACT matches of held-out real variants (never seen by the generator),
    per dataset, for the full generator vs the 1-gram ablation vs random."""
    import json as _json
    p = "benchmark/results/generator_validation.json"
    if not os.path.exists(p):
        print("  genval: no results file, skipped")
        return
    data = _json.load(open(p, encoding="utf-8"))
    DS = [("D1\nSepsis", "D1"), ("D2\nBPI 2013", "D2"), ("D3\nBPI 2017", "D3"),
          ("D4\nBPI 2018", "D4"), ("D5\nBPI 2019", "D5")]
    SERIES = [("shadow log (N=6, mle)", "v26_mle_N6", "#1D9E75"),
              ("1-gram ablation", "N1_ablation", "#9BC4E8"),
              ("random traces", "random", "0.6")]
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    x = np.arange(len(DS)); w = 0.26
    for k, (name, key, color) in enumerate(SERIES):
        vals = [100 * data[d][key]["hit_rate_mean"] if d in data and key in data[d] else np.nan
                for _, d in DS]
        bars = ax.bar(x + (k - 1) * w, vals, w, label=name, color=color)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.35,
                        f"{v:.2f}" if v >= 0.005 else "0", ha="center",
                        va="bottom", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels([d[0] for d in DS], fontsize=8)
    ax.set_ylabel("% of generated traces that are\nheld-out REAL variants", fontsize=8.5)
    ax.set_ylim(0, 30)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", ls=":", alpha=.5)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_genval.pdf", bbox_inches="tight")
    plt.close(fig)

def fig_genval_box():
    """Generator premise over the full 21-log catalog (breadth): exact-match
    hit-rate against log representativeness (TLRA), ShadowGen crosses coloured by
    alphabet size vs the uniformly random floor (open circles), one point per log.
    Joins genval_21logs.json (hit-rate), tlra.json (per-log TLRA) and
    alphabet.json (per-log |A|), all keyed D1..D21; skipped if any is absent.
    Shows the two structural drivers of exact-matchability at once: hit-rate rises
    with representativeness (x) and falls with alphabet size (colour), so the
    large-alphabet municipality logs sit on the floor regardless of TLRA."""
    import json as _json
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D
    pg = "benchmark/results/genval_21logs.json"
    pt = "benchmark/results/tlra.json"
    pa = "benchmark/results/alphabet.json"
    if not all(os.path.exists(p) for p in (pg, pt, pa)):
        print("  genval_box: missing 21-log / tlra / alphabet results, skipped")
        return
    g = _json.load(open(pg, encoding="utf-8"))
    tl = _json.load(open(pt, encoding="utf-8"))
    al = _json.load(open(pa, encoding="utf-8"))
    keys = [k for k in g if k in tl and k in al and "error" not in g[k]]
    tlra = [tl[k]["tlra"] for k in keys]
    A = [al[k]["n_activities"] for k in keys]
    shadow = [100 * g[k]["v26_mle_N6"]["hit_rate_mean"] for k in keys]
    rnd = [100 * g[k]["random"]["hit_rate_mean"] for k in keys]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.scatter(tlra, rnd, marker="o", s=30, facecolor="none", edgecolor="0.55",
               linewidth=1.0, zorder=2)
    sc = ax.scatter(tlra, shadow, marker="X", s=78, c=A, cmap="viridis",
                    norm=LogNorm(vmin=min(A), vmax=max(A)), edgecolor="0.3",
                    linewidth=0.5, zorder=3)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(r"alphabet size $|\mathcal{A}|$ (log scale)", fontsize=11)
    cb.ax.tick_params(labelsize=9)
    ax.set_xlabel("TLRA (log representativeness)", fontsize=12)
    ax.set_ylabel("exact-match hit-rate (%)", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.set_xlim(0, 1.0); ax.set_ylim(-2, max(shadow) * 1.12)
    handles = [Line2D([0], [0], marker="X", color="w", markerfacecolor="0.4",
                      markersize=10, label="ShadowGen"),
               Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                      markeredgecolor="0.55", markersize=8, label="Random")]
    ax.legend(handles=handles, fontsize=10, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(ls=":", alpha=.4)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_genval_box.pdf", bbox_inches="tight")
    plt.close(fig)

def fig_runtime_scale():
    """Median time per model vs event-log size, all five logs (log-log).
    Missing points are the feasibility story: SpeciAL crashed on D5, the
    Entropia tool timed out on D4. R1 is the ground-truth reference."""
    import json as _json
    EVENTS = {"Sepsis": 15214, "BPI2013_Incidents": 65533, "BPI2017": 1202267,
              "BPI2018": 2514266, "BPI2019": 1595923}
    MINERS8 = ["Trace_Filtered"] + REAL + ["Flower"]

    def rt_stats(ds, meth):
        """(median, min, max) cell runtime over miners with valid scores."""
        ts = []
        for m in MINERS8:
            p = f"{CFG_V1}/{ds}__{m}__{meth}.json"
            if not os.path.exists(p):
                continue
            r = _json.load(open(p, encoding="utf-8")).get("results", {})
            t = r.get("runtime_s")
            s = None
            for k in ("mean", "score", "gen_score"):
                if r.get(k) is not None:
                    s = float(r[k]); break
            if t is not None and float(t) > 0 and (s is None or s >= 0):
                ts.append(float(t))
        if not ts:
            return (np.nan, np.nan, np.nan)
        return (float(np.median(ts)), float(min(ts)), float(max(ts)))

    METHS = [("ShadowGen v2.6-mle (ours)", "M1g", "#1D9E75", "o-"),
             ("PM4Py", "M2", "#378ADD", "s-"),
             ("SpeciAL", "M7", "#9673a6", "^-"),
             ("Entropia -bgen", "M6original", "#E8943A", "D-"),
             ("R1 ground truth", "R1", "0.35", "v--")]
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for name, meth, c, mk in METHS:
        pts = []
        for _, ds in [("D1", "Sepsis"), ("D2", "BPI2013_Incidents"),
                      ("D3", "BPI2017"), ("D4", "BPI2018"), ("D5", "BPI2019")]:
            med, lo, hi = rt_stats(ds, meth)
            if not np.isnan(med):
                pts.append((EVENTS[ds], med, lo, hi))
        pts.sort()  # connect points in log-size order, not dataset order
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.plot(xs, ys, mk, color=c, label=name, ms=4.5, lw=1.4)
        # min-max whiskers: the median hides the tail; show the worst cell too
        ax.errorbar(xs, ys,
                    yerr=[[y - p[2] for y, p in zip(ys, pts)],
                          [p[3] - y for y, p in zip(ys, pts)]],
                    fmt="none", ecolor=c, elinewidth=0.8, capsize=2, alpha=0.55)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("event log size (events, log scale)", fontsize=9)
    ax.set_ylabel("time per model (s, log scale)\nmarker = median, whiskers = min to max", fontsize=8.5)
    ax.text(0.98, 0.04, "SpeciAL: crashed on D5 (1.6M events)\n"
                        "Entropia -bgen: timeout on D4 (2.5M events)",
            transform=ax.transAxes, fontsize=7.5, color="#b8403e",
            ha="right", va="bottom")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(ls=":", alpha=.4)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_runtime_scale.pdf", bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Sepsis")
    args = ap.parse_args()
    ds = args.dataset
    tag = "d1" if ds == "Sepsis" else ds.lower()

    # used by the report
    fig_calibration(ds, tag); fig_calibration_v2(ds, tag); fig_accept(ds, tag); fig_nsweep(); fig_pareto()
    # cross-dataset summaries (D1-D5)
    fig_scale(); fig_accept_scale(); fig_accept_landscape(); fig_calibration_scale()
    fig_runtime_scale(); fig_calibration_grid_scale(); fig_pareto_scale(); fig_genval(); fig_genval_box()
    # kept for slides / reuse (not referenced by the current report)
    fig_landscape(ds, tag); fig_mae(ds, tag); fig_metric_corr(ds, tag)
    fig_ladder(); fig_runtime()
    print(f"Figures written to {OUT}/ for dataset {ds}")

if __name__ == "__main__":
    main()
