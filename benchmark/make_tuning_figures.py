"""Visualise the parameter search (TUNING_FINDINGS.md) from the result JSONs.

fig_tuning_sweeps.png  N, tau, weighting-temperature, theta: every knob against
                       MAE, with the shipped default marked. The story is that
                       the defaults sit at optima or inside noise bands.
fig_tuning_novelty.png the three results that matter more than any knob:
                       (a) alpha per log: the logs DISAGREE about novelty and
                           the effects cancel in the mean,
                       (b) are the novel traces plausible? (near-match split),
                       (c) tuning does not transfer (leave-one-log-out).

Usage: python benchmark/make_tuning_figures.py
"""
import os, json, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = RES
# vector copies for the report appendix (report/figures is the convention)
REPFIG = os.path.join(os.path.dirname(HERE), "report", "figures")
GREEN, BLUE, ORANGE, RED, PURPLE = "#1D9E75", "#378ADD", "#E8943A", "#b8403e", "#9673a6"
GREY = "0.55"
SHIP = dict(max_n=6, safe_threshold=5, num_traces=1000, weighting="mle",
            alpha=1.0, pu_clamp=1.0, temp=1.0, cap_mult=2.0, mut_uniform=False)
LOGMARK = {"D1": ("o", BLUE, "L1 Sepsis"), "D2": ("s", GREEN, "L2 BPI2013"),
           "D3": ("^", PURPLE, "L3 BPI2017"), "D4": ("D", ORANGE, "L4 BPI2018"),
           "D5": ("v", RED, "L5 BPI2019")}


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


def series(res, vary):
    """Configs that differ from the shipped default ONLY in `vary`."""
    out = []
    for r in res:
        c = r["cfg"]
        if all(c[k] == v for k, v in SHIP.items() if k != vary):
            out.append((c[vary], r["agg"], r))
    return sorted(out, key=lambda t: t[0])


def runtimes(logpath):
    """label -> seconds, parsed from the sweep log."""
    t = {}
    if not os.path.exists(logpath):
        return t
    for line in open(logpath, encoding="utf-8", errors="ignore"):
        m = re.match(r"\[\d+/\d+\]\s+(\S+)\s+mean\w+=[\d.]+\s+\((\d+)s\)", line)
        if m:
            t[m.group(1)] = int(m.group(2))
    return t


def mark_shipped(ax, x, y, label="shipped"):
    ax.scatter([x], [y], s=220, marker="*", color=GREEN, edgecolor="white",
               linewidth=0.8, zorder=6)
    ax.annotate(label, (x, y), textcoords="offset points", xytext=(8, 8),
                fontsize=8.5, color=GREEN, fontweight="bold")


NOISE = 0.004   # the metric's own sampling variance (report, Sect. threats)


def noise_band(ax, ys, floor=None):
    """Shade the +-noise band around the best value, and force the y-range to be
    at least the noise band. Without this, autoscale turns a 0.002 null result
    into a dramatic-looking V and the panel lies about its own conclusion."""
    lo = floor if floor is not None else min(ys)
    ax.axhspan(lo, lo + NOISE, color=GREEN, alpha=0.10, zorder=1)
    top = max(max(ys), lo + NOISE * 1.35)
    ax.set_ylim(lo - NOISE * 0.25, top + (top - lo) * 0.12)


# --------------------------------------------------------------------------- #
def fig_sweeps():
    coord = load("tune_mae_coord.json")["results"]
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.2))

    # (a) N, swept to 20
    ax = axes[0][0]
    s = series(coord, "max_n")
    xs, ys = [a for a, _, _ in s], [b for _, b, _ in s]
    ax.plot(xs, ys, "-o", color=BLUE, ms=4.5, lw=1.4, zorder=3)
    noise_band(ax, ys)
    mark_shipped(ax, 6, dict(zip(xs, ys))[6])
    ax.set_xlabel("context bound N", fontsize=9.5)
    ax.set_ylabel("mean MAE vs R1  (lower = better)", fontsize=9.5)
    ax.set_title("N: plateau from 4, nothing above 8", fontsize=10, loc="left")
    ax.text(0.97, 0.93, "shaded = within the metric's\nown sampling noise (0.004)",
            transform=ax.transAxes, ha="right", va="top", fontsize=7, color="0.45")

    # (b) tau
    ax = axes[0][1]
    s = series(coord, "safe_threshold")
    xs, ys = [a for a, _, _ in s], [b for _, b, _ in s]
    ax.plot(xs, ys, "-o", color=BLUE, ms=4.5, lw=1.4, zorder=3)
    noise_band(ax, ys)
    mark_shipped(ax, 5, dict(zip(xs, ys))[5])
    ax.set_xlabel(r"support threshold $\tau$", fontsize=9.5)
    ax.set_ylabel("mean MAE vs R1", fontsize=9.5)
    ax.set_title(r"$\tau$: shallow basin; only $\tau\geq10$ clearly degrades", fontsize=10, loc="left")

    # (c) weighting temperature  w(c) = c^(1/T); T=1 IS the shipped mle
    ax = axes[1][0]
    pts = []
    for r in coord:
        c = r["cfg"]
        if c["weighting"] == "temp" and all(
                c[k] == v for k, v in SHIP.items() if k not in ("weighting", "temp")):
            pts.append((c["temp"], r["agg"]))
    mle = next(r["agg"] for r in coord if r["label"] == "N6_t5_mle")
    logw = next((r["agg"] for r in coord if r["label"] == "N6_t5_log"), None)
    pts.append((1.0, mle))
    pts = sorted(pts)
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    fin = [(x, y) for x, y in zip(xs, ys) if x < 900]
    ax.plot([x for x, _ in fin], [y for _, y in fin], "-o", color=BLUE, ms=4.5, lw=1.4, zorder=3)
    unif = [y for x, y in zip(xs, ys) if x >= 900]
    if unif:
        ax.axhline(unif[0], ls=":", color=GREY, lw=1)
        ax.text(3.4, unif[0] + 0.001, f"uniform weighting ({unif[0]:.3f})",
                fontsize=7.5, color=GREY)
    if logw:
        ax.axhline(logw, ls="--", color=ORANGE, lw=1)
        ax.text(3.4, logw + 0.001, f"'log' weighting ({logw:.3f})", fontsize=7.5, color=ORANGE)
    mark_shipped(ax, 1.0, mle, "shipped (mle)")
    ax.set_xlabel(r"weighting temperature $T$   in   $w(c)=c^{1/T}$", fontsize=9.5)
    ax.set_ylabel("mean MAE vs R1", fontsize=9.5)
    ax.set_title("weighting: MLE is the bottom of a symmetric well", fontsize=10, loc="left")
    ax.annotate("sharper", xy=(0.55, 0.052), fontsize=7.5, color="0.45")
    ax.annotate("flatter", xy=(4.4, 0.052), fontsize=7.5, color="0.45")

    # (d) theta, with runtime on a second axis
    ax = axes[1][1]
    s = series(coord, "num_traces")
    xs, ys = [a for a, _, _ in s], [b for _, b, _ in s]
    ax.plot(xs, ys, "-o", color=BLUE, ms=4.5, lw=1.4, zorder=3, label="MAE")
    noise_band(ax, ys)
    mark_shipped(ax, 1000, dict(zip(xs, ys))[1000])
    ax.set_xscale("log")
    ax.set_xlabel(r"shadow-log size $\theta$ (traces, log scale)", fontsize=9.5)
    ax.set_ylabel("mean MAE vs R1", fontsize=9.5)
    ax.set_title(r"$\theta$: inert (cost grows with $\theta$; accuracy does not)",
                 fontsize=10, loc="left")
    # No runtime overlay: the five-log sweep runs in parallel and does not record
    # per-config wall time, and borrowing the three-log serial timings would put two
    # different runs in one panel. The cost claim lives in the caption instead.
    ax.text(0.03, 0.06, "every point is inside the noise band:\n"
                        "16x the shadow log does not move MAE, so the\n"
                        "residual error is systematic, not sampling",
            transform=ax.transAxes, fontsize=7.5, color="0.45")

    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(ls=":", alpha=.45)
        ax.tick_params(labelsize=8.5)
    fig.suptitle("ShadowGen parameter search: every shipped default is at an optimum or inside the noise band",
                 fontsize=11.5, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{OUT}/fig_tuning_sweeps.png", dpi=190, bbox_inches="tight")
    fig.savefig(os.path.join(REPFIG, "fig_tuning_sweeps.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  fig_tuning_sweeps.png")


# --------------------------------------------------------------------------- #
def fig_novelty():
    al = load("tune_mae_alpha.json")["results"]
    near = load("nearmatch_novelty_split.json")
    coord = load("tune_mae_coord.json")["results"]
    # Two-row layout: (a) full width on top, (b)/(c) below. At \textwidth in the
    # report each panel is ~2x the area of the old 1x3 strip, so the in-print
    # font size clears the legibility bar.
    fig = plt.figure(figsize=(10.5, 7.2))
    gs = fig.add_gridspec(2, 2)
    axes = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

    # (a) alpha per log: the logs disagree, the mean cancels.
    # Label the lines at their right-hand ends instead of using a legend: with six
    # series, a legend collided with the title, the shipped marker and the callouts.
    ax = axes[0]
    order = sorted(al, key=lambda r: r["cfg"]["alpha"])
    xs = [r["cfg"]["alpha"] for r in order]
    END = {"D1": ("L1 Sepsis  HURT by novelty", "bold"),
           "D5": ("L5 BPI2019  HELPED by it", "bold"),
           "D2": ("L2", "normal"), "D3": ("L3", "normal"), "D4": ("L4", "normal")}
    for dk in ("D1", "D2", "D3", "D4", "D5"):
        mk, c, _ = LOGMARK[dk]
        ys = [r["per_log"][dk]["mae"] for r in order]
        ax.plot(xs, ys, "-", marker=mk, color=c, ms=4.5, lw=1.4)
        txt, w = END[dk]
        ax.annotate(txt, (xs[-1], ys[-1]), textcoords="offset points", xytext=(7, -2),
                    fontsize=7.8, color=c, fontweight=w, va="center")
    means = [r["agg"] for r in order]
    ax.plot(xs, means, ls="--", color="0.25", lw=2.4, zorder=5)
    ax.annotate("mean: FLAT" + chr(10) + "(they cancel)", (xs[-1], means[-1]),
                textcoords="offset points",
                xytext=(7, -2), fontsize=7.8, color="0.25", fontweight="bold", va="center")
    ax.axvline(1.0, color=GREEN, lw=1.2, ls=":")
    ax.annotate("shipped" + chr(10) + "(Good-Turing rate)", (1.0, 0.0068),
                textcoords="offset points",
                xytext=(5, 0), fontsize=7.5, color=GREEN, fontweight="bold", va="center")
    ax.set_xlim(-0.15, 4.1)
    ax.set_ylim(0.004, 0.047)
    ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
    ax.set_xlabel(r"novelty scale $\alpha$   (0 = novelty OFF)", fontsize=9.5)
    ax.set_ylabel("MAE vs R1  (lower = better)", fontsize=9.5)
    ax.set_title(r"(a) $\alpha$ is NOT inert: the logs disagree", fontsize=10, loc="left")

    # (b) are the novel traces plausible?
    ax = axes[1]
    logs = ["D1", "D2", "D3", "D5"]
    labs = [LOGMARK[d][2].split()[0] for d in logs]
    reg = [near[d]["near_regular_pct"]["3"] for d in logs]
    mut = [near[d]["near_mutated_pct"]["3"] for d in logs]
    x = np.arange(len(logs)); w = 0.36
    ax.bar(x - w/2, reg, w, color=BLUE, label="recombination traces")
    ax.bar(x + w/2, mut, w, color=ORANGE, label="MUTATED traces (novel event)")
    ax.axhline(4.6, ls="--", color=RED, lw=1.2)
    ax.text(2.6, 6.0, "random-trace floor (4.6%)", fontsize=7.5, color=RED)
    for i, v in enumerate(mut):
        ax.text(i + w/2, v + 1.6, f"{v:.0f}%", ha="center", fontsize=8, color=ORANGE,
                fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=9)
    ax.set_ylabel("share within 3 edits of a REAL held-out variant (%)", fontsize=9)
    ax.set_title("(b) the novel traces are plausible, not noise", fontsize=10, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.set_ylim(0, 105)

    # (c) tuning does not transfer
    ax = axes[2]
    base = next(r for r in coord if r["label"] == "N6_t5_mle")
    logs3 = ["D1", "D2", "D5"]
    tuned, ship, names = [], [], []
    for held in logs3:
        others = [l for l in logs3 if l != held]
        pick = min(coord, key=lambda r: np.mean([r["per_log"][l]["mae"] for l in others]))
        tuned.append(pick["per_log"][held]["mae"])
        ship.append(base["per_log"][held]["mae"])
        names.append(f"{LOGMARK[held][2].split()[0]}\ntuned: {pick['label'].replace('N6_t5_mle_','')}")
    x = np.arange(len(logs3)); w = 0.36
    ax.bar(x - w/2, ship, w, color=GREEN, label="shipped defaults (untuned)")
    ax.bar(x + w/2, tuned, w, color=RED, label="tuned on the OTHER logs")
    for i, (s_, t_) in enumerate(zip(ship, tuned)):
        ax.annotate("", xy=(i + w/2, t_), xytext=(i - w/2, s_),
                    arrowprops=dict(arrowstyle="->", color="0.35", lw=1.0))
        ax.text(i, max(s_, t_) + 0.004, f"+{(t_-s_)*1000:.0f}e-3", ha="center",
                fontsize=8, color=RED, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7.8)
    ax.set_ylabel("MAE on the held-out log", fontsize=9.5)
    ax.set_title("(c) tuning does NOT transfer: it makes it worse", fontsize=10, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.set_ylim(0, max(tuned) * 1.3)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(ls=":", alpha=.45, axis="y")
        ax.tick_params(labelsize=8.5)
    fig.suptitle("The three findings that matter more than any parameter value",
                 fontsize=11.5, y=1.00)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{OUT}/fig_tuning_novelty.png", dpi=190, bbox_inches="tight")
    fig.savefig(os.path.join(REPFIG, "fig_tuning_novelty.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  fig_tuning_novelty.png")


if __name__ == "__main__":
    fig_sweeps()
    fig_novelty()
    print(f"written to {OUT}/")
