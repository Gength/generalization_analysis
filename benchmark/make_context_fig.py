"""Deck graphic: what a CONTEXT is, and why a shorter one offers more options.

The same trace looked up two ways:
  order 5 (the last 5 events)  -> matched rarely -> few next activities
  order 4 (one event shorter)  -> matched often  -> more next activities

The extra activities the shorter context offers are exactly the novel events:
attested somewhere, never after the full 5-event context. This is the primitive
the two-backoff slide then uses (same 35x count as that slide).

Writes two variants to presentation/figures/:
  deck_context.png       standalone (title + punchline), for its own slide
  deck_context_flat.png  wide and short, to sit ABOVE the two backoff panels

Usage: python benchmark/make_context_fig.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "presentation", "figures")
os.makedirs(OUT, exist_ok=True)

GREEN, BLUE, GREY, DARK = "#1A7A3A", "#1F497D", "#9A9A9A", "#3A3A3A"
LIGHTG, LIGHTB = "#E8F4EC", "#E6EDF6"
TRACE = ["p", "q", "r", "s", "t", "u"]


def draw(ax, y, window, colour, fill, label, count, nexts, extra_from,
         bw=0.52, gap=0.08, x_evt=1.55, x_cnt=5.5, x_pill=8.0, fs=14):
    """One row: label | trace with the last `window` events framed | count | pills."""
    ax.text(0.12, y + bw / 2, label, fontsize=fs - 2, color=colour,
            fontweight="bold", va="center")
    for i, a in enumerate(TRACE):
        x = x_evt + i * (bw + gap)
        inwin = i >= len(TRACE) - window
        ax.add_patch(FancyBboxPatch((x, y), bw, bw,
                     boxstyle="round,pad=0.01,rounding_size=0.05",
                     linewidth=1.1, edgecolor=colour if inwin else "#CCCCCC",
                     facecolor=fill if inwin else "white", zorder=2))
        ax.text(x + bw / 2, y + bw / 2, a, ha="center", va="center", fontsize=fs,
                color=DARK if inwin else GREY,
                fontweight="bold" if inwin else "normal", zorder=3)
    ax.add_patch(FancyArrowPatch((x_cnt - 0.35, y + bw / 2), (x_cnt + 0.05, y + bw / 2),
                 arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color=colour))
    ax.text(x_cnt + 0.2, y + bw / 2, "matched", fontsize=fs - 3, color=GREY, va="center")
    ax.text(x_cnt + 1.35, y + bw / 2, count, fontsize=fs, color=colour, va="center",
            fontweight="bold")
    for i, a in enumerate(nexts):
        xx = x_pill + i * 0.64
        is_new = i >= extra_from
        ax.add_patch(FancyBboxPatch((xx, y), 0.5, 0.5,
                     boxstyle="round,pad=0.01,rounding_size=0.16",
                     linewidth=1.7 if is_new else 1.1,
                     edgecolor=BLUE if is_new else GREY,
                     facecolor=LIGHTB if is_new else "white", zorder=2))
        ax.text(xx + 0.25, y + 0.25, a, ha="center", va="center", fontsize=fs - 2,
                color=BLUE if is_new else DARK,
                fontweight="bold" if is_new else "normal", zorder=3)
    ax.text(x_pill + len(nexts) * 0.64 + 0.15, y + bw / 2,
            f"{len(nexts)} options", fontsize=fs - 3, color=DARK,
            fontweight="bold", va="center")


def flat():
    """Wide and short: sits above the two backoff panels on the same slide."""
    fig, ax = plt.subplots(figsize=(13.0, 2.35))
    ax.set_xlim(0, 13.0); ax.set_ylim(0, 2.35); ax.axis("off")
    draw(ax, 1.35, 5, GREEN, LIGHTG, "order 5", "35x", ["v", "w", "x"], 3)
    draw(ax, 0.35, 4, BLUE, LIGHTB, "order 4", "210x",
         ["v", "w", "x", "y", "z"], 3)
    ax.text(8.0, 1.98, "what followed it in the log:", fontsize=10, color=GREY)
    ax.text(0.12, 0.02, "a shorter context matches more often, so it offers more",
            fontsize=9.5, color=GREY, style="italic")
    p = os.path.join(OUT, "deck_context_flat.png")
    fig.savefig(p, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("written:", p)


def standalone():
    """With title and punchline, for use on its own slide."""
    fig, ax = plt.subplots(figsize=(13.0, 4.6))
    ax.set_xlim(0, 13.0); ax.set_ylim(0, 4.6); ax.axis("off")
    ax.text(0.12, 4.25, "The context is just the last N events. The model looks up "
            "what followed it in the log.", fontsize=13.5, color=DARK,
            fontweight="bold")
    draw(ax, 2.75, 5, GREEN, LIGHTG, "order 5", "35x", ["v", "w", "x"], 3)
    draw(ax, 1.55, 4, BLUE, LIGHTB, "order 4", "210x",
         ["v", "w", "x", "y", "z"], 3)
    ax.text(0.12, 0.75, "A shorter context matches more often, so it offers more "
            "options.", fontsize=12.5, color=DARK, fontweight="bold")
    ax.text(0.12, 0.28, "The two extra activities never followed the full 5-event "
            "context. Valid here, new there: the novel events.",
            fontsize=11.5, color=BLUE)
    p = os.path.join(OUT, "deck_context.png")
    fig.savefig(p, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("written:", p)


if __name__ == "__main__":
    flat()
    standalone()
