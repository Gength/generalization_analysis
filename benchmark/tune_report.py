"""Digest tune_shadowgen.py results.

Reports, for each swept configuration:
  * mean MAE to R1 (or mean genval hit-rate), and the delta against the SHIPPED
    default as measured in the same harness (never against the committed matrix,
    which used cached nets that no longer exist locally),
  * the per-log breakdown, since a config can win on average by winning one log,
  * the realised mutation rate, reported only as a side effect,
  * a leave-one-log-out test: choose the best config on four logs, then score it
    on the fifth. If tuning does not transfer, the winner was overfitting to the
    logs that selected it, and that is exactly what we need to know before
    proposing any change of default.

Usage: python benchmark/tune_report.py results/tune_mae_alpha.json [...]
"""
import sys, json, os
import numpy as np

DEFAULT_LABEL = "N6_t5_mle"


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def show(path):
    d = load(path)
    obj, logs, res = d["objective"], d["logs"], d["results"]
    better = min if obj == "mae" else max
    key = "mae" if obj == "mae" else "hit_rate"
    unit = "MAE" if obj == "mae" else "hit%"

    base = next((r for r in res if r["label"] == DEFAULT_LABEL), None)
    print(f"\n=== {os.path.basename(path)}  [{obj}, grid={d['grid']}, logs={','.join(logs)}] ===")
    if base is None:
        print("  (shipped default not in this grid; deltas unavailable)")

    rows = sorted(res, key=lambda r: r["agg"], reverse=(obj != "mae"))
    print(f"{'config':30} {unit:>8} {'delta':>8}  " +
          " ".join(f"{l:>7}" for l in logs) + "   mut%")
    for r in rows:
        per = r["per_log"]
        delta = (r["agg"] - base["agg"]) if base else float("nan")
        mark = " <= SHIPPED" if r["label"] == DEFAULT_LABEL else ""
        mut = np.mean([per[l]["mutrate"] for l in logs]) * 100
        cells = " ".join(f"{per[l][key]:>7.4f}" if obj == "mae" else f"{per[l][key]:>7.2f}"
                         for l in logs)
        print(f"{r['label']:30} {r['agg']:>8.4f} {delta:>+8.4f}  {cells}  {mut:>5.1f}{mark}")

    # leave-one-log-out: does a tuned winner transfer to a log that did not pick it?
    if base and len(logs) >= 3:
        print("\n  leave-one-log-out (pick the winner on the other logs, score it here):")
        transfers = []
        for held in logs:
            others = [l for l in logs if l != held]
            pick = better(res, key=lambda r: float(np.mean([r["per_log"][l][key] for l in others])))
            got = pick["per_log"][held][key]
            ref = base["per_log"][held][key]
            gain = (ref - got) if obj == "mae" else (got - ref)
            transfers.append(gain)
            verdict = "beats" if gain > 0 else "loses to"
            print(f"    {held}: winner-on-others = {pick['label']:24} -> {got:.4f} "
                  f"({verdict} shipped {ref:.4f}, gain {gain:+.4f})")
        m = float(np.mean(transfers))
        print(f"    mean out-of-sample gain over shipped: {m:+.4f}  "
              f"({'tuning transfers' if m > 0 else 'tuning does NOT transfer: overfitting'})")


if __name__ == "__main__":
    paths = sys.argv[1:]
    if not paths:
        here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        paths = [os.path.join(here, f) for f in sorted(os.listdir(here))
                 if f.startswith("tune_") and f.endswith(".json")]
    for p in paths:
        show(p)
