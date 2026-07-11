"""Provenance for the stochastic-stability / single-draw claims in the report
(Sect. 6.4 threats, and the K=1-default framing). Reads the K=5 config cells,
which store the five per-iteration scores (raw_iterations), and treats each
iteration as an independent draw (the RNG advances between iterations, so this is
the same as re-seeding). Reports:
  - the per-cell score std across the five draws (median / 90th / max),
  - the four-criteria agreement (Pearson, Spearman, MAE) recomputed on each single
    draw separately, per log, so K=1 vs the K=5 mean can be compared directly.
No reruns: everything comes from the already-gathered K=5 data. Writes
results/seed_robustness.json.

Usage: python benchmark/seed_robustness.py
"""
import os
import json
import statistics

CFG = "benchmark/results/configs"
OUT = "benchmark/results/seed_robustness.json"
REAL = ["Alpha", "Alpha+", "Heuristics", "Heuristics_Strict",
        "Inductive_Infrequent", "Inductive_Strict"]
DS = [("L1", "Sepsis"), ("L2", "BPI2013_Incidents"), ("L3", "BPI2017"),
      ("L4", "BPI2018"), ("L5", "BPI2019")]


def _res(ds, m, meth):
    p = f"{CFG}/{ds}__{m}__{meth}.json"
    return json.load(open(p, encoding="utf-8"))["results"] if os.path.exists(p) else None


def _pearson(a, b):
    n = len(a); ma = sum(a) / n; mb = sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((y - mb) ** 2 for y in b) ** 0.5
    return cov / (va * vb) if va and vb else float("nan")


def _spearman(a, b):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    return _pearson(rank(a), rank(b))


def main():
    all_std = []
    per_log = {}
    for dk, ds in DS:
        iters, r1, stds = [], [], []
        ok = True
        for m in REAL:
            c = _res(ds, m, "M1g"); rr = _res(ds, m, "R1")
            if not c or not rr or not c.get("raw_iterations"):
                ok = False
                break
            iters.append(c["raw_iterations"]); r1.append(rr["mean"]); stds.append(c["std"])
        if not ok:
            continue
        all_std += stds
        k = len(iters[0])
        pear = [_pearson([iters[m][i] for m in range(len(REAL))], r1) for i in range(k)]
        mae = [statistics.mean(abs(iters[m][i] - r1[m]) for m in range(len(REAL))) for i in range(k)]
        spear = [_spearman([iters[m][i] for m in range(len(REAL))], r1) for i in range(k)]
        mean5 = [statistics.mean(iters[m]) for m in range(len(REAL))]
        per_log[dk] = {
            "cell_std_median": round(statistics.median(stds), 4),
            "cell_std_max": round(max(stds), 4),
            "pearson_5iter_mean": round(_pearson(mean5, r1), 4),
            "pearson_per_draw_min": round(min(pear), 4),
            "pearson_per_draw_max": round(max(pear), 4),
            "pearson_per_draw_std": round(statistics.pstdev(pear), 4),
            "mae_per_draw_min": round(min(mae), 4),
            "mae_per_draw_max": round(max(mae), 4),
            "spearman_per_draw": [round(s, 3) for s in spear],
        }
    summary = {
        "cell_std_median_all": round(statistics.median(all_std), 4),
        "cell_std_p90_all": round(sorted(all_std)[int(0.9 * len(all_std))], 4),
        "cell_std_max_all": round(max(all_std), 4),
        "note": "each of the 5 raw_iterations is an independent 1000-trace draw; "
                "single-draw = K=1, the shipped default; 5-draw mean = K=5, benchmarked.",
        "per_log": per_log,
    }
    json.dump(summary, open(OUT, "w", encoding="utf-8"), indent=1)
    print(json.dumps(summary, indent=1))
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
