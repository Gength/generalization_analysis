"""
Experiment 4 (cheap half): bootstrap confidence intervals for the four-criteria
agreement, from the committed matrix. No new compute.

The report's headline calibration (ShadowGen vs R1) is a Pearson correlation over
six miners per log. Six points is a small sample, which is exactly why the report
uses four criteria and both poles rather than a single r. This script quantifies
that: it puts a bootstrap 95% CI on the per-log correlations (resampling the six
miners), pools all 5x6 = 30 calibration points for a tighter global CI, and does
the same for PM4Py's M2 as a contrast. It also reports the L1 eleven-miner CI
from exp4_miners.json if present.

Output: benchmark/results/exp4_bootstrap.json
"""
import os, sys, json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from datasets import DATASETS

CFG = os.path.join(_HERE, "results", "configs")
REAL = ["Alpha", "Alpha+", "Heuristics", "Heuristics_Strict",
        "Inductive_Strict", "Inductive_Infrequent"]
LOGS = ["D1", "D2", "D3", "D4", "D5"]
B = 10000
RNG = np.random.default_rng(42)


def cfg_val(dsname, miner, method):
    p = os.path.join(CFG, f"{dsname}__{miner}__{method}.json")
    if not os.path.exists(p):
        return None
    res = json.load(open(p))["results"]
    v = res.get("mean")
    if v is None:                        # single-value methods (M2, ...) use "score"
        v = res.get("score")
    if v is None or (isinstance(v, (int, float)) and v <= -1):
        return None
    return float(v)


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    rank = lambda v: np.argsort(np.argsort(np.asarray(v, float))).astype(float)
    return pearson(rank(x), rank(y))


def boot_ci(x, y, stat=pearson):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    vals = []
    for _ in range(B):
        idx = RNG.integers(0, n, n)
        v = stat(x[idx], y[idx])
        if not np.isnan(v):
            vals.append(v)
    vals = np.array(vals)
    if len(vals) == 0:
        return {"point": None, "lo": None, "hi": None, "median": None,
                "n": n, "n_boot_valid": 0}
    pt = stat(x, y)
    return {"point": None if np.isnan(pt) else float(pt),
            "lo": float(np.percentile(vals, 2.5)),
            "hi": float(np.percentile(vals, 97.5)), "median": float(np.median(vals)),
            "n": n, "n_boot_valid": len(vals)}


def collect(method):
    """Return dict log -> (shadowgen_or_method_values, r1_values) over REAL miners."""
    out = {}
    pooled_m, pooled_r = [], []
    for dk in LOGS:
        name = DATASETS[dk]["name"]
        ms, rs = [], []
        for mn in REAL:
            mv = cfg_val(name, mn, method)
            rv = cfg_val(name, mn, "R1")
            if mv is not None and rv is not None:
                ms.append(mv); rs.append(rv)
        out[dk] = (ms, rs)
        pooled_m += ms; pooled_r += rs
    return out, pooled_m, pooled_r


def run():
    result = {"config": {"logs": LOGS, "miners": REAL, "B": B}, "per_log": {}, "pooled": {}}
    for method, label in [("M1g", "shadowgen_vs_R1"), ("M2", "m2_vs_R1"), ("R3", "random_vs_R1")]:
        per_log, pm, pr = collect(method)
        result["per_log"][label] = {}
        for dk, (ms, rs) in per_log.items():
            if len(ms) >= 3:
                result["per_log"][label][dk] = {
                    "pearson": boot_ci(ms, rs, pearson),
                    "spearman": boot_ci(ms, rs, spearman),
                    "mae": float(np.mean(np.abs(np.array(ms) - np.array(rs)))),
                }
            else:
                result["per_log"][label][dk] = {"n": len(ms), "note": "insufficient cells"}
        result["pooled"][label] = {
            "pearson": boot_ci(pm, pr, pearson),
            "spearman": boot_ci(pm, pr, spearman),
            "mae": float(np.mean(np.abs(np.array(pm) - np.array(pr)))) if pm else None,
        }

    # Paired: P(shadowgen more calibrated than M2) via pooled paired bootstrap.
    # Only cells where M1g, M2 and R1 are all present, kept in aligned order.
    sm, mm, sr = [], [], []
    for dk in LOGS:
        name = DATASETS[dk]["name"]
        for mn in REAL:
            a = cfg_val(name, mn, "M1g"); b = cfg_val(name, mn, "M2"); c = cfg_val(name, mn, "R1")
            if a is not None and b is not None and c is not None:
                sm.append(a); mm.append(b); sr.append(c)
    if sm and mm and len(sm) == len(mm):
        sm, sr, mm = np.array(sm), np.array(sr), np.array(mm)
        wins = 0; diffs = []
        n = len(sm)
        for _ in range(B):
            idx = RNG.integers(0, n, n)
            rs_ = pearson(sm[idx], sr[idx]); rm_ = pearson(mm[idx], sr[idx])
            if not (np.isnan(rs_) or np.isnan(rm_)):
                diffs.append(rs_ - rm_)
                if rs_ > rm_:
                    wins += 1
        result["pooled"]["paired_shadowgen_gt_m2"] = {
            "p_win": wins / max(len(diffs), 1),
            "mean_pearson_gap": float(np.mean(diffs)),
            "gap_lo": float(np.percentile(diffs, 2.5)),
            "gap_hi": float(np.percentile(diffs, 97.5)),
        }

    # L1 eleven-miner CI, computed from the stored ShadowGen/R1 values
    p11 = os.path.join(_HERE, "results", "exp4_miners.json")
    if os.path.exists(p11):
        try:
            d11 = json.load(open(p11))["D1"]
            ms = [m for m in d11["miners"] if m in d11["ShadowGen"] and m in d11["R1"]]
            sg = [d11["ShadowGen"][m] for m in ms]; r1 = [d11["R1"][m] for m in ms]
            result["eleven_miner_L1"] = {
                "n_miners": len(ms), "miners": ms,
                "pearson": boot_ci(sg, r1, pearson),
                "spearman": boot_ci(sg, r1, spearman),
                "mae": float(np.mean(np.abs(np.array(sg) - np.array(r1))))}
        except Exception as e:
            result["eleven_miner_L1"] = {"error": repr(e)}

    outp = os.path.join(_HERE, "results", "exp4_bootstrap.json")
    json.dump(result, open(outp, "w"), indent=2)

    def fmt(ci):
        if ci.get("point") is None:
            return f"n/a [n={ci['n']}]"
        return f"{ci['point']:.3f} [{ci['lo']:.3f}, {ci['hi']:.3f}] (n={ci['n']})"
    print("=== Per-log ShadowGen vs R1 (Pearson, 95% CI) ===")
    for dk in LOGS:
        c = result["per_log"]["shadowgen_vs_R1"].get(dk, {})
        if "pearson" in c:
            print(f"  {dk}: {fmt(c['pearson'])}  MAE={c['mae']:.3f}")
    print("=== Pooled (30 cells) ===")
    print(f"  ShadowGen vs R1: Pearson {fmt(result['pooled']['shadowgen_vs_R1']['pearson'])}"
          f"  Spearman {fmt(result['pooled']['shadowgen_vs_R1']['spearman'])}"
          f"  MAE={result['pooled']['shadowgen_vs_R1']['mae']:.3f}")
    print(f"  M2        vs R1: Pearson {fmt(result['pooled']['m2_vs_R1']['pearson'])}"
          f"  MAE={result['pooled']['m2_vs_R1']['mae']:.3f}")
    if "paired_shadowgen_gt_m2" in result["pooled"]:
        pw = result["pooled"]["paired_shadowgen_gt_m2"]
        print(f"  Paired: P(ShadowGen r > M2 r) = {pw['p_win']:.3f}; "
              f"mean gap {pw['mean_pearson_gap']:.3f} [{pw['gap_lo']:.3f}, {pw['gap_hi']:.3f}]")
    if "eleven_miner_L1" in result and "pearson" in result["eleven_miner_L1"]:
        e = result["eleven_miner_L1"]
        print(f"=== L1 eleven-miner check ({e['n_miners']} configs) ===")
        print(f"  ShadowGen vs R1: Pearson {fmt(e['pearson'])}  MAE={e['mae']:.3f}")
    print(f"-> {outp}")


if __name__ == "__main__":
    run()
