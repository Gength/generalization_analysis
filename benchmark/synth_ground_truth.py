"""
Experiment 1: Synthetic-system ground truth for ShadowGen.

Motivation. The benchmark validates ShadowGen against a log-derived reference
(R1/R2, leave-one-variant-out replay fitness). Two threats follow: the metric
and the reference share a token-replay core, and both are computed from the same
log. This experiment removes both by validating against a KNOWN generating
system, where "future valid behaviour" is not inferred from the log but sampled
from the system itself.

Design. For each synthetic system S (a random process tree converted to an
accepting Petri net):
  1. F ~ S : a large fresh playout, the future valid behaviour of S.
  2. L ~ S : a small, deliberately incomplete training log (|L| << |F|).
  3. Discover the eight benchmark models from L (six miners + trace/flower poles).
  4. TRUE generalisation of a model M = recall of S = mean token-replay fitness
     of F on M (graded); strict = fraction of F perfectly fitting.
  5. ShadowGen(L, M) is computed from L only and never sees S or F.
  6. TRUE precision of M vs S = mean fitness of a fresh M-playout replayed on S
     (best-effort; for the generalisation-vs-precision plot).
  7. PM4Py generalisation (M2) per model, as a baseline that should NOT track
     the known-system recall.

The headline is the agreement between ShadowGen (from L) and true recall
(against S), across models, per system and pooled.

Output: benchmark/results/synth_ground_truth.json
Usage:  python synth_ground_truth.py [n_seeds_per_mode]   (default 12 -> 60 systems)
"""
import os, sys, json, random, time, math
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ["TQDM_DISABLE"] = "1"        # silence pm4py replay progress bars
import numpy as np
import pm4py
from collections import Counter
try:
    import pm4py.util.constants as _pmc
    _pmc.SHOW_PROGRESS_BAR = False
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)
from miners import MINERS
from shadowgen import gen_shadow
from pm4py.algo.evaluation.generalization import algorithm as gen_eval
from pm4py.algo.simulation.playout.petri_net import algorithm as pn_playout
from pm4py.algo.simulation.playout.petri_net.variants import basic_playout as _bp

REAL_MINERS = ["Alpha", "Alpha+", "Heuristics", "Heuristics_Strict",
               "Inductive_Strict", "Inductive_Infrequent"]
POLES = ["Trace_Filtered", "Flower"]

# System complexity axis: most-frequent visible-activity count.
MODES = [6, 8, 10, 12, 15]
TREE_PROBS = {"sequence": 0.40, "choice": 0.20, "parallel": 0.20,
              "loop": 0.20, "or": 0.0, "silent": 0.10, "duplicate": 0}
N_L = 300      # incomplete training log
N_F = 2500     # fresh future reference
N_M = 1000     # playout per model for the precision estimate
MAX_LEN = 200  # trace-length cap for (possibly unbounded) discovered nets


def _tree_params(mode):
    p = {"mode": mode, "min": max(3, mode - 3), "max": mode + 4, "no_models": 1}
    p.update(TREE_PROBS)
    return p


def gen_system(mode, seed):
    random.seed(seed); np.random.seed(seed)
    tree = pm4py.generate_process_tree(parameters=_tree_params(mode))
    net, im, fm = pm4py.convert_to_petri_net(tree)
    return tree, net, im, fm


def playout(net, im, fm, n, seed, max_len=MAX_LEN):
    random.seed(seed); np.random.seed(seed)
    params = {_bp.Parameters.NO_TRACES: n}
    if hasattr(_bp.Parameters, "MAX_TRACE_LENGTH"):
        params[_bp.Parameters.MAX_TRACE_LENGTH] = max_len
    return pn_playout.apply(net, im, fm, variant=pn_playout.Variants.BASIC_PLAYOUT,
                            parameters=params)


def variants(log):
    return Counter(tuple(e["concept:name"] for e in t) for t in log)


def avg_fit(log, net, im, fm):
    r = pm4py.fitness_token_based_replay(log, net, im, fm)
    return float(r["average_trace_fitness"]), float(r["perc_fit_traces"]) / 100.0


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    def rank(v):
        order = np.argsort(np.argsort(np.asarray(v, float)))
        return order.astype(float)
    return pearson(rank(x), rank(y))


def mae(x, y):
    return float(np.mean(np.abs(np.asarray(x, float) - np.asarray(y, float))))


def run(n_seeds):
    systems = [(m, 1000 + 37 * i) for m in MODES for i in range(n_seeds)]
    records = []
    t_start = time.time()
    for idx, (mode, seed) in enumerate(systems):
        try:
            tree, S_net, S_im, S_fm = gen_system(mode, seed)
            F = playout(S_net, S_im, S_fm, N_F, seed + 1)
            L = playout(S_net, S_im, S_fm, N_L, seed + 2)
            vL, vF = variants(L), variants(F)
            if len(vF) < 3:
                print(f"[{idx+1}/{len(systems)}] mode={mode} seed={seed} SKIP (degenerate system)")
                continue
            cov = sum(c for v, c in vF.items() if v in vL) / max(sum(vF.values()), 1)
            models = {}
            for name, fn in MINERS.items():
                m = {"error": None}
                try:
                    net, im, fm = fn(L)
                    m["n_transitions"] = len(net.transitions)
                    m["shadowgen"] = float(gen_shadow(L, net, im, fm))
                    rg, rs = avg_fit(F, net, im, fm)
                    m["true_recall"] = rg
                    m["true_recall_strict"] = rs
                    try:
                        m["m2"] = float(gen_eval.apply(L, net, im, fm))
                    except Exception as e:
                        m["m2"] = None
                    try:
                        Mp = playout(net, im, fm, N_M, seed + 3)
                        m["true_precision"] = avg_fit(Mp, S_net, S_im, S_fm)[0]
                    except Exception:
                        m["true_precision"] = None
                except Exception as e:
                    m["error"] = f"{type(e).__name__}: {e}"
                models[name] = m
            rec = {"mode": mode, "seed": seed, "tree": str(tree)[:200],
                   "n_L": len(L), "n_F": len(F), "var_L": len(vL), "var_F": len(vF),
                   "coverage": cov, "models": models}
            records.append(rec)
            ok = [n for n in MINERS if models[n]["error"] is None]
            sg = [models[n]["shadowgen"] for n in REAL_MINERS if models[n]["error"] is None]
            tr = [models[n]["true_recall"] for n in REAL_MINERS if models[n]["error"] is None]
            pr = pearson(sg, tr)
            print(f"[{idx+1}/{len(systems)}] mode={mode} seed={seed} cov={cov:.2f} "
                  f"ok={len(ok)}/8 r(sg,recall|6miners)={'na' if pr is None else round(pr,3)} "
                  f"[{time.time()-t_start:.0f}s]")
        except Exception as e:
            print(f"[{idx+1}/{len(systems)}] mode={mode} seed={seed} SYSTEM FAIL: {type(e).__name__}: {e}")

    agg = aggregate(records)
    out = {"config": {"modes": MODES, "n_seeds": n_seeds, "tree_probs": TREE_PROBS,
                      "N_L": N_L, "N_F": N_F, "pm4py": pm4py.__version__},
           "n_systems": len(records), "aggregate": agg, "systems": records}
    outpath = os.path.join(_HERE, "results", "synth_ground_truth.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== AGGREGATE ===")
    print(json.dumps(agg, indent=2))
    print(f"\nWrote {outpath}  ({len(records)} systems, {time.time()-t_start:.0f}s)")


def aggregate(records):
    """Pooled and per-system agreement of ShadowGen and M2 vs true recall."""
    def collect(miner_set, metric):
        pooled_x, pooled_y, per_sys = [], [], []
        for rec in records:
            xs, ys = [], []
            for n in miner_set:
                m = rec["models"][n]
                if m["error"] is None and m.get(metric) is not None:
                    xs.append(m[metric]); ys.append(m["true_recall"])
            if len(xs) >= 3:
                pr = pearson(xs, ys)
                if pr is not None:
                    per_sys.append(pr)
            pooled_x += xs; pooled_y += ys
        return {
            "pooled_pearson": pearson(pooled_x, pooled_y),
            "pooled_spearman": spearman(pooled_x, pooled_y),
            "pooled_mae": mae(pooled_x, pooled_y) if metric == "shadowgen" else None,
            "pooled_n": len(pooled_x),
            "per_system_pearson_mean": float(np.mean(per_sys)) if per_sys else None,
            "per_system_pearson_median": float(np.median(per_sys)) if per_sys else None,
            "per_system_pearson_min": float(np.min(per_sys)) if per_sys else None,
            "n_systems_scored": len(per_sys),
            "per_system_pearson_values": [float(v) for v in per_sys],
        }
    return {
        "shadowgen_vs_recall_6miners": collect(REAL_MINERS, "shadowgen"),
        "shadowgen_vs_recall_8models": collect(REAL_MINERS + POLES, "shadowgen"),
        "m2_vs_recall_6miners": collect(REAL_MINERS, "m2"),
        "m2_vs_recall_8models": collect(REAL_MINERS + POLES, "m2"),
        "mean_coverage": float(np.mean([r["coverage"] for r in records])) if records else None,
    }


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    run(n)
