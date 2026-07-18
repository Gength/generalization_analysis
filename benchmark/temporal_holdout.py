"""
Experiment 2: Temporal hold-out ground truth for ShadowGen.

The benchmark validates against variant-based cross-validation (R1/R2), which
shuffles variants across folds. This experiment uses a strictly temporal
reference instead, the most literal reading of "future behaviour": train on the
earliest cases, test on the latest.

For each real log:
  1. sort cases by start timestamp, split earliest 70% (train) / latest 30% (test).
  2. discover the eight benchmark models from the train split.
  3. ground truth = token-replay fitness of the TEST (future) cases on each model
     (graded average, and strict fraction fitting).
  4. ShadowGen(train, model) is computed from the train split only.
  5. agreement of ShadowGen (and PM4Py M2, as a baseline) with the temporal
     hold-out fitness, over the six non-degenerate miners, poles reported for the
     litmus.

Output: benchmark/results/temporal_holdout.json   (written incrementally per log)
Usage:  python temporal_holdout.py D1 D2 D5 D3 D4   (cheap logs first)
"""
import os, sys, json, time
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ["TQDM_DISABLE"] = "1"
import numpy as np
import pm4py
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)
from miners import MINERS
from datasets import DATASETS, get_log_path
from shadowgen import gen_shadow
from pm4py.algo.evaluation.generalization import algorithm as gen_eval
try:
    import pm4py.util.constants as _pmc
    _pmc.SHOW_PROGRESS_BAR = False
except Exception:
    pass

REAL_MINERS = ["Alpha", "Alpha+", "Heuristics", "Heuristics_Strict",
               "Inductive_Strict", "Inductive_Infrequent"]
POLES = ["Trace_Filtered", "Flower"]
TRAIN_FRAC = 0.70
OUT = os.path.join(_HERE, "results", "temporal_holdout.json")


def case_start(trace):
    ts = [e["time:timestamp"] for e in trace if "time:timestamp" in e]
    return min(ts) if ts else None


def temporal_split(log, frac=TRAIN_FRAC):
    keyed = [(case_start(t), i, t) for i, t in enumerate(log)]
    if any(k is None for k, _, _ in keyed):
        raise ValueError("log has traces without timestamps")
    keyed.sort(key=lambda x: (x[0], x[1]))
    cut = int(len(keyed) * frac)
    train = pm4py.objects.log.obj.EventLog([t for _, _, t in keyed[:cut]])
    test = pm4py.objects.log.obj.EventLog([t for _, _, t in keyed[cut:]])
    return train, test, keyed[0][0], keyed[cut - 1][0], keyed[cut][0], keyed[-1][0]


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
    rank = lambda v: np.argsort(np.argsort(np.asarray(v, float))).astype(float)
    return pearson(rank(x), rank(y))


def mae(x, y):
    return float(np.mean(np.abs(np.asarray(x, float) - np.asarray(y, float))))


def process(dkey):
    t0 = time.time()
    name = DATASETS[dkey]["name"]
    print(f"\n=== {dkey} {name} ===", flush=True)
    log = pm4py.convert_to_event_log(pm4py.read_xes(get_log_path(dkey)))
    train, test, tr0, tr1, te0, te1 = temporal_split(log)
    vtr, vte = variants(train), variants(test)
    novel = sum(c for v, c in vte.items() if v not in vtr) / max(sum(vte.values()), 1)
    print(f"  |train|={len(train)} ({len(vtr)} var)  |test|={len(test)} ({len(vte)} var)  "
          f"novel-in-test={novel:.3f}", flush=True)
    print(f"  train {tr0} .. {tr1}  |  test {te0} .. {te1}", flush=True)
    models = {}
    for mn, fn in MINERS.items():
        m = {"error": None}
        try:
            tm = time.time()
            net, im, fm = fn(train)
            m["shadowgen"] = float(gen_shadow(train, net, im, fm))
            g, s = avg_fit(test, net, im, fm)
            m["holdout_fit"] = g
            m["holdout_fit_strict"] = s
            try:
                m["m2"] = float(gen_eval.apply(train, net, im, fm))
            except Exception:
                m["m2"] = None
            print(f"    {mn:22s} sg={m['shadowgen']:.3f}  holdout={g:.3f}  "
                  f"m2={m['m2'] if m['m2'] is None else round(m['m2'],3)}  [{time.time()-tm:.0f}s]",
                  flush=True)
        except Exception as e:
            m["error"] = f"{type(e).__name__}: {e}"
            print(f"    {mn:22s} FAIL: {m['error']}", flush=True)
        models[mn] = m

    def corr(metric):
        xs = [models[n][metric] for n in REAL_MINERS
              if models[n]["error"] is None and models[n].get(metric) is not None]
        ys = [models[n]["holdout_fit"] for n in REAL_MINERS
              if models[n]["error"] is None and models[n].get(metric) is not None]
        return {"pearson": pearson(xs, ys), "spearman": spearman(xs, ys),
                "mae": mae(xs, ys) if metric == "shadowgen" and len(xs) else None, "n": len(xs)}

    rec = {"dataset": name, "n_train": len(train), "n_test": len(test),
           "var_train": len(vtr), "var_test": len(vte), "novel_in_test": novel,
           "train_range": [str(tr0), str(tr1)], "test_range": [str(te0), str(te1)],
           "poles": {p: {"shadowgen": models[p].get("shadowgen"),
                         "holdout_fit": models[p].get("holdout_fit")} for p in POLES},
           "shadowgen_vs_holdout": corr("shadowgen"),
           "m2_vs_holdout": corr("m2"),
           "models": models, "seconds": time.time() - t0}
    all_res = {}
    if os.path.exists(OUT):
        all_res = json.load(open(OUT))
    all_res[dkey] = rec
    json.dump(all_res, open(OUT, "w"), indent=2)
    sg = rec["shadowgen_vs_holdout"]; m2 = rec["m2_vs_holdout"]
    print(f"  ShadowGen vs holdout: Pearson={sg['pearson']} Spearman={sg['spearman']} "
          f"MAE={sg['mae']}", flush=True)
    print(f"  M2        vs holdout: Pearson={m2['pearson']} Spearman={m2['spearman']}", flush=True)
    print(f"  poles: " + ", ".join(f"{p}(sg={rec['poles'][p]['shadowgen']:.2f},"
          f"hold={rec['poles'][p]['holdout_fit']:.2f})" for p in POLES), flush=True)
    print(f"  [{dkey} done in {time.time()-t0:.0f}s -> {OUT}]", flush=True)


if __name__ == "__main__":
    keys = sys.argv[1:] or ["D1", "D2", "D5", "D3", "D4"]
    for k in keys:
        try:
            process(k)
        except Exception as e:
            print(f"[{k}] DATASET FAIL: {type(e).__name__}: {e}", flush=True)
