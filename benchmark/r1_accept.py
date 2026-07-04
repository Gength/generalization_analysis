"""R1-acceptance: fraction of HELD-OUT traces perfectly replayed (variant-based 5-fold x 3).

The acceptance-based ground truth that validates gen_accept (report Sect. 6.6).
Reference metric: EXEMPT from the compute budget (no timeout).

Parallel over (miner, shuffle, fold) tasks — 8 x 3 x 5 = 120 independent
fold jobs. Numbers are IDENTICAL to the original serial script: each fold
worker re-derives the exact serial fold partition by replaying the seeded
shuffle sequence (seed 42 per miner, shuffles applied cumulatively), keeps
the same train-log trace order, and results are aggregated in the same
order (folds -> shuffle mean -> mean/std over shuffles). Verified on D1.

Usage (from the repo root, dataset keys per benchmark/datasets.py):
    python benchmark/r1_accept.py D3
    python benchmark/r1_accept.py D4 --output benchmark/results/configs --workers 32
    python benchmark/r1_accept.py D1 --workers 1        # serial fallback

Writes one sidecar config per miner (<Name>__<miner>__R1accept.json), then
correlates with M1f/M1g gen_accept (looked up in --output first, then
configs) where available.
"""
import os, sys, time, random, json, argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import multiprocessing as mp
import numpy as np
import pm4py
from pm4py.objects.log.obj import EventLog
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay

from miners import MINERS
from datasets import DATASETS

SEED = 42
K, SHUFFLES = 5, 3

# fork-shared globals (read-only after fork)
_LOG = None
_VMAP = None
_VORDER = None   # variant keys in log order (serial script's initial list order)


def _build_variant_map(log):
    vmap = defaultdict(list)
    for trace in log:
        vmap[tuple(e["concept:name"] for e in trace)].append(trace)
    return vmap


def _fold_partition(s_idx, fold_idx):
    """Re-derive the serial script's exact fold: seed once, apply s_idx+1
    cumulative shuffles, slice fold fold_idx. Returns (ordered_variants, test_set)."""
    rng = random.Random(SEED)
    variants = list(_VORDER)
    for _ in range(s_idx + 1):
        rng.shuffle(variants)
    n = len(variants)
    fold_size = max(1, n // K)
    start = fold_idx * fold_size
    end = (fold_idx + 1) * fold_size if fold_idx < K - 1 else n
    return variants, set(variants[start:end])


def _fold_worker(task):
    miner_name, s_idx, fold_idx = task
    t0 = time.time()
    random.seed(SEED)
    np.random.seed(SEED)
    variants, test_variants = _fold_partition(s_idx, fold_idx)
    train_log = EventLog([t for v in variants if v not in test_variants for t in _VMAP[v]])
    test_log = EventLog([t for v in variants if v in test_variants for t in _VMAP[v]])
    try:
        net, im, fm = MINERS[miner_name](train_log)
        replayed = token_replay.apply(test_log, net, im, fm)
        acc = float(np.mean([1.0 if r["trace_is_fit"] else 0.0 for r in replayed]))
    except Exception:
        acc = 0.0
    return miner_name, s_idx, fold_idx, acc, time.time() - t0


def main():
    global _LOG, _VMAP, _VORDER
    ap = argparse.ArgumentParser(description="R1-accept ground truth (parallel)")
    ap.add_argument("dataset", nargs="?", default="D1", choices=list(DATASETS.keys()))
    ap.add_argument("--output", default="benchmark/results/configs",
                    help="Config output dir (default: benchmark/results/configs)")
    ap.add_argument("--workers", type=int, default=os.cpu_count(),
                    help="Parallel fold workers (default: all cores; 1 = serial)")
    args = ap.parse_args()

    name = DATASETS[args.dataset]["name"]
    os.makedirs(args.output, exist_ok=True)

    _LOG = pm4py.read_xes(DATASETS[args.dataset]["log_path"])
    _LOG = pm4py.convert_to_event_log(_LOG)
    _VMAP = _build_variant_map(_LOG)
    _VORDER = list(_VMAP.keys())
    print(f"R1-accept — {args.dataset} {name} ({len(_LOG)} traces, "
          f"{len(_VORDER)} variants, workers={args.workers})", flush=True)

    tasks = [(m, s, f) for m in MINERS for s in range(SHUFFLES) for f in range(K)]
    fold_acc = {}
    fold_cpu = defaultdict(float)
    wall0 = time.time()

    can_fork = hasattr(os, "fork")
    if args.workers > 1 and can_fork:
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
            futs = {ex.submit(_fold_worker, t): t for t in tasks}
            done = 0
            for fu in as_completed(futs):
                mn, s, f, acc, dt = fu.result()
                fold_acc[(mn, s, f)] = acc
                fold_cpu[mn] += dt
                done += 1
                if done % 15 == 0:
                    print(f"  {done}/{len(tasks)} folds  ({time.time()-wall0:.0f}s)", flush=True)
    else:
        if args.workers > 1:
            print("  (no fork on this platform; running serial)", flush=True)
        for t in tasks:
            mn, s, f, acc, dt = _fold_worker(t)
            fold_acc[(mn, s, f)] = acc
            fold_cpu[mn] += dt

    # Aggregate exactly like the serial script: folds -> shuffle mean -> mean/std
    results = {}
    for mn in MINERS:
        shuffle_accepts = []
        for s in range(SHUFFLES):
            fold_accepts = [fold_acc[(mn, s, f)] for f in range(K)]
            shuffle_accepts.append(float(np.mean(fold_accepts)))
        results[mn] = (float(np.mean(shuffle_accepts)), float(np.std(shuffle_accepts)))
        cfg = {
            "dataset": name, "miner": mn, "method": "R1accept",
            "method_label": "R1-accept (variant 5-fold x3, perfect replay)",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "host": "local", "seed": SEED,
            "parameters": {"folds": K, "shuffles": SHUFFLES},
            "results": {"accept_mean": results[mn][0], "accept_std": results[mn][1],
                        "shuffle_means": shuffle_accepts,
                        "runtime_s": fold_cpu[mn], "wall_s": time.time() - wall0},
            "notes": "r1_accept.py (parallel folds; reference, budget-exempt)",
        }
        with open(os.path.join(args.output, f"{name}__{mn}__R1accept.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"R1_accept[{mn}] = {results[mn][0]:.4f} +- {results[mn][1]:.4f}"
              f"  (cpu {fold_cpu[mn]:.0f}s)", flush=True)
    print(f"Total wall: {time.time()-wall0:.0f}s", flush=True)

    # Correlate with M1f/M1g gen_accept (from --output first, then configs)
    def pearson(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if x.std() == 0 or y.std() == 0: return float("nan")
        return float(np.corrcoef(x, y)[0, 1])
    def spearman(x, y):
        return pearson(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))

    real = ["Alpha", "Alpha+", "Heuristics", "Heuristics_Strict",
            "Inductive_Infrequent", "Inductive_Strict"]
    for method in ("M1f", "M1g"):
        gen_acc, missing = {}, []
        for m in MINERS:
            v = None
            for d in (args.output, "benchmark/results/configs"):
                p = os.path.join(d, f"{name}__{m}__{method}.json")
                if os.path.exists(p):
                    v = json.load(open(p, encoding="utf-8"))["results"].get("gen_accept")
                    if v is not None:
                        break
            if v is None:
                missing.append(m)
            else:
                gen_acc[m] = v
        if missing:
            print(f"\n{method}: gen_accept missing for {missing} — "
                  f"re-run run_m1_family.py on this dataset first; skipping comparison.")
            continue
        x = [gen_acc[m] for m in real]
        y = [results[m][0] for m in real]
        mae = float(np.mean(np.abs(np.array(x) - np.array(y))))
        print(f"\n{method} gen_accept vs R1_accept (real miners): "
              f"Pearson={pearson(x, y):.3f} Spearman={spearman(x, y):.3f} MAE={mae:.3f}")
        print("  miner                  gen_accept  R1_accept")
        for m in real + ["Trace_Filtered", "Flower"]:
            print(f"  {m:22s} {gen_acc[m]:9.4f}  {results[m][0]:9.4f}")


if __name__ == "__main__":
    main()
