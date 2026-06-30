"""
M1-family — HybridGen methods M1a–M1g (Methodology v2)
=======================================================
Architecture (pure multi-process):
  Phase 1 — Model discovery:   serial in the main process (8 miners)
  Phase 2 — Method evaluation:  56 sub-processes (8 miners × 7 methods),
                                ``--workers`` caps concurrency (default 8).

  Each sub-process sets its own random seed and reads the cached model +
  log via fork-shared globals.  ``evaluate_miner(seed=None)`` skips
  internal re-seeding, so the sub-process-level seed is inherited
  deterministically.  Isolated processes → no thread RNG interference.

Provides run() for job wrappers (job_m1.py). CLI via main().
"""
import os, sys, json, time, signal
# Fix Python hash randomization so set/dict iteration order is deterministic
# across runs.  Without this, PYTHONHASHSEED changes on every interpreter
# startup, making any code that iterates over sets or dicts non-reproducible
# even with the same random.seed().
os.environ.setdefault("PYTHONHASHSEED", "0")
from collections import defaultdict
from datetime import datetime, timezone
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed)

import multiprocessing as mp
import numpy as np
import pm4py
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED = 42
NUM_SHADOW = 1000
ITERATIONS = 5
TRACE_TOP_K = 50

METHODS = {
    "M1a": {"version": "v1.0", "kwargs": {},                                "label": "HybridGen v1.0 (1-gram DFG)"},
    "M1b": {"version": "v2.1", "kwargs": {"max_n": 3},                      "label": "HybridGen v2.1 (N=3)"},
    "M1c": {"version": "v2.1", "kwargs": {"max_n": 6},                      "label": "HybridGen v2.1 (N=6)"},
    "M1d": {"version": "v2.4", "kwargs": {},                                "label": "HybridGen v2.4 (uniform)"},
    "M1e": {"version": "v2.5", "kwargs": {},                                "label": "HybridGen v2.5 (Katz)"},
    "M1f": {"version": "v2.6", "kwargs": {},                                "label": "HybridGen v2.6 (log)"},
    "M1g": {"version": "v2.6", "kwargs": {"successor_weighting": "mle"},    "label": "HybridGen v2.6 (MLE)"},
}

# ── Miners ──────────────────────────────────────────────────────────────────
def _flower(log):
    net = PetriNet("Flower"); p = PetriNet.Place("mid"); net.places.add(p)
    for a in set(e["concept:name"] for t in log for e in t):
        tr = PetriNet.Transition(f"t_{a}", a); net.transitions.add(tr)
        petri_utils.add_arc_from_to(p, tr, net); petri_utils.add_arc_from_to(tr, p, net)
    return net, Marking({p: 1}), Marking({p: 1})

def _trace(log, top_k=TRACE_TOP_K):
    net = PetriNet("Trace"); s = PetriNet.Place("s"); e = PetriNet.Place("e"); net.places.update([s, e])
    vc = defaultdict(int)
    for t in log:
        vc[tuple(x["concept:name"] for x in t)] += 1
    for i, v in enumerate(sorted(vc, key=vc.get, reverse=True)[:top_k]):
        prev = s
        for j, a in enumerate(v):
            tr = PetriNet.Transition(f"t_{i}_{j}", a); net.transitions.add(tr)
            petri_utils.add_arc_from_to(prev, tr, net)
            if j == len(v) - 1:
                petri_utils.add_arc_from_to(tr, e, net)
            else:
                pn = PetriNet.Place(f"p_{i}_{j}"); net.places.add(pn)
                petri_utils.add_arc_from_to(tr, pn, net); prev = pn
    return net, Marking({s: 1}), Marking({e: 1})

MINERS = {
    "Trace_Filtered": lambda l: _trace(l),
    "Alpha": lambda l: pm4py.discover_petri_net_alpha(l),
    "Alpha+": lambda l: pm4py.discover_petri_net_alpha_plus(l),
    "Heuristics": lambda l: pm4py.discover_petri_net_heuristics(l),
    "Heuristics_Strict": lambda l: pm4py.discover_petri_net_heuristics(l, dependency_threshold=0.99),
    "Inductive_Strict": lambda l: pm4py.discover_petri_net_inductive(l, noise_threshold=0.0),
    "Inductive_Infrequent": lambda l: pm4py.discover_petri_net_inductive(l, noise_threshold=0.2),
    "Flower": lambda l: _flower(l),
}

# ── Fork-shared globals (read-only after fork) ──────────────────────────────
_MP_LOG = None
_MP_DNAME = None
_MP_OUTPUT_DIR = None
_MP_CACHE = None          # miner_name -> (net, im, fm)


def _write_one_config(dname, output_dir, mn, mid, spec, r):
    """Write one config JSON (called from sub-processes — safe, unique filenames)."""
    cfg = {
        "dataset": dname, "miner": mn, "method": mid, "method_label": spec["label"],
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": "local", "seed": SEED,
        "parameters": {"algorithm_version": spec["version"],
                       "num_shadow_traces": NUM_SHADOW, "iterations": ITERATIONS,
                       **({"successor_weighting": spec["kwargs"]["successor_weighting"]}
                          if "successor_weighting" in spec["kwargs"] else {})},
        "results": {"mean": r["gen_shadow_mean"], "std": r["gen_shadow_std"],
                    "raw_iterations": r.get("gen_shadow_raw_iterations"), "runtime_s": r["runtime_s"]},
        "notes": "run_m1_family.py",
    }
    path = os.path.join(output_dir, f"{dname}__{mn}__{mid}.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def _eval_worker(miner_name, method_id):
    """Sub-process: evaluate ONE (miner, method) pair.

    Each sub-process:
      1. Sets random.seed(SEED) + np.random.seed(SEED)  ← isolated RNG
      2. Reads cached model from fork-shared _MP_CACHE
      3. Calls evaluate_miner(seed=None)  ← skips internal re-seed
      4. Writes config JSON

    Returns (miner_name, method_id, result_dict).
    """
    import random
    random.seed(SEED)
    np.random.seed(SEED)

    from HybridGen.algorithm import load_algorithm

    spec = METHODS[method_id]
    algo = load_algorithm(spec["version"])
    model_triple = _MP_CACHE[miner_name]

    def _cached_miner(_log):
        return model_triple

    r = algo.evaluate_miner(_MP_LOG, miner_name, _cached_miner,
                            num_shadow_traces=NUM_SHADOW,
                            iterations=ITERATIONS, seed=None,
                            **spec["kwargs"])

    _write_one_config(_MP_DNAME, _MP_OUTPUT_DIR, miner_name, method_id, spec, r)
    return miner_name, method_id, r


def run(dataset_key, workdir, output_dir, methods=None, miners=None, workers=8):
    """Run M1a–M1g in pure multi-process architecture.

    Phase 1: serial model discovery for all 8 miners in the main process.
    Phase 2: submit every (miner, method) pair to a process pool
             (``workers`` caps concurrency, default 8).

    If ``miners`` is provided (list of miner names), only those miners
    are run (see ``benchmark/statistics/_miner_availability.json``).
    """
    from job_prepare import prepare_workdir

    # ── Filter miners if subset requested ────────────────────────────────
    _miners_dict = MINERS
    if miners:
        _miners_dict = {k: v for k, v in MINERS.items() if k in miners}
        missing = set(miners) - set(MINERS.keys())
        if missing:
            print(f"  Warning: unknown miner(s): {missing}")
    _miner_names = list(_miners_dict.keys())
    mp_path = os.path.join(workdir, "manifest.json")
    if os.path.exists(mp_path):
        with open(mp_path) as f:
            mf = json.load(f)
        dname, xes_path = mf["dataset"], mf["xes_file"]
    else:
        ctx = prepare_workdir(workdir, dataset_key, copy_xes=True)
        dname, xes_path = ctx["dataset_name"], ctx["xes_path"]

    print("=" * 78)
    print(f"M1-family — {dataset_key} {dname}, seed {SEED}, {ITERATIONS}x{NUM_SHADOW}")
    print("=" * 78)

    log = pm4py.read_xes(xes_path)
    log = pm4py.convert_to_event_log(log)
    print(f"Loaded {len(log)} traces")

    target_methods = methods or list(METHODS.keys())
    n_miners = len(_miner_names)
    n_tasks = n_miners * len(target_methods)
    n_workers = min(workers, n_tasks, os.cpu_count() or 8)
    os.makedirs(output_dir, exist_ok=True)

    # ── Phase 1: parallel model discovery (8 threads) ───────────────────
    print("\nDiscovering models (8 threads):")

    def _discover_one(mn_fn):
        mn, fn = mn_fn
        t0 = time.time()
        result = fn(log)
        return mn, result, time.time() - t0

    cache = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(_discover_one, (mn, fn)) for mn, fn in _miners_dict.items()]
        for f in as_completed(futs):
            mn, result, elapsed = f.result()
            cache[mn] = result
            net = result[0]
            print(f"  {mn:22s} {len(net.transitions)}t/{len(net.places)}p  ({elapsed:.1f}s)")

    # ── Set fork-shared globals ──────────────────────────────────────────
    global _MP_LOG, _MP_DNAME, _MP_OUTPUT_DIR, _MP_CACHE
    _MP_LOG = log
    _MP_DNAME = dname
    _MP_OUTPUT_DIR = output_dir
    _MP_CACHE = cache

    # ── Phase 2: submit all (miner, method) pairs to process pool ────────
    print(f"\n--- Evaluating {n_tasks} (miner×method) tasks ({n_workers} workers) ---")

    all_results = {}
    ctx = mp.get_context("fork")
    old_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            futs = {}
            for mn in _miner_names:
                for mid in target_methods:
                    f = ex.submit(_eval_worker, mn, mid)
                    futs[f] = (mn, mid)
            for f in as_completed(futs):
                mn, mid, r = f.result()
                all_results[(mid, mn)] = r
                # Progress indicator: print first miner when a method finishes
                first_mn = _miner_names[0]
                if mn == first_mn:
                    print(f"  ✓ {mid} ({METHODS[mid]['label']})  "
                          f"{first_mn}={r.get('gen_shadow_mean', '?'):.4f}", flush=True)
    finally:
        signal.signal(signal.SIGINT, old_handler)

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 98)
    print(f"SUMMARY — {dataset_key} {dname}")
    h = f"{'Method':6s} " + " ".join(f"{m[:13]:>15s}" for m in _miner_names)
    print(h); print("-" * 98)
    for mid in target_methods:
        cells = [all_results.get((mid, m), {}) for m in _miner_names]
        vals = " ".join(
            f"{c.get('gen_shadow_mean', -1):>7.4f}+-{c.get('gen_shadow_std', -1):<6.4f}"
            if c else "     ---     "
            for c in cells
        )
        print(f"{mid:6s} {vals}")
    print("=" * 98)
    print(f"Configs → {output_dir}/")

    return all_results


def main():
    import argparse, shutil, secrets
    from datetime import datetime as dt
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--methods", nargs="+", default=list(METHODS.keys()), choices=list(METHODS.keys()))
    ap.add_argument("--miners", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=8, help="Parallel workers (default: 8)")
    args = ap.parse_args()
    workdir = f"/tmp/benchmark_M1_{args.dataset}_{dt.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}/"
    output_dir = args.output or os.path.join(workdir, "results"); os.makedirs(output_dir, exist_ok=True)
    run(args.dataset, workdir, output_dir, methods=args.methods, miners=args.miners, workers=args.workers)
    shutil.rmtree(workdir); print(f"  [clean] removed {workdir}")


if __name__ == "__main__":
    main()
