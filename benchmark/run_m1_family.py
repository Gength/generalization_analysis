"""
M1-family — HybridGen methods M1a–M1g (Methodology v2)
=======================================================
Parallelized: each algorithm version runs on its own core.
Default parallelism: 8 workers (--workers).
Provides run() for job wrappers. CLI via main().
"""
import os, sys, json, time, random, argparse, signal
from collections import defaultdict
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED

import multiprocessing as mp
import numpy as np
import pm4py
from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness

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
REAL = ["Alpha", "Alpha+", "Heuristics", "Heuristics_Strict", "Inductive_Infrequent", "Inductive_Strict"]

# ── Module-level globals for fork sharing (read-only after fork) ────────────
_SHARED_LOG = None
_SHARED_CACHE = None       # miner_name -> (net, im, fm)
_SHARED_DNAME = None
_SHARED_OUTPUT_DIR = None
_SHARED_SEED = None


def _write_one_config(dname, output_dir, mn, mid, spec, r):
    """Write one config JSON."""
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


def _m1_worker(method_id):
    """Worker process: run one method version across all 8 miners.

    Accesses module-level globals (_SHARED_LOG, _SHARED_CACHE) inherited via fork.
    Returns dict of results keyed by (method_id, miner_name).
    """
    from HybridGen.algorithm import load_algorithm
    spec = METHODS[method_id]
    algo = load_algorithm(spec["version"])
    results = {}
    for mn in MINERS:
        # Build a callable that returns the cached model for this miner
        miner_fn = lambda l, n=mn: _SHARED_CACHE[n]
        r = algo.evaluate_miner(_SHARED_LOG, mn, miner_fn,
                                num_shadow_traces=NUM_SHADOW,
                                iterations=ITERATIONS, seed=_SHARED_SEED,
                                **spec["kwargs"])
        results[(method_id, mn)] = r
        # Write config from worker (unique filename, no race)
        _write_one_config(_SHARED_DNAME, _SHARED_OUTPUT_DIR, mn, method_id, spec, r)
    return method_id, results


def run(dataset_key, workdir, output_dir, methods=None, workers=8):
    """Run M1a–M1g in parallel. Reads XES from workdir, writes configs to output_dir."""
    from job_prepare import prepare_workdir
    mp_path = os.path.join(workdir, "manifest.json")
    if os.path.exists(mp_path):
        with open(mp_path) as f:
            mf = json.load(f)
        dname, xes_path = mf["dataset"], mf["xes_file"]
    else:
        ctx = prepare_workdir(workdir, dataset_key, mode="minimal")
        dname, xes_path = ctx["dataset_name"], ctx["xes_path"]

    print("=" * 78)
    print(f"M1-family — {dataset_key} {dname}, seed {SEED}, {ITERATIONS}x{NUM_SHADOW}")
    print("=" * 78)

    log = pm4py.read_xes(xes_path)
    log = pm4py.convert_to_event_log(log)
    print(f"Loaded {len(log)} traces")

    # ── Serial: discover models ───────────────────────────────────────────
    print("\nDiscovering models:")
    cache = {}
    for mn, fn in MINERS.items():
        t0 = time.time()
        cache[mn] = fn(log)
        net = cache[mn][0]
        print(f"  {mn:22s} {len(net.transitions)}t/{len(net.places)}p  ({time.time()-t0:.1f}s)")

    # ── Serial: R1 ground truth ───────────────────────────────────────────
    print("\nGround truth R1:")
    r1 = {}
    for mn, fn in MINERS.items():
        t0 = time.time()
        random.seed(SEED); np.random.seed(SEED)
        vm = defaultdict(list)
        for t in log:
            vm[tuple(e["concept:name"] for e in t)].append(t)
        vs = list(vm.keys())
        K, SH = 5, 3
        all_fits = []
        for _ in range(SH):
            random.shuffle(vs)
            fs = max(1, len(vs) // K)
            folds = []
            for i in range(K):
                s, e2 = i * fs, (i + 1) * fs if i < K - 1 else len(vs)
                tv = set(vs[s:e2])
                train = EventLog([t for v in vs if v not in tv for t in vm[v]])
                test = EventLog([t for v in tv for t in vm[v]])
                try:
                    net, im, fm = fn(train)
                    fit = replay_fitness.apply(test, net, im, fm,
                                               variant=replay_fitness.Variants.TOKEN_BASED)["log_fitness"]
                    folds.append(fit)
                except Exception:
                    folds.append(0.0)
            all_fits.append(float(np.mean(folds)))
        mean, std = float(np.mean(all_fits)), float(np.std(all_fits))
        r1[mn] = mean
        print(f"  R1[{mn}] = {mean:.4f}  ({(time.time()-t0):.0f}s)")
        r1cfg = {
            "dataset": dname, "miner": mn, "method": "R1",
            "method_label": "K-Fold CV (k=5)",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "host": "local", "seed": SEED,
            "parameters": {"k": 5, "shuffles": 3, "variant_based": True},
            "results": {"mean": mean, "std": std, "raw_shuffles": all_fits, "runtime_s": time.time()-t0},
            "notes": "computed by run_m1_family.py",
        }
        with open(os.path.join(output_dir, f"{dname}__{mn}__R1.json"), "w") as f:
            json.dump(r1cfg, f, indent=2)

    os.makedirs(output_dir, exist_ok=True)
    target_methods = methods or list(METHODS.keys())
    t_par_start = time.time()

    if len(target_methods) <= 1:
        # Serial path (single method) — no overhead
        results = {}
        for mid in target_methods:
            from HybridGen.algorithm import load_algorithm
            spec = METHODS[mid]
            algo = load_algorithm(spec["version"])
            print(f"\n--- {mid}: {spec['label']} ---")
            for mn in MINERS:
                cfn = lambda l, n=mn: cache[n]
                r = algo.evaluate_miner(log, mn, cfn, num_shadow_traces=NUM_SHADOW,
                                        iterations=ITERATIONS, seed=SEED, **spec["kwargs"])
                results[(mid, mn)] = r
                _write_one_config(dname, output_dir, mn, mid, spec, r)
                print(f"    {mn:22s} {r['gen_shadow_mean']:.4f} +- {r['gen_shadow_std']:.4f}  ({r['runtime_s']:.1f}s)")
    else:
        # Parallel: set globals → fork workers
        global _SHARED_LOG, _SHARED_CACHE, _SHARED_DNAME, _SHARED_OUTPUT_DIR, _SHARED_SEED
        _SHARED_LOG = log
        _SHARED_CACHE = cache
        _SHARED_DNAME = dname
        _SHARED_OUTPUT_DIR = output_dir
        _SHARED_SEED = SEED

        n_workers = min(workers, len(target_methods), os.cpu_count() or 8)
        print(f"\n--- Parallel evaluation ({n_workers} workers, {len(target_methods)} methods) ---")
        ctx = mp.get_context("fork")
        results = {}
        old_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
                futs = {ex.submit(_m1_worker, mid): mid for mid in target_methods}
                pending = set(futs.keys())
                while pending:
                    done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                    for f in done:
                        mid, m_results = f.result()
                        results.update(m_results)
                        # Log first miner result for this method as progress indicator
                        first_mn = next(iter(MINERS))
                        first_r = m_results.get((mid, first_mn), {})
                        print(f"  ✓ {mid} ({METHODS[mid]['label']})  "
                              f"{first_mn}={first_r.get('gen_shadow_mean', '?'):.4f}  "
                              f"({len(MINERS)} miners)", flush=True)
                ex.shutdown(wait=True)
        finally:
            signal.signal(signal.SIGINT, old_handler)

        print(f"\n  Parallel eval: {time.time()-t_par_start:.1f}s")

    # ── Summary ───────────────────────────────────────────────────────────
    def _pearson(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if x.std() == 0 or y.std() == 0: return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    print("\n" + "=" * 98)
    print(f"SUMMARY — {dataset_key} {dname}")
    h = f"{'Method':6s} " + " ".join(f"{m[:13]:>15s}" for m in MINERS)
    print(h); print("-" * 98)
    print(f"{'R1':6s} " + " ".join(f"{r1[m]:>15.4f}" for m in MINERS))
    for mid in target_methods:
        cells = [results[(mid, m)] for m in MINERS]
        print(f"{mid:6s} " + " ".join(f"{c['gen_shadow_mean']:>7.4f}+-{c['gen_shadow_std']:<6.4f}" for c in cells))

    print(f"\nAgreement with R1 (6 real miners):")
    print(f"{'Method':6s} {'Pearson':>8s} {'Spearman':>9s} {'MAE':>7s} {'Flower':>7s}")
    y = [r1[m] for m in REAL]
    for mid in target_methods:
        x = [results[(mid, m)]["gen_shadow_mean"] for m in REAL]
        ma = float(np.mean(np.abs(np.array(x) - np.array(y))))
        fl = results[(mid, "Flower")]["gen_shadow_mean"]
        sp = _pearson(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))
        print(f"{mid:6s} {_pearson(x, y):8.3f} {sp:9.3f} {ma:7.3f} {fl:7.3f}")
    print("=" * 98)
    print(f"Configs → {output_dir}/")


def main():
    import argparse, shutil, secrets
    from datetime import datetime as dt
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--methods", nargs="+", default=list(METHODS.keys()), choices=list(METHODS.keys()))
    ap.add_argument("--workers", type=int, default=8, help="Parallel workers (default: 8)")
    args = ap.parse_args()
    workdir = f"/tmp/benchmark_M1_{args.dataset}_{dt.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}/"
    output_dir = args.output or os.path.join(workdir, "results"); os.makedirs(output_dir, exist_ok=True)
    run(args.dataset, workdir, output_dir, methods=args.methods, workers=args.workers)
    shutil.rmtree(workdir); print(f"  [clean] removed {workdir}")


if __name__ == "__main__":
    main()
