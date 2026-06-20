"""
R-family — Reference / Sanity-Check Metrics (R1–R3)
====================================================
Three independent functions: run_r1(), run_r2(), run_r3().
Each is fully self-contained (discovers models, runs, writes configs).
R2 is parallelized with ProcessPoolExecutor (fork context).
"""
import os, sys, json, time, random, signal
from collections import defaultdict
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED

import multiprocessing as mp
import numpy as np
import pm4py
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miners import MINERS
from utils import compute_kfold_fitness

SEED = 42

# ── R2 module-level globals (shared with worker processes via fork) ────────
_R2_LOG = None          # EventLog (full)
_R2_VARIANT_MAP = None  # dict: variant_tuple -> list[Trace]
_R2_MINER_FN = None     # miner discovery function
_R2_INTERRUPTED = False


def _load_workdir(workdir, dataset_key):
    """Load manifest and XES log from workdir. Returns (dname, log)."""
    mp = os.path.join(workdir, "manifest.json")
    if os.path.exists(mp):
        with open(mp) as f:
            mf = json.load(f)
        dname, xes_path = mf["dataset"], mf["xes_file"]
    else:
        from job_prepare import prepare_workdir
        ctx = prepare_workdir(workdir, dataset_key, mode="minimal")
        dname, xes_path = ctx["dataset_name"], ctx["xes_path"]
    log = pm4py.read_xes(xes_path)
    log = pm4py.convert_to_event_log(log)
    return dname, log


def _write_config(output_dir, dname, miner, method_id, label, params, results, notes=""):
    cfg = {
        "dataset": dname, "miner": miner, "method": method_id, "method_label": label,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": "local", "seed": SEED,
        "parameters": params, "results": results, "notes": notes,
    }
    path = os.path.join(output_dir, f"{dname}__{miner}__{method_id}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"    ✓ {path}")


def run_r1(dataset_key, workdir, output_dir, seed=SEED, miners=None):
    """R1 — K-Fold CV (k=5, variant-based, 3 shuffles).

    Reads XES from workdir, discovers models inline, writes configs to output_dir.
    """
    dname, log = _load_workdir(workdir, dataset_key)
    random.seed(seed); np.random.seed(seed)
    print(f"R1 — K-Fold CV (k=5, 3 shuffles) — {dname}  ({len(log)} traces)")

    target = dict(MINERS) if miners is None else {k: v for k, v in MINERS.items() if k in miners}
    os.makedirs(output_dir, exist_ok=True)

    for mn, fn in target.items():
        t0 = time.time()
        fits = [compute_kfold_fitness(log, fn, k=5) for _ in range(3)]
        mean, std = float(np.mean(fits)), float(np.std(fits))
        rt = time.time() - t0
        print(f"  {mn:22s} {mean:.4f} ± {std:.4f} ({rt:.1f}s)")
        _write_config(output_dir, dname, mn, "R1", "K-Fold CV (k=5)",
                      {"k": 5, "shuffles": 3, "variant_based": True},
                      {"mean": mean, "std": std, "raw_shuffles": fits, "runtime_s": rt})
    print(f"\nDone → {output_dir}/")


def _r2_sigint_handler(signum, frame):
    """Set interrupt flag for graceful shutdown."""
    global _R2_INTERRUPTED
    _R2_INTERRUPTED = True
    raise KeyboardInterrupt


def _r2_worker_task(held_out_variant):
    """Worker process task: compute fitness for one held-out variant.

    Reads module-level globals (_R2_LOG, _R2_VARIANT_MAP, _R2_MINER_FN)
    which are inherited via fork. No pickling of complex objects needed.
    """
    global _R2_VARIANT_MAP, _R2_MINER_FN
    train_traces = []
    for v, traces in _R2_VARIANT_MAP.items():
        if v != held_out_variant:
            train_traces.extend(traces)
    test_traces = _R2_VARIANT_MAP[held_out_variant]

    train_log = EventLog(train_traces)
    test_log = EventLog(test_traces)
    try:
        net, im, fm = _R2_MINER_FN(train_log)
        rp = token_replay.apply(test_log, net, im, fm)
        return sum(r["trace_fitness"] for r in rp) / len(rp) if rp else 0.0
    except Exception:
        return 0.0


def run_r2(dataset_key, workdir, output_dir, seed=SEED, miners=None,
           sample_n=0, workers=8):
    """R2 — Leave-One-Variant-Out Fitness (parallelized).

    Uses ProcessPoolExecutor with fork context. Workers share the event log
    and variant map via module-level globals (zero-copy after fork).
    """
    dname, log = _load_workdir(workdir, dataset_key)
    random.seed(seed); np.random.seed(seed)
    lab = "Leave-One-Variant-Out" + (f" (sampled {sample_n})" if sample_n > 0 else "")
    print(f"R2 — {lab} — {dname}  ({len(log)} traces)")

    target = dict(MINERS) if miners is None else {k: v for k, v in MINERS.items() if k in miners}
    os.makedirs(output_dir, exist_ok=True)

    # Group by variant (once, shared across miners)
    vg = defaultdict(list)
    for t in log:
        vg[tuple(e["concept:name"] for e in t)].append(t)
    all_v = list(vg.keys())
    test_v = random.sample(all_v, sample_n) if 0 < sample_n < len(all_v) else all_v
    print(f"  Variants: {len(all_v)} total, testing {len(test_v)} with {workers} workers")

    n_workers = min(workers, len(test_v), os.cpu_count() or 8)
    ctx = mp.get_context("fork")

    for mn, fn in target.items():
        # Set module-level globals for worker processes (inherited via fork)
        global _R2_LOG, _R2_VARIANT_MAP, _R2_MINER_FN
        _R2_LOG = log
        _R2_VARIANT_MAP = vg
        _R2_MINER_FN = fn

        t0 = time.time()
        fits = []
        global _R2_INTERRUPTED
        _R2_INTERRUPTED = False
        old_handler = signal.signal(signal.SIGINT, _r2_sigint_handler)

        try:
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
                futs = {ex.submit(_r2_worker_task, v): v for v in test_v}
                pending = set(futs.keys())
                done_count = 0
                total = len(pending)
                while pending:
                    if _R2_INTERRUPTED:
                        raise KeyboardInterrupt
                    done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                    for f in done:
                        try:
                            fits.append(f.result())
                        except Exception:
                            fits.append(0.0)
                        done_count += 1
                        if done_count % max(1, total // 10) == 0 or done_count == total:
                            print(f"    {mn:22s}  {done_count}/{total} variants done "
                                  f"({100*done_count//total}%)", flush=True)
                ex.shutdown(wait=True)
        except KeyboardInterrupt:
            print(f"\n  ⚠ Interrupted — {done_count}/{total} variants evaluated for {mn}")
            if fits:
                mean = float(np.mean(fits))
                std = float(np.std(fits))
                rt = time.time() - t0
                _write_config(output_dir, dname, mn, "R2", lab,
                              {"k": len(fits), "variants_sampled": len(fits), "variant_based": True},
                              {"mean": mean, "std": std, "raw_fits": fits, "runtime_s": rt},
                              notes=f"Interrupted after {done_count}/{total} variants")
            return
        finally:
            signal.signal(signal.SIGINT, old_handler)

        mean, std = float(np.mean(fits)), float(np.std(fits))
        rt = time.time() - t0
        print(f"  {mn:22s} {mean:.4f} ± {std:.4f} ({rt:.1f}s, n={len(fits)})")
        _write_config(output_dir, dname, mn, "R2", lab,
                      {"k": len(fits), "variants_sampled": len(fits), "variant_based": True},
                      {"mean": mean, "std": std, "raw_fits": fits, "runtime_s": rt})
    print(f"\nDone → {output_dir}/")


def run_r3(dataset_key, workdir, output_dir, seed=SEED, miners=None, num_traces=1000):
    """R3 — Naive Random Baseline.

    Reads XES from workdir, discovers models inline, writes configs to output_dir.
    """
    dname, log = _load_workdir(workdir, dataset_key)
    random.seed(seed); np.random.seed(seed)
    print(f"R3 — Random Baseline ({num_traces} traces, 5 iters) — {dname}  ({len(log)} traces)")

    target = dict(MINERS) if miners is None else {k: v for k, v in MINERS.items() if k in miners}
    os.makedirs(output_dir, exist_ok=True)

    acts = list(set(e["concept:name"] for t in log for e in t))
    lengths = [len(t) for t in log]

    for mn, fn in target.items():
        # Discover model
        net, im, fm = fn(log)
        t0 = time.time()
        scores = []
        for _ in range(5):
            shadow = EventLog()
            for i in range(num_traces):
                seq = random.choices(acts, k=random.choice(lengths))
                tr = Trace(attributes={"concept:name": f"r_{i}"})
                for a in seq:
                    tr.append(Event({"concept:name": a}))
                shadow.append(tr)
            try:
                rp = token_replay.apply(shadow, net, im, fm)
                fits = [r["trace_fitness"] for r in rp]
                scores.append(sum(fits) / len(fits) if fits else 0.0)
            except Exception:
                scores.append(0.0)
        mean, std = float(np.mean(scores)), float(np.std(scores))
        rt = time.time() - t0
        print(f"  {mn:22s} {mean:.4f} ± {std:.4f} ({rt:.1f}s)")
        _write_config(output_dir, dname, mn, "R3", "Naive Random Baseline",
                      {"num_traces": num_traces, "iterations": 5},
                      {"mean": mean, "std": std, "runtime_s": rt})
    print(f"\nDone → {output_dir}/")


def main():
    import argparse, shutil, secrets
    from datetime import datetime as dt

    ap = argparse.ArgumentParser(description="R-family (R1–R3)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--method", required=True, choices=["R1", "R2", "R3"])
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--miners", nargs="*", default=None)
    ap.add_argument("--r2-sample", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8, help="Worker processes for R2 (default: 8)")
    ap.add_argument("--num-traces", type=int, default=1000)
    args = ap.parse_args()

    workdir = f"/tmp/benchmark_{args.method}_{args.dataset}_{dt.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}/"
    output_dir = args.output or os.path.join(workdir, "results")
    os.makedirs(output_dir, exist_ok=True)

    if args.method == "R1":
        run_r1(args.dataset, workdir, output_dir, seed=args.seed, miners=args.miners)
    elif args.method == "R2":
        run_r2(args.dataset, workdir, output_dir, seed=args.seed, miners=args.miners,
               sample_n=args.r2_sample, workers=args.workers)
    elif args.method == "R3":
        run_r3(args.dataset, workdir, output_dir, seed=args.seed, miners=args.miners, num_traces=args.num_traces)

    shutil.rmtree(workdir)
    print(f"  [clean] removed {workdir}")


if __name__ == "__main__":
    main()
