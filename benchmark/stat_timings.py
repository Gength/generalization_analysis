"""
stat_timings — Measure model-discovery time for all 8 miners × datasets.

Output: per-dataset JSON files in benchmark/statistics/{DATASET_NAME}.json

Design:
  - Each miner runs in its own subprocess (to allow hard kill on timeout).
  - Sequential when workers=1, parallel when workers>1.
  - Per-dataset JSON → no file-locking issues when multiple nodes run.

Usage:
  # Local: single dataset, sequential
  uv run python benchmark/stat_timings.py --dataset D1

  # All datasets, sequential
  uv run python benchmark/stat_timings.py --all

  # HPC node: one dataset, 8 miners in parallel
  uv run python benchmark/stat_timings.py --dataset D3 --workers 8

  # Print summary table from existing results
  uv run python benchmark/stat_timings.py --summary
"""
import os, sys, json, time, argparse, multiprocessing
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Lazy-import miners here only for miner-name list (names only, not the fns)
from miners import MINERS as _MINERS_DICT
from datasets import DATASETS

# ── Configuration ────────────────────────────────────────────────────────
SKIP_DATASETS = {}
TIMEOUT = 3600  # 1 hour per miner
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "statistics")

# Stabilise miner order: fast synthetic miners last
_MINER_NAMES = sorted(_MINERS_DICT.keys(), key=lambda n: (
    1 if n in ("Flower", "Trace_Filtered") else 0, n
))


def _worker(dataset_key, log_path, miner_name, result_queue):
    """Run model discovery in a child process (module-level for pickling)."""
    import sys
    sys.path.insert(0, SCRIPT_DIR)
    from miners import MINERS
    import pm4py

    t0 = time.time()
    try:
        log = pm4py.read_xes(log_path)
        log = pm4py.convert_to_event_log(log)
        net, im, fm = MINERS[miner_name](log)
        elapsed = time.time() - t0
        result_queue.put({
            "status": "ok",
            "time_seconds": round(elapsed, 2),
            "n_transitions": len(net.transitions),
            "n_places": len(net.places),
        })
    except Exception as e:
        elapsed = time.time() - t0
        result_queue.put({
            "status": "error",
            "time_seconds": round(elapsed, 2),
            "error": repr(e),
        })


def discover_miner(dataset_key, log_path, miner_name, timeout=TIMEOUT):
    """Run ONE miner with a wall-clock timeout. Returns result dict."""
    queue = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_worker,
        args=(dataset_key, log_path, miner_name, queue),
    )
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return {
            "status": "timeout",
            "time_seconds": timeout,
            "error": f"exceeded {timeout}s limit",
        }

    try:
        return queue.get_nowait()
    except Exception:
        return {
            "status": "error",
            "time_seconds": -1,
            "error": "process exited without result",
        }


def run_dataset(dataset_key, workers=1):
    """Run all miners on one dataset, write per-dataset JSON."""
    if dataset_key in SKIP_DATASETS:
        print(f"[{dataset_key}] SKIP (known model-discovery issues)")
        return None

    ds = DATASETS[dataset_key]
    dname = ds["name"]
    log_path = ds["log_path"]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"[{dataset_key}] {dname}")
    print(f"       log: {log_path}")
    print(f"       workers: {workers}  |  timeout: {TIMEOUT}s")
    print(f"{'='*70}")

    results = {}

    if workers <= 1:
        # ── Sequential ────────────────────────────────────────────────
        for mn in _MINER_NAMES:
            print(f"  [{mn:22s}] discovering...", end=" ", flush=True)
            res = discover_miner(dataset_key, log_path, mn)
            tag = {"ok": "✓", "timeout": "⌛", "error": "✗"}.get(res["status"], "?")
            t = res["time_seconds"]
            err = res.get("error", "")
            print(f"{tag}  {t:>8.1f}s  {err}")
            results[mn] = res
    else:
        # ── Parallel (launch all, poll until all done) ────────────────
        pending = {}
        for mn in _MINER_NAMES:
            q = multiprocessing.Queue()
            p = multiprocessing.Process(
                target=_worker,
                args=(dataset_key, log_path, mn, q),
            )
            p.start()
            pending[mn] = (p, q)
            print(f"  [{mn:22s}] launched")

        deadline = time.time() + TIMEOUT
        while pending and time.time() < deadline:
            time.sleep(1.0)
            for mn in list(pending.keys()):
                proc, queue = pending[mn]
                if not proc.is_alive():
                    proc.join(timeout=2)
                    try:
                        res = queue.get_nowait()
                    except Exception:
                        res = {
                            "status": "error",
                            "time_seconds": -1,
                            "error": "no result from process",
                        }
                    tag = {"ok": "✓", "timeout": "⌛", "error": "✗"}.get(res["status"], "?")
                    t = res["time_seconds"]
                    err = res.get("error", "")
                    print(f"  [{mn:22s}] {tag}  {t:>8.1f}s  {err}")
                    results[mn] = res
                    del pending[mn]

        # Any remaining = timeout
        for mn, (proc, _queue) in pending.items():
            proc.terminate()
            proc.join()
            res = {
                "status": "timeout",
                "time_seconds": TIMEOUT,
                "error": f"exceeded {TIMEOUT}s limit",
            }
            print(f"  [{mn:22s}] ⌛  {TIMEOUT:>8.1f}s  timeout")
            results[mn] = res

    # ── Write JSON ──────────────────────────────────────────────────────
    out = {
        "dataset_key": dataset_key,
        "dataset_name": dname,
        "timestamp": datetime.now().isoformat(),
        "timeout_seconds": TIMEOUT,
        "miners": results,
    }
    out_path = os.path.join(OUTPUT_DIR, f"{dname}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  → {out_path}")

    # ── Short summary ──────────────────────────────────────────────────
    ok_n = sum(1 for r in results.values() if r["status"] == "ok")
    to_n = sum(1 for r in results.values() if r["status"] == "timeout")
    er_n = sum(1 for r in results.values() if r["status"] == "error")
    print(f"  summary: {ok_n} ok, {to_n} timeout, {er_n} error / {len(results)} miners")

    return out


def summary_table(results_dir=OUTPUT_DIR):
    """Read all result JSONs and print a Markdown summary table."""
    import glob
    jsons = sorted(j for j in glob.glob(os.path.join(results_dir, "*.json"))
                   if not os.path.basename(j).startswith("_"))
    if not jsons:
        print("No results found in", results_dir)
        return

    miners = _MINER_NAMES

    # Header
    cols = "| Dataset | " + " | ".join(miners) + " |"
    sep = "|" + ":" + "-" * 5 + "-|" + ":".join("-" * max(len(m), 5) for m in miners) + ":|"
    print()
    print("# Model-discovery timing summary")
    print()
    print(cols)
    print(sep)

    for jp in jsons:
        with open(jp) as f:
            data = json.load(f)
        dk = data["dataset_key"]
        row = f"| {dk} "
        for mn in miners:
            r = data["miners"].get(mn, {})
            st = r.get("status", "?")
            t = r.get("time_seconds", -1)
            if st == "ok":
                row += f"| {t:.1f}s "
            elif st == "timeout":
                row += f"| >{TIMEOUT//60}m "
            elif st == "error":
                err_short = str(r.get("error", "err"))[:18]
                row += f"| err:{err_short} "
            else:
                row += "| ? "
        row += "|"
        print(row)

    # Footer: if any timeouts, note which miners are problematic
    timeout_miners = set()
    error_miners = set()
    for jp in jsons:
        with open(jp) as f:
            data = json.load(f)
        for mn, r in data["miners"].items():
            if r.get("status") == "timeout":
                timeout_miners.add(mn)
            elif r.get("status") == "error":
                error_miners.add(mn)
    if timeout_miners:
        print(f"\n_Timeout on: {', '.join(sorted(timeout_miners))}_")
    if error_miners:
        print(f"_Error on: {', '.join(sorted(error_miners))}_")


def format_config_json(results_dir=OUTPUT_DIR):
    """Generate a config JSON that records per-dataset miner availability.

    The config maps dataset_key → list of miner names that succeeded
    within timeout.  This can be consumed by benchmark jobs to skip
    miners known to time out on specific datasets.
    """
    import glob
    jsons = sorted(j for j in glob.glob(os.path.join(results_dir, "*.json"))
                   if not os.path.basename(j).startswith("_"))
    if not jsons:
        return None

    config = {}
    for jp in jsons:
        with open(jp) as f:
            data = json.load(f)
        dk = data["dataset_key"]
        good = []
        for mn, r in data["miners"].items():
            if r.get("status") == "ok":
                good.append(mn)
        config[dk] = {
            "dataset_name": data["dataset_name"],
            "available_miners": good,
        }
    return config


# ── CLI ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Measure model-discovery timings per miner×dataset")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", type=str, default=None, help="Dataset key, e.g. D1")
    group.add_argument("--all", action="store_true", help="Run on all datasets")
    group.add_argument("--summary", action="store_true", help="Print summary table from existing results")
    ap.add_argument("--workers", type=int, default=8, help="Parallel workers (default: 8)")
    ap.add_argument("--timeout", type=int, default=TIMEOUT, help=f"Per-miner timeout seconds (default: {TIMEOUT})")
    args = ap.parse_args()

    if args.summary:
        summary_table()
        sys.exit(0)

    workers = max(1, args.workers)
    print(f"Settings: workers={workers}, timeout={args.timeout}s, output={OUTPUT_DIR}")
    print(f"Miners ({len(_MINER_NAMES)}): {', '.join(_MINER_NAMES)}")

    if args.all:
        for dk in sorted(DATASETS.keys()):
            if dk in SKIP_DATASETS:
                print(f"[{dk}] SKIP (known model-discovery issues)")
                continue
            run_dataset(dk, workers=workers)
    else:
        run_dataset(args.dataset, workers=workers)

    # Summary
    summary_table()

    # Also write a convenience config JSON
    cfg = format_config_json()
    if cfg:
        cfg_path = os.path.join(OUTPUT_DIR, "_miner_availability.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"\nConfig (miner availability): {cfg_path}")
