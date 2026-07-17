"""
M2 — PM4Py Built-in Generalization
===================================
Provides run() for job_prepare. CLI via main().
"""
import os, sys, json, time, argparse
from datetime import datetime, timezone

import pm4py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miners import MINERS


def run(dataset_key, workdir, output_dir, miners=None):
    """Run M2. Reads XES from workdir, writes configs to output_dir."""
    mp = os.path.join(workdir, "manifest.json")
    if os.path.exists(mp):
        with open(mp) as f:
            mf = json.load(f)
        dname, xes_path = mf["dataset"], mf["xes_file"]
    else:
        from job_prepare import prepare_workdir
        ctx = prepare_workdir(workdir, dataset_key, copy_xes=True)
        dname, xes_path = ctx["dataset_name"], ctx["xes_path"]

    log = pm4py.read_xes(xes_path)
    log = pm4py.convert_to_event_log(log)
    print(f"M2 — {dname}  ({len(log)} traces)")

    target = {k: v for k, v in MINERS.items() if miners is None or k in miners}

    for name, fn in target.items():
        t0 = time.time()
        try:
            net, im, fm = fn(log)
            score = pm4py.algo.evaluation.generalization.algorithm.apply(log, net, im, fm)
            rt = time.time() - t0
            print(f"  [{name}] score={score:.4f} ({rt:.2f}s)")
        except Exception as e:
            score = -1
            rt = time.time() - t0
            print(f"  [{name}] ERROR: {e}")

        config = {
            "dataset": dname, "miner": name, "method": "M2",
            "method_label": "PM4Py Built-in Gen",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "host": "local", "seed": 42,
            "parameters": {},
            "results": {"score": score, "runtime_s": rt},
            "notes": "",
        }
        path = os.path.join(output_dir, f"{dname}__{name}__M2.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump(config, f, indent=2)

    print(f"\nDone! {len(target)} configs → {output_dir}/")


def main():
    ap = argparse.ArgumentParser(description="M2 PM4Py Built-in Gen")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--miners", nargs="*", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    import tempfile, shutil, secrets
    from datetime import datetime as dt
    workdir = f"/tmp/benchmark_M2_{args.dataset}_{dt.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}/"
    output_dir = args.output or os.path.join(workdir, "results")

    run(args.dataset, workdir, output_dir, miners=args.miners)
    shutil.rmtree(workdir)
    print(f"  [clean] removed {workdir}")


if __name__ == "__main__":
    main()
