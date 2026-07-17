"""
M7 — SpeciAL4PM (Species-based Generalization)
===============================================
Provides run() for job_prepare. CLI via main().
"""
import os, sys, json, time, argparse
from datetime import datetime, timezone
from functools import partial

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "src/SpeciAL-core")
from special4pm.estimation import SpeciesEstimator
from special4pm.species import retrieve_species_n_gram, retrieve_species_trace_variant
from special4pm.simulation.simulation import simulate_model
import pm4py


def run(dataset_key, workdir, output_dir, miners=None, cell_timeout=None):
    """Run M7. Reads manifest/PNMLs from workdir, writes configs to output_dir.

    cell_timeout (seconds) bounds each miner's METRIC time (simulation +
    coverage; model discovery happened in prepare_workdir and is excluded).
    Timed-out cells are written as -1 sentinels. Unix-only (SIGALRM)."""
    with open(os.path.join(workdir, "manifest.json")) as f:
        manifest = json.load(f)
    dname = manifest["dataset"]
    log_path = manifest["xes_file"]

    log = pm4py.read_xes(log_path)
    log = pm4py.convert_to_event_log(log)
    print(f"M7 — SpeciAL4PM ({len(log)} traces)")

    estimator = SpeciesEstimator(step_size=None, d0=False, d1=False, d2=False, c0=True, c1=True)
    for sp in ["1-gram", "2-gram", "3-gram"]:
        estimator.register(sp, partial(retrieve_species_n_gram, n=int(sp[0])))
    estimator.register("tv", retrieve_species_trace_variant)
    estimator.apply(log, verbose=False)

    orig_c1 = {}
    for sp in ["1-gram", "2-gram", "3-gram", "tv"]:
        if "incidence_c1" in estimator.metrics[sp]:
            orig_c1[sp] = estimator.metrics[sp]["incidence_c1"][-1]
    print(f"  Original C1: {orig_c1}")

    target = miners or list(manifest["miners"].keys())

    # Sentinel pre-write: if the job dies mid-run (OOM, wall-clock kill), the
    # unfinished miners keep a -1 config instead of leaving silent holes
    # (on BPI2019 only Trace_Filtered was written before the job died).
    for mname in target:
        if mname in manifest["miners"]:
            _write_config(output_dir, dname, mname,
                          {"gen_score": -1, "runtime_s": -1},
                          "did not complete (crash or budget)")

    import signal
    use_alarm = cell_timeout and hasattr(signal, "SIGALRM")
    if use_alarm:
        def _on_alarm(signum, frame):
            raise TimeoutError(f"exceeds cell budget ({cell_timeout}s)")
        signal.signal(signal.SIGALRM, _on_alarm)

    for mname in target:
        minfo = manifest["miners"].get(mname)
        if not minfo:
            print(f"  [{mname}] SKIP — not in manifest")
            continue
        print(f"  [{mname}]", end=" ", flush=True)
        t0 = time.time()
        if use_alarm:
            signal.alarm(int(cell_timeout))
        try:
            from pm4py.objects.petri_net.importer import importer as pnml_importer
            net, im, fm = pnml_importer.apply(minfo["pnml"])
            sim_log = simulate_model(net, im, fm, size=len(log))
            se = SpeciesEstimator(step_size=None, d0=False, d1=False, d2=False, c0=True, c1=True)
            for sp in ["1-gram", "2-gram", "3-gram"]:
                se.register(sp, partial(retrieve_species_n_gram, n=int(sp[0])))
            se.register("tv", retrieve_species_trace_variant)
            se.apply(sim_log, verbose=False)
            sim_c1 = {}
            for sp in ["1-gram", "2-gram", "3-gram", "tv"]:
                if "incidence_c1" in se.metrics[sp]:
                    sim_c1[sp] = se.metrics[sp]["incidence_c1"][-1]
            ratios = [min(sim_c1[sp] / orig_c1[sp], 1.0) for sp in orig_c1 if sp in sim_c1 and orig_c1[sp] > 0]
            score = float(np.mean(ratios)) if ratios else 0.0
            el = time.time() - t0
            _write_config(output_dir, dname, mname, {
                "c1_original": orig_c1, "c1_simulated": sim_c1, "gen_score": score, "runtime_s": el,
            })
            print(f" C1_ratio={score:.4f} ({el:.1f}s)")
        except Exception as e:
            _write_config(output_dir, dname, mname, {"gen_score": -1, "runtime_s": time.time() - t0},
                          f"SpeciAL4PM error: {e}")
            print(f" ERROR: {e}")
        finally:
            if use_alarm:
                signal.alarm(0)

    print(f"\nDone → {output_dir}/")


def _write_config(output_dir, dname, miner, results, notes=""):
    cfg = {
        "dataset": dname, "miner": miner, "method": "M7",
        "method_label": "SpeciAL4PM",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": "local", "seed": 42,
        "parameters": {"n_grams": ["1-gram", "2-gram", "3-gram", "tv"]},
        "results": results, "notes": notes,
    }
    path = os.path.join(output_dir, f"{dname}__{miner}__M7.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description="M7 SpeciAL4PM")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--miners", nargs="*", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--cell-timeout", type=int, default=3600,
                    help="Per-cell METRIC budget in seconds (discovery excluded; "
                         "protocol default 3600, 0 = unlimited)")
    args = ap.parse_args()

    import shutil, secrets
    from datetime import datetime as dt
    workdir = f"/tmp/benchmark_M7_{args.dataset}_{dt.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}/"
    output_dir = args.output or os.path.join(workdir, "results")
    os.makedirs(output_dir, exist_ok=True)

    from job_prepare import prepare_workdir
    prepare_workdir(workdir, args.dataset, copy_xes=True, discover_pnmls=True)
    run(args.dataset, workdir, output_dir, miners=args.miners,
        cell_timeout=args.cell_timeout)
    shutil.rmtree(workdir)
    print(f"  [clean] removed {workdir}")


if __name__ == "__main__":
    main()
