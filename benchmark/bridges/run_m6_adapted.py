"""
M6 (adapted) — Bootstrap Generalization: bsgen breeding + PM4Py token replay
=============================================================================
The construct-faithful adaptation reported as M6 in the report (Sect. 5.4):
run the genuine bsgen bootstrap-sampling-with-breeding sampler and score each
bred, deduplicated sample by PM4Py token-replay fitness (a graded
model-system recall). This is NOT the Entropia -bgen precision/recall
F-measure; for that reading see bridges/run_m6_bgen.py.

Same code path and parameters as the D1/D2 runs
(archive/Tianhao/benchmark/bridges/run_m6.py): seed 42, 10 replicates,
10 breeding generations, k=2, p=1.0, sample size 200, score = token-replay
log_fitness of the deduplicated bred sample, mean over replicates.
One deliberate change: the RNG is re-seeded per miner (the archived script
seeded once at import), so every (dataset, miner) cell is reproducible
independent of miner order or subsetting.

Requires src/bsgen/bsgen_eval.py (vendor code, gitignored; present on the
benchmark runner). Import is resolved from the repo root, so run from there.

Output goes to benchmark/results/configs/ by default, next to the D1
adapted-M6 configs. Do NOT point --output at configs/: the Entropia
-bgen results share the <dataset>__<miner>__M6adapted.json filename there and
would be overwritten.

Sentinel protocol: a -1 sentinel config is written for every target miner
BEFORE its evaluation starts and overwritten on completion, so a killed job
leaves "-1 (did not complete)" cells instead of silent holes.

Provides run() for job_prepare. CLI via main().
"""
import os, sys, json, math, time, random, argparse
from datetime import datetime, timezone

import numpy as np
import pm4py
from pm4py.algo.evaluation.replay_fitness import algorithm as rf_eval

# Sampler resolution order: the partner's src/bsgen copy wins if present;
# otherwise the committed port next to this bridge (bridges/bsgen_eval.py).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "src/bsgen")

SEED = 42
N_BOOTSTRAP = 10     # bootstrap replicates
N_GENERATIONS = 10   # breeding generations
K = 2                # subtrace length for crossover
P = 1.0              # breeding probability
N_SAMPLE = 200       # sample size per replicate


# fork-shared globals for the miner pool (read-only after fork)
_MP = {}


def _eval_miner(mname):
    """Evaluate ONE miner: seed, breed, replay. Runs in its own process
    (or inline when workers=1). Per-miner reseeding makes the result
    independent of pool scheduling and miner order."""
    import signal
    g = _MP
    t0 = time.time()
    random.seed(SEED)
    np.random.seed(SEED)
    use_alarm = g["cell_timeout"] and hasattr(signal, "SIGALRM")
    if use_alarm:
        def _on_alarm(signum, frame):
            raise TimeoutError(f"exceeds cell budget ({g['cell_timeout']}s)")
        signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(int(g["cell_timeout"]))
    try:
        net, im, fm = pm4py.read_pnml(g["manifest"]["miners"][mname]["pnml"])
        fitnesses = []
        for i in range(g["replicates"]):
            try:
                bred = g["breed"](g["log"], g["generations"], g["sample_size"],
                                  g["k"], g["p"])
                uniq = g["dedup"](bred)
                fit = rf_eval.apply(uniq, net, im, fm,
                                    variant=rf_eval.Variants.TOKEN_BASED)
                fitnesses.append(fit["log_fitness"])
            except TimeoutError:
                raise
            except Exception as e:
                print(f"    [{mname}] replicate {i}: {e}", flush=True)
                continue
        elapsed = time.time() - t0
        if not fitnesses:
            return mname, {"gen_score": -1, "runtime_s": elapsed}, \
                   "all bootstrap replicates failed"
        mean = float(np.mean(fitnesses))
        std = float(np.std(fitnesses))
        ci = 1.96 * std / math.sqrt(len(fitnesses))
        return mname, {"gen_score": mean, "std": std, "ci_95": ci,
                       "n_replicates": len(fitnesses),
                       "raw_fitnesses": fitnesses, "runtime_s": elapsed}, ""
    except Exception as e:
        return mname, {"gen_score": -1, "runtime_s": time.time() - t0}, f"error: {e}"
    finally:
        if use_alarm:
            signal.alarm(0)


def run(dataset_key, workdir, output_dir, miners=None,
        replicates=N_BOOTSTRAP, generations=N_GENERATIONS,
        k=K, p=P, sample_size=N_SAMPLE, cell_timeout=None, workers=1):
    """Run adapted M6. Reads manifest/PNMLs from workdir, writes configs to output_dir.

    cell_timeout (seconds) bounds each miner's METRIC time (breeding + replay;
    model discovery happened in prepare_workdir and is excluded). Timed-out
    cells keep their -1 sentinel. Unix-only (SIGALRM).

    workers > 1 evaluates miners in parallel processes (fork). Numerically
    safe: each miner re-seeds, so results are identical to the serial run."""
    from bsgen_eval import log_sample_with_breeding, dedup

    with open(os.path.join(workdir, "manifest.json")) as f:
        manifest = json.load(f)
    dname = manifest["dataset"]

    log = pm4py.read_xes(manifest["xes_file"])
    log = pm4py.convert_to_event_log(log)
    print(f"M6 adapted — bsgen breeding + token replay ({len(log)} traces)")
    print(f"  Params: replicates={replicates}, generations={generations}, "
          f"k={k}, p={p}, sample_size={sample_size}, seed={SEED} (per miner), "
          f"workers={workers}")

    target = [m for m in (miners or list(manifest["miners"].keys()))
              if m in manifest["miners"]]

    # Sentinel pre-write: a killed job leaves -1 cells, not holes.
    for mname in target:
        _write_config(output_dir, dname, mname, replicates, generations, k, p,
                      sample_size, {"gen_score": -1, "runtime_s": -1},
                      "did not complete (crash or budget)")

    global _MP
    _MP = {"manifest": manifest, "log": log, "breed": log_sample_with_breeding,
           "dedup": dedup, "replicates": replicates, "generations": generations,
           "k": k, "p": p, "sample_size": sample_size, "cell_timeout": cell_timeout}

    def _finish(mname, res, note):
        _write_config(output_dir, dname, mname, replicates, generations, k, p,
                      sample_size, res, note)
        if res.get("gen_score", -1) >= 0:
            print(f"  [{mname}] gen={res['gen_score']:.4f} +- {res['std']:.4f} "
                  f"[{res['runtime_s']:.0f}s]", flush=True)
        else:
            print(f"  [{mname}] SENTINEL: {note}", flush=True)

    if workers > 1 and hasattr(os, "fork"):
        import multiprocessing as mp_mod
        from concurrent.futures import ProcessPoolExecutor, as_completed
        ctx = mp_mod.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = {ex.submit(_eval_miner, m): m for m in target}
            for fu in as_completed(futs):
                _finish(*fu.result())
    else:
        for mname in target:
            _finish(*_eval_miner(mname))

    print(f"\nDone -> {output_dir}/")


def _write_config(output_dir, dname, miner, replicates, generations, k, p,
                  sample_size, results, notes=""):
    cfg = {
        "dataset": dname, "miner": miner, "method": "M6adapted",
        "method_label": "Bootstrap Generalization (adapted)",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": "local", "seed": SEED,
        "parameters": {"generations": generations, "k": k, "p": p,
                       "replicates": replicates, "sample_size": sample_size,
                       "source": "bsgen (breeding + token replay)"},
        "results": results, "notes": notes,
    }
    path = os.path.join(output_dir, f"{dname}__{miner}__M6adapted.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


def main():
    ap = argparse.ArgumentParser(description="M6 adapted (bsgen + token replay)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", default=None,
                    help="Default: benchmark/results/configs (NOT configs, "
                         "which holds the Entropia -bgen results)")
    ap.add_argument("--miners", nargs="+", default=None)
    ap.add_argument("--replicates", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--generations", type=int, default=N_GENERATIONS)
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--p", type=float, default=P)
    ap.add_argument("--sample-size", type=int, default=N_SAMPLE)
    ap.add_argument("--cell-timeout", type=int, default=3600,
                    help="Per-cell METRIC budget in seconds (discovery excluded; "
                         "protocol default 3600, 0 = unlimited)")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel miner processes (per-miner reseeding keeps "
                         "results identical to serial; default 1)")
    args = ap.parse_args()

    import shutil, secrets
    from datetime import datetime as dt
    workdir = f"/tmp/benchmark_M6A_{args.dataset}_{dt.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}/"
    output_dir = args.output or os.path.join("benchmark", "results", "configs")
    os.makedirs(output_dir, exist_ok=True)

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from job_prepare import prepare_workdir
    prepare_workdir(workdir, args.dataset, copy_xes=True, discover_pnmls=True)
    run(args.dataset, workdir, output_dir, miners=args.miners,
        replicates=args.replicates, generations=args.generations,
        k=args.k, p=args.p, sample_size=args.sample_size,
        cell_timeout=args.cell_timeout, workers=args.workers)
    shutil.rmtree(workdir)
    print(f"  [clean] removed {workdir}")


if __name__ == "__main__":
    main()
