"""
M6 — Bootstrap Generalization (Entropia -bgen)
===============================================
Provides run() for job wrappers. CLI via main().
"""
import os, sys, json, re, time, argparse, subprocess
from datetime import datetime, timezone

ENTROPIA_DIR = os.path.join(os.path.dirname(__file__), "..", "..",
                            "src", "codebase", "jbpt-pm", "entropia")
JAR_FIXED = os.path.join(ENTROPIA_DIR, "jbpt-pm-entropia-1.7.1.jar")
JAR_VANILLA = os.path.join(ENTROPIA_DIR, "jbpt-pm-entropia-1.7.jar")

RE_REPLICATE = re.compile(
    r"Model-log precision and recall calculated for bootstrap sample\s+\d+:\s+"
    r"([\d.]+(?:E[+-]?\d+)?),\s*([\d.]+(?:E[+-]?\d+)?)")
RE_SUMMARY = re.compile(
    r"Model-system precision:\s*([\d.]+(?:E[+-]?\d+)?)\s*\+/-\s*([\d.]+(?:E[+-]?\d+)?)\s*\n"
    r"Model-system recall:\s*([\d.]+(?:E[+-]?\d+)?)\s*\+/-\s*([\d.]+(?:E[+-]?\d+)?)")
RE_RUNTIME = re.compile(r"Generalization calculated in\s*(\d+)\s*ms")


def run(dataset_key, workdir, output_dir, jar="fixed",
        k=2, m=10, n=200, g=10, p=1.0, miners=None):
    """Run M6. Reads manifest/per-miner DFGs from workdir, writes configs to output_dir."""
    with open(os.path.join(workdir, "manifest.json")) as f:
        manifest = json.load(f)
    dname = manifest["dataset"]
    xes_plain = os.path.join(workdir, f"{dname.lower().replace(' ', '_')}.xes")

    jarmap = {"vanilla": JAR_VANILLA, "fixed": JAR_FIXED, "1.7": JAR_VANILLA, "1.7.1": JAR_FIXED}
    jar_path = jarmap.get(jar, JAR_FIXED)
    jar_label = os.path.basename(jar_path)

    dfg_dir = os.path.join(workdir, "dfg_models")
    target = miners or list(manifest["miners"].keys())

    print(f"M6 — Bootstrap Gen ({jar_label})")
    print(f"  Params: k={k}, m={m}, n={n}, g={g}, p={p}")
    print()

    for mname in target:
        dfg = os.path.join(dfg_dir, f"{mname}_dfg.json")
        if not os.path.exists(dfg):
            print(f"  [{mname}] SKIP — DFG not found")
            continue
        print(f"  [{mname}] ", end="", flush=True)
        try:
            cmd = ["java", "-jar", jar_path, "-bgen",
                   f"-rel={xes_plain}", f"-ret={dfg}",
                   f"-n={n}", f"-m={m}", f"-g={g}", f"-k={k}", f"-p={p}"]
            t0 = time.time()
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=ENTROPIA_DIR)
            elapsed = time.time() - t0

            precisions = [float(x.group(1)) for x in RE_REPLICATE.finditer(r.stdout)]
            recalls = [float(x.group(2)) for x in RE_REPLICATE.finditer(r.stdout)]
            sm = RE_SUMMARY.search(r.stdout)
            p_mean = p_std = r_mean = r_std = None
            if sm:
                p_mean, p_std, r_mean, r_std = map(float, sm.groups())
            rm = RE_RUNTIME.search(r.stdout)
            runtime_ms = int(rm.group(1)) if rm else int(elapsed * 1000)

            res = {
                "precisions": precisions, "recalls": recalls,
                "precision_mean": p_mean, "precision_std": p_std,
                "recall_mean": r_mean, "recall_std": r_std,
                "n_replicates": len(precisions),
                "runtime_s": round(runtime_ms / 1000, 1),
                "exit_code": r.returncode,
            }
            _write_config(output_dir, dname, mname, res, jar_label)
            ps = f"{p_mean:.4f}±{p_std:.4f}" if p_mean else "N/A"
            rs = f"{r_mean:.4f}±{r_std:.4f}" if r_mean else "N/A"
            print(f"p={ps}, r={rs} [{res['runtime_s']}s]")
        except subprocess.TimeoutExpired:
            print("TIMEOUT (>600s)")
            _write_config(output_dir, dname, mname, {
                "precision_mean": None, "precision_std": None,
                "recall_mean": None, "recall_std": None,
                "n_replicates": 0, "runtime_s": 600, "precisions": [], "recalls": [],
            }, jar_label, "Timed out")
        except Exception as e:
            print(f"ERROR: {e}")
            _write_config(output_dir, dname, mname, {
                "precision_mean": None, "precision_std": None,
                "recall_mean": None, "recall_std": None,
                "n_replicates": 0, "runtime_s": -1, "precisions": [], "recalls": [],
            }, jar_label, str(e))

    print(f"\nDone → {output_dir}/")


def _write_config(output_dir, dname, miner, results, jar_label, notes=""):
    p, r = results.get("precision_mean"), results.get("recall_mean")
    gen = round(2 * p * r / (p + r), 6) if (p and r and p + r > 0) else 0.0
    cfg = {
        "dataset": dname, "miner": miner, "method": "M6",
        "method_label": "Bootstrap Gen (Entropia -bgen)",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": "local", "seed": 42,
        "parameters": {"jar": jar_label.replace(".jar", ""), "flag": "-bgen"},
        "results": {
            "gen_score": gen,
            "precision": round(results["precision_mean"], 6) if results["precision_mean"] is not None else None,
            "precision_std": round(results["precision_std"], 6) if results["precision_std"] is not None else None,
            "recall": round(results["recall_mean"], 6) if results["recall_mean"] is not None else None,
            "recall_std": round(results["recall_std"], 6) if results["recall_std"] is not None else None,
            "n_replicates": results["n_replicates"],
            "runtime_s": results["runtime_s"],
        },
        "notes": notes,
    }
    path = os.path.join(output_dir, f"{dname}__{miner}__M6.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


def main():
    ap = argparse.ArgumentParser(description="M6 Bootstrap Gen (-bgen)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--jar", choices=["vanilla", "fixed", "1.7", "1.7.1"], default="fixed")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--m", type=int, default=10)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--g", type=int, default=10)
    ap.add_argument("--p", type=float, default=1.0)
    ap.add_argument("--miners", nargs="+", default=None)
    args = ap.parse_args()

    import shutil, secrets
    from datetime import datetime as dt
    workdir = f"/tmp/benchmark_M6_{args.dataset}_{dt.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}/"
    output_dir = args.output or os.path.join(workdir, "results")
    os.makedirs(output_dir, exist_ok=True)

    from job_prepare import prepare_workdir
    prepare_workdir(workdir, args.dataset, copy_xes=True, decompress_xes=True, discover_pnmls=True, per_miner_dfgs=True)
    run(args.dataset, workdir, output_dir, jar=args.jar, k=args.k, m=args.m,
        n=args.n, g=args.g, p=args.p, miners=args.miners)
    shutil.rmtree(workdir)
    print(f"  [clean] removed {workdir}")


if __name__ == "__main__":
    main()
