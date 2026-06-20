"""
M3 — Entropic Relevance (Entropia JAR)
=======================================
Provides run() for job wrappers. CLI via main().
"""
import os, sys, json, subprocess, time, argparse, re
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from job_prepare import prepare_workdir, get_miner_names

ENTROPIA_JAR = "src/codebase/jbpt-pm/entropia/jbpt-pm-entropia-1.7.jar"


def run(dataset_key, workdir, output_dir, miners=None):
    """Run M3. Reads manifest/DFG/XES from workdir, writes configs to output_dir."""
    with open(os.path.join(workdir, "manifest.json")) as f:
        manifest = json.load(f)
    dname = manifest["dataset"]
    xes = manifest["xes_file"]
    dfg = manifest["dfg_json"]

    target = miners or get_miner_names()

    print(f"M3 — Entropic Relevance")
    print(f"  java -jar {ENTROPIA_JAR} -r -s -rel={xes} -ret={dfg}")
    t0 = time.time()
    r = subprocess.run(
        ["java", "-jar", ENTROPIA_JAR, "-r", "-s", f"-rel={xes}", f"-ret={dfg}"],
        capture_output=True, text=True, timeout=60)
    elapsed = time.time() - t0

    if r.returncode != 0:
        print(f"  ERROR: {r.stderr[:200]}")
        for m in target:
            _write_config(output_dir, dname, m, {
                "entropic_relevance_raw": -1, "entropic_relevance_normalized": None, "runtime_s": elapsed,
            }, f"JVM error: {r.stderr[:200]}")
        return

    rel = float(re.search(r'([\d.Ee+-]+)', r.stdout.strip()).group(1))
    print(f"  Relevance: {rel:.4f}  ({elapsed:.1f}s)")

    for m in target:
        _write_config(output_dir, dname, m, {
            "entropic_relevance_raw": rel, "entropic_relevance_normalized": None, "runtime_s": elapsed,
        })
    print(f"  → {len(target)} configs → {output_dir}/")


def _write_config(output_dir, dname, miner, results, notes=""):
    cfg = {
        "dataset": dname, "miner": miner, "method": "M3",
        "method_label": "Entropic Relevance",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": "local", "seed": 42,
        "parameters": {"jar": "entropia-1.7", "flag": "-r", "model": "DFG"},
        "results": results, "notes": notes,
    }
    path = os.path.join(output_dir, f"{dname}__{miner}__M3.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  ✓ {miner}")


def main():
    ap = argparse.ArgumentParser(description="M3 Entropic Relevance")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--miners", nargs="*", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    import shutil, secrets
    from datetime import datetime as dt
    workdir = f"/tmp/benchmark_M3_{args.dataset}_{dt.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}/"
    output_dir = args.output or os.path.join(workdir, "results")
    os.makedirs(output_dir, exist_ok=True)

    prepare_workdir(workdir, args.dataset, mode="log_dfg")
    run(args.dataset, workdir, output_dir, miners=args.miners)
    shutil.rmtree(workdir)
    print(f"  [clean] removed {workdir}")


if __name__ == "__main__":
    main()
