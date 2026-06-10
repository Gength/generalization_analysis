#!/usr/bin/env python3
"""
M4: Anti-Alignment Generalization
Usage:  uv run python benchmark/m4.sh [--mode single|parallel]

--mode single (default): Run miners one-at-a-time. Gurobi uses all cores (8 on WSL).
                         For local machine with 16G RAM.

--mode parallel: Run all miners concurrently. For HPC (128G+ RAM, SLURM array job).
                 Each miner as independent subprocess with MaxHeap=16G.

Config JSONs written to benchmark/results/configs/{Dataset}__{Miner}__M4.json
"""
import argparse, glob, json, os, subprocess, sys, time
from datetime import datetime, timezone

# =============================================================================
# CONFIGURATION — Change these for your experiment
# =============================================================================
DATASET_KEY = "D1"
MINER_LIST = None
TIMEOUT_S = 86400
PARALLEL = False

DATASETS = {
    "D1": {
        "model_dir": "benchmark/models",
        "manifest": "benchmark/models/manifest.json",
    },
}

# System paths
PROJ_DIR = "/home/gengtianhao/Process Mining"
GUROBI_HOME = "/home/gengtianhao/gurobi1100/linux64"

# =============================================================================
# EXECUTION — Do not edit below this line
# =============================================================================

import argparse, glob, json, os, subprocess, sys, time
from datetime import datetime, timezone

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['single', 'parallel'], default='parallel' if PARALLEL else 'single')
parser.add_argument('--timeout', type=int, default=TIMEOUT_S)
args = parser.parse_args()

info = DATASETS.get(DATASET_KEY, list(DATASETS.values())[0])
MODEL_DIR = os.path.join(PROJ_DIR, info["model_dir"])
CONFIG_DIR = os.path.join(PROJ_DIR, "benchmark/results/configs")
os.makedirs(CONFIG_DIR, exist_ok=True)

CLASSPATH = ":".join([
    f"{GUROBI_HOME}/lib/gurobi.jar",
    f"{PROJ_DIR}/build",
    f"{PROJ_DIR}/src/AntiAlignments/dist/AntiAlignments-20260609.jar",
    *glob.glob(f"{PROJ_DIR}/src/AntiAlignments/ivy/*.jar"),
    f"{PROJ_DIR}/src/AutomataConformance/out/production/AutomataConformance",
    *glob.glob(f"{PROJ_DIR}/src/prom_workspace_link/dist/*.jar"),
    *glob.glob(f"{PROJ_DIR}/src/prom_workspace_link/lib/*.jar"),
    f"{PROJ_DIR}/src/prom_workspace_link/packages/logfiltering-6.13.2/lib/fake-context-1.0.20180719.jar",
])

with open(f"{MODEL_DIR}/manifest.json") as f:
    manifest = json.load(f)
DATASET = manifest["dataset"]
XES = f"{MODEL_DIR}/sepsis.xes.gz"

all_miners = [(os.path.basename(p).replace(".pnml", ""), p)
              for p in sorted(glob.glob(f"{MODEL_DIR}/*.pnml"))]
if MINER_LIST is not None:
    miners = [(n, p) for (n, p) in all_miners if n in MINER_LIST]
else:
    miners = all_miners

def run_miner(name, pnml):
    """Run a single miner, return (score, runtime_s, notes)."""
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{GUROBI_HOME}/lib:{env.get('LD_LIBRARY_PATH', '')}"
    t0 = time.time()
    proc = subprocess.Popen(
        ["java", "-Xmx16G",
         f"-Djava.library.path={PROJ_DIR}/lib:{GUROBI_HOME}/lib",
         "-cp", CLASSPATH, "run.M4Runner", XES, pnml],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = proc.communicate(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        return -1, args.timeout, "TIMEOUT"
    elapsed = time.time() - t0
    score = -1
    for line in stdout.split("\n"):
        if line.startswith("Generalization:"):
            try:
                score = float(line.split(":")[1].strip())
            except ValueError:
                pass
            break
    notes = "" if score >= 0 else "FAILED"
    return score, elapsed, notes

def save_config(name, score, runtime_s, notes):
    config = {
        "dataset": DATASET, "miner": name, "method": "M4",
        "method_label": "Anti-Alignment Generalization",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": "local", "seed": 42,
        "parameters": {"solver": "gurobi_11.0", "threads": "auto",
                       "mode": args.mode, "timeout_s": args.timeout},
        "results": {"score": score, "runtime_s": runtime_s},
        "notes": notes,
    }
    out_path = f"{CONFIG_DIR}/{DATASET}__{name}__M4.json"
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)
    return out_path

if args.mode == "parallel":
    # HPC mode: all miners at once
    print(f"PARALLEL mode — launching {len(miners)} miners simultaneously")
    print(f"WARNING: Needs 16G × {len(miners)} = {16*len(miners)}G RAM total")
    procs = {}
    for name, pnml in miners:
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{GUROBI_HOME}/lib:{env.get('LD_LIBRARY_PATH', '')}"
        p = subprocess.Popen(
            ["java", "-Xmx16G",
             f"-Djava.library.path={PROJ_DIR}/lib:{GUROBI_HOME}/lib",
             "-cp", CLASSPATH, "run.M4Runner", XES, pnml],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        procs[name] = {"proc": p, "t_start": time.time()}
        print(f"  Started {name} (pid={p.pid})")
    # Poll all
    remaining = dict(procs)
    end_time = time.time() + args.timeout
    while remaining and time.time() < end_time:
        for n in list(remaining.keys()):
            if remaining[n]["proc"].poll() is not None:
                out, _ = remaining[n]["proc"].communicate()
                el = time.time() - remaining[n]["t_start"]
                score = -1
                for line in out.split("\n"):
                    if line.startswith("Generalization:"):
                        try: score = float(line.split(":")[1].strip())
                        except: pass
                        break
                save_config(n, score, el, "" if score >= 0 else "FAILED")
                print(f"  {'✅' if score>=0 else '❌'} {n} score={score:.4f} ({el:.0f}s)")
                del remaining[n]
        time.sleep(2)
    for n in list(remaining.keys()):
        remaining[n]["proc"].kill()
        remaining[n]["proc"].communicate()
        el = time.time() - remaining[n]["t_start"]
        save_config(n, -1, el, "TIMEOUT")
        print(f"  ❌ {n} TIMEOUT ({el:.0f}s)")
else:
    # SINGLE mode (default): one miner at a time, Gurobi uses all cores
    print(f"SINGLE mode — running {len(miners)} miners sequentially")
    print(f"Gurobi Threads=0 → auto-uses all {os.cpu_count()} cores per miner")
    for name, pnml in miners:
        print(f"\n  [{name}] ", end="", flush=True)
        score, elapsed, notes = run_miner(name, pnml)
        path = save_config(name, score, elapsed, notes)
        marker = "✅" if score >= 0 else "❌"
        print(f"{marker} score={score:.4f} ({elapsed:.0f}s) {notes}")
        print(f"       -> {path}")

print("\nDone!")