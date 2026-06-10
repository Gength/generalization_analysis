#!/usr/bin/env python3
"""
M8: Pattern-based Generalization
Usage:  uv run python benchmark/m8.sh [--mode single|parallel]

--mode single (default): Run miners one-at-a-time. Gurobi uses all cores.
                         For local machine with 16G RAM.

--mode parallel: Run all miners concurrently. For HPC (128G+ RAM, SLURM).
"""
import argparse, glob, json, os, subprocess, sys, time
from datetime import datetime, timezone

# =============================================================================
# CONFIGURATION — Change these for your experiment
# =============================================================================
DATASET_KEY = "D1"
MINER_LIST = None
JAR_TIMEOUT_MIN = 1440         # 24 hours

DATASETS = {
    "D1": {
        "model_dir": "benchmark/models",
        "xes_file": "sepsis.xes.gz",
        "manifest": "benchmark/models/manifest.json",
    },
}

# System paths (change if your setup differs)
PROJ_DIR = "/home/gengtianhao/Process Mining"
GUROBI_HOME = "/home/gengtianhao/gurobi1100/linux64"

# =============================================================================
# EXECUTION — Do not edit below this line
# =============================================================================

import argparse, glob, json, os, subprocess, sys, time
from datetime import datetime, timezone

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['single', 'parallel'], default='single')
parser.add_argument('--timeout', type=int, default=86400)
args = parser.parse_args()

info = DATASETS.get(DATASET_KEY, list(DATASETS.values())[0])
MODEL_DIR = os.path.join(PROJ_DIR, info["model_dir"])
CONFIG_DIR = os.path.join(PROJ_DIR, "benchmark/results/configs")
os.makedirs(CONFIG_DIR, exist_ok=True)
XES = info["xes_file"]

CLASSPATH = ":".join([
    f"{GUROBI_HOME}/lib/gurobi.jar",
    *glob.glob(f"{PROJ_DIR}/src/prom_workspace_link/dist/*.jar"),
    *glob.glob(f"{PROJ_DIR}/src/prom_workspace_link/lib/*.jar"),
    f"{PROJ_DIR}/src/prom_workspace_link/packages/logfiltering-6.13.2/lib/fake-context-1.0.20180719.jar",
    f"{PROJ_DIR}/src/prom_workspace_link/packages/nlplogutils-6.9.77/lib/dafsa.jar",
    f"{PROJ_DIR}/src/prom_workspace_link/packages/nlplogutils-6.9.77/NLPLogUtils.jar",
    f"{PROJ_DIR}/src/prom_workspace_link/packages/apromore-6.9.61/AProMore.jar",
    *glob.glob(f"{PROJ_DIR}/src/AntiAlignments/ivy/*.jar"),
    *glob.glob(f"{PROJ_DIR}/src/AutomataConformance/out/artifacts/PatternGeneralization_jar/*.jar"),
])

with open(f"{MODEL_DIR}/manifest.json") as f:
    manifest = json.load(f)
DATASET = manifest["dataset"]
XES = "sepsis.xes.gz"

all_miners = [(os.path.basename(p).replace(".pnml", ""), os.path.basename(p))
              for p in sorted(glob.glob(f"{MODEL_DIR}/*.pnml"))]
if MINER_LIST is not None:
    miners = [(n, p) for (n, p) in all_miners if n in MINER_LIST]
else:
    miners = all_miners

SAVE_DIR = f"{CONFIG_DIR}"

def run_miner(name, pnml_name):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{gurobi_home}/lib:{env.get('LD_LIBRARY_PATH', '')}"
    t0 = time.time()
    proc = subprocess.Popen(
        ["xvfb-run", "--auto-servernum", "java", "-Xmx16G",
         f"-Djava.library.path={proj}/lib:{gurobi_home}/lib",
         "-cp", CLASSPATH,
         "au.unimelb.evaluation.GeneralizationEvaluation",
         f"{MODEL_DIR}/", XES, pnml_name,
         "PatternBasedGeneralization", "global", "PartialMatching", "0.02",
         f"{args.timeout // 60}", "MINUTES"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = proc.communicate(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        return -1, args.timeout, "TIMEOUT", ""
    elapsed = time.time() - t0
    result_line = ""
    for line in stdout.strip().split("\n"):
        if line.startswith(XES):
            result_line = line.strip()
            break
    notes = "FAILED" if "t/out" in result_line else ""
    if stderr.strip():
        notes = (notes + "; " if notes else "") + "STDERR: " + stderr.strip()[:200]
    return 0 if result_line and "t/out" not in result_line else -1, elapsed, notes, result_line

def save_config(name, score, runtime_s, notes, raw_line):
    config = {
        "dataset": DATASET, "miner": name, "method": "M8",
        "method_label": "Pattern-based Generalization",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": "local", "seed": 42,
        "parameters": {"jar": "PatternGeneralization", "oracle": "global",
                       "matching": "PartialMatching", "noise_threshold": 0.02,
                       "solver": "gurobi_11.0", "mode": args.mode},
        "results": {"raw_line": raw_line, "runtime_s": runtime_s},
        "notes": notes,
    }
    out_path = f"{SAVE_DIR}/{DATASET}__{name}__M8.json"
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)
    return out_path

if args.mode == "parallel":
    print(f"PARALLEL mode — {len(miners)} miners at once")
    procs = {}
    for name, pn in miners:
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{gurobi_home}/lib:{env.get('LD_LIBRARY_PATH', '')}"
        p = subprocess.Popen(
            ["xvfb-run", "--auto-servernum", "java", "-Xmx16G",
             f"-Djava.library.path={proj}/lib:{gurobi_home}/lib",
             "-cp", CLASSPATH,
             "au.unimelb.evaluation.GeneralizationEvaluation",
             f"{MODEL_DIR}/", XES, pn,
             "PatternBasedGeneralization", "global", "PartialMatching", "0.02",
             f"{args.timeout // 60}", "MINUTES"],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        procs[name] = {"proc": p, "t_start": time.time()}
        print(f"  Started {name} (pid={p.pid})")
    remaining = dict(procs)
    end = time.time() + args.timeout
    while remaining and time.time() < end:
        for n in list(remaining.keys()):
            if remaining[n]["proc"].poll() is not None:
                out, _ = remaining[n]["proc"].communicate()
                el = time.time() - remaining[n]["t_start"]
                rl = ""
                for l in out.strip().split("\n"):
                    if l.startswith(XES): rl = l.strip(); break
                save_config(n, 0 if rl and "t/out" not in rl else -1, el, "", rl)
                print(f"  {n} ({el:.0f}s) {rl[:80] if rl else 'no result'}")
                del remaining[n]
        time.sleep(2)
    for n in list(remaining.keys()):
        remaining[n]["proc"].kill()
        remaining[n]["proc"].communicate()
        save_config(n, -1, args.timeout, "TIMEOUT", "")
        print(f"  {n} TIMEOUT ({args.timeout}s)")
else:
    # SINGLE mode (default)
    print(f"SINGLE mode — {len(miners)} miners sequentially, Gurobi all cores")
    for name, pn in miners:
        print(f"\n  [{name}] ", end="", flush=True)
        score, elapsed, notes, raw = run_miner(name, pn)
        path = save_config(name, score, elapsed, notes, raw)
        marker = "✅" if score >= 0 else "❌"
        status = notes if notes else (f"result={raw[:60]}" if raw else "no result")
        print(f"{marker} ({elapsed:.0f}s) {status}")
        print(f"       -> {path}")

print("\nDone!")