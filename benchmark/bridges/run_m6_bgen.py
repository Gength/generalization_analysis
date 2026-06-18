"""
M6 — Bootstrap Generalization (Entropia -bgen) Runner
=====================================================
Invokes the Entropia JAR's -bgen flag for each miner in the manifest,
parses output (per-replicate precision/recall ± std), writes config JSON.

Uses the FIXED JAR (jbpt-pm-entropia-1.7.1.jar) which adds a null guard
at EventLogSampling.logBreeding:101 — enabling k=2 on all datasets.

Usage:
    uv run python benchmark/bridges/run_m6_bgen.py --dataset D1
    uv run python benchmark/bridges/run_m6_bgen.py --dataset D2 --jar version
"""

import os, sys, json, re, time, argparse, subprocess
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from datasets import DATASETS, get_info, CONFIG_DIR_V2

# ── Paths ────────────────────────────────────────────────────────────────────
ENTROPIA_DIR = os.path.join(os.path.dirname(__file__), "..", "..",
                            "src", "codebase", "jbpt-pm", "entropia")
JAR_VANILLA = os.path.join(ENTROPIA_DIR, "jbpt-pm-entropia-1.7.jar")
JAR_FIXED = os.path.join(ENTROPIA_DIR, "jbpt-pm-entropia-1.7.1.jar")

# ── Defaults ─────────────────────────────────────────────────────────────────
DATASET_KEY = "D1"
N_SAMPLE = 200
N_REPLICATES = 10
N_GENERATIONS = 10
K = 2
P = 1.0
USE_FIXED = True  # use the fixed JAR by default

# ── Regex to parse -bgen output ──────────────────────────────────────────────
RE_REPLICATE = re.compile(
    r"Model-log precision and recall calculated for bootstrap sample\s+\d+:\s+"
    r"([\d.]+(?:E[+-]?\d+)?),\s*([\d.]+(?:E[+-]?\d+)?)"
)
RE_SUMMARY = re.compile(
    r"Model-system precision:\s*([\d.]+(?:E[+-]?\d+)?)\s*\+/-\s*([\d.]+(?:E[+-]?\d+)?)\s*\n"
    r"Model-system recall:\s*([\d.]+(?:E[+-]?\d+)?)\s*\+/-\s*([\d.]+(?:E[+-]?\d+)?)"
)
RE_RUNTIME = re.compile(
    r"Generalization calculated in\s*(\d+)\s*ms"
)

# ── CLI ──────────────────────────────────────────────────────────────────────
_cli = argparse.ArgumentParser(description="M6 Bootstrap Gen (-bgen) Runner")
_cli.add_argument("--dataset", default=DATASET_KEY, choices=list(DATASETS.keys()))
_cli.add_argument("--jar", choices=["vanilla", "fixed", "1.7", "1.7.1"], default="fixed")
_cli.add_argument("--k", type=int, default=K)
_cli.add_argument("--m", type=int, default=N_REPLICATES, help="Number of bootstrap samples")
_cli.add_argument("--n", type=int, default=N_SAMPLE, help="Sample size")
_cli.add_argument("--g", type=int, default=N_GENERATIONS, help="Generations")
_cli.add_argument("--p", type=float, default=P, help="Breeding probability")
_cli.add_argument("--miners", nargs="+", default=None, help="Restrict to specific miners")
_args = _cli.parse_args()

DATASET_KEY = _args.dataset
K = _args.k
N_REPLICATES = _args.m
N_SAMPLE = _args.n
N_GENERATIONS = _args.g
P = _args.p

jar_map = {
    "vanilla": JAR_VANILLA,
    "fixed": JAR_FIXED,
    "1.7": JAR_VANILLA,
    "1.7.1": JAR_FIXED,
}
JAR_PATH = jar_map[_args.jar]
JAR_LABEL = os.path.basename(JAR_PATH)

# ── Load manifest ────────────────────────────────────────────────────────────
info = get_info(DATASET_KEY)
manifest_path = info["manifest"]
log_path = info["log_path"]

# Decompress XES if needed
if log_path.endswith(".gz"):
    xes_tmp = f"/tmp/{DATASET_KEY.lower()}_{os.path.basename(log_path)[:-3]}"
    if not os.path.exists(xes_tmp):
        import gzip, shutil
        with gzip.open(log_path, "rb") as fin, open(xes_tmp, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        print(f"  Decompressed → {xes_tmp}")
    XES_PATH = os.path.abspath(xes_tmp)
    print(f"  Decompressed → {XES_PATH}")
else:
    XES_PATH = os.path.abspath(log_path)

with open(manifest_path) as f:
    manifest = json.load(f)

DATASET = manifest["dataset"]
DFG_DIR = os.path.abspath(os.path.join(info["model_dir"], "dfg_models"))
os.makedirs(CONFIG_DIR_V2, exist_ok=True)
os.makedirs(DFG_DIR, exist_ok=True)

# Warn if DFG files are missing
_missing = [m for m in manifest["miners"] if not os.path.exists(os.path.join(DFG_DIR, f"{m}_dfg.json"))]
if _missing:
    print(f"  ⚠ {len(_missing)} miner(s) missing DFG JSON: run 'uv run python benchmark/02_gen_per_miner_dfgs.py --dataset {DATASET_KEY}' first.")
    print(f"     Missing: {', '.join(_missing)}")


def run_bgen(miner_name, dfg_path):
    """Run -bgen and return parsed results dict."""
    cmd = [
        "java", "-jar", JAR_PATH,
        "-bgen",
        f"-rel={XES_PATH}",
        f"-ret={dfg_path}",
        f"-n={N_SAMPLE}",
        f"-m={N_REPLICATES}",
        f"-g={N_GENERATIONS}",
        f"-k={K}",
        f"-p={P}",
    ]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                       cwd=ENTROPIA_DIR)  # JAR needs CWD=entropia dir for lib resolution
    elapsed = time.time() - t0
    stdout = r.stdout

    # Parse replicate values
    precisions = []
    recalls = []
    for m in RE_REPLICATE.finditer(stdout):
        precisions.append(float(m.group(1)))
        recalls.append(float(m.group(2)))

    # Parse summary line
    sm = RE_SUMMARY.search(stdout)
    if sm:
        p_mean, p_std, r_mean, r_std = float(sm.group(1)), float(sm.group(2)), float(sm.group(3)), float(sm.group(4))
    else:
        p_mean, p_std, r_mean, r_std = None, None, None, None

    rm = RE_RUNTIME.search(stdout)
    runtime_ms = int(rm.group(1)) if rm else int(elapsed * 1000)

    return {
        "precisions": precisions,
        "recalls": recalls,
        "precision_mean": p_mean,
        "precision_std": p_std,
        "recall_mean": r_mean,
        "recall_std": r_std,
        "n_replicates": len(precisions),
        "runtime_s": round(runtime_ms / 1000, 1),
        "exit_code": r.returncode,
        "stderr": r.stderr[:500] if r.stderr else "",
    }


def write_config(miner, results, notes=""):
    p = results["precision_mean"]
    r = results["recall_mean"]
    gen_score = round(2 * p * r / (p + r), 6) if (p is not None and r is not None and (p + r) > 0) else 0.0
    config = {
        "dataset": DATASET,
        "miner": miner,
        "method": "M6",
        "method_label": f"Bootstrap Generalization (Entropia -bgen)",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": "local",
        "seed": 42,
        "parameters": {
            "jar": JAR_LABEL.replace(".jar", ""),
            "flag": "-bgen",
            "sample_size": N_SAMPLE,
            "replicates": N_REPLICATES,
            "generations": N_GENERATIONS,
            "k": K,
            "p": P,
            "model_format": "DFG JSON (simulated from PNML)",
        },
        "results": {
            "gen_score": gen_score,
            "precision": round(results["precision_mean"], 6) if results["precision_mean"] is not None else None,
            "precision_std": round(results["precision_std"], 6) if results["precision_std"] is not None else None,
            "recall": round(results["recall_mean"], 6) if results["recall_mean"] is not None else None,
            "recall_std": round(results["recall_std"], 6) if results["recall_std"] is not None else None,
            "n_replicates": results["n_replicates"],
            "raw_precisions": [round(v, 8) for v in results["precisions"]],
            "raw_recalls": [round(v, 8) for v in results["recalls"]],
            "runtime_s": results["runtime_s"],
        },
        "notes": notes,
    }
    path = f"{CONFIG_DIR_V2}/{DATASET}__{miner}__M6.json"
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    return path


# ── Main ─────────────────────────────────────────────────────────────────────
target_miners = _args.miners or list(manifest["miners"].keys())
print(f"M6 — Bootstrap Generalization ({JAR_LABEL})")
print(f"  Dataset: {DATASET} ({DATASET_KEY})")
print(f"  Parameters: k={K}, m={N_REPLICATES}, n={N_SAMPLE}, g={N_GENERATIONS}, p={P}")
print(f"  XES: {XES_PATH}")
print()

for miner_name in target_miners:
    dfg_path = os.path.join(DFG_DIR, f"{miner_name}_dfg.json")
    if not os.path.exists(dfg_path):
        print(f"  [{miner_name}] SKIP — DFG not found: {dfg_path}")
        continue

    print(f"  [{miner_name}] ", end="", flush=True)
    try:
        results = run_bgen(miner_name, dfg_path)
        if results["exit_code"] != 0:
            print(f"ERROR (code {results['exit_code']}): {results['stderr'][:200]}")
            write_config(miner_name, results, notes=f"Exit code {results['exit_code']}")
        else:
            p_str = f"{results['precision_mean']:.4f}±{results['precision_std']:.4f}" if results['precision_mean'] else "N/A"
            r_str = f"{results['recall_mean']:.4f}±{results['recall_std']:.4f}" if results['recall_mean'] else "N/A"
            print(f"p={p_str}, r={r_str} [{results['runtime_s']}s]")
            path = write_config(miner_name, results)
            print(f"    → {path}")
    except subprocess.TimeoutExpired:
        print("TIMEOUT (>600s)")
        write_config(miner_name, {"precision_mean": None, "precision_std": None,
                                   "recall_mean": None, "recall_std": None,
                                   "n_replicates": 0, "runtime_s": 600,
                                   "precisions": [], "recalls": []},
                     notes="Timed out after 600s")
    except Exception as e:
        print(f"ERROR: {e}")
        write_config(miner_name, {"precision_mean": None, "precision_std": None,
                                   "recall_mean": None, "recall_std": None,
                                   "n_replicates": 0, "runtime_s": -1,
                                   "precisions": [], "recalls": []},
                     notes=str(e))

print(f"\nDone → {CONFIG_DIR_V2}/")
