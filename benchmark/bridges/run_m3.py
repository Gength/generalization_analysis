"""
M3 — Entropic Relevance (Entropia JAR)
=======================================
Reads manifest.json, calls Entropia with -r flag.
No pm4py dependency.
"""
import os, sys, json, subprocess, time, argparse, re
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from datasets import get_info, CONFIG_DIR_V2

# =============================================================================
# CONFIGURATION — Change these for your experiment
# =============================================================================
DATASET_KEY = "D1"          # Which dataset (D1-D5). None = read from manifest.json
MINER_LIST = None            # None = all miners, or ["Alpha", ...]
CONFIG_DIR = "benchmark/results/configs_v2"
SEED = 42

from datasets import DATASETS

# ── CLI ─────────────────────────────────────────────────────────────────────
_cli = argparse.ArgumentParser(description="M3 Entropic Relevance")
_cli.add_argument("--dataset", default=None, choices=list(DATASETS.keys()),
                  help="Override DATASET_KEY (default: D1)")
_cli.add_argument("--miners", nargs="*", default=None,
                  help="Restrict to specific miners (default: all)")
_cli_args, _ = _cli.parse_known_args()
if _cli_args.dataset:
    DATASET_KEY = _cli_args.dataset
if _cli_args.miners is not None:
    MINER_LIST = _cli_args.miners

os.makedirs(CONFIG_DIR, exist_ok=True)

# =============================================================================
# EXECUTION — Do not edit below this line
# =============================================================================

ENTROPIA_JAR = "src/codebase/jbpt-pm/entropia/jbpt-pm-entropia-1.7.jar"

info = get_info(DATASET_KEY)
manifest_path = info["manifest"]

with open(manifest_path) as f:
    manifest = json.load(f)

DATASET = manifest["dataset"]
XES = manifest["xes_file"]
DFG = manifest["dfg_json"]

def write_config(miner, results, notes=""):
    config = {
        "dataset": DATASET, "miner": miner, "method": "M3",
        "method_label": "Entropic Relevance",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": "local", "seed": 42,
        "parameters": {"jar": "entropia-1.7", "flag": "-r", "model": "DFG"},
        "results": results, "notes": notes,
    }
    with open(f"{CONFIG_DIR}/{DATASET}__{miner}__M3.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ {miner}")

print("M3 — Entropic Relevance")
print(f"  java -jar {ENTROPIA_JAR} -r -rel={XES} -ret={DFG}")
t0 = time.time()
r = subprocess.run(
    ["java", "-jar", ENTROPIA_JAR, "-r", "-s",
     f"-rel={XES}", f"-ret={DFG}"],
    capture_output=True, text=True, timeout=60)
elapsed = time.time() - t0

target_miners = list(manifest["miners"].keys()) if MINER_LIST is None else MINER_LIST

if r.returncode != 0:
    print(f"  ERROR: {r.stderr[:200]}")
    for m in target_miners:
        write_config(m, {
            "entropic_relevance_raw": -1,
            "entropic_relevance_normalized": None,
            "runtime_s": elapsed,
        }, f"JVM error: {r.stderr[:200]}")
    sys.exit(1)

output_text = r.stdout.strip()
# With -s flag, first line is the raw relevance number
# Without -s, it's in format "Relevance:  XX.XXX"
rel_match = re.search(r'([\d.Ee+-]+)', output_text)
relevance = float(rel_match.group(1)) if rel_match else -1.0
# The JAR -r flag does not output costOfBackgroundModel, so we cannot
# compute the theoretically correct normalized score (relevance / costOfBackgroundModel).
# Store normalized as None; the notebook's score_m3() uses a per-dataset global max fallback.
normalized = None

print(f"  Relevance: {relevance:.4f}")
print(f"  Runtime: {elapsed:.1f}s")

for m in target_miners:
    write_config(m, {
        "entropic_relevance_raw": relevance,
        "entropic_relevance_normalized": normalized,
        "runtime_s": elapsed,
        "note": "Raw entropic relevance (unbounded). Same for all miners (DFG-based). Normalized version requires costOfBackgroundModel which the JAR -r flag does not output.",
    })
print(f"  → {len(target_miners)} configs written")
