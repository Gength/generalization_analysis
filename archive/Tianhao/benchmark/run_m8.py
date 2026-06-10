"""
M8 — Pattern-based Generalization (PatternGeneralization JAR)
=============================================================
Reads manifest.json, calls GeneralizationEvaluation with
PatternBasedGeneralization approach.

Status: All miners on D1 Sepsis return "t/out".
  - Alpha: t/out at 1043s (17min)
  - Alpha+: never returned within 30min
  - Root cause: PatternGeneralization JAR's ILP-based pattern matching
    is too slow on real models AND the JAR catches all exceptions
    (NPE, OOM, etc.) and converts them to "t/out"
"""
import os, json, subprocess, time, glob
from datetime import datetime, timezone

JAR_DIR = "src/AutomataConformance/out/artifacts/PatternGeneralization_jar"
MAIN_CLASS = "au.unimelb.evaluation.GeneralizationEvaluation"
CONFIG_DIR = "benchmark/results/configs"
MODEL_DIR = "benchmark/models"
os.makedirs(CONFIG_DIR, exist_ok=True)

CP = f"{JAR_DIR}/PatternGeneralization.jar"
for j in sorted(glob.glob(f"{JAR_DIR}/*.jar")):
    if "PatternGeneralization" not in j:
        CP += f":{j}"

with open(f"{MODEL_DIR}/manifest.json") as f:
    manifest = json.load(f)

DATASET = manifest["dataset"]
XES = os.path.basename(manifest["xes_file"])

NOTE = ("Pattern-based Generalization unavailable: "
        "PatternGeneralization JAR returns 't/out' for all miners. "
        "Alpha took 1043s then crashed; Alpha+ never returned within 30min. "
        "JAR catches all exceptions and converts to 't/out'.")

print("M8 — Quick test on Inductive_Strict")
t0 = time.time()
r = subprocess.run(
    ["java", "-cp", CP, MAIN_CLASS,
     f"{MODEL_DIR}/", XES, "Inductive_Strict.pnml",
     "PatternBasedGeneralization", "global", "PartialMatching", "0.02",
     "1", "MINUTES"],
    capture_output=True, text=True, timeout=90)
elapsed = time.time() - t0

out = r.stdout.strip()
print(f"  Result ({elapsed:.0f}s): {out[:100]}")
print()

print("M8 — Writing configs (all miners unavailable)...")
for miner_name in manifest["miners"]:
    config = {
        "dataset": DATASET, "miner": miner_name, "method": "M8",
        "method_label": "Pattern-based Generalization",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": "local", "seed": 42,
        "parameters": {"jar": "PatternGeneralization", "oracle": "global",
                       "matching": "PartialMatching", "noise_threshold": 0.02},
        "results": {"overall_gen": -1, "runtime_s": 0, "note": NOTE},
        "notes": NOTE,
    }
    with open(f"{CONFIG_DIR}/{DATASET}__{miner_name}__M8.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ {miner_name}")

print(f"Done → {CONFIG_DIR}/")
