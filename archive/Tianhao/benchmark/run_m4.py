"""
M4 — Anti-Alignment Generalization (PatternGeneralization JAR)
==============================================================
No hard timeout — let JAR run to completion.
Passes 120 min to JAR so it doesn't self-kill.
If runtime > 10 min, mark ⚠️ SLOW but keep the score.
"""
import os, json, subprocess, time, glob
from datetime import datetime, timezone

# =============================================================================
# CONFIGURATION — Change these for your experiment
# =============================================================================
DATASET_KEY = "D1"            # Which dataset (D1-D5)
MINER_LIST = None              # None = all miners, or ["Alpha", ...]
JAR_TIMEOUT_MIN = 120          # JAR internal timeout (minutes)
SLOW_THRESHOLD_S = 600         # 10 min — mark SLOW if exceeded

DATASETS = {
    "D1": {
        "name": "Sepsis",
        "model_dir": "benchmark/models",
        "manifest": "benchmark/models/manifest.json",
    },
}
JAR_DIR = "src/AutomataConformance/out/artifacts/PatternGeneralization_jar"
MAIN_CLASS = "au.unimelb.evaluation.GeneralizationEvaluation"
CONFIG_DIR = "benchmark/results/configs"
os.makedirs(CONFIG_DIR, exist_ok=True)

# =============================================================================
# EXECUTION — Do not edit below this line
# =============================================================================

info = DATASETS.get(DATASET_KEY, list(DATASETS.values())[0])
DATASET = info["name"]
MODEL_DIR = info["model_dir"]

CP = f"{JAR_DIR}/PatternGeneralization.jar"
for j in sorted(glob.glob(f"{JAR_DIR}/*.jar")):
    if "PatternGeneralization" not in j:
        CP += f":{j}"

with open(f"{MODEL_DIR}/manifest.json") as f:
    manifest = json.load(f)

DATASET = manifest["dataset"]
XES = os.path.basename(manifest["xes_file"])

print("M4 — Anti-Alignment Generalization (no hard timeout)")
print()

target_miners = list(manifest["miners"].keys()) if MINER_LIST is None else MINER_LIST
for miner_name in target_miners:
    info = manifest["miners"][miner_name]
    pnml = os.path.basename(info["pnml"])
    print(f"  [{miner_name}]", end=" ", flush=True)

    t0 = time.time()
    r = subprocess.run(
        ["java", "-cp", CP, MAIN_CLASS,
         f"{MODEL_DIR}/", XES, pnml,
         "AntiAlignmentsGeneralization",
         "120", "MINUTES"],                      # JAR internal timeout = 120 min
        capture_output=True, text=True)           # no subprocess timeout — wait forever

    elapsed = time.time() - t0
    notes = ""
    if elapsed > SLOW_THRESHOLD_S:
        notes = f"⚠️ SLOW ({elapsed:.0f}s)"

    out = r.stdout.strip()
    err = r.stderr.strip()

    # Parse CSV: Log,Model,approach,Execution time,generalization
    data_line = ""
    for line in out.split("\n"):
        line = line.strip()
        if line and line.split(",")[0].strip() == XES:
            data_line = line
            break

    if not data_line:
        print(f"NO OUTPUT ({elapsed:.0f}s)")
        score = -1
        notes += " NO_OUTPUT"
    else:
        parts = data_line.split(",")
        if len(parts) >= 5 and parts[4].strip():
            try:
                score = float(parts[4].strip())
                print(f"score={score:.4f} ({elapsed:.0f}s)")
            except ValueError:
                score = -1
                print(f"t/out or crash ({elapsed:.0f}s)")
        else:
            score = -1
            print(f"no gen col ({elapsed:.0f}s)")

    config = {
        "dataset": DATASET, "miner": miner_name, "method": "M4",
        "method_label": "Anti-Alignment Generalization",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": "local", "seed": 42,
        "parameters": {"jar": "PatternGeneralization", "jar_timeout_min": 120},
        "results": {"score": score, "runtime_s": elapsed},
        "notes": notes.strip(),
    }
    with open(f"{CONFIG_DIR}/{DATASET}__{miner_name}__M4.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"    ✓ {CONFIG_DIR}/{DATASET}__{miner_name}__M4.json")

print(f"\nDone!")
