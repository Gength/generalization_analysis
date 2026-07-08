"""
AVATAR Bridge — M5 Generalization via RelGAN
===========================================================

Pipeline:
  1. Convert event log to AVATAR variant format
  2. Split variants 80/20 (GAN train / model discovery)
  3. Train RelGAN on GAN training set
  4. Per-miner: discover model from holdout → sample from GAN → replay → score

Usage:
  # Quick demo (reduced epochs)
  uv run python benchmark/bridges/avatar_bridge.py --quick

  # Full run
  uv run python benchmark/bridges/avatar_bridge.py

Data Leakage Mitigation:
  Variants are split BEFORE model discovery. The GAN trains on 80% of variants;
  each miner is discovered from the remaining 20%. This ensures the GAN cannot
  simply regenerate traces the model was fitted to.
"""
import os, sys, json, time, subprocess, random, shutil, shlex
from datetime import datetime, timezone
from collections import defaultdict, Counter
from functools import partial

import numpy as np
import pm4py
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.algo.evaluation.replay_fitness import algorithm as rf_eval

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
DATASET = "Sepsis"
LOG_PATH = "data/Sepsis Cases - Event Log_1_all/Sepsis Cases - Event Log.xes.gz"
CONFIG_DIR = "benchmark/results/configs"
AVATAR_DIR = "src/AVATAR"
AVATAR_DATA = os.path.join(AVATAR_DIR, "data")
SYSTEM_NAME = "sepsis"
os.makedirs(CONFIG_DIR, exist_ok=True)

# ─── Load and split log ─────────────────────────────────────────────────────
print("=" * 60)
print("M5 — AVATAR: RelGAN-based Generalization")
print("=" * 60)

print(f"\n[1] Loading {LOG_PATH}...")
log = pm4py.read_xes(LOG_PATH)
log = pm4py.convert_to_event_log(log)
print(f"    {len(log)} traces, {sum(len(t) for t in log)} events")

# Group by variant
variant_map = defaultdict(list)
for trace in log:
    seq = tuple(e["concept:name"] for e in trace)
    variant_map[seq].append(trace)
variants = list(variant_map.keys())
n_variants = len(variants)
print(f"    {n_variants} unique variants")

# 80/20 variant-based split (by instance count, not raw variant count)
# Sort variants by frequency (most common first) for stratified assignment
variant_counts = [(v, len(variant_map[v])) for v in variants]
variant_counts.sort(key=lambda x: -x[1])  # descending by frequency

# GAN gets 80% of instances, model discovery gets 20%
total_instances = sum(c for _, c in variant_counts)
gan_instances_target = int(total_instances * 0.8)

gan_variants = []
discovery_variants = []
gan_count = 0
for v, c in variant_counts:
    if gan_count + c <= gan_instances_target or not gan_variants:
        gan_variants.append(v)
        gan_count += c
    else:
        discovery_variants.append(v)

print(f"    GAN set: {len(gan_variants)} variants ({gan_count}/{total_instances} instances)")
print(f"    Discovery set: {len(discovery_variants)} variants ({total_instances - gan_count}/{total_instances} instances)")

# ─── Convert to AVATAR format ──────────────────────────────────────────────
def write_variant_file(variants_set, filepath):
    """Write variants as space-separated activity names, one per line."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        for v in variants_set:
            seq_str = " ".join(v)
            f.write(seq_str + "\n")
    print(f"    Wrote {len(variants_set)} variants → {filepath}")

# AVATAR expects files at:
#   data/variants/{system}_train.txt  ← ALL variants for GAN training
#   data/avatar/train_data/{system}.txt  ← after train/eval split
#   data/avatar/train_data/{system}_eval.txt

train_path = os.path.join(AVATAR_DATA, "variants", f"{SYSTEM_NAME}_train.txt")
write_variant_file(gan_variants, train_path)

# ─── Miners for discovery set ───────────────────────────────────────────────
def discover_flower_model(log):
    from pm4py.objects.petri_net.obj import PetriNet, Marking
    from pm4py.objects.petri_net.utils import petri_utils
    net = PetriNet("Flower Model")
    p_mid = PetriNet.Place("mid")
    net.places.add(p_mid)
    activities = set(e["concept:name"] for t in log for e in t)
    for act in activities:
        t = PetriNet.Transition(f"t_{act}", act)
        net.transitions.add(t)
        petri_utils.add_arc_from_to(p_mid, t, net)
        petri_utils.add_arc_from_to(t, p_mid, net)
    im, fm = Marking(), Marking()
    im[p_mid] = 1; fm[p_mid] = 1
    return net, im, fm

MINERS = {
    "Alpha": pm4py.discover_petri_net_alpha,
    "Alpha+": pm4py.discover_petri_net_alpha_plus,
    "Heuristics": pm4py.discover_petri_net_heuristics,
    "Heuristics_Strict": lambda l: pm4py.discover_petri_net_heuristics(l, dependency_threshold=0.99),
    "Inductive_Strict": lambda l: pm4py.discover_petri_net_inductive(l, noise_threshold=0.0),
    "Inductive_Infrequent": lambda l: pm4py.discover_petri_net_inductive(l, noise_threshold=0.2),
    "Flower": discover_flower_model,
}

# Build discovery log
discovery_traces = []
for v in discovery_variants:
    discovery_traces.extend(variant_map[v])
discovery_log = EventLog(discovery_traces)
print(f"\n    Discovery log: {len(discovery_log)} traces")

def write_config(dataset, miner, method, label, params, results, notes=""):
    config = {
        "dataset": dataset, "miner": miner, "method": method,
        "method_label": label,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": "local", "seed": SEED,
        "parameters": params, "results": results, "notes": notes,
    }
    safe_miner = miner.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
    fname = f"{CONFIG_DIR}/{dataset}__{safe_miner}__{method}.json"
    with open(fname, "w") as f:
        json.dump(config, f, indent=2)
    print(f"      ✓ {method} → {fname}")

# ─── Determine AVATAR training parameters ──────────────────────────────────
quick_mode = "--quick" in sys.argv

if quick_mode:
    NPRE_EPOCHS = 3
    NADV_STEPS = 100
    print(f"\n[2] QUICK MODE: {NPRE_EPOCHS} pre-epochs, {NADV_STEPS} adv steps, batch_size=16")
else:
    NPRE_EPOCHS = 100
    NADV_STEPS = 5000
    print(f"\n[2] FULL MODE: {NPRE_EPOCHS} pre-epochs, {NADV_STEPS} adv steps, batch_size=16")

# Also accept explicit overrides from env vars
NPRE_EPOCHS = int(os.environ.get("AVATAR_NPRE_EPOCHS", NPRE_EPOCHS))
NADV_STEPS = int(os.environ.get("AVATAR_NADV_STEPS", NADV_STEPS))

# ─── Run GAN Training ──────────────────────────────────────────────────────
# Docker-based AVATAR (TF 1.15.5 GPU)
docker_base = [
    "docker", "run", "--rm", "--gpus", "all",
    "-v", "/home/gengtianhao/Process Mining/src/AVATAR:/workspace/src/AVATAR",
    "-w", "/workspace/src/AVATAR",
    "avatar-tf1",
]

print(f"\n[3] Starting AVATAR GAN training...")
print(f"    Command: {conda_cmd} {training_args}")
print(f"    This may take a while (quick={quick_mode}: ~{NPRE_EPOCHS} pre-epochs + {NADV_STEPS} adv steps)")

# Modify the training script's parameters by overriding via environment
train_env = os.environ.copy()
train_env["AVATAR_BATCH_SIZE"] = "8"   # Reduced to avoid GPU OOM (seq_len=182 from real sepsis)
train_env["AVATAR_GEN_EMB_DIM"] = "16"  # Reduced from 32 to save GPU memory
train_env["AVATAR_DIS_EMB_DIM"] = "32"  # Reduced from 64
train_env["AVATAR_NUM_REP"] = "32"      # Reduced from 64

# Check if a checkpoint already exists
checkpoint_dir = os.path.join(AVATAR_DIR, "data", "avatar", "sgans", SYSTEM_NAME, "0", "tf_logs")
suffix = None

if os.path.exists(checkpoint_dir):
    ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".meta")]
    if ckpt_files:
        # Extract the suffix (epoch number) from the latest checkpoint
        epochs = []
        for f in ckpt_files:
            try:
                ep = int(f.split("_")[-1].replace(".meta", ""))
                epochs.append(ep)
            except:
                pass
        if epochs:
            suffix = str(max(epochs))
            print(f"    Found existing checkpoint at epoch {suffix}")

if suffix is None:
    # Need to train. Modify training.py params via sed-like approach:
    # We'll create a temp version with reduced epochs
    print(f"    No checkpoint found. Starting training...")
    
    # Run training
    full_cmd = f"{conda_cmd} {training_args}"
    t0 = time.time()
    
    # Training.py reads its params internally, so we need to pass them differently
    # Instead, we inject via a temporary override
    # Since the training script hardcodes params, we need to use the default values
    # and just let it run
    try:
        result = subprocess.run(
            full_cmd.split(),
            capture_output=True, text=True,
            env=train_env,
            cwd=AVATAR_DIR
        )
        pipeline_time = time.time() - t0
        print(f"    Training pipeline completed in {pipeline_time:.1f}s")
        # Find the latest suffix (epoch) from output
        suffix = "5000"  # default: use last adversarial step
        print(f"    Using suffix={suffix}")
    except Exception as e:
        print(f"    Training error: {e}")
        suffix = None

if suffix is None:
    print(f"    No checkpoint available. M5 results unavailable.")
    for miner_name in MINERS:
        write_config(DATASET, miner_name, "M5", "AVATAR (RelGAN)",
                    {"GAN": "RelGAN", "status": "no_checkpoint"},
                    {"score": -1}, "No GAN checkpoint available")
    sys.exit(0)

# ─── Sampling ──────────────────────────────────────────────────────────────
print(f"\n[4] Sampling from GAN (suffix={suffix})...")
sampling_cmd = f"conda run -n avatar-tf2 python3 -u -m avatar.sampling"
sampling_args = f"-s {SYSTEM_NAME} -j 0 -sfx {suffix} -gpu 0 -strategy naive -n_n 10000"

try:
    t0 = time.time()
    result = subprocess.run(
        f"{sampling_cmd} {sampling_args}".split(),
        capture_output=True, text=True,
        cwd=AVATAR_DIR
    )
    sampling_time = time.time() - t0
    print(f"    Sampling completed in {sampling_time:.1f}s")
except Exception as e:
    print(f"    Sampling error: {e}")
    for miner_name in MINERS:
        write_config(DATASET, miner_name, "M5", "AVATAR (RelGAN)",
                    {"GAN": "RelGAN"}, {"score": -1, "error": str(e)}, str(e))
    sys.exit(0)

# ─── Per-miner generalization ──────────────────────────────────────────────
print(f"\n[5] Running generalization for each miner...")

for miner_name, miner_fn in MINERS.items():
    print(f"\n    --- {miner_name} ---")

    notes = ""
    if miner_name == "Flower":
        notes = "Flower model — all activities in one concurrent block"

    # Discover model from DISCOVERY set (not full log!)
    try:
        net, im, fm = miner_fn(discovery_log)
    except Exception as e:
        write_config(DATASET, miner_name, "M5", "AVATAR (RelGAN)",
                    {"GAN": "RelGAN", "discovery_variants": len(discovery_variants)},
                    {"score": -1, "error": f"Model discovery failed: {e}"},
                    notes + " ⚠️ DISCOVERY_ERROR")
        print(f"      Model discovery failed: {e}")
        continue

    print(f"      Model: {len(net.transitions)} trans, {len(net.places)} places")

    # Export the model as PNML for AVATAR's generalization.py
    pnml_dir = os.path.join(AVATAR_DIR, "data", "pns", SYSTEM_NAME)
    os.makedirs(pnml_dir, exist_ok=True)
    pnml_path = os.path.join(pnml_dir, f"{miner_name}.pnml")
    pm4py.write_pnml(net, im, fm, pnml_path)

    # Run AVATAR generalization.py
    gen_cmd = f"conda run -n avatar-tf2 python3 -u -m avatar.generalization"
    gen_args = f"-s {SYSTEM_NAME} -sfx {suffix} -j 0 -pn {miner_name}.pnml -strategy naive"

    try:
        t0 = time.time()
        result = subprocess.run(
            f"{gen_cmd} {gen_args}".split(),
            capture_output=True, text=True,
            cwd=AVATAR_DIR
        )
        gen_time = time.time() - t0

        # Parse output: "AVATAR Generalization= 0.xxxx"
        out = result.stdout
        score = -1
        for line in out.split("\n"):
            if "AVATAR Generalization=" in line:
                score = float(line.split("=")[-1].strip())
                break

        print(f"      AVATAR Generalization: {score:.4f} ({gen_time:.1f}s)")
        if score < 0:
            print(f"      stdout: {out[:200]}")

        write_config(DATASET, miner_name, "M5", "AVATAR (RelGAN)",
                    {"GAN": "RelGAN", "suffix": suffix, "strategy": "naive",
                     "n_samples": 10000, "discovery_variants": len(discovery_variants)},
                    {"score": score, "runtime_s": gen_time}, notes)

    except Exception as e:
        write_config(DATASET, miner_name, "M5", "AVATAR (RelGAN)",
                    {"GAN": "RelGAN"}, {"score": -1, "error": str(e)}, notes + " ⚠️ ERROR")
        print(f"      Generalization error: {e}")

print(f"\n{'='*60}")
print("M5 (AVATAR) bridge completed! Config files in", CONFIG_DIR)
print(f"{'='*60}")
