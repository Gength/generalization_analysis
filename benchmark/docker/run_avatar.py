#!/usr/bin/env python3
"""Run AVATAR (M5) via Docker — single command.
Usage:  uv run python benchmark/docker/run_avatar.py [--quick]
"""
import subprocess, os, sys, json, time, glob, re, argparse
from datetime import datetime, timezone
from collections import defaultdict

# =============================================================================
# CONFIGURATION — Change these for your experiment
# =============================================================================
DATASET_KEY = "D1"            # Which dataset (D1-D5)
QUICK_MODE = False              # True = 3 pre-epochs + 100 adv steps; False = 100 + 5000
GPU_ID = "0"
DOCKER_IMAGE = "avatar-tf1"
MINER_LIST = None
EVAL_ONLY = False
TF2_MODE = False

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from datasets import DATASETS, get_info, CONFIG_DIR_V2

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AVATAR_DIR = "src/AVATAR"

# ── CLI ─────────────────────────────────────────────────────────────────────
_cli = argparse.ArgumentParser(description="AVATAR (M5) via Docker")
_cli.add_argument("--dataset", default=None, choices=list(DATASETS.keys()),
                  help="Override DATASET_KEY (default: D1)")
_cli.add_argument("--miners", nargs="*", default=None,
                  help="Restrict to specific miners (default: all)")
_cli.add_argument("--eval-only", action="store_true",
                  help="Skip training/sampling, reuse existing checkpoint")
_cli.add_argument("--tf2", action="store_true",
                  help="Use TF2 container (avatar-tf2) instead of TF1 (avatar-tf1)")
_args, _ = _cli.parse_known_args()
if _args.dataset:
    DATASET_KEY = _args.dataset
if _args.miners is not None:
    MINER_LIST = _args.miners
if _args.eval_only:
    EVAL_ONLY = True
if getattr(_args, "tf2", False):
    TF2_MODE = True
    DOCKER_IMAGE = "avatar-tf2"
    AVATAR_DIR = "src/AVATAR_tf2"

# =============================================================================
# EXECUTION — Do not edit below this line
# =============================================================================

def docker_kill_all():
    """Kill stale AVATAR Docker containers (avatar-tf1 / avatar-tf2) to prevent GPU conflicts."""
    import subprocess
    for image in ["avatar-tf1", "avatar-tf2"]:
        try:
            running = subprocess.run(
                ["docker", "ps", "-q", "--filter", f"ancestor={image}"],
                capture_output=True, text=True, timeout=10
            )
            ids = running.stdout.strip().split()
            if ids:
                subprocess.run(["docker", "kill"] + ids, capture_output=True, timeout=10)
                print(f"🧹 Killed {len(ids)} stale {image} container(s)")
        except Exception as e:
            print(f"⚠️  docker_kill_all({image}) warning: {e}")

info = get_info(DATASET_KEY)
SYSTEM_NAME = info["system_name"]
LOG_PATH = os.path.join(PROJ, info["log_path"])
CONFIG_DIR = os.path.join(PROJ, CONFIG_DIR_V2)
AVATAR_ABS = os.path.join(PROJ, AVATAR_DIR)
VARIANT_DIR = os.path.join(AVATAR_ABS, "data/variants")
os.makedirs(VARIANT_DIR, exist_ok=True)
variant_file = os.path.join(VARIANT_DIR, f"{SYSTEM_NAME}_train.txt")

import pm4py

if EVAL_ONLY:
    print(f"--eval-only: reusing existing artifacts (variant, checkpoint, sampling)")
elif not os.path.exists(variant_file):
    print(f"Creating variant file: {variant_file}")
    log = pm4py.read_xes(LOG_PATH)
    log = pm4py.convert_to_event_log(log)
    variant_map = defaultdict(list)
    for t in log:
        seq = tuple(e["concept:name"] for e in t)
        variant_map[seq].append(t)
    # Write full population
    with open(variant_file, "w") as f:
        for v in variant_map:
            f.write(" ".join(v) + "\n")
    # Also write as _pop.txt (full population)
    pop_file = os.path.join(VARIANT_DIR, f"{SYSTEM_NAME}_pop.txt")
    with open(pop_file, "w") as f:
        for v in variant_map:
            f.write(" ".join(v) + "\n")
    # 80/20 train/test split
    variants = list(variant_map.keys())
    split = int(len(variants) * 0.8)
    train_variants = variants[:split]
    test_variants = variants[split:]
    with open(variant_file, "w") as f:
        for v in train_variants:
            f.write(" ".join(v) + "\n")
    test_file = os.path.join(VARIANT_DIR, f"{SYSTEM_NAME}_test.txt")
    with open(test_file, "w") as f:
        for v in test_variants:
            f.write(" ".join(v) + "\n")
    print(f"  Wrote {len(variant_map)} variants ({len(train_variants)} train, {len(test_variants)} test)")
else:
    print(f"Variant file exists: {variant_file}")

npre = "3" if QUICK_MODE else "100"
nadv = "100" if QUICK_MODE else "5000"
mode = "QUICK" if QUICK_MODE else "FULL"

AVATAR_ABS = os.path.join(PROJ, AVATAR_DIR)

if EVAL_ONLY:
    # Find suffix from existing sampling output
    from miners import filtered_trace_miner
    variant_dir_samples = os.path.join(AVATAR_ABS, "data/avatar/variants")
    sampled_files = glob.glob(os.path.join(variant_dir_samples, f"{SYSTEM_NAME}_relgan_*_j0_naive.txt"))
    suffix = None
    for f in sampled_files:
        m = re.search(r'_relgan_(\d+)_j0_naive\.txt$', f)
        if m:
            ckpts = glob.glob(os.path.join(AVATAR_ABS, f"data/avatar/sgans/{SYSTEM_NAME}/0/tf_logs/ckpt/*.meta"))
            if any(f"adv_model-{m.group(1)}." in c for c in ckpts):
                suffix = m.group(1)
                break
    if suffix:
        print(f"--eval-only: suffix={suffix} (reusing existing checkpoint + sampling)")
    else:
        print("--eval-only failed: no matching checkpoint+sampling pair found!")
        sys.exit(1)
else:
    # Full training + sampling
    print(f"AVATAR {mode}: {npre} pre-epochs, {nadv} adv steps")
    docker_cmd = [
        "docker", "run", "--rm", "--gpus", "all", "--ipc=host",
        "-v", f"{AVATAR_ABS}:/workspace/src/AVATAR",
        "-w", "/workspace/src/AVATAR",
        "-e", "AVATAR_NPRE_EPOCHS=" + npre,
        "-e", "AVATAR_NADV_STEPS=" + nadv,
        "-e", "AVATAR_BATCH_SIZE=16",
        "-e", "CUDA_VISIBLE_DEVICES=0",
        "-e", "TF_XLA_FLAGS=--tf_xla_auto_jit=2",
        DOCKER_IMAGE,
        "python", "-u", "-m", "avatar.training",
        "-s", SYSTEM_NAME, "-j", "0", "-gpu", "0", "-n", "10000",
    ]
    print(f"Training with Docker image '{DOCKER_IMAGE}'...")
    docker_kill_all()
    t0 = time.time()
    r = subprocess.run(docker_cmd, capture_output=True, text=True)
    print(f"Training: {time.time()-t0:.0f}s  Exit={r.returncode}")
    lines = r.stdout.strip().split("\n")
    for l in lines[-5:]:
        print(f"  {l}")
    if r.stderr:
        err = [l for l in r.stderr.split("\n") if "Traceback" in l or "Error" in l or "error" in l]
        for l in err[-3:]:
            print(f"  ERR: {l}")

    # Find checkpoint suffix from checkpoints
    ckpts = glob.glob(os.path.join(AVATAR_ABS, f"data/avatar/sgans/{SYSTEM_NAME}/0/tf_logs/ckpt/*.meta"))
    if ckpts:
        suffixes = []
        for f in ckpts:
            m = re.search(r'\.(\d+)\.meta$', f)
            if m: suffixes.append(int(m.group(1)))
        suffix = str(max(suffixes)) if suffixes else "5000"
        print(f"Checkpoint: suffix={suffix}")
    else:
        print("No checkpoints found!")
        sys.exit(1)

    print("\nSampling...")
    docker_cmd = [
        "docker", "run", "--rm", "--gpus", "all", "--ipc=host",
        "-v", f"{AVATAR_ABS}:/workspace/src/AVATAR",
        "-w", "/workspace/src/AVATAR",
        "-e", "CUDA_VISIBLE_DEVICES=0",
        "-e", "TF_XLA_FLAGS=--tf_xla_auto_jit=2",
        DOCKER_IMAGE,
        "python", "-u", "-m", "avatar.sampling",
        "-s", SYSTEM_NAME, "-j", "0", "-sfx", suffix, "-gpu", "0",
        "-strategy", "naive", "-n_n", "10000",
    ]
    docker_kill_all()
    t0 = time.time()
    r = subprocess.run(docker_cmd, capture_output=True, text=True)
    print(f"Sampling: {time.time()-t0:.0f}s  Exit={r.returncode}")

    from miners import filtered_trace_miner

# Generalization for each miner
miners = [
    ("Alpha", pm4py.discover_petri_net_alpha),
    ("Alpha+", pm4py.discover_petri_net_alpha_plus),
    ("Heuristics", pm4py.discover_petri_net_heuristics),
    ("Heuristics_Strict", lambda l: pm4py.discover_petri_net_heuristics(l, dependency_threshold=0.99)),
    ("Inductive_Strict", lambda l: pm4py.discover_petri_net_inductive(l, noise_threshold=0.0)),
    ("Inductive_Infrequent", lambda l: pm4py.discover_petri_net_inductive(l, noise_threshold=0.2)),
    ("Flower", lambda l: (None, None, None)),
    ("Trace_Filtered", lambda l: filtered_trace_miner(l, top_k=50)),
]

from pm4py.objects.log.obj import EventLog

# Load discovery log
log = pm4py.read_xes(LOG_PATH)
log = pm4py.convert_to_event_log(log)
# 80/20 split
variant_map2 = defaultdict(list)
for t in log:
    variant_map2[tuple(e["concept:name"] for e in t)].append(t)
variants = list(variant_map2.keys())
split = int(len(variants) * 0.8)
discovery_variants = variants[split:]
disc_traces = []
for v in discovery_variants:
    disc_traces.extend(variant_map2[v])
discovery_log = EventLog(disc_traces)

os.makedirs(CONFIG_DIR, exist_ok=True)
PNML_DIR = os.path.join(AVATAR_ABS, "data/pns", SYSTEM_NAME)
os.makedirs(PNML_DIR, exist_ok=True)

from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils

def flower_model(log):
    acts = set(e["concept:name"] for t in log for e in t)
    net = PetriNet("Flower")
    p = PetriNet.Place("p0")
    net.places.add(p)
    im = Marking({p: 1})
    fm = Marking({p: 1})
    for a in acts:
        t = PetriNet.Transition(a, a)
        net.transitions.add(t)
        petri_utils.add_arc_from_to(p, t, net)
        petri_utils.add_arc_from_to(t, p, net)
    return net, im, fm

if MINER_LIST is not None:
    miners = [(n, f) for n, f in miners if n in MINER_LIST]

for name, miner_fn in miners:
    print(f"\n[{name}] ", end="", flush=True)
    try:
        if name == "Flower":
            net, im, fm = flower_model(discovery_log)
        else:
            net, im, fm = miner_fn(discovery_log)
        pm4py.write_pnml(net, im, fm, f"{PNML_DIR}/{name}.pnml")
    except Exception as e:
        print(f"Discovery error: {e}")
        continue

    # Run generalization in Docker
    docker_kill_all()
    docker_cmd = [
        "docker", "run", "--rm", "--gpus", "all", "--ipc=host",
        "-v", f"{AVATAR_ABS}:/workspace/src/AVATAR",
        "-w", "/workspace/src/AVATAR",
        "-e", "CUDA_VISIBLE_DEVICES=0",
        "-e", "TF_XLA_FLAGS=--tf_xla_auto_jit=2",
        DOCKER_IMAGE,
        "python", "-u", "-m", "avatar.generalization",
        "-s", SYSTEM_NAME, "-sfx", suffix, "-j", "0",
        "-pn", f"{name}.pnml", "-strategy", "naive",
    ]
    t0 = time.time()
    r = subprocess.run(docker_cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    score = -1
    for line in r.stdout.split("\n"):
        if "AVATAR Generalization=" in line:
            try: score = float(line.split("=")[-1].strip())
            except: pass

    config = {
        "dataset": info["name"], "miner": name, "method": "M5",
        "method_label": "AVATAR (RelGAN)",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": "docker", "seed": 42,
        "parameters": {"GAN": "RelGAN", "suffix": suffix, "strategy": "naive",
                       "npre_epochs": npre, "nadv_steps": nadv, "env": "docker-tf1.15"},
        "results": {"score": score, "runtime_s": elapsed},
        "notes": "",
    }
    tag = "_TF2" if TF2_MODE else ""
    path = os.path.join(CONFIG_DIR, f"{info['name']}__{name}__M5{tag}.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    marker = "✅" if score >= 0 else "❌"
    print(f"{marker} score={score:.4f} ({elapsed:.0f}s)")
    print(f"  -> {path}")

print("\nDone!")
