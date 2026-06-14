"""
Demo: D1 Sepsis — Full Benchmark Run (Methodology v2)
=========================================================
Runs all M1-family methods (M1a–M1g: v1.0→v2.6) + M2 + R3 on Sepsis (1,050 traces).
Output: benchmark/results/configs_v2/{Dataset}__{Miner}__{Method}.json
"""

import os, sys, json, time, random, argparse
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np
import pm4py
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.algo.evaluation.generalization import algorithm as gen_eval
from pm4py.algo.evaluation.replay_fitness import algorithm as rf_eval
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay

# Ensure HybridGen is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from HybridGen.algorithm import load_algorithm

# =============================================================================
# CONFIGURATION — Change these for your experiment
# =============================================================================
DATASET_KEY = "D1"            # Which dataset (D1-D5)
MINER_LIST = None              # None = all miners, or ["Alpha", ...]
SEED = 42

# HybridGen hyperparams (from BenchmarkDesign.md)
HPARAMS = {
    "max_n": 6,
    "safe_threshold": 5,
    "num_shadow_traces": 1000,
    "iterations": 5,
}

DATASETS = {
    "D1": {
        "name": "Sepsis",
        "log_path": "data/Sepsis Cases - Event Log_1_all/Sepsis Cases - Event Log.xes.gz",
        "results_dir": "benchmark/results",
        "config_dir": "benchmark/results/configs_v2",
    },
}
random.seed(SEED)
np.random.seed(SEED)

# ── CLI override for MINER_LIST ─────────────────────────────────────────────
_cli = argparse.ArgumentParser(description="D1 Sepsis benchmark (M1a–M1g + M2 + R3)")
_cli.add_argument("--miners", nargs="*", default=None,
                  help="Restrict to specific miners (default: all)")
_args, _ = _cli.parse_known_args()
if _args.miners is not None:
    MINER_LIST = _args.miners

# =============================================================================
# EXECUTION — Do not edit below this line
# =============================================================================

info = DATASETS.get(DATASET_KEY, list(DATASETS.values())[0])
DATASET_NAME = info["name"]
LOG_PATH = info["log_path"]
RESULTS_DIR = info["results_dir"]
CONFIG_DIR = info["config_dir"]
BRIDGES_DIR = "benchmark/bridges"
ENTROPIA_JAR = "src/codebase/jbpt-pm/entropia/jbpt-pm-entropia-1.7.jar"

os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(BRIDGES_DIR, exist_ok=True)

# =========================================================================
# 1. Load log & discover models
# =========================================================================
print("=" * 70)
print("D1 Sepsis — Full Benchmark Demo")
print("=" * 70)

print(f"\n[1] Loading {LOG_PATH}...")
log = pm4py.read_xes(LOG_PATH)
log = pm4py.convert_to_event_log(log)
print(f"    {len(log)} traces, {sum(len(t) for t in log)} events")

from miners import MINERS  # defined in benchmark/miners.py, imported here to avoid circular imports

# =========================================================================
# Helpers
# =========================================================================
FLOWER_MODEL_NOTE = "Flower model — all activities in one loop; Gen ≈ 1.0 (construct-purity litmus)"
TRACE_MODEL_NOTE = "Filtered Trace Model — top-50 variants; 0.0 pole (memorization)"

def write_config(dataset, miner, method, method_label, params, results, notes=""):
    """Write a JSON config file for one cell."""
    config = {
        "dataset": dataset,
        "miner": miner,
        "method": method,
        "method_label": method_label,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": "local",
        "seed": SEED,
        "parameters": params,
        "results": results,
        "notes": notes,
    }
    safe_miner = miner.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
    fname = f"{CONFIG_DIR}/{dataset}__{safe_miner}__{method}.json"
    with open(fname, "w") as f:
        json.dump(config, f, indent=2)
    print(f"      ✓ wrote {fname}")
    return config

# =========================================================================
# 2. Run methods for each miner
# =========================================================================
target_miners = dict(MINERS)
if MINER_LIST is not None:
    target_miners = {k: v for k, v in MINERS.items() if k in MINER_LIST}

for miner_name, miner_fn in target_miners.items():
    print(f"\n[2] Miner: {miner_name}")
    print(f"    {'─' * 60}")

    notes = ""
    if miner_name == "Flower":
        notes = FLOWER_MODEL_NOTE
    if miner_name == "Trace_Filtered":
        notes = TRACE_MODEL_NOTE

    # --- 2a. Discover model ---
    t0 = time.time()
    net, im, fm = miner_fn(log)
    disc_time = time.time() - t0
    print(f"    Discovered: {len(net.transitions)} transitions, {len(net.places)} places ({disc_time:.1f}s)")

    # Check fitness — skip if < 0.5
    try:
        fit_res = rf_eval.apply(log, net, im, fm, variant=rf_eval.Variants.TOKEN_BASED)
        base_fitness = fit_res["log_fitness"]
        print(f"    PM4Py Fitness: {base_fitness:.4f}")
        if base_fitness < 0.5:
            notes += " ⚠️ LOW_FITNESS"
            print(f"    ⚠️  Fitness < 0.5 — results may be unreliable")
    except Exception as e:
        base_fitness = 0.0
        notes += f" ⚠️ FITNESS_ERROR"
        print(f"    ⚠️  Fitness error: {e}")

    # --- 2b. M2: PM4Py Built-in Generalization ---
    t0 = time.time()
    try:
        m2_score = gen_eval.apply(log, net, im, fm)
        m2_time = time.time() - t0
    except Exception as e:
        m2_score = 0.0
        m2_time = time.time() - t0
        notes += " ⚠️ M2_ERROR"
    print(f"    M2 (PM4Py Gen): {m2_score:.4f} ({m2_time:.1f}s)")
    write_config(DATASET_NAME, miner_name, "M2", "PM4Py Built-in Gen",
                 {}, {"score": m2_score, "runtime_s": m2_time}, notes)

    # --- 2c. M1a: HybridGen v1.0 (1-gram, no max_n) ---
    v1 = load_algorithm("v1.0")
    t0 = time.time()
    try:
        m1a_mean, m1a_std, m1a_raw = v1.calculate_gen_shadow_stable(
            log, net, im, fm,
            num_traces=HPARAMS["num_shadow_traces"],
            iterations=HPARAMS["iterations"],
        )[:3]
        m1a_time = time.time() - t0
    except Exception as e:
        m1a_mean = m1a_std = 0.0; m1a_raw = []; m1a_time = 0
        notes += " ⚠️ M1a_ERROR"
    print(f"    M1a (v1.0): {m1a_mean:.4f} ± {m1a_std:.4f} ({m1a_time:.1f}s)")
    write_config(DATASET_NAME, miner_name, "M1a", "HybridGen v1.0",
                 {"safe_threshold": HPARAMS["safe_threshold"],
                  "num_shadow_traces": HPARAMS["num_shadow_traces"],
                  "iterations": HPARAMS["iterations"]},
                 {"mean": m1a_mean, "std": m1a_std,
                  "raw_iterations": m1a_raw, "runtime_s": m1a_time}, notes)

    # --- 2d. M1b: HybridGen v2.1 (N=3, flat termination) ---
    v21 = load_algorithm("v2.1")
    t0 = time.time()
    try:
        m1b_mean, m1b_std, m1b_raw = v21.calculate_gen_shadow_stable(
            log, net, im, fm,
            num_traces=HPARAMS["num_shadow_traces"],
            iterations=HPARAMS["iterations"],
            safe_threshold=HPARAMS["safe_threshold"],
            max_n=3,
        )[:3]
        m1b_time = time.time() - t0
    except Exception as e:
        m1b_mean = m1b_std = 0.0; m1b_raw = []; m1b_time = 0
        notes += " ⚠️ M1b_ERROR"
    print(f"    M1b (v2.1 N=3): {m1b_mean:.4f} ± {m1b_std:.4f} ({m1b_time:.1f}s)")
    write_config(DATASET_NAME, miner_name, "M1b", "HybridGen v2.1 (N=3)",
                 {"max_n": 3, "safe_threshold": HPARAMS["safe_threshold"],
                  "num_shadow_traces": HPARAMS["num_shadow_traces"],
                  "iterations": HPARAMS["iterations"]},
                 {"mean": m1b_mean, "std": m1b_std,
                  "raw_iterations": m1b_raw, "runtime_s": m1b_time}, notes)

    # --- 2e. M1c: HybridGen v2.1 (N=6, flat termination) ---
    t0 = time.time()
    try:
        m1c_mean, m1c_std, m1c_raw = v21.calculate_gen_shadow_stable(
            log, net, im, fm,
            num_traces=HPARAMS["num_shadow_traces"],
            iterations=HPARAMS["iterations"],
            safe_threshold=HPARAMS["safe_threshold"],
            max_n=6,
        )[:3]
        m1c_time = time.time() - t0
    except Exception as e:
        m1c_mean = m1c_std = 0.0; m1c_raw = []; m1c_time = 0
        notes += " ⚠️ M1c_ERROR"
    print(f"    M1c (v2.1 N=6): {m1c_mean:.4f} ± {m1c_std:.4f} ({m1c_time:.1f}s)")
    write_config(DATASET_NAME, miner_name, "M1c", "HybridGen v2.1 (N=6)",
                 {"max_n": 6, "safe_threshold": HPARAMS["safe_threshold"],
                  "num_shadow_traces": HPARAMS["num_shadow_traces"],
                  "iterations": HPARAMS["iterations"]},
                 {"mean": m1c_mean, "std": m1c_std,
                  "raw_iterations": m1c_raw, "runtime_s": m1c_time}, notes)

    # --- 2f. M1d: HybridGen v2.4 (N=6, context-aware termination) ---
    v24 = load_algorithm("v2.3")  # v2.3 = latest (v2.4 is bug-compatible)
    t0 = time.time()
    try:
        m1g_result = v24.evaluate_miner(
            log, miner_name, miner_fn,
            w=0.5, num_shadow_traces=HPARAMS["num_shadow_traces"],
            iterations=HPARAMS["iterations"], seed=SEED, max_n=HPARAMS["max_n"],
        )
        m1g_time = time.time() - t0
        m1g_mean = m1g_result["gen_shadow_mean"]
        m1g_std = m1g_result["gen_shadow_std"]
        m1g_raw = m1g_result["gen_shadow_raw_iterations"]
    except Exception as e:
        m1g_mean = m1g_std = 0.0; m1g_raw = []; m1g_time = 0
        notes += " ⚠️ M1g_ERROR"
        m1g_result = {"gen_total": 0, "gen_shadow_mean": 0, "gen_shadow_std": 0}
    print(f"    M1d (v2.4 N=6): {m1g_mean:.4f} ± {m1g_std:.4f} ({m1g_time:.1f}s)")
    write_config(DATASET_NAME, miner_name, "M1d", "HybridGen v2.4 (N=6)",
                 HPARAMS,
                 {"gen_total": m1g_result.get("gen_total", m1g_mean),
                  "mean": m1g_mean, "std": m1g_std,
                  "raw_iterations": m1g_raw, "runtime_s": m1g_time}, notes)

    # --- 2g. M1e: HybridGen v2.5 (Katz-consistent mutation proposal) ---
    v25 = load_algorithm("v2.5")
    t0 = time.time()
    try:
        m1g_result = v25.evaluate_miner(
            log, miner_name, miner_fn,
            w=0.5, num_shadow_traces=HPARAMS["num_shadow_traces"],
            iterations=HPARAMS["iterations"], seed=SEED, max_n=HPARAMS["max_n"],
        )
        m1g_time = time.time() - t0
        m1g_mean = m1g_result["gen_shadow_mean"]
        m1g_std = m1g_result["gen_shadow_std"]
        m1g_raw = m1g_result.get("gen_shadow_raw_iterations")
    except Exception as e:
        m1g_mean = m1g_std = 0.0; m1g_raw = []; m1g_time = 0
        notes += " ⚠️ M1g_ERROR"
        m1g_result = {"gen_total": 0, "gen_shadow_mean": 0, "gen_shadow_std": 0}
    print(f"    M1e (v2.5 Katz): {m1g_mean:.4f} ± {m1g_std:.4f} ({m1g_time:.1f}s)")
    write_config(DATASET_NAME, miner_name, "M1e", "HybridGen v2.5 (Katz proposal)",
                 HPARAMS,
                 {"gen_total": m1g_result.get("gen_total", m1g_mean),
                  "mean": m1g_mean, "std": m1g_std,
                  "raw_iterations": m1g_raw, "runtime_s": m1g_time}, notes)

    # --- 2h. M1f: HybridGen v2.6 (log weighting) ---
    v26 = load_algorithm("v2.6")
    t0 = time.time()
    try:
        m1g_result = v26.evaluate_miner(
            log, miner_name, miner_fn,
            w=0.5, num_shadow_traces=HPARAMS["num_shadow_traces"],
            iterations=HPARAMS["iterations"], seed=SEED, max_n=HPARAMS["max_n"],
            successor_weighting="log",
        )
        m1g_time = time.time() - t0
        m1g_mean = m1g_result["gen_shadow_mean"]
        m1g_std = m1g_result["gen_shadow_std"]
        m1g_raw = m1g_result.get("gen_shadow_raw_iterations")
    except Exception as e:
        m1g_mean = m1g_std = 0.0; m1g_raw = []; m1g_time = 0
        notes += " ⚠️ M1g_ERROR"
        m1g_result = {"gen_total": 0, "gen_shadow_mean": 0, "gen_shadow_std": 0}
    print(f"    M1f (v2.6 log): {m1g_mean:.4f} ± {m1g_std:.4f} ({m1g_time:.1f}s)")
    write_config(DATASET_NAME, miner_name, "M1f", "HybridGen v2.6 (log weighting)",
                 {"max_n": HPARAMS["max_n"], "safe_threshold": HPARAMS["safe_threshold"],
                  "num_shadow_traces": HPARAMS["num_shadow_traces"],
                  "iterations": HPARAMS["iterations"],
                  "successor_weighting": "log"},
                 {"gen_total": m1g_result.get("gen_total", m1g_mean),
                  "mean": m1g_mean, "std": m1g_std,
                  "raw_iterations": m1g_raw, "runtime_s": m1g_time}, notes)

    # --- 2i. M1g: HybridGen v2.6 (MLE weighting) ---
    t0 = time.time()
    try:
        m1g_result = v26.evaluate_miner(
            log, miner_name, miner_fn,
            w=0.5, num_shadow_traces=HPARAMS["num_shadow_traces"],
            iterations=HPARAMS["iterations"], seed=SEED, max_n=HPARAMS["max_n"],
            successor_weighting="mle",
        )
        m1g_time = time.time() - t0
        m1g_mean = m1g_result["gen_shadow_mean"]
        m1g_std = m1g_result["gen_shadow_std"]
        m1g_raw = m1g_result.get("gen_shadow_raw_iterations")
    except Exception as e:
        m1g_mean = m1g_std = 0.0; m1g_raw = []; m1g_time = 0
        notes += " ⚠️ M1g_ERROR"
        m1g_result = {"gen_total": 0, "gen_shadow_mean": 0, "gen_shadow_std": 0}
    print(f"    M1g (v2.6 mle): {m1g_mean:.4f} ± {m1g_std:.4f} ({m1g_time:.1f}s)")
    write_config(DATASET_NAME, miner_name, "M1g", "HybridGen v2.6 (MLE weighting)",
                 {"max_n": HPARAMS["max_n"], "safe_threshold": HPARAMS["safe_threshold"],
                  "num_shadow_traces": HPARAMS["num_shadow_traces"],
                  "iterations": HPARAMS["iterations"],
                  "successor_weighting": "mle"},
                 {"gen_total": m1g_result.get("gen_total", m1g_mean),
                  "mean": m1g_mean, "std": m1g_std,
                  "raw_iterations": m1g_raw, "runtime_s": m1g_time}, notes)

    # --- 2j. R3: Naive Random Baseline ---
    t0 = time.time()
    try:
        activities = list(set(e["concept:name"] for t in log for e in t))
        lengths = [len(t) for t in log]
        r3_scores = []
        for _ in range(5):
            shadow = EventLog()
            for i in range(HPARAMS["num_shadow_traces"]):
                length = random.choice(lengths)
                seq = random.choices(activities, k=length)
                trace = Trace(attributes={"concept:name": f"rand_{i}"})
                for act in seq:
                    trace.append(Event({"concept:name": act}))
                shadow.append(trace)
            replayed = token_replay.apply(shadow, net, im, fm)
            fits = [r["trace_fitness"] for r in replayed]
            r3_scores.append(sum(fits) / len(fits) if fits else 0.0)
        r3_mean = np.mean(r3_scores)
        r3_std = np.std(r3_scores)
        r3_time = time.time() - t0
    except Exception as e:
        r3_mean = r3_std = 0.0; r3_time = 0; r3_scores = []
        notes += " ⚠️ R3_ERROR"
    print(f"    R3 (Random): {r3_mean:.4f} ± {r3_std:.4f} ({r3_time:.1f}s)")
    write_config(DATASET_NAME, miner_name, "R3", "Naive Random Baseline",
                 {"num_traces": HPARAMS["num_shadow_traces"]},
                 {"mean": r3_mean, "std": r3_std,
                  "raw_iterations": r3_scores, "runtime_s": r3_time}, notes)

    # --- 2k. R2: Sampled Leave-One-Variant-Out ---
    R2_VARIANTS_SAMPLED = 50
    t0 = time.time()
    try:
        variant_groups = defaultdict(list)
        for t in log:
            variant_groups[tuple(e["concept:name"] for e in t)].append(t)
        all_variants = list(variant_groups.keys())
        sampled = random.sample(all_variants, min(R2_VARIANTS_SAMPLED, len(all_variants)))
        r2_fits = []
        for held_out_variant in sampled:
            train_log = EventLog()
            test_traces = []
            for variant, traces in variant_groups.items():
                if variant == held_out_variant:
                    test_traces = traces
                else:
                    for t in traces:
                        train_log.append(t)
            # Discover model on training log
            r2_net, r2_im, r2_fm = miner_fn(train_log)
            # Token replay on held-out traces
            test_log = EventLog()
            for t in test_traces:
                test_log.append(t)
            replayed = token_replay.apply(test_log, r2_net, r2_im, r2_fm)
            fit = sum(r["trace_fitness"] for r in replayed) / len(replayed) if replayed else 0.0
            r2_fits.append(fit)
        r2_mean = float(np.mean(r2_fits))
        r2_std = float(np.std(r2_fits))
        r2_time = time.time() - t0
    except Exception as e:
        r2_mean = r2_std = 0.0; r2_time = 0; r2_fits = []
        notes += " ⚠️ R2_ERROR"
    print(f"    R2 (LOVO sampled {R2_VARIANTS_SAMPLED}/{len(all_variants)}): {r2_mean:.4f} ± {r2_std:.4f} ({r2_time:.1f}s)")
    write_config(DATASET_NAME, miner_name, "R2", "Leave-One-Variant-Out",
                 {"k": len(all_variants), "variants_sampled": R2_VARIANTS_SAMPLED,
                  "variant_based": True},
                 {"mean": r2_mean, "std": r2_std,
                  "raw_fits": r2_fits, "runtime_s": r2_time}, notes)

print(f"\n{'=' * 70}")
print(f"Completed! Config JSONs in {CONFIG_DIR}/")
print(f"Methods: M1a (v1.0), M1b (v2.1 N=3), M1c (v2.1 N=6),")
print(f"         M1d (v2.4), M1e (v2.5 Katz), M1f (v2.6 log), M1g (v2.6 mle),")
print(f"         M2 (PM4Py built-in), R2 (LOVO sampled), R3 (random baseline)")
print(f"Miners: 8 (incl. Trace_Filtered top-50)")
print(f"{'=' * 70}")
