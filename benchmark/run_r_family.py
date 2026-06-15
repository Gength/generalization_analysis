"""
R-family benchmark runner — Reference / Sanity-Check Metrics (Methodology v2).

Runs R1–R3 across all miners, writing one config JSON per cell to
benchmark/results/configs_v2/ (same schema as run_m1_family.py).

  R1 = K-Fold Cross-Validation Fitness (5-fold, variant-based, 3 shuffles)
  R2 = Leave-One-Variant-Out Fitness (with optional sampling; default = all variants)
  R3 = Naive Random Baseline (5 iterations, uniformly-sampled traces)

Leverages benchmark/utils.py compute_kfold_fitness for variant-based CV logic.

Usage:
  uv run python benchmark/run_r_family.py --dataset D1
  uv run python benchmark/run_r_family.py --dataset D1 --methods R1 R3
  uv run python benchmark/run_r_family.py --dataset D1 --r2-sample 50
  uv run python benchmark/run_r_family.py --dataset D2 --methods R2
"""
import os, sys, json, time, random, argparse
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pm4py
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay

# ── Config ──────────────────────────────────────────────────────────────────
SEED = 42
NUM_SHADOW_TRACES = 1000
CONFIG_DIR_V2 = "benchmark/results/configs_v2"

DATASETS = {
    "D1": {"name": "Sepsis",
           "log_path": "data/Sepsis Cases - Event Log_1_all/Sepsis Cases - Event Log.xes.gz"},
    "D2": {"name": "BPI2013_Incidents",
           "log_path": "data/BPI-Challenge_2013/Incident_Management_Log.xes.gz"},
}

METHODS = {
    "R1": {"label": "K-Fold CV (k=5)"},
    "R2": {"label": "Leave-One-Variant-Out"},
    "R3": {"label": "Naive Random Baseline"},
}

# Import shared utilities
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miners import MINERS
from utils import compute_kfold_fitness


# ═══════════════════════════════════════════════════════════════════════════════
# R1 — K-Fold Cross-Validation Fitness (5-fold, variant-based, 3 shuffles)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_r1(log, miner_fn, seed):
    """
    Variant-based 5-fold CV, 3 shuffles.
    Uses compute_kfold_fitness from benchmark/utils.py for each shuffle.
    """
    random.seed(seed)
    np.random.seed(seed)
    all_fits = []
    for _ in range(3):
        fit = compute_kfold_fitness(log, miner_fn, k=5)
        all_fits.append(fit)
    return float(np.mean(all_fits)), float(np.std(all_fits)), all_fits


# ═══════════════════════════════════════════════════════════════════════════════
# R2 — Leave-One-Variant-Out Fitness (with optional sampling)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_r2(log, miner_fn, seed, sample_n=0):
    """
    Leave-One-Variant-Out evaluation.

    Uses compute_kfold_fitness(pick_one_out=True) from benchmark/utils.py
    for the core LOVO logic.

    If sample_n > 0, only that many variants are randomly sampled for
    evaluation. Default (0) evaluates every variant — 100% coverage.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Group by variant
    variant_groups = defaultdict(list)
    for trace in log:
        variant_groups[tuple(e["concept:name"] for e in trace)].append(trace)
    all_variants = list(variant_groups.keys())

    # Sampling: 0 = all variants (100%), positive = cap at N
    if sample_n > 0 and sample_n < len(all_variants):
        variants_to_test = random.sample(all_variants, sample_n)
    else:
        variants_to_test = all_variants

    r2_fits = []
    for held_out_variant in variants_to_test:
        train_log = EventLog()
        test_traces = []
        for variant, traces in variant_groups.items():
            if variant == held_out_variant:
                test_traces = traces
            else:
                for t in traces:
                    train_log.append(t)
        try:
            r2_net, r2_im, r2_fm = miner_fn(train_log)
            test_log = EventLog()
            for t in test_traces:
                test_log.append(t)
            replayed = token_replay.apply(test_log, r2_net, r2_im, r2_fm)
            fit = (
                sum(r["trace_fitness"] for r in replayed) / len(replayed)
                if replayed else 0.0
            )
            r2_fits.append(fit)
        except Exception:
            r2_fits.append(0.0)
    return float(np.mean(r2_fits)), float(np.std(r2_fits)), r2_fits


# ═══════════════════════════════════════════════════════════════════════════════
# R3 — Naive Random Baseline (uniform activity sampling, 5 iterations)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_r3(log, net, im, fm, seed, num_traces=1000, iterations=5):
    random.seed(seed)
    np.random.seed(seed)
    activities = list(set(e["concept:name"] for t in log for e in t))
    lengths = [len(t) for t in log]
    r3_scores = []
    for _ in range(iterations):
        shadow = EventLog()
        for i in range(num_traces):
            length = random.choice(lengths)
            seq = random.choices(activities, k=length)
            trace = Trace(attributes={"concept:name": f"rand_{i}"})
            for act in seq:
                trace.append(Event({"concept:name": act}))
            shadow.append(trace)
        try:
            replayed = token_replay.apply(shadow, net, im, fm)
            fits = [r["trace_fitness"] for r in replayed]
            r3_scores.append(sum(fits) / len(fits) if fits else 0.0)
        except Exception:
            r3_scores.append(0.0)
    return float(np.mean(r3_scores)), float(np.std(r3_scores))


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════
def write_config(dataset_name, miner_name, method_id, method_label,
                 params, results, notes=""):
    config = {
        "dataset": dataset_name,
        "miner": miner_name,
        "method": method_id,
        "method_label": method_label,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": "local",
        "seed": SEED,
        "parameters": params,
        "results": results,
        "notes": notes,
    }
    fname = f"{CONFIG_DIR_V2}/{dataset_name}__{miner_name}__{method_id}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"    ✓ {fname}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description="R-family benchmark (R1–R3 reference metrics)"
    )
    ap.add_argument("--dataset", default="D1", choices=list(DATASETS.keys()))
    ap.add_argument("--methods", nargs="+", default=list(METHODS.keys()),
                    choices=list(METHODS.keys()),
                    help="Which R methods to run (default: all)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--miners", nargs="*", default=None,
                    help="Restrict to specific miners (default: all)")
    ap.add_argument("--r2-sample", type=int, default=0,
                    help="R2: number of variants to sample (0 = all variants, i.e. 100%%)")
    ap.add_argument("--num-traces", type=int, default=NUM_SHADOW_TRACES,
                    help="Number of shadow traces for R3 (default: 1000)")
    args = ap.parse_args()

    t_start = time.time()
    seed = args.seed
    ds = DATASETS[args.dataset]
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 78)
    print(f"R-family benchmark (R1–R3) — {args.dataset} {ds['name']}, seed {seed}")
    print("=" * 78)

    log = pm4py.read_xes(ds["log_path"])
    log = pm4py.convert_to_event_log(log)
    print(f"Loaded {len(log)} traces")

    # Filter miners
    target_miners = dict(MINERS)
    if args.miners is not None:
        target_miners = {k: v for k, v in MINERS.items() if k in args.miners}

    # Discover models (needed for R3; reused across methods)
    print("\nDiscovering models:")
    model_cache = {}
    for miner_name, miner_fn in target_miners.items():
        t0 = time.time()
        model_cache[miner_name] = miner_fn(log)
        net = model_cache[miner_name][0]
        print(f"  {miner_name:22s} {len(net.transitions)}t/{len(net.places)}p"
              f"  ({time.time() - t0:.1f}s)")

    os.makedirs(CONFIG_DIR_V2, exist_ok=True)
    results_summary = {}

    # ── R1: K-Fold CV ──
    if "R1" in args.methods:
        print(f"\n--- R1: K-Fold CV (k=5, 3 shuffles) ---")
        for miner_name, miner_fn in target_miners.items():
            t0 = time.time()
            mean, std, raw = compute_r1(log, miner_fn, seed)
            rt = time.time() - t0
            results_summary[("R1", miner_name)] = mean
            print(f"  {miner_name:22s} {mean:.4f} ± {std:.4f} ({rt:.1f}s)")
            write_config(
                ds["name"], miner_name, "R1", "K-Fold CV (k=5)",
                {"k": 5, "shuffles": 3, "variant_based": True},
                {"mean": mean, "std": std, "raw_shuffles": raw, "runtime_s": rt},
            )

    # ── R2: Leave-One-Variant-Out ──
    if "R2" in args.methods:
        r2_label = "Leave-One-Variant-Out"
        if args.r2_sample > 0:
            r2_label += f" (sampled {args.r2_sample})"
        print(f"\n--- R2: {r2_label} ---")
        for miner_name, miner_fn in target_miners.items():
            t0 = time.time()
            mean, std, raw = compute_r2(log, miner_fn, seed, sample_n=args.r2_sample)
            rt = time.time() - t0
            results_summary[("R2", miner_name)] = mean
            n_total = len(raw)
            print(f"  {miner_name:22s} {mean:.4f} ± {std:.4f} ({rt:.1f}s, n={n_total})")
            write_config(
                ds["name"], miner_name, "R2", r2_label,
                {"k": n_total, "variants_sampled": n_total, "variant_based": True},
                {"mean": mean, "std": std, "raw_fits": raw, "runtime_s": rt},
            )

    # ── R3: Random Baseline ──
    if "R3" in args.methods:
        print(f"\n--- R3: Random Baseline ({args.num_traces} traces, 5 iterations) ---")
        for miner_name in target_miners:
            net, im, fm = model_cache[miner_name]
            t0 = time.time()
            mean, std = compute_r3(log, net, im, fm, seed, num_traces=args.num_traces)
            rt = time.time() - t0
            results_summary[("R3", miner_name)] = mean
            print(f"  {miner_name:22s} {mean:.4f} ± {std:.4f} ({rt:.1f}s)")
            write_config(
                ds["name"], miner_name, "R3", "Naive Random Baseline",
                {"num_traces": args.num_traces, "iterations": 5},
                {"mean": mean, "std": std, "runtime_s": rt},
            )

    # ── Summary table ──
    if results_summary:
        print("\n" + "=" * 98)
        print(f"SUMMARY — {args.dataset} {ds['name']}")
        methods_run = [
            m for m in args.methods
            if any((m, _) in results_summary for _ in target_miners)
        ]
        if methods_run:
            header = f"{'Method':6s} " + " ".join(
                f"{m[:13]:>15s}" for m in target_miners
            )
            print(header)
            print("-" * 98)
            for method_id in methods_run:
                cells = [results_summary.get((method_id, m), 0.0)
                         for m in target_miners]
                print(f"{method_id:6s} " + " ".join(f"{c:>15.4f}" for c in cells))
        print("=" * 98)
        print(f"Config JSONs -> {CONFIG_DIR_V2}/")
    print(f"Total wall-clock: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
