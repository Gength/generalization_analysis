"""
Version comparison benchmark: HybridGen v24 vs v25 vs v26 (+mle) — multi-dataset, multi-seed.

Mirrors the master benchmark protocol (same 7 miners incl. Flower, 5 iterations x
1000 shadow traces) but limited to the HybridGen versions, so algorithmic changes
can be judged quickly and robustly:

  v24      — uniform mutation proposal (benchmark M1 baseline)
  v25      — Katz-consistent mutation proposal + probe-integrity counters
  v26      — v25 + acceptance rate + data-driven length cap (ln-damped sampling)
  v26-mle  — v26 with successor_weighting='mle' (unbiased probe distribution)

Ground truth R1 (variant-based 5-fold CV, 3 shuffles, seed 42) is loaded from
benchmark/results/configs/{Dataset}__{Miner}__R1.json; if missing it is computed
with the identical protocol (r1_demo.py) and a config JSON is written.

Usage:
  uv run python benchmark/version_comparison.py --dataset D1 --seeds 42 1 7 99
  uv run python benchmark/version_comparison.py --dataset D2 --seeds 42

Output: benchmark/results/version_comparison_{D}.csv (one row per
dataset x seed x version x miner) + console summary aggregated across seeds.
"""
import os, sys, json, time, random, argparse
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pm4py
from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from HybridGen.algorithm import load_algorithm

# ── Configuration ────────────────────────────────────────────────────────────
R1_SEED = 42          # ground truth is always the seed-42 protocol
NUM_SHADOW_TRACES = 1000
ITERATIONS = 5
CONFIG_DIR = "benchmark/results/configs"

from datasets import DATASETS

VERSIONS = [
    ("v24", "v24", {}),
    ("v25", "v25", {}),
    ("v26", "v26", {}),
    ("v26-mle", "v26", {"successor_weighting": "mle"}),
]

from miners import MINERS  # defined in benchmark/miners.py, imported here to avoid circular imports

# ── R1 ground truth ──────────────────────────────────────────────────────────
def compute_r1(log, miner_fn):
    random.seed(R1_SEED); np.random.seed(R1_SEED)
    variant_map = defaultdict(list)
    for trace in log:
        variant_map[tuple(e["concept:name"] for e in trace)].append(trace)
    variants = list(variant_map.keys())
    n_variants, K, SHUFFLES = len(variants), 5, 3
    all_fits = []
    for _ in range(SHUFFLES):
        random.shuffle(variants)
        fold_size = max(1, n_variants // K)
        fold_fits = []
        for i in range(K):
            start = i * fold_size
            end = (i + 1) * fold_size if i < K - 1 else n_variants
            test_variants = set(variants[start:end])
            train_log = EventLog([t for v in variants if v not in test_variants for t in variant_map[v]])
            test_log = EventLog([t for v in test_variants for t in variant_map[v]])
            try:
                net, im, fm = miner_fn(train_log)
                fit = replay_fitness.apply(test_log, net, im, fm,
                                           variant=replay_fitness.Variants.TOKEN_BASED)["log_fitness"]
                fold_fits.append(fit)
            except Exception:
                fold_fits.append(0.0)
        all_fits.append(float(np.mean(fold_fits)))
    return float(np.mean(all_fits)), float(np.std(all_fits)), all_fits

def load_or_compute_r1(dataset_name, log, miner_name, miner_fn):
    path = f"{CONFIG_DIR}/{dataset_name}__{miner_name}__R1.json"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)["results"]["mean"], "config"
    t0 = time.time()
    mean, std, raw = compute_r1(log, miner_fn)
    config = {
        "dataset": dataset_name, "miner": miner_name, "method": "R1",
        "method_label": "K-Fold CV (k=5)",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": "local", "seed": R1_SEED,
        "parameters": {"k": 5, "shuffles": 3, "variant_based": True},
        "results": {"mean": mean, "std": std, "raw_shuffles": raw,
                    "runtime_s": time.time() - t0},
        "notes": "computed by version_comparison.py",
    }
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return mean, f"fresh ({time.time() - t0:.0f}s)"

# ── Agreement statistics ─────────────────────────────────────────────────────
def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0: return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return pearson(rx, ry)

# ── Run ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="D1", choices=list(DATASETS.keys()))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = ap.parse_args()

    ds = DATASETS[args.dataset]
    out_csv = f"benchmark/results/version_comparison_{args.dataset}.csv"
    t_start = time.time()

    print("=" * 78)
    print(f"Version comparison — {args.dataset} {ds['name']}, seeds {args.seeds}, "
          f"{ITERATIONS}x{NUM_SHADOW_TRACES} shadow traces")
    print("=" * 78)
    log = pm4py.read_xes(ds["log_path"])
    log = pm4py.convert_to_event_log(log)
    print(f"Loaded {len(log)} traces, "
          f"{len(set(tuple(e['concept:name'] for e in t) for t in log))} variants")

    # Discover each model ONCE; reuse across versions and seeds.
    print("\nDiscovering models (cached for all versions/seeds):")
    model_cache = {}
    for miner_name, miner_fn in MINERS.items():
        t0 = time.time()
        model_cache[miner_name] = miner_fn(log)
        print(f"  {miner_name:22s} {time.time() - t0:6.1f}s")
    cached_fn = {name: (lambda l, n=name: model_cache[n]) for name in MINERS}

    print("\nGround truth R1 (variant-based 5-fold CV, 3 shuffles, seed 42):")
    r1 = {}
    for miner_name, miner_fn in MINERS.items():
        val, source = load_or_compute_r1(ds["name"], log, miner_name, miner_fn)
        r1[miner_name] = val
        print(f"  R1[{miner_name}] = {val:.4f} ({source})")

    rows = []
    for seed in args.seeds:
        for label, version, kwargs in VERSIONS:
            algo = load_algorithm(version)
            print(f"\n--- seed {seed} | {label} ---")
            for miner_name in MINERS:
                r = algo.evaluate_miner(log, miner_name, cached_fn[miner_name],
                                        num_shadow_traces=NUM_SHADOW_TRACES,
                                        iterations=ITERATIONS, seed=seed, **kwargs)
                rows.append({
                    "Dataset": args.dataset, "Seed": seed,
                    "Version": label, "Miner": miner_name,
                    "Gen": round(r["gen_total"], 4),
                    "Std": round(r["gen_shadow_std"], 4),
                    "Accept": round(r["gen_accept"], 4) if "gen_accept" in r else "",
                    "Fit_Regular": round(r["gen_shadow_regular"], 4) if "gen_shadow_regular" in r else "",
                    "Fit_Mutated": round(r["gen_shadow_mutated"], 4) if "gen_shadow_mutated" in r else "",
                    "Acc_Regular": round(r["gen_accept_regular"], 4) if "gen_accept_regular" in r else "",
                    "Acc_Mutated": round(r["gen_accept_mutated"], 4) if "gen_accept_mutated" in r else "",
                    "Mut_Traces": str(r.get("mutated_traces_per_iteration", "")),
                    "Dup_Kept": r.get("duplicates_kept", ""),
                    "Truncated": r.get("truncated_traces", ""),
                    "Cap": r.get("max_trace_length_used", ""),
                    "R1": round(r1[miner_name], 4),
                    "Runtime_s": round(r["runtime_s"], 2),
                })
                print(f"    {miner_name:22s} gen={r['gen_total']:.4f} "
                      f"({r['runtime_s']:.1f}s)")

    # ── CSV ──
    import csv
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV -> {out_csv}")

    # ── Summary aggregated across seeds ──
    miners_nf = [m for m in MINERS if m != "Flower"]
    y = [r1[m] for m in miners_nf]
    by = {(s, l): {row["Miner"]: row for row in rows
                   if row["Seed"] == s and row["Version"] == l}
          for s in args.seeds for l, _, _ in VERSIONS}

    print("\n" + "=" * 78)
    print(f"SUMMARY — {args.dataset}, aggregated over {len(args.seeds)} seed(s); "
          f"fitness score vs R1, flower excluded")
    print(f"{'Version':10s} {'Pearson':>14s} {'Spearman':>14s} {'MAE':>14s} "
          f"{'Spread':>14s} {'Flower':>7s} {'s/cell':>7s}")
    print("-" * 78)
    for label, _, _ in VERSIONS:
        ps, sps, maes, sprs, flowers, rts = [], [], [], [], [], []
        for s in args.seeds:
            cells = by[(s, label)]
            x = [float(cells[m]["Gen"]) for m in miners_nf]
            ps.append(pearson(x, y)); sps.append(spearman(x, y))
            maes.append(float(np.mean(np.abs(np.array(x) - np.array(y)))))
            sprs.append(max(x) - min(x))
            flowers.append(float(cells["Flower"]["Gen"]))
            rts += [float(cells[m]["Runtime_s"]) for m in MINERS]
        def ms(v): return f"{np.mean(v):.3f}+-{np.std(v):.3f}"
        print(f"{label:10s} {ms(ps):>14s} {ms(sps):>14s} {ms(maes):>14s} "
              f"{ms(sprs):>14s} {min(flowers):7.3f} {np.mean(rts):7.1f}")
    print("=" * 78)
    print(f"Total wall-clock: {(time.time() - t_start) / 60:.1f} min")
    print("Litmus: a pure generalization score should give Flower ~ 1.0.")

if __name__ == "__main__":
    main()
