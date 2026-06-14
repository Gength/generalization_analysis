"""
M1-family benchmark runner — Benchmark Methodology v2 (see BenchmarkDesign_v2.md).

Runs ALL HybridGen versions (M1, M1a-M1f) across EIGHT miners — the original
seven plus the Filtered Trace Model (top-50 variants, the 0.0 pole opposite the
Flower Model) — under the official protocol (seed 42, 5 iterations x 1000 shadow
traces), and writes one config JSON per cell in the official schema.

Results go to a NEW directory (benchmark/results/configs_v2/) so the original
benchmark results (configs/) are never overwritten.

Method mapping (Methodology v2):
  M1a  = HybridGen v1.0 (1-gram DFG)
  M1b  = HybridGen v2.1 (N=3, flat termination)
  M1c  = HybridGen v2.1 (N=6, flat termination)
  M1   = HybridGen v2.4 (uniform mutation proposal, ln-damped)   [v1-methodology baseline]
  M1d  = HybridGen v2.5 (Katz-consistent mutation proposal)
  M1e  = HybridGen v2.6 (acceptance + data-driven cap, ln-damped sampling)
  M1f  = HybridGen v2.6 (successor_weighting='mle')               [headline candidate]

Ground truth R1 (variant-based 5-fold CV, 3 shuffles, seed 42) is copied from
the v1 configs where available and computed fresh otherwise (e.g. for the new
Trace_Filtered miner); R1 configs are also written to configs_v2/.

Usage:
  uv run python benchmark/run_m1_family.py --dataset D1
  uv run python benchmark/run_m1_family.py --dataset D2
  uv run python benchmark/run_m1_family.py --dataset D1 --methods M1d M1e M1f

Summary printed at the end: per-method mean +- std for every miner, plus
agreement with R1 (Pearson / Spearman / MAE / spread over the six real miners)
and the two litmus checks (Flower ~ 1.0, Trace low).
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

# ── Protocol constants (Methodology v2) ─────────────────────────────────────
SEED = 42
NUM_SHADOW_TRACES = 1000
ITERATIONS = 5
TRACE_TOP_K = 50
CONFIG_DIR_V1 = "benchmark/results/configs"
CONFIG_DIR_V2 = "benchmark/results/configs_v2"

DATASETS = {
    "D1": {"name": "Sepsis",
           "log_path": "data/Sepsis Cases - Event Log_1_all/Sepsis Cases - Event Log.xes.gz"},
    "D2": {"name": "BPI2013_Incidents",
           "log_path": "data/BPI-Challenge_2013/Incident_Management_Log.xes.gz"},
}

METHODS = {
    "M1a": {"version": "v1.0", "kwargs": {},                                "label": "HybridGen v1.0 (1-gram DFG)"},
    "M1b": {"version": "v2.1", "kwargs": {"max_n": 3},                      "label": "HybridGen v2.1 (N=3)"},
    "M1c": {"version": "v2.1", "kwargs": {"max_n": 6},                      "label": "HybridGen v2.1 (N=6)"},
    "M1":  {"version": "v2.4", "kwargs": {},                                "label": "HybridGen v2.4 (uniform proposal)"},
    "M1d": {"version": "v2.5", "kwargs": {},                                "label": "HybridGen v2.5 (Katz proposal)"},
    "M1e": {"version": "v2.6", "kwargs": {},                                "label": "HybridGen v2.6 (log weighting)"},
    "M1f": {"version": "v2.6", "kwargs": {"successor_weighting": "mle"},    "label": "HybridGen v2.6 (MLE weighting)"},
}

# ── Miners (Methodology v2: original seven + Filtered Trace Model) ──────────
def flower_miner(log):
    net = PetriNet("Flower Model")
    p_mid = PetriNet.Place("mid")
    net.places.add(p_mid)
    for act in set(e["concept:name"] for t in log for e in t):
        t = PetriNet.Transition(f"t_{act}", act)
        net.transitions.add(t)
        petri_utils.add_arc_from_to(p_mid, t, net)
        petri_utils.add_arc_from_to(t, p_mid, net)
    im, fm = Marking(), Marking()
    im[p_mid] = 1; fm[p_mid] = 1
    return net, im, fm

def filtered_trace_miner(log, top_k=TRACE_TOP_K):
    """Trace Model over the top-K variants (identical to master_benchmark_v24.py).

    The 0.0 pole: memorizes observed behavior, accepts nothing unseen.
    K=50 keeps the net small enough for token replay (full trace models on
    variant-rich logs have 10^4-10^5 transitions and are intractable)."""
    net = PetriNet("Filtered Trace Model")
    p_start, p_end = PetriNet.Place("start"), PetriNet.Place("end")
    net.places.update([p_start, p_end])
    variant_counts = defaultdict(int)
    for t in log:
        variant_counts[tuple(e["concept:name"] for e in t)] += 1
    top_variants = [v for v, c in sorted(variant_counts.items(),
                                         key=lambda i: i[1], reverse=True)[:top_k]]
    for i, variant in enumerate(top_variants):
        prev = p_start
        for j, act in enumerate(variant):
            t = PetriNet.Transition(f"t_{i}_{j}", act)
            net.transitions.add(t)
            petri_utils.add_arc_from_to(prev, t, net)
            if j == len(variant) - 1:
                petri_utils.add_arc_from_to(t, p_end, net)
            else:
                p_next = PetriNet.Place(f"p_{i}_{j}")
                net.places.add(p_next)
                petri_utils.add_arc_from_to(t, p_next, net)
                prev = p_next
    im, fm = Marking(), Marking()
    im[p_start] = 1; fm[p_end] = 1
    return net, im, fm

MINERS = {
    "Trace_Filtered":       filtered_trace_miner,
    "Alpha":                lambda l: pm4py.discover_petri_net_alpha(l),
    "Alpha+":               lambda l: pm4py.discover_petri_net_alpha_plus(l),
    "Heuristics":           lambda l: pm4py.discover_petri_net_heuristics(l),
    "Heuristics_Strict":    lambda l: pm4py.discover_petri_net_heuristics(l, dependency_threshold=0.99),
    "Inductive_Strict":     lambda l: pm4py.discover_petri_net_inductive(l, noise_threshold=0.0),
    "Inductive_Infrequent": lambda l: pm4py.discover_petri_net_inductive(l, noise_threshold=0.2),
    "Flower":               flower_miner,
}
REAL_MINERS = ["Alpha", "Alpha+", "Heuristics", "Heuristics_Strict",
               "Inductive_Infrequent", "Inductive_Strict"]

# ── R1 ground truth ──────────────────────────────────────────────────────────
def compute_r1(log, miner_fn):
    random.seed(SEED); np.random.seed(SEED)
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

def get_r1(dataset_name, log, miner_name, miner_fn):
    """Copy from v1 configs when present, compute fresh otherwise; write to configs_v2."""
    v2_path = f"{CONFIG_DIR_V2}/{dataset_name}__{miner_name}__R1.json"
    if os.path.exists(v2_path):
        with open(v2_path, encoding="utf-8") as f:
            return json.load(f)["results"]["mean"], "configs_v2"
    v1_path = f"{CONFIG_DIR_V1}/{dataset_name}__{miner_name}__R1.json"
    if os.path.exists(v1_path):
        with open(v1_path, encoding="utf-8") as f:
            config = json.load(f)
        config["notes"] = (config.get("notes", "") + " [copied from configs/]").strip()
        source = "copied from configs/"
    else:
        t0 = time.time()
        mean, std, raw = compute_r1(log, miner_fn)
        config = {
            "dataset": dataset_name, "miner": miner_name, "method": "R1",
            "method_label": "K-Fold CV (k=5)",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "host": "local", "seed": SEED,
            "parameters": {"k": 5, "shuffles": 3, "variant_based": True},
            "results": {"mean": mean, "std": std, "raw_shuffles": raw,
                        "runtime_s": time.time() - t0},
            "notes": "computed by run_m1_family.py",
        }
        source = f"fresh ({time.time() - t0:.0f}s)"
    os.makedirs(CONFIG_DIR_V2, exist_ok=True)
    with open(v2_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return config["results"]["mean"], source

# ── Agreement statistics ─────────────────────────────────────────────────────
def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0: return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

def spearman(x, y):
    return pearson(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))

# ── Run ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="D1", choices=list(DATASETS.keys()))
    ap.add_argument("--methods", nargs="+", default=list(METHODS.keys()),
                    choices=list(METHODS.keys()))
    args = ap.parse_args()

    ds = DATASETS[args.dataset]
    t_start = time.time()
    print("=" * 78)
    print(f"M1-family benchmark (Methodology v2) — {args.dataset} {ds['name']}, "
          f"seed {SEED}, {ITERATIONS}x{NUM_SHADOW_TRACES} shadow traces")
    print("=" * 78)
    log = pm4py.read_xes(ds["log_path"])
    log = pm4py.convert_to_event_log(log)
    print(f"Loaded {len(log)} traces")

    print("\nDiscovering models (cached across methods):")
    model_cache = {}
    for miner_name, miner_fn in MINERS.items():
        t0 = time.time()
        model_cache[miner_name] = miner_fn(log)
        net = model_cache[miner_name][0]
        print(f"  {miner_name:22s} {len(net.transitions)}t/{len(net.places)}p"
              f"  ({time.time() - t0:.1f}s)")
    cached_fn = {name: (lambda l, n=name: model_cache[n]) for name in MINERS}

    print("\nGround truth R1:")
    r1 = {}
    for miner_name, miner_fn in MINERS.items():
        val, source = get_r1(ds["name"], log, miner_name, miner_fn)
        r1[miner_name] = val
        print(f"  R1[{miner_name}] = {val:.4f} ({source})")

    os.makedirs(CONFIG_DIR_V2, exist_ok=True)
    results = {}
    for method_id in args.methods:
        spec = METHODS[method_id]
        algo = load_algorithm(spec["version"])
        print(f"\n--- {method_id}: {spec['label']} ---")
        for miner_name in MINERS:
            r = algo.evaluate_miner(log, miner_name, cached_fn[miner_name],
                                    num_shadow_traces=NUM_SHADOW_TRACES,
                                    iterations=ITERATIONS, seed=SEED,
                                    **spec["kwargs"])
            results[(method_id, miner_name)] = r
            config = {
                "dataset": ds["name"], "miner": miner_name, "method": method_id,
                "method_label": spec["label"],
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "host": "local", "seed": SEED,
                "parameters": {
                    "algorithm_version": spec["version"],
                    "max_n": spec["kwargs"].get("max_n", 6 if spec["version"] in ("v24", "v25", "v26") else 1),
                    "safe_threshold": 5,
                    "num_shadow_traces": NUM_SHADOW_TRACES,
                    "iterations": ITERATIONS,
                    **({"successor_weighting": spec["kwargs"]["successor_weighting"]}
                       if "successor_weighting" in spec["kwargs"] else {}),
                    **({"trace_model_top_k": TRACE_TOP_K} if miner_name == "Trace_Filtered" else {}),
                },
                "results": {
                    "mean": r["gen_shadow_mean"],
                    "std": r["gen_shadow_std"],
                    "raw_iterations": r.get("gen_shadow_raw_iterations"),
                    "runtime_s": r["runtime_s"],
                    **({"gen_accept": r["gen_accept"],
                        "gen_accept_std": r["gen_accept_std"],
                        "gen_shadow_regular": r["gen_shadow_regular"],
                        "gen_shadow_mutated": r["gen_shadow_mutated"],
                        "duplicates_kept": r["duplicates_kept"],
                        "truncated_traces": r["truncated_traces"],
                        "max_trace_length_used": r["max_trace_length_used"]}
                       if "gen_accept" in r else {}),
                    **({"duplicates_kept": r["duplicates_kept"],
                        "truncated_traces": r["truncated_traces"]}
                       if "gen_accept" not in r and "duplicates_kept" in r else {}),
                },
                "notes": "Benchmark Methodology v2 (run_m1_family.py)",
            }
            fname = f"{CONFIG_DIR_V2}/{ds['name']}__{miner_name}__{method_id}.json"
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            print(f"    {miner_name:22s} {r['gen_shadow_mean']:.4f} "
                  f"+- {r['gen_shadow_std']:.4f}  ({r['runtime_s']:.1f}s)")

    # ── Summary ──
    print("\n" + "=" * 98)
    print(f"SUMMARY — {args.dataset} {ds['name']} (mean +- std over "
          f"{ITERATIONS} iterations; seed {SEED})")
    header = f"{'Method':6s} " + " ".join(f"{m[:13]:>15s}" for m in MINERS)
    print(header)
    print("-" * 98)
    print(f"{'R1':6s} " + " ".join(f"{r1[m]:>15.4f}" for m in MINERS))
    for method_id in args.methods:
        cells = [results[(method_id, m)] for m in MINERS]
        print(f"{method_id:6s} " + " ".join(
            f"{c['gen_shadow_mean']:>7.4f}+-{c['gen_shadow_std']:<6.4f}" for c in cells))

    print("\nAgreement with R1 (six real miners; Flower & Trace excluded as poles):")
    print(f"{'Method':6s} {'Pearson':>8s} {'Spearman':>9s} {'MAE':>7s} {'Spread':>7s} "
          f"{'Flower':>7s} {'Trace':>7s} {'s/cell':>7s}")
    y = [r1[m] for m in REAL_MINERS]
    for method_id in args.methods:
        x = [results[(method_id, m)]["gen_shadow_mean"] for m in REAL_MINERS]
        mae = float(np.mean(np.abs(np.array(x) - np.array(y))))
        fl = results[(method_id, "Flower")]["gen_shadow_mean"]
        tr = results[(method_id, "Trace_Filtered")]["gen_shadow_mean"]
        rt = np.mean([results[(method_id, m)]["runtime_s"] for m in MINERS])
        print(f"{method_id:6s} {pearson(x, y):8.3f} {spearman(x, y):9.3f} {mae:7.3f} "
              f"{max(x) - min(x):7.3f} {fl:7.3f} {tr:7.3f} {rt:7.1f}")
    print("=" * 98)
    print(f"Config JSONs -> {CONFIG_DIR_V2}/")
    print(f"Total wall-clock: {(time.time() - t_start) / 60:.1f} min")
    print("Litmus: pure generalization => Flower ~ 1.0 and Trace low "
          "(Trace > 0 because token replay grants partial credit).")

if __name__ == "__main__":
    main()
