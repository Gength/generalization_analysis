"""Realized mutation rate vs the context bound N, per log.

For each N this generates K=5 independent 1,000-trace shadow logs (generation
only, no net, no replay) and reports, per log:
  per_step  - mutation events / generation steps (the report's x1e-3 quantity)
  per_trace - share of shadow traces carrying at least one mutation
  avg_len   - mean generated trace length

The per-step count needs two counters inside generate_shadow_log, which the
shipped v2.6 does not expose. Instead of maintaining a duplicate, this script
loads HybridGen/algorithm/v26.py, textually injects the counters (and disables
the package registration tail), and execs the result: the instrumented
generator IS the frozen v2.6 plus counters. Every injection is asserted, so a
change to v26.py fails loudly here rather than measuring the wrong thing.

Usage (repo root): PYTHONHASHSEED=0 python benchmark/nt_perstep_sweep.py D1 D2 D3 D4 D5
Writes/updates results/nt_perstep_sweep.json (resumable: finished logs are kept).
"""
import os, sys, json, random, argparse
import numpy as np
import pm4py

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "benchmark"))
from datasets import DATASETS

SEED = 42

# ---- build the instrumented generator from the real v2.6 source ------------
_PATCHES = [
    ("    duplicates_kept = 0\n    truncated = 0\n",
     "    duplicates_kept = 0\n    truncated = 0\n"
     "    total_muts = 0\n    total_steps = 0\n"),
    ("            seq = [curr_act]\n            had_mutation = False\n",
     "            seq = [curr_act]\n            had_mutation = False\n"
     "            n_mut = 0\n"),
    ("                    had_mutation = True\n",
     "                    had_mutation = True\n"
     "                    n_mut += 1\n"),
    ("        shadow_log.append(trace)\n        mutation_flags.append(had_mutation)\n",
     "        shadow_log.append(trace)\n        mutation_flags.append(had_mutation)\n"
     "        total_muts += n_mut\n        total_steps += len(seq) - 1\n"),
    ("    return shadow_log, mutation_flags, duplicates_kept, truncated, max_trace_length\n",
     "    return shadow_log, mutation_flags, duplicates_kept, truncated, max_trace_length, total_muts, total_steps\n"),
    ("from . import register_algorithm\nregister_algorithm(\"v2.6\")",
     "# registration disabled in the instrumented exec copy"),
]


def instrumented_generator():
    src_path = os.path.join(REPO, "HybridGen", "algorithm", "v26.py")
    src = open(src_path, encoding="utf-8").read()
    for old, new in _PATCHES:
        assert old in src, f"v26.py changed; instrumentation anchor not found: {old[:60]!r}"
        src = src.replace(old, new, 1)
    ns = {}
    exec(compile(src, src_path + " [instrumented]", "exec"), ns)
    return ns["generate_shadow_log"]


def run(dk, Ns, gen, iters=5, num_traces=1000):
    ds = DATASETS[dk]
    log = pm4py.convert_to_event_log(pm4py.read_xes(ds["log_path"]))
    print(f"[{dk} {ds['name']}] loaded", flush=True)
    out = {}
    for N in Ns:
        random.seed(SEED); np.random.seed(SEED)
        tm = ts = ntr = 0
        for _ in range(iters):
            _, flags, _d, _t, _cap, muts, steps = gen(
                log, num_traces=num_traces, safe_threshold=5, max_n=N,
                successor_weighting="mle")
            tm += muts; ts += steps; ntr += int(sum(flags))
        out[N] = {"per_step": tm / ts if ts else 0.0,
                  "per_trace": ntr / (iters * num_traces),
                  "avg_len": ts / (iters * num_traces)}
        print(f"  N={N}: per-step={out[N]['per_step']*1000:7.3f}e-3   "
              f"per-trace={out[N]['per_trace']*1000:6.1f}e-3   "
              f"avg_len={out[N]['avg_len']:5.1f}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=None)
    ap.add_argument("--Ns", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8])
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", "nt_perstep_sweep.json"))
    a = ap.parse_args()
    gen = instrumented_generator()
    res = {}
    if os.path.exists(a.out):
        res = json.load(open(a.out, encoding="utf-8"))
        print("resuming; have:", list(res.keys()), flush=True)
    for dk in (a.datasets or ["D1", "D2", "D3", "D4", "D5"]):
        if dk in res:
            print(f"skip {dk} (done)", flush=True); continue
        res[dk] = run(dk, a.Ns, gen)
        json.dump(res, open(a.out, "w"), indent=1)
    print("->", a.out, flush=True)
