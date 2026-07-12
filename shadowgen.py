"""ShadowGen -- generative N-gram generalization metric (release module).

This is the shipped configuration of the metric described in the report:
a single generate-and-replay draw (K=1) of 1,000 shadow traces, context bound
N=6, support threshold tau=5, MLE successor weighting, seeded. It returns ONE
number: the graded generalization score Gen_shadow in [0, 1] (mean token-replay
trace fitness of the shadow log on the model). The strict acceptance reading
used in the report is a validation diagnostic of the benchmark harness and is
deliberately not part of the release.

Programmatic use:
    import pm4py
    from shadowgen import gen_shadow

    log = pm4py.read_xes("log.xes")
    net, im, fm = pm4py.read_pnml("model.pnml")
    score = gen_shadow(log, net, im, fm)                 # K=1, the default
    score, info = gen_shadow(log, net, im, fm, iterations=5, details=True)

Command line:
    python shadowgen.py LOG.xes MODEL.pnml [--iterations K] [--seed S] [--details]

More iterations are an option, not a need: a single draw reproduces the
benchmark result (Pearson within 0.003 of the 5-draw mean on every log);
iterations>1 buys an error bar (details["std"]).
"""
import os
import sys
import random
import argparse

os.environ.setdefault("PYTHONHASHSEED", "0")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

DEFAULTS = {
    "iterations": 1,          # K: single draw is the shipped operating point
    "num_shadow_traces": 1000,  # theta (capped by |L| via dedup behaviour)
    "context_bound": 6,       # N
    "support_threshold": 5,   # tau
    "weighting": "mle",       # successor weighting (expected-future mode)
    "seed": 42,
}


def gen_shadow(event_log, net, initial_marking, final_marking, *,
               iterations=DEFAULTS["iterations"],
               num_shadow_traces=DEFAULTS["num_shadow_traces"],
               context_bound=DEFAULTS["context_bound"],
               support_threshold=DEFAULTS["support_threshold"],
               weighting=DEFAULTS["weighting"],
               seed=DEFAULTS["seed"],
               details=False):
    """Score a Petri-net model's generalization against an event log.

    Returns the graded score Gen_shadow (float in [0, 1]); with details=True,
    returns (score, info) where info carries the interpretability internals:
      std (over iterations), raw_iterations, mutation share, the graded score
      split into regular (recombination) vs mutated (novel-event) shadow
      traces, and the integrity counters duplicates_kept / truncated_traces.
    """
    import numpy as np
    from HybridGen.algorithm import load_algorithm

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    core = load_algorithm("v2.6").calculate_gen_shadow_stable
    r = core(event_log, net, initial_marking, final_marking,
             num_traces=num_shadow_traces, iterations=iterations,
             safe_threshold=support_threshold, max_n=context_bound,
             successor_weighting=weighting)
    score = float(r["mean"])
    if not details:
        return score
    n_mut = sum(r["mutation_counts"])
    info = {
        "std": float(r["std"]),
        "raw_iterations": [float(x) for x in r["raw_scores"]],
        "gen_shadow_regular": float(r["reg_mean"]),
        "gen_shadow_mutated": float(r["mut_mean"]),
        "mutated_trace_share": n_mut / max(iterations * num_shadow_traces, 1),
        "duplicates_kept": int(r["duplicates_kept"]),
        "truncated_traces": int(r["truncated"]),
        "max_trace_length_used": r["max_trace_length_used"],
        "parameters": {"iterations": iterations,
                       "num_shadow_traces": num_shadow_traces,
                       "context_bound": context_bound,
                       "support_threshold": support_threshold,
                       "weighting": weighting, "seed": seed},
    }
    return score, info


def main():
    ap = argparse.ArgumentParser(
        description="ShadowGen: graded generalization score of a Petri net "
                    "against an event log (single-draw default).")
    ap.add_argument("log", help="event log (.xes or .xes.gz)")
    ap.add_argument("model", help="Petri net (.pnml)")
    ap.add_argument("--iterations", type=int, default=DEFAULTS["iterations"],
                    help="independent draws K (default 1; >1 adds an error bar)")
    ap.add_argument("--traces", type=int, default=DEFAULTS["num_shadow_traces"],
                    help="shadow traces per draw (default 1000)")
    ap.add_argument("--context-bound", type=int, default=DEFAULTS["context_bound"])
    ap.add_argument("--support-threshold", type=int, default=DEFAULTS["support_threshold"])
    ap.add_argument("--weighting", choices=["mle", "log"], default=DEFAULTS["weighting"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--details", action="store_true",
                    help="also print the interpretability internals")
    args = ap.parse_args()

    import pm4py
    log = pm4py.convert_to_event_log(pm4py.read_xes(args.log))
    net, im, fm = pm4py.read_pnml(args.model)

    out = gen_shadow(log, net, im, fm, iterations=args.iterations,
                     num_shadow_traces=args.traces,
                     context_bound=args.context_bound,
                     support_threshold=args.support_threshold,
                     weighting=args.weighting, seed=args.seed,
                     details=args.details)
    if args.details:
        score, info = out
        print(f"Gen_shadow = {score:.4f}")
        for k, v in info.items():
            print(f"  {k}: {v}")
    else:
        print(f"Gen_shadow = {out:.4f}")


if __name__ == "__main__":
    main()
