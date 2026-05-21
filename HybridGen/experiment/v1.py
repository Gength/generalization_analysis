"""
Hybrid Generalization Metric - Experiment Runner v1 (DFG-based)
"""

import time

from . import utils

algo = None  # injected by load_experiment(name, algo)


def parse_args():
    _ver = algo.__name__.split('.')[-1] if algo else "v1"       # v1, v2, v21
    p = utils.base_parse_args(
        f"Hybrid Gen-Struct Eval - {_ver} (DFG + Good-Turing)",
        output_dir_default=f"output/{_ver}",
    )
    return p.parse_args()


def main(args=None):
    """Run experiment. If args is None, parse from CLI. Otherwise use provided Namespace."""
    if args is None:
        args = parse_args()
    t_start = time.time()

    active_miners = utils.resolve_miners(args.miner)
    utils.print_header(active_miners, args)
    event_log = utils.load_event_log(args.data_path)

    print("\n[2/3] Running Evaluations...")
    all_results = []

    for w in args.weights:
        print("\n--- Testing Weight: w=" + str(w) + " ---")
        for miner_name, miner_fn in active_miners.items():
            for run_idx in range(1, args.runs + 1):
                current_seed = (args.seed + run_idx) if args.seed is not None else None
                print("   Run " + str(run_idx) + "/" + str(args.runs) + " | Seed: " + str(current_seed))

                res = algo.evaluate_miner(
                    event_log, miner_name, miner_fn, w=w,
                    num_shadow_traces=args.shadow_traces,
                    iterations=args.iterations, seed=current_seed,
                )
                res["run_id"] = run_idx
                res["base_seed"] = args.seed
                all_results.append(res)

    utils.print_summary_table(all_results)
    utils.print_stratified_table(all_results)
    out_path = utils.export_results(all_results, args, version="1")
    print("\n Results saved to " + out_path + " (Total time: " + str(round(time.time()-t_start, 1)) + "s)")


if __name__ == "__main__":
    main()

# HybridGen Registry
from . import register_experiment
register_experiment("v1")
