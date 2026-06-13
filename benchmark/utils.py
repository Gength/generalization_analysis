import random
import numpy as np
from pm4py.objects.log.obj import EventLog
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness

# ─── K-Fold Cross Validation Function ───────────────────────────────────────
def compute_kfold_fitness(log, miner_fn, k=3, pick_one_out=False):
    """
    Compute cross-validated fitness.
    
    Both modes are variant-based: traces are grouped by their unique activity
    sequence (variant), ensuring that all traces of the same variant stay
    together in either train or test — never split across folds.
    
    Args:
        log: Event log
        miner_fn: Process discovery function
        k: Number of variant groups (folds) to partition the variants into.
           Used only when pick_one_out=False.
        pick_one_out: If True, leave-one-variant-out (each variant is its own
                      fold). If False (default), variants are split into k
                      groups, each group serving as test in one fold.
    """
    try:
        from collections import defaultdict
        
        # ── Group traces by variant (unique activity sequence) ──
        variant_map = defaultdict(list)
        for trace in log:
            seq = tuple(e["concept:name"] for e in trace)
            variant_map[seq].append(trace)
        
        variants = list(variant_map.keys())
        n_variants = len(variants)
        if n_variants <= 1:
            print(f"       ⚠️ pick_one_out: only {n_variants} variant(s), skipping")
            return 0.0
        
        if pick_one_out:
            # ── Leave-one-variant-out: each variant is one fold ──
            fitnesses = []
            for variant in variants:
                test_traces = variant_map[variant]
                train_traces = [t for v in variants if v != variant for t in variant_map[v]]
                
                train_log = EventLog(train_traces)
                test_log = EventLog(test_traces)
                
                net, im, fm = miner_fn(train_log)
                fit = replay_fitness.apply(test_log, net, im, fm,
                                           variant=replay_fitness.Variants.TOKEN_BASED)['log_fitness']
                fitnesses.append(fit)
            return np.mean(fitnesses)
        
        else:
            # ── Variant-based K-fold: partition variants into k groups ──
            random.shuffle(variants)
            fold_size = max(1, n_variants // k)
            fitnesses = []
            
            for i in range(k):
                start = i * fold_size
                end = (i + 1) * fold_size if i < k - 1 else n_variants
                test_variants = variants[start:end]
                
                test_traces = [t for v in test_variants for t in variant_map[v]]
                train_traces = [t for v in variants if v not in test_variants for t in variant_map[v]]
                
                train_log = EventLog(train_traces)
                test_log = EventLog(test_traces)
                
                net, im, fm = miner_fn(train_log)
                fit = replay_fitness.apply(test_log, net, im, fm,
                                           variant=replay_fitness.Variants.TOKEN_BASED)['log_fitness']
                fitnesses.append(fit)
            return np.mean(fitnesses)
    except Exception as e:
        print(f"       ⚠️ K-Fold Error: {e}")
        return 0.0