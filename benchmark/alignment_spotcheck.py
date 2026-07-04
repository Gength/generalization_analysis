"""Acceptance-proxy spot-check: trace_is_fit (token replay) vs cost-zero alignments.

The strict acceptance reading uses PM4Py's token-replay flag as a proxy for
language membership; the report commits to spot-checking it against alignments
(Sect. 5.5, threats). This does exactly that on D1: generate the v2.6-mle
shadow log, replay it on cached models, and compare per-trace perfect-replay
flags with per-trace alignment fitness == 1.0.

Sound nets only get a clean verdict (alignments require an easy final marking);
unsound Heuristics nets are attempted with a time cap and reported as such,
which itself documents why the proxy is used.

Usage (repo root): PYTHONHASHSEED=0 python benchmark/alignment_spotcheck.py
Writes benchmark/results/alignment_spotcheck.json.
"""
import os, sys, json, time, random
import numpy as np
import pm4py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import DATASETS
from HybridGen.algorithm.v26 import generate_shadow_log
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments

SEED = 42
MINERS = ["Inductive_Strict", "Inductive_Infrequent", "Flower", "Heuristics"]
CAP_S = 1200  # per-miner alignment time cap


def main():
    ds = DATASETS["D1"]
    log = pm4py.convert_to_event_log(pm4py.read_xes(ds["log_path"]))
    random.seed(SEED); np.random.seed(SEED)
    shadow, *_ = generate_shadow_log(log, num_traces=1000, successor_weighting="mle")
    print(f"shadow log: {len(shadow)} traces", flush=True)

    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "D1")
    out = {}
    for m in MINERS:
        pnml = os.path.join(cache, f"{m}.pnml")
        if not os.path.exists(pnml):
            out[m] = {"error": "no cached model"}
            continue
        net, im, fm = pm4py.read_pnml(pnml)
        replayed = token_replay.apply(shadow, net, im, fm)
        proxy = [bool(r["trace_is_fit"]) for r in replayed]

        t0 = time.time()
        align_ok = []
        err = None
        try:
            for i, tr in enumerate(shadow):
                if time.time() - t0 > CAP_S:
                    err = f"time cap {CAP_S}s after {i} traces"
                    break
                a = alignments.apply(tr, net, im, fm)
                align_ok.append(a["fitness"] >= 1.0 - 1e-9)
        except Exception as e:
            err = f"alignment error after {len(align_ok)} traces: {e}"

        n = len(align_ok)
        if n == 0:
            out[m] = {"proxy_accept_rate": float(np.mean(proxy)), "error": err}
            print(f"{m}: alignments unavailable ({err})", flush=True)
            continue
        agree = sum(1 for p, a in zip(proxy[:n], align_ok) if p == a)
        proxy_only = sum(1 for p, a in zip(proxy[:n], align_ok) if p and not a)
        align_only = sum(1 for p, a in zip(proxy[:n], align_ok) if a and not p)
        out[m] = {"n_compared": n, "agreement": agree / n,
                  "proxy_accept_rate": float(np.mean(proxy[:n])),
                  "align_accept_rate": float(np.mean(align_ok)),
                  "proxy_yes_align_no": proxy_only, "align_yes_proxy_no": align_only,
                  "runtime_s": time.time() - t0, "note": err}
        print(f"{m}: agreement {agree/n:.4f} over {n} traces "
              f"(proxy+/align- {proxy_only}, align+/proxy- {align_only})"
              + (f" [{err}]" if err else ""), flush=True)

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", "alignment_spotcheck.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
