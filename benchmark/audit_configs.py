"""Audit a config folder against the full benchmark matrix.

For every (dataset, miner, method) cell, report: OK (valid score), SENT
(explicit -1 sentinel), BAD (present but no usable score and no sentinel),
or MISS (no file). The goal state is a folder with zero MISS and zero BAD:
every cell is either a measurement or a documented infeasibility.

Usage: python benchmark/audit_configs.py [config_dir]
       (default: benchmark/results/configs)
"""
import os, sys, json, glob
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
CFG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "configs")

DATASETS = ["Sepsis", "BPI2013_Incidents", "BPI2017", "BPI2018", "BPI2019"]
MINERS = ["Trace_Filtered", "Alpha", "Alpha+", "Heuristics", "Heuristics_Strict",
          "Inductive_Infrequent", "Inductive_Strict", "Flower"]
METHODS = ["M1a", "M1b", "M1c", "M1d", "M1e", "M1f", "M1g",
           "M2", "M3", "M4", "M5", "M6adapted", "M6original", "M7", "M8",
           "R1", "R2", "R3", "R1accept"]
# Methods whose configs must also carry these result fields to count as OK
REQUIRED_FIELDS = {"M1e": ["duplicates_kept", "truncated_traces"],
                   "M1f": ["gen_accept", "duplicates_kept", "truncated_traces"],
                   "M1g": ["gen_accept", "duplicates_kept", "truncated_traces"]}


def status(path, meth):
    if not os.path.exists(path):
        return "MISS", ""
    try:
        cfg = json.load(open(path, encoding="utf-8"))
    except Exception as e:
        return "BAD", f"unreadable: {e}"
    r = cfg.get("results", {})
    score = None
    for k in ("mean", "score", "gen_score", "accept_mean", "entropic_relevance_raw"):
        if r.get(k) is not None:
            score = float(r[k])
            break
    if score is None:
        return "BAD", "no score field"
    if score < 0:
        return "SENT", (cfg.get("notes") or "")[:60]
    missing = [f for f in REQUIRED_FIELDS.get(meth, []) if r.get(f) is None]
    if missing:
        return "BAD", f"missing fields: {missing}"
    return "OK", ""


counts = defaultdict(int)
problems = []
print(f"Audit: {CFG}\n")
print(f"{'method':8}", " ".join(f"{d[:9]:>14}" for d in DATASETS))
for meth in METHODS:
    row = []
    for ds in DATASETS:
        ok = sent = bad = miss = 0
        for m in MINERS:
            st, note = status(os.path.join(CFG, f"{ds}__{m}__{meth}.json"), meth)
            counts[st] += 1
            if st == "OK":
                ok += 1
            elif st == "SENT":
                sent += 1
            elif st == "BAD":
                bad += 1
                problems.append(f"BAD  {ds}__{m}__{meth}: {note}")
            else:
                miss += 1
                problems.append(f"MISS {ds}__{m}__{meth}")
        cell = f"{ok}ok"
        if sent: cell += f"+{sent}s"
        if bad:  cell += f"+{bad}B"
        if miss: cell += f"+{miss}M"
        row.append(f"{cell:>14}")
    print(f"{meth:8}", " ".join(row))

extra = [os.path.basename(p) for p in glob.glob(os.path.join(CFG, "*.json"))
         if os.path.basename(p)[:-5].split("__")[2] not in METHODS]
print(f"\nTotals: {counts['OK']} OK, {counts['SENT']} sentinels, "
      f"{counts['BAD']} BAD, {counts['MISS']} missing"
      + (f", {len(extra)} unexpected files" if extra else ""))
if problems:
    print(f"\n{len(problems)} cells pending:")
    for p in problems[:60]:
        print(" ", p)
    if len(problems) > 60:
        print(f"  ... and {len(problems) - 60} more")
else:
    print("\nDATASET COMPLETE: every cell is a score or a documented sentinel.")
