"""Materialize the M9 (negative-event generalization, CoBeFra) results from
m9_negative_events.jsonl into the per-cell config convention the figures read,
so M9 is a first-class method like M2-M7 (e.g. appears in the speed/accuracy
pareto). Additive only: writes new configs/{name}__{miner}__M9.json files, never
touches existing cells. Timeout cells become the -1 budget sentinel (score_of
maps that to NaN, and the pareto excludes it), matching how the benchmark records
methods that exceed the budget.

Usage: python benchmark/m9/m9_to_configs.py
"""
import os
import json
import statistics

SRC = "benchmark/results/m9_negative_events.jsonl"
CFG = "benchmark/results/configs"
NAME = {"D1": "Sepsis", "D2": "BPI2013_Incidents", "D3": "BPI2017",
        "D4": "BPI2018", "D5": "BPI2019"}


def main():
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
    written = 0
    for r in rows:
        name = NAME.get(r["dataset"])
        if not name:
            continue
        mean = r.get("mean")
        rts = r.get("runtimes_s") or []
        if mean is not None and mean >= 0:
            score = float(mean)
            runtime = float(statistics.mean(rts)) if rts else float(r.get("budget_spent_s", 0.0))
        else:
            score = -1.0  # budget sentinel (timeout); excluded from agreement + pareto
            runtime = float(r.get("budget_spent_s", 0.0))
        out = {
            "dataset": name, "miner": r["miner"], "method": "M9",
            "method_label": "Weighted Negative Events (CoBeFra)",
            "host": "cibox", "seed": 42, "parameters": {"config": r.get("config", "cobefra")},
            "results": {"score": score, "runtime_s": runtime},
            "notes": "converted from m9_negative_events.jsonl",
        }
        path = f"{CFG}/{name}__{r['miner']}__M9.json"
        json.dump(out, open(path, "w", encoding="utf-8"), indent=1)
        written += 1
    print(f"wrote {written} M9 config cells to {CFG}/")


if __name__ == "__main__":
    main()
