"""Consolidate the honest M4 (anti-alignment) and M8 (pattern-based) reruns into
one per-cell JSON. Runs on cibox, where the raw sweep output lives:
  M4 D1:      ~/m4_run/results/<miner>.out            (M4Progress driver)
  M8 D1:      ~/m8_run/results/<miner>.out            (+ results_big/ for the
              three cells re-run at -Xmx16g)
  M4/M8 D2-D5:~/m45_run/<D>/<method>/cell_<miner>/out.txt

Each cell is reduced to {outcome, value?, progress_pct?, error?, wall_s}. This
is the regenerable artifact behind the Sect. 6.2 / appendix M4/M8 claims.

Usage (on cibox): python3 consolidate.py   ->  ~/m4m8_rerun.json
"""
import os, re, json

HOME = os.path.expanduser("~")
MINERS = ["Alpha", "Alpha+", "Heuristics", "Heuristics_Strict",
          "Inductive_Infrequent", "Inductive_Strict", "Trace_Filtered", "Flower"]
DS = ["D1", "D2", "D3", "D4", "D5"]


def loc(method, ds, miner):
    if ds == "D1":
        if method == "M4":
            return f"{HOME}/m4_run/results/{miner}.out"
        big = f"{HOME}/m8_run/results_big/{miner}.out"
        return big if os.path.exists(big) else f"{HOME}/m8_run/results/{miner}.out"
    return f"{HOME}/m45_run/{ds}/{method}/cell_{miner}/out.txt"


def parse(f):
    if not os.path.exists(f):
        return {"outcome": "missing_or_running"}
    t = open(f, errors="ignore").read()
    g4 = re.findall(r"DONE generalization=([0-9.eEN+-]+)", t)
    g8 = re.search(r"^log\.xes,model\.pnml,PatternBasedGeneralization,.*,([0-9.eEN+-]+)\s*$", t, re.M)
    ex = re.search(r"___EXIT=([0-9]+)", t)
    ex = ex.group(1) if ex else None
    wall = re.search(r"WALL=([0-9]+)s", t)
    wall = int(wall.group(1)) if wall else None
    crash = re.search(r"ArrayIndexOutOfBoundsException|OutOfMemoryError|NullPointerException", t)
    prog = re.findall(r"([0-9.]+)% \(", t)
    pct = float(prog[-1]) if prog else None
    if g4:
        return {"outcome": "complete", "value": float(g4[-1]), "wall_s": wall}
    if g8 and ex == "0":
        return {"outcome": "complete", "value": float(g8.group(1)), "wall_s": wall}
    if "t/out" in t:
        return {"outcome": "timeout", "progress_pct": pct, "wall_s": wall}
    if crash:
        return {"outcome": "crash", "error": crash.group(0), "progress_pct": pct, "wall_s": wall}
    if ex == "124":
        return {"outcome": "timeout", "progress_pct": pct, "wall_s": wall}
    if ex is None and wall is None:
        return {"outcome": "missing_or_running"}
    return {"outcome": "other", "exit": ex, "wall_s": wall}


def main():
    out = {"_note": "Honest M4/M8 reruns, 1h cap per cell, seed-fixed; see README.md",
           "M4": {}, "M8": {}}
    for method in ("M4", "M8"):
        for ds in DS:
            out[method][ds] = {m: parse(loc(method, ds, m)) for m in MINERS}
    path = f"{HOME}/m4m8_rerun.json"
    json.dump(out, open(path, "w"), indent=1)
    for method in ("M4", "M8"):
        for ds in DS:
            cells = out[method][ds]
            comp = sum(1 for c in cells.values() if c["outcome"] == "complete")
            real = sum(1 for m, c in cells.items()
                       if c["outcome"] == "complete" and m not in ("Trace_Filtered", "Flower"))
            print(f"{method} {ds}: {comp}/8 complete ({real}/6 real miners)")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
