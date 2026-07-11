#!/usr/bin/env python3
# M9 negative-event generalization: full 5-log matrix, N-run variance, budget sentinels.
# Runs M9Runner (CoBeFra config) up to N times per cell within a 1h cell budget.
# Incremental JSONL, resumable (skips cells already recorded).
import subprocess, json, time, os, threading, sys
from concurrent.futures import ThreadPoolExecutor

HOME = os.path.expanduser("~")
GENB = HOME + "/genbench"
CP = (HOME + "/m9_neg/classes:" + HOME + "/m8_attempt/out/production/AutomataConformance:"
      + HOME + "/m8_attempt/Libraries/*")
JAVA = HOME + "/jdk8/bin/java"
MODELS = GENB + "/benchmark/models"
OUT = HOME + "/m9_neg/m9_results.jsonl"

MINERS = ["Flower", "Trace_Filtered", "Alpha", "Alpha+", "Heuristics",
          "Heuristics_Strict", "Inductive_Infrequent", "Inductive_Strict"]

DATASETS = {
    "D1": {"log": GENB + "/data/Sepsis Cases - Event Log_1_all/Sepsis Cases - Event Log.xes.gz", "heap": "8g",  "conc": 8},
    "D2": {"log": GENB + "/data/BPI-Challenge_2013/Incident_Management_Log.xes.gz",               "heap": "10g", "conc": 6},
    "D3": {"log": GENB + "/data/BPI-Challenge_2017/BPI Challenge 2017.xes.gz",                    "heap": "26g", "conc": 3},
    "D4": {"log": GENB + "/data/BPI-Challenge_2018/BPI Challenge 2018.xes.gz",                    "heap": "40g", "conc": 2},
    "D5": {"log": GENB + "/data/BPI-Challenge_2019/BPI_Challenge_2019.xes.gz",                    "heap": "40g", "conc": 2},
}

N_RUNS = 5
CELL_BUDGET = 3600.0  # seconds, total per cell (1h protocol)
lock = threading.Lock()

def done_cells():
    s = set()
    if os.path.exists(OUT):
        for ln in open(OUT):
            try:
                r = json.loads(ln); s.add((r["dataset"], r["miner"]))
            except Exception:
                pass
    return s

def run_once(logp, modelp, label, heap, timeout):
    t0 = time.time()
    env = dict(os.environ); env["DISPLAY"] = ":0"
    try:
        p = subprocess.run([JAVA, "-Xmx" + heap, "-cp", CP, "m9.M9Runner", logp, modelp, label, "cobefra"],
                           capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return None, time.time() - t0, "timeout"
    dt = time.time() - t0
    for ln in p.stdout.splitlines():
        if ln.startswith("M9_RESULT"):
            for pt in ln.split("\t"):
                if pt.startswith("gen="):
                    try:
                        return float(pt[4:]), dt, "ok"
                    except Exception:
                        pass
    out = p.stdout + p.stderr
    st = "oom" if "OutOfMemory" in out else ("error_rc" + str(p.returncode))
    return None, dt, st

def do_cell(ds, miner, cfg):
    logp, heap = cfg["log"], cfg["heap"]
    modelp = MODELS + "/" + ds + "/" + miner + ".pnml"
    label = ds + "/" + miner
    vals, runtimes, statuses, spent = [], [], [], 0.0
    for i in range(N_RUNS):
        remaining = CELL_BUDGET - spent
        if i > 0 and remaining < 30:
            break
        v, dt, st = run_once(logp, modelp, label, heap, int(max(30, remaining)))
        spent += dt; runtimes.append(round(dt, 1)); statuses.append(st)
        if v is not None:
            vals.append(v)
        else:
            break  # first failure (timeout/oom/error) ends the cell
    rec = {"dataset": ds, "miner": miner, "config": "cobefra", "n": len(vals),
           "values": [round(x, 6) for x in vals], "runtimes_s": runtimes,
           "statuses": statuses, "budget_spent_s": round(spent, 1)}
    if vals:
        m = sum(vals) / len(vals); var = sum((x - m) ** 2 for x in vals) / len(vals)
        rec["mean"] = round(m, 6); rec["std"] = round(var ** 0.5, 6); rec["gen_score"] = round(m, 6)
    else:
        rec["gen_score"] = -1; rec["sentinel"] = statuses[0] if statuses else "none"
    with lock:
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print("[" + label + "] n=" + str(rec["n"]) + " gen=" + str(rec.get("mean", -1))
              + " std=" + str(rec.get("std", "-")) + " status=" + str(statuses)
              + " spent=" + str(round(spent)) + "s", flush=True)
    return rec

def main():
    order = sys.argv[1:] or ["D1", "D2", "D3", "D4", "D5"]
    done = done_cells()
    for ds in order:
        cfg = DATASETS[ds]
        todo = [m for m in MINERS if (ds, m) not in done]
        print("=== " + ds + " heap=" + cfg["heap"] + " conc=" + str(cfg["conc"])
              + " todo=" + str(len(todo)) + " ===", flush=True)
        if not todo:
            continue
        with ThreadPoolExecutor(max_workers=cfg["conc"]) as ex:
            list(ex.map(lambda mn: do_cell(ds, mn, cfg), todo))
    print("BATCH_DONE", flush=True)

main()
