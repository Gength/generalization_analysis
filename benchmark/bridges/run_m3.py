"""
M3 — Entropic Relevance (relevance.jar JDFG2Aut + Relevance)
=============================================================
Computes entropic relevance per miner: mines PNML → simulates →
generates a model-level DFG → JDFG2Aut (DFG→probability automaton)
→ Relevance (entropic relevance against the log XES).

Uses the open-source ``promtecmx/relevance`` library instead of
the closed-source Entropia JAR ``-r`` flag.  The pipeline is
transparent and the result identical.
"""
import os, sys, json, subprocess, time, argparse, tempfile
from datetime import datetime, timezone
from collections import Counter

import pm4py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from job_prepare import prepare_workdir, get_miner_names

# ── Java & JAR paths ─────────────────────────────────────────────────────
_JAVA21 = "/usr/lib/jvm/java-21-openjdk-amd64/bin/java"
_RELEVANCE_JAR = os.path.join(
    os.path.dirname(__file__), "..", "..", "src", "relevance", "relevance.jar")
_OPENXES_JAR = os.path.join(
    os.path.dirname(__file__), "..", "..", "src", "relevance", "lib", "OpenXES-20180810.jar")
_RELEVANCE_CP = f"{_RELEVANCE_JAR}:{_OPENXES_JAR}"

SIMULATION_TRACES = 5000


def _model_to_dfg_json(net, im, fm, no_traces=SIMULATION_TRACES):
    """Simulate a Petri net and produce a DFG JSON dict (nodes/arcs/freq)."""
    sim = pm4py.play_out(net, im, fm, no_traces=no_traces)
    dfg, sa, ea = pm4py.discover_dfg(sim)
    af = Counter()
    for t in sim:
        for e in t:
            af[e["concept:name"]] += 1
    acts = sorted(af.keys())
    aid = {a: i + 1 for i, a in enumerate(acts)}
    nodes = [{"id": aid[a], "label": a, "freq": af[a]} for a in acts]
    nodes.append({"id": len(acts) + 1, "label": "INPUT", "freq": len(sim)})
    nodes.append({"id": len(acts) + 2, "label": "OUTPUT", "freq": len(sim)})
    arcs = [{"from": aid[a], "to": aid[b], "freq": f} for (a, b), f in dfg.items()]
    fa, la = Counter(), Counter()
    for t in sim:
        seq = [e["concept:name"] for e in t]
        if seq:
            fa[seq[0]] += 1
            la[seq[-1]] += 1
    for a, f in fa.items():
        arcs.append({"from": len(acts) + 1, "to": aid[a], "freq": f})
    for a, f in la.items():
        arcs.append({"from": aid[a], "to": len(acts) + 2, "freq": f})
    return {"nodes": nodes, "arcs": arcs}


def run(dataset_key, workdir, output_dir, miners=None):
    """Run M3.  Reads manifest + per-miner PNMLs from workdir, writes
    per-miner configs to output_dir.

    Pipeline per miner:
      1. Simulate PNML → model-level DFG JSON
      2. JDFG2Aut <dfg.json> <aut_dir/>> → AutTransition JSON (with probs)
      3. Relevance <aut.json> <log.xes> → CSV → parse entropic relevance
    """
    with open(os.path.join(workdir, "manifest.json")) as f:
        manifest = json.load(f)
    dname = manifest["dataset"]
    xes = manifest["xes_file"]
    # Use decompressed .xes if available (Java Relevance can't read .xes.gz)
    xes_plain = os.path.join(workdir, f"{dname.lower().replace(' ', '_')}.xes")
    if os.path.exists(xes_plain):
        xes = xes_plain
    all_miners = manifest.get("miners", {})

    target = miners or get_miner_names()

    print(f"M3 — Entropic Relevance (JDFG2Aut + Relevance)")

    # Temporary aut dir for JDFG2Aut output (one file per miner)
    aut_dir = tempfile.mkdtemp(prefix="m3_aut_")

    for mname in target:
        minfo = all_miners.get(mname)
        if not minfo or "pnml" not in minfo:
            print(f"  {mname}: SKIP (no PNML in manifest)")
            _write_config(
                output_dir, dname, mname,
                {"entropic_relevance_raw": -1, "entropic_relevance_normalized": None, "runtime_s": 0},
                notes="No PNML available in manifest",
            )
            continue

        pnml_path = minfo["pnml"]
        if not os.path.exists(pnml_path):
            print(f"  {mname}: SKIP (PNML not found at {pnml_path})")
            _write_config(
                output_dir, dname, mname,
                {"entropic_relevance_raw": -1, "entropic_relevance_normalized": None, "runtime_s": 0},
                notes=f"PNML not found: {pnml_path}",
            )
            continue

        try:
            t0 = time.time()

            # ── Step 1: Simulate PNML → DFG JSON ───────────────────────
            net, im, fm = pm4py.read_pnml(pnml_path)
            dfg_data = _model_to_dfg_json(net, im, fm)

            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
            json.dump(dfg_data, tmp)
            dfg_path = tmp.name
            tmp.close()

            # ── Step 2: JDFG2Aut (DFG → probability automaton) ─────────
            r2 = subprocess.run(
                [_JAVA21, "-cp", _RELEVANCE_CP, "org.jbpt.relevance.JDFG2Aut",
                 dfg_path, aut_dir],
                capture_output=True, text=True,
            )
            if r2.returncode != 0:
                raise RuntimeError(f"JDFG2Aut failed: {r2.stderr[:200]}")
            os.unlink(dfg_path)

            # Automaton JSON is written as <aut_dir>/<input_filename>
            aut_path = os.path.join(aut_dir, os.path.basename(dfg_path))

            # ── Step 3: Relevance (entropic relevance) ─────────────────
            r3 = subprocess.run(
                [_JAVA21, "-cp", _RELEVANCE_CP, "org.jbpt.relevance.Relevance",
                 aut_path, xes],
                capture_output=True, text=True,
            )
            os.unlink(aut_path)
            elapsed = time.time() - t0

            if r3.returncode != 0:
                raise RuntimeError(f"Relevance failed: {r3.stderr[:200]}")

            # Parse CSV: states,transitions,totalTraces,nonFitting,
            #            FittingFraction,EntropicRelevance,...
            cols = r3.stdout.strip().split(",")
            if len(cols) < 6:
                raise RuntimeError(
                    f"Unexpected output: {r3.stdout.strip()[:200]}")

            rel = float(cols[5])
            print(f"  {mname}: {rel:.4f}  ({elapsed:.1f}s)")
            _write_config(
                output_dir, dname, mname,
                {"entropic_relevance_raw": rel, "entropic_relevance_normalized": None,
                 "runtime_s": elapsed},
            )

        except Exception as e:
            print(f"  {mname}: ERROR ({e})")
            _write_config(
                output_dir, dname, mname,
                {"entropic_relevance_raw": -1, "entropic_relevance_normalized": None, "runtime_s": 0},
                notes=str(e),
            )

    # Clean up aut dir
    import shutil
    shutil.rmtree(aut_dir, ignore_errors=True)

    print(f"  → {len(target)} miners processed → {output_dir}/")


def _write_config(output_dir, dname, miner, results, notes=""):
    cfg = {
        "dataset": dname, "miner": miner, "method": "M3",
        "method_label": "Entropic Relevance",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": "local", "seed": 42,
        "parameters": {"jar": "relevance", "flag": "JDFG2Aut+Relevance",
                       "model": "per-miner DFG → automaton (probability)"},
        "results": results, "notes": notes,
    }
    path = os.path.join(output_dir, f"{dname}__{miner}__M3.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  ✓ {miner}")


def main():
    ap = argparse.ArgumentParser(description="M3 Entropic Relevance (JDFG2Aut + Relevance)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--miners", nargs="*", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    import shutil, secrets
    from datetime import datetime as dt
    workdir = f"/tmp/benchmark_M3_{args.dataset}_{dt.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}/"
    output_dir = args.output or os.path.join(workdir, "results")
    os.makedirs(output_dir, exist_ok=True)

    prepare_workdir(workdir, args.dataset, copy_xes=True, discover_pnmls=True)
    run(args.dataset, workdir, output_dir, miners=args.miners)
    shutil.rmtree(workdir)
    print(f"  [clean] removed {workdir}")


if __name__ == "__main__":
    main()
