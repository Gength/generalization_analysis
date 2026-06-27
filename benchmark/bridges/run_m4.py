"""
M4 — Anti-Alignment Generalization (ProcessM TwoPhaseDFS)
==========================================================
Provides run() for job wrappers. CLI via main().

Pipeline:
  1. Load PNML → export as ProcessM-compatible JSON
  2. Export log traces as text (one line per trace, space-separated)
  3. Call: java -jar m4bridge.jar <petri.json> <traces.txt>
  4. Parse stdout → score
"""
import os, sys, json, subprocess, time, argparse, tempfile
from datetime import datetime, timezone

import pm4py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from job_prepare import prepare_workdir, get_miner_names

_M4_JAR = os.path.join(
    os.path.dirname(__file__), "..", "..", "src", "processm",
    "processm.m4bridge", "target", "m4bridge-1.0.jar")
_JAVA_HOME = "/usr/lib/jvm/java-21-openjdk-amd64/bin/java"


def _pnml_to_processm_json(net, im, fm):
    """Convert a PM4Py Petri net to ProcessM bridge JSON format."""
    all_places = sorted(net.places, key=lambda p: str(p.name or ""))
    place_index = {p: i for i, p in enumerate(all_places)}

    transitions = []
    for t in net.transitions:
        tname = t.name or "__silent__"
        is_silent = tname.startswith("_") or tname.startswith("tau") or tname == "__silent__"
        in_place_indices = sorted(set(
            place_index[a.source] for a in net.arcs if a.target == t and a.source in place_index
        ))
        out_place_indices = sorted(set(
            place_index[a.target] for a in net.arcs if a.source == t and a.target in place_index
        ))
        transitions.append({
            "name": tname,
            "isSilent": is_silent,
            "inPlaces": in_place_indices,
            "outPlaces": out_place_indices,
        })

    im_indices = [place_index[p] for p in im if p in place_index]
    fm_indices = [place_index[p] for p in fm if p in place_index]

    return {
        "numPlaces": len(all_places),
        "initialMarkingIndices": im_indices,
        "finalMarkingIndices": fm_indices,
        "transitions": transitions,
    }


def _log_to_traces_text(log, path, max_traces=10000):
    """Write traces from a PM4Py event log to a text file (one per line)."""
    count = 0
    with open(path, "w") as f:
        for i, t in enumerate(log):
            if i >= max_traces:
                break
            line = " ".join(e["concept:name"] for e in t)
            f.write(line + "\n")
            count += 1
    return count


def run(dataset_key, workdir, output_dir, miners=None):
    """Run M4.  Reads manifest + per-miner PNMLs from workdir, writes
    per-miner configs to output_dir.

    Each miner: export PNML → JSON + traces → Java bridge → parse score.
    """
    with open(os.path.join(workdir, "manifest.json")) as f:
        manifest = json.load(f)
    dname = manifest["dataset"]
    xes = manifest.get("xes_file")
    all_miners = manifest.get("miners", {})

    target = miners or get_miner_names()

    print(f"M4 — Anti-Alignment Generalization (ProcessM)")

    # Read the log (needed for trace text export)
    log = pm4py.read_xes(xes.replace(".gz", "") if xes.endswith(".gz") else xes)
    log = pm4py.convert_to_event_log(log)

    for mname in target:
        minfo = all_miners.get(mname)
        if not minfo or "pnml" not in minfo:
            print(f"  {mname}: SKIP (no PNML in manifest)")
            _write_config(
                output_dir, dname, mname,
                {"gen_score": -1, "runtime_s": 0},
                notes="No PNML available in manifest",
            )
            continue

        pnml_path = minfo["pnml"]
        if not os.path.exists(pnml_path):
            print(f"  {mname}: SKIP (PNML not found at {pnml_path})")
            _write_config(
                output_dir, dname, mname,
                {"gen_score": -1, "runtime_s": 0},
                notes=f"PNML not found: {pnml_path}",
            )
            continue

        try:
            t0 = time.time()

            # ── Export PNML → JSON ──────────────────────────────────
            net, im, fm = pm4py.read_pnml(pnml_path)
            pn_json = _pnml_to_processm_json(net, im, fm)

            tmpdir = tempfile.mkdtemp(prefix="m4_")
            petri_json = os.path.join(tmpdir, "model.json")
            with open(petri_json, "w") as f:
                json.dump(pn_json, f, indent=2)

            # ── Export traces → text ─────────────────────────────────
            traces_txt = os.path.join(tmpdir, "traces.txt")
            n_traces = _log_to_traces_text(log, traces_txt)

            # ── Run Java bridge (no timeout — caller manages) ─────────
            r = subprocess.run(
                [_JAVA_HOME, "-jar", _M4_JAR, petri_json, traces_txt],
                capture_output=True, text=True,
            )
            elapsed = time.time() - t0

            # Cleanup temp dir
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

            if r.returncode != 0:
                stderr_short = r.stderr[:200]
                print(f"  {mname}: bridge error ({stderr_short})")
                _write_config(
                    output_dir, dname, mname,
                    {"gen_score": -1, "runtime_s": elapsed},
                    notes=f"Bridge error (rc={r.returncode}): {stderr_short}",
                )
                continue

            try:
                score = float(r.stdout.strip())
            except (ValueError, TypeError):
                print(f"  {mname}: unexpected output: {r.stdout.strip()[:100]}")
                _write_config(
                    output_dir, dname, mname,
                    {"gen_score": -1, "runtime_s": elapsed},
                    notes=f"Unexpected output: {r.stdout.strip()[:100]}",
                )
                continue

            print(f"  {mname}: {score:.4f}  ({elapsed:.1f}s)")
            _write_config(
                output_dir, dname, mname,
                {"gen_score": score, "runtime_s": elapsed},
            )

        except Exception as e:
            print(f"  {mname}: ERROR ({e})")
            _write_config(
                output_dir, dname, mname,
                {"gen_score": -1, "runtime_s": 0},
                notes=str(e),
            )

    print(f"  → {len(target)} miners processed → {output_dir}/")


def _write_config(output_dir, dname, miner, results, notes=""):
    cfg = {
        "dataset": dname, "miner": miner, "method": "M4",
        "method_label": "Anti-Alignment Generalization",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": "local", "seed": 42,
        "parameters": {"jar": "m4bridge"},
        "results": results, "notes": notes,
    }
    path = os.path.join(output_dir, f"{dname}__{miner}__M4.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  ✓ {miner}")


def main():
    ap = argparse.ArgumentParser(description="M4 Anti-Alignment Generalization")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--miners", nargs="*", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    import shutil, secrets
    from datetime import datetime as dt
    workdir = f"/tmp/benchmark_M4_{args.dataset}_{dt.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}/"
    output_dir = args.output or os.path.join(workdir, "results")
    os.makedirs(output_dir, exist_ok=True)

    prepare_workdir(workdir, args.dataset, copy_xes=True, discover_pnmls=True)
    run(args.dataset, workdir, output_dir, miners=args.miners)
    shutil.rmtree(workdir)
    print(f"  [clean] removed {workdir}")


if __name__ == "__main__":
    main()
