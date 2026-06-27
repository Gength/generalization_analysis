"""
Job Preparation Module — prepare_workdir() for self-contained benchmark jobs.

Each bool flag directly controls one data product:

  copy_xes=True:         copy source XES to workdir (native format — .xes or .xes.gz)
  decompress_xes=True:   additionally decompress to .xes_plain (M6 Java JAR only)
  discover_pnmls=True:   discover PNML for all 8 miners
  per_miner_dfgs=True:   simulate PNMLs → per-miner DFG JSONs (M6 only)

All I/O is contained in `workdir`. Returns a dict of paths.
"""
import os, sys, json, shutil, gzip
from collections import Counter
from datetime import datetime

import pm4py

_LOADED_MINERS = None


def _get_miners():
    global _LOADED_MINERS
    if _LOADED_MINERS is None:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from miners import MINERS
        _LOADED_MINERS = MINERS
    return _LOADED_MINERS


def _dump_dfg(log, path):
    dfg, sa, ea = pm4py.discover_dfg(log)
    af = Counter()
    for t in log:
        for e in t:
            af[e["concept:name"]] += 1
    acts = sorted(af.keys())
    aid = {a: i + 1 for i, a in enumerate(acts)}
    nodes = [{"id": aid[a], "label": a, "freq": af[a]} for a in acts]
    nodes.append({"id": len(acts) + 1, "label": "INPUT", "freq": len(log)})
    nodes.append({"id": len(acts) + 2, "label": "OUTPUT", "freq": len(log)})
    arcs = [{"from": aid[a], "to": aid[b], "freq": f} for (a, b), f in dfg.items()]
    fa, la = Counter(), Counter()
    for t in log:
        seq = [e["concept:name"] for e in t]
        if seq:
            fa[seq[0]] += 1
            la[seq[-1]] += 1
    for a, f in fa.items():
        arcs.append({"from": len(acts) + 1, "to": aid[a], "freq": f})
    for a, f in la.items():
        arcs.append({"from": aid[a], "to": len(acts) + 2, "freq": f})
    with open(path, "w") as f:
        json.dump({"nodes": nodes, "arcs": arcs}, f, indent=2)


def prepare_workdir(workdir, dataset_key,
                    copy_xes=True,
                    decompress_xes=False,
                    discover_pnmls=False,
                    per_miner_dfgs=False):
    """Prepare `workdir` with data for a benchmark method.

    Args:
        workdir: Absolute path to a (possibly non-existent) temp directory.
                 Created if needed; caller is responsible for cleanup.
        dataset_key: e.g. "D1", "D2".
        copy_xes: Copy the source XES to workdir (native format — .xes or .xes.gz).
        decompress_xes: Also produce a decompressed .xes_plain (M6 Java JAR only).
        discover_pnmls: Discover PNML for all 8 miners.
        per_miner_dfgs: Simulate PNMLs → per-miner DFG JSONs (implies discover_pnmls).

    Returns:
        dict {
            "dataset_name": str,
            "xes_path": str,
            "xes_plain": str,        # decompressed copy (same as xes_path if not decompressed)
            "dfg_path": str | None,  # always None — log-level DFG no longer produced
            "manifest_path": str,
            "manifest": dict,
            "miner_names": list[str],
        }
    """
    os.makedirs(workdir, exist_ok=True)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from datasets import DATASETS

    ds = DATASETS[dataset_key]
    dname = ds["name"]
    slug = dname.lower().replace(" ", "_")
    log_path = ds["log_path"]

    # ── XES copy (always retains original extension) ──────────────────────
    xes_path = os.path.join(workdir, f"{slug}{os.path.splitext(log_path)[1]}")
    if copy_xes:
        shutil.copy2(log_path, xes_path)
        xes_manifest = xes_path
    else:
        xes_manifest = log_path

    # ── Decompressed XES plain (M6 Java JAR only) ─────────────────────────
    xes_plain = xes_path
    if decompress_xes and copy_xes:
        slug_xes = os.path.join(workdir, f"{slug}.xes")
        if log_path.endswith(".gz"):
            with gzip.open(log_path, "rb") as f_in:
                with open(slug_xes, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            xes_plain = slug_xes
        else:
            # Already plain .xes, just symlink or reference the copy
            xes_plain = xes_path

    print(f"  [prep] XES → {xes_manifest}" +
          (f"  plain → {xes_plain}" if xes_plain != xes_manifest else ""))

    manifest = {
        "dataset": dname,
        "xes_file": xes_manifest,
        "miners": {},
    }

    # ── PNML discovery for all miners ─────────────────────────────────────
    if discover_pnmls:
        log = pm4py.read_xes(log_path)
        log = pm4py.convert_to_event_log(log)
        MINERS = _get_miners()
        for mname, mfn in MINERS.items():
            try:
                t0 = datetime.now()
                net, im, fm = mfn(log)
                pn = os.path.join(workdir, f"{mname}.pnml")
                pm4py.write_pnml(net, im, fm, pn)
                manifest["miners"][mname] = {
                    "pnml": pn,
                    "n_transitions": len(net.transitions),
                    "n_places": len(net.places),
                }
                el = (datetime.now() - t0).total_seconds()
                print(f"  [prep] {mname:22s} → {pn}  ({el:.1f}s)")
            except Exception as e:
                print(f"  [prep] {mname:22s} SKIP ({e})")

    # ── Per-miner DFGs (simulate PNML → DFG JSON) ─────────────────────────
    if per_miner_dfgs:
        dfg_dir = os.path.join(workdir, "dfg_models")
        os.makedirs(dfg_dir, exist_ok=True)
        if "log" not in locals():
            log = pm4py.read_xes(log_path)
            log = pm4py.convert_to_event_log(log)
        for mname, minfo in list(manifest["miners"].items()):
            pn = minfo.get("pnml")
            if not pn or not os.path.exists(pn):
                continue
            try:
                net, im, fm = pm4py.read_pnml(pn)
                sim = pm4py.play_out(net, im, fm, no_traces=5000)
                sdfg, _, _ = pm4py.discover_dfg(sim)
                af = Counter()
                for t in sim:
                    for e in t:
                        af[e["concept:name"]] += 1
                acts = sorted(af.keys())
                aid = {a: i + 1 for i, a in enumerate(acts)}
                nodes = [{"id": aid[a], "label": a, "freq": af[a]} for a in acts]
                nodes.append({"id": len(acts) + 1, "label": "INPUT", "freq": len(sim)})
                nodes.append({"id": len(acts) + 2, "label": "OUTPUT", "freq": len(sim)})
                arcs = [{"from": aid[a], "to": aid[b], "freq": f} for (a, b), f in sdfg.items()]
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
                out = os.path.join(dfg_dir, f"{mname}_dfg.json")
                with open(out, "w") as f:
                    json.dump({"nodes": nodes, "arcs": arcs}, f, indent=2)
                print(f"  [prep] {mname:22s} DFG → {out}")
            except Exception as e:
                print(f"  [prep] {mname:22s} DFG SKIP ({e})")

    # ── Write manifest ────────────────────────────────────────────────────
    manifest_path = os.path.join(workdir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  [prep] Manifest → {manifest_path}")

    return {
        "dataset_name": dname,
        "xes_path": xes_manifest,
        "xes_plain": xes_plain,
        "dfg_path": None,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "miner_names": list(_get_miners().keys()),
    }


def get_miner_names():
    """Return canonical list of 8 miner names."""
    return list(_get_miners().keys())
