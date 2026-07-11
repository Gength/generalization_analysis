#!/usr/bin/env python3
"""M5 — AVATAR (RelGAN) via Docker
===================================
Provides run() for job wrappers. CLI via main().
"""
import subprocess, os, sys, json, time, glob, re, argparse, csv
from datetime import datetime, timezone
from collections import defaultdict

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def docker_kill_all():
    for img in ["avatar-tf1"]:
        try:
            out = subprocess.run(["docker", "ps", "-q", "--filter", f"ancestor={img}"],
                                 capture_output=True, text=True, timeout=10)
            ids = out.stdout.strip().split()
            if ids:
                subprocess.run(["docker", "kill"] + ids, capture_output=True, timeout=10)
        except Exception:
            pass


def run(dataset_key, workdir, output_dir, quick=False,
        eval_only=False, miners=None, n_runs=2):
    """Run M5. Reads XES from workdir, uses AVATAR source for Docker, writes configs."""
    mp = os.path.join(workdir, "manifest.json")
    if os.path.exists(mp):
        with open(mp) as f:
            mf = json.load(f)
        dname, xes_path = mf["dataset"], mf["xes_file"]
    else:
        from job_prepare import prepare_workdir
        ctx = prepare_workdir(workdir, dataset_key, copy_xes=True)
        dname, xes_path = ctx["dataset_name"], ctx["xes_path"]

    DOCKER_IMAGE = "avatar-tf1"
    AVATAR_DIR = "src/AVATAR"
    AVATAR_ABS = os.path.join(PROJ, AVATAR_DIR)

    from datasets import DATASETS
    SYSTEM_NAME = DATASETS[dataset_key]["system_name"]
    import pm4py
    from pm4py.objects.log.obj import EventLog
    from miners import filtered_trace_miner
    from pm4py.objects.petri_net.obj import PetriNet, Marking
    from pm4py.objects.petri_net.utils import petri_utils

    # ── Variants (to AVATAR dir — Docker mount needs them there) ─────────
    VARIANT_DIR = os.path.join(AVATAR_ABS, "data/variants")
    os.makedirs(VARIANT_DIR, exist_ok=True)
    vfile = os.path.join(VARIANT_DIR, f"{SYSTEM_NAME}_train.txt")

    if not eval_only and not os.path.exists(vfile):
        print(f"Creating variant file: {vfile}")
        log = pm4py.read_xes(xes_path)
        log = pm4py.convert_to_event_log(log)
        vm = defaultdict(list)
        for t in log:
            vm[tuple(e["concept:name"] for e in t)].append(t)
        vs = list(vm.keys())
        split = int(len(vs) * 0.8)
        with open(os.path.join(VARIANT_DIR, f"{SYSTEM_NAME}_pop.txt"), "w") as f:
            for v in vm:
                f.write(" ".join(v) + "\n")
        with open(vfile, "w") as f:
            for v in vs[:split]:
                f.write(" ".join(v) + "\n")
        with open(os.path.join(VARIANT_DIR, f"{SYSTEM_NAME}_test.txt"), "w") as f:
            for v in vs[split:]:
                f.write(" ".join(v) + "\n")
        print(f"  {len(vs)} variants ({split} train, {len(vs)-split} test)")
    else:
        print(f"Variant file exists: {vfile}" if not eval_only else "--eval-only mode")

    npre = "3" if quick else "100"
    nadv = "100" if quick else "5000"

    training_elapsed = 0.0  # populated during training, reused in config

    # ── Step 1: Discover suffix from checkpoint ──────────
    ckpt_glob = os.path.join(AVATAR_ABS, f"data/avatar/sgans/{SYSTEM_NAME}/0/tf_logs/ckpt/*.meta")
    ckpts = glob.glob(ckpt_glob)
    suffix = None
    if ckpts:
        nums = [int(re.search(r'-(\d+)\.meta$', f).group(1)) for f in ckpts if re.search(r'-(\d+)\.meta$', f)]
        suffix = str(max(nums)) if nums else None
    if eval_only and suffix is None:
        print("--eval-only failed: no checkpoint found!")
        return
    if suffix:
        print(f"Existing checkpoint: suffix={suffix}")

    # ── Step 2: Training (skip when --eval-only) ─────────
    training_elapsed = 0.0
    if not eval_only:
        print(f"AVATAR {'QUICK' if quick else 'FULL'}: {npre} pre-epochs, {nadv} adv steps")
        docker_kill_all()
        t0 = time.time()
        r = subprocess.run(["docker", "run", "--rm", "--gpus", "all", "--ipc=host",
                            "-v", f"{AVATAR_ABS}:/workspace/src/AVATAR",
                            "-w", "/workspace/src/AVATAR",
                            "-e", f"AVATAR_NPRE_EPOCHS={npre}",
                            "-e", f"AVATAR_NADV_STEPS={nadv}",
                            "-e", f"AVATAR_BATCH_SIZE={os.environ.get('AVATAR_BATCH_SIZE', '16')}",
                            "-e", "CUDA_VISIBLE_DEVICES=0",
                            "-e", "TF_GPU_ALLOCATOR=cuda_malloc_async",
                            DOCKER_IMAGE, "python", "-u", "-m", "avatar.training",
                            "-s", SYSTEM_NAME, "-j", "0", "-gpu", "0", "-n", "10000"],
                           capture_output=True, text=True)
        training_elapsed = time.time() - t0
        if r.returncode != 0:
            print(f"Training FAILED (exit={r.returncode})")
            stderr_lines = r.stderr.strip().split('\n')
            print('\n'.join(stderr_lines[-20:]))
            return
        print(f"Training done ({training_elapsed:.0f}s)")
        # Re-discover suffix from new checkpoint
        ckpts = glob.glob(ckpt_glob)
        if not ckpts:
            print("No checkpoints after training!")
            return
        nums = [int(re.search(r'-(\d+)\.meta$', f).group(1)) for f in ckpts if re.search(r'-(\d+)\.meta$', f)]
        suffix = str(max(nums)) if nums else "5000"
        print(f"Checkpoint: suffix={suffix}")
    else:
        print(f"--eval-only: reusing checkpoint suffix={suffix}")

    # ── Step 3: Sampling (always, n_runs times) ──────────
    print(f"Sampling ({n_runs} run(s))...")
    for jid in range(n_runs):
        t_samp = time.time()
        docker_kill_all()
        r2 = subprocess.run(["docker", "run", "--rm", "--gpus", "all", "--ipc=host",
                             "-v", f"{AVATAR_ABS}:/workspace/src/AVATAR",
                             "-w", "/workspace/src/AVATAR",
                             "-e", "CUDA_VISIBLE_DEVICES=0",
                             DOCKER_IMAGE, "python", "-u", "-m", "avatar.sampling",
                             "-s", SYSTEM_NAME, "-j", str(jid), "-sfx", suffix, "-gpu", "0",
                             "-strategy", "naive", "-n_n", "10000"],
                            capture_output=True, text=True)
        if r2.returncode != 0:
            print(f"Sampling j{jid} FAILED (exit={r2.returncode})")
            stderr_lines = r2.stderr.strip().split('\n')
            print('\n'.join(stderr_lines[-20:]))
            return
        print(f"  j{jid} done ({time.time()-t_samp:.0f}s)")
        sp = os.path.join(AVATAR_ABS, f"data/avatar/variants/{SYSTEM_NAME}_relgan_{suffix}_j{jid}_naive.txt")
        if not os.path.exists(sp):
            print(f"WARNING: sampling output not found at {sp}")

    # ── Generalization ───────────────────────────────────────────────────
    def _flower(l):
        acts = set(e["concept:name"] for t in l for e in t)
        net = PetriNet("Flower"); p = PetriNet.Place("p0"); net.places.add(p)
        im = Marking({p: 1}); fm = Marking({p: 1})
        for a in acts:
            tr = PetriNet.Transition(a, a); net.transitions.add(tr)
            petri_utils.add_arc_from_to(p, tr, net); petri_utils.add_arc_from_to(tr, p, net)
        return net, im, fm

    log = pm4py.read_xes(xes_path)
    log = pm4py.convert_to_event_log(log)
    vm2 = defaultdict(list)
    for t in log:
        vm2[tuple(e["concept:name"] for e in t)].append(t)
    vs = list(vm2.keys())
    split = int(len(vs) * 0.8)
    disc_log = EventLog([t for v in vs[split:] for t in vm2[v]])

    PNML_DIR = os.path.join(AVATAR_ABS, "data/pns", SYSTEM_NAME)
    os.makedirs(PNML_DIR, exist_ok=True)

    all_miners = [
        ("Alpha", pm4py.discover_petri_net_alpha),
        ("Alpha+", pm4py.discover_petri_net_alpha_plus),
        ("Heuristics", pm4py.discover_petri_net_heuristics),
        ("Heuristics_Strict", lambda l: pm4py.discover_petri_net_heuristics(l, dependency_threshold=0.99)),
        ("Inductive_Strict", lambda l: pm4py.discover_petri_net_inductive(l, noise_threshold=0.0)),
        ("Inductive_Infrequent", lambda l: pm4py.discover_petri_net_inductive(l, noise_threshold=0.2)),
        ("Flower", lambda l: (None, None, None)),
        ("Trace_Filtered", lambda l: filtered_trace_miner(l, top_k=50)),
    ]
    target = [(n, f) for n, f in all_miners if miners is None or n in miners]

    # ── Corrected CSV for multi-word activity names ─────────────────────
    def _build_vocab(event_log):
        acts = set()
        for t in event_log:
            for e in t:
                acts.add(e["concept:name"])
        vocab = {tuple(a.lower().split()): a for a in acts}
        sorted_keys = sorted(vocab.keys(), key=lambda x: -len(x))
        return vocab, sorted_keys

    def _decode_tokens(tokens, vocab, sorted_keys):
        """Greedy longest-match decoding: map token list to activity names."""
        decoded = []
        pos = 0
        while pos < len(tokens):
            matched = False
            for act_words in sorted_keys:
                n = len(act_words)
                if pos + n <= len(tokens):
                    candidate = tuple(tokens[pos:pos+n])
                    if candidate == act_words:
                        decoded.append(vocab[act_words])
                        pos += n
                        matched = True
                        break
            if not matched:
                pos += 1  # skip unknown/unmatched token
        return decoded

    def _create_corrected_csv(samp_path, csv_path, event_log):
        """Read generated samples, decode with greedy_longest_match,
        merge with train variants (dedup), write corrected CSV."""
        vocab, sorted_keys = _build_vocab(event_log)
        with open(samp_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        rows = [["concept:name", "case:concept:name", "time:timestamp"]]
        case_id = 0
        seen_seqs = set()
        for line in lines:
            decoded = _decode_tokens(line.strip().split(), vocab, sorted_keys)
            seq = " ".join(decoded)
            if seq in seen_seqs:
                continue
            seen_seqs.add(seq)
            ts = 0
            for act_name in decoded:
                ts += 1
                dt_obj = datetime.fromtimestamp(ts)
                rows.append([act_name, str(case_id), str(dt_obj)])
            case_id += 1
        # Add training variants (dedup, matching generalization.py behaviour)
        train_path = os.path.join(AVATAR_ABS, "data/variants", f"{SYSTEM_NAME}_train.txt")
        if os.path.exists(train_path):
            with open(train_path) as f:
                for t_line in f:
                    t_line = t_line.strip()
                    if not t_line:
                        continue
                    decoded = _decode_tokens(t_line.split(), vocab, sorted_keys)
                    seq = " ".join(decoded)
                    if seq in seen_seqs:
                        continue
                    seen_seqs.add(seq)
                    ts = 1000000  # offset from generated traces
                    for act_name in decoded:
                        ts += 1
                        dt_obj = datetime.fromtimestamp(ts)
                        rows.append([act_name, str(case_id), str(dt_obj)])
                    case_id += 1
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerows(rows)
        n_traces = len(rows) - 1
        print(f"  Corrected CSV: {case_id} traces, {n_traces} events → {csv_path}")

    # ── Create corrected CSV per job ────────────────────────────────
    csv_paths = {}  # job → csv path
    available_jobs = []
    for jid in range(n_runs):
        sp = os.path.join(AVATAR_ABS, f"data/avatar/variants/{SYSTEM_NAME}_relgan_{suffix}_j{jid}_naive.txt")
        cp = os.path.join(AVATAR_ABS, f"data/avatar/variants/{SYSTEM_NAME}_relgan_{suffix}_j{jid}_naive_decoded.csv")
        if os.path.exists(sp):
            _create_corrected_csv(sp, cp, log)
            csv_paths[jid] = cp
            available_jobs.append(jid)
        else:
            print(f"  WARNING: no sample file at {sp} — job {jid} skipped")

    os.makedirs(output_dir, exist_ok=True)
    t_gen = time.time()
    for name, fn in target:
        print(f"\n[{name}] ", end="", flush=True)
        try:
            net, im, fm = _flower(disc_log) if name == "Flower" else fn(disc_log)
            pm4py.write_pnml(net, im, fm, os.path.join(PNML_DIR, f"{name}.pnml"))
        except Exception as e:
            print(f"Discovery error: {e}")
            continue

        scores = []
        runtimes = []
        for jid in available_jobs:
            docker_kill_all()
            gen_args = ["docker", "run", "--rm", "--gpus", "all", "--ipc=host",
                         "-v", f"{AVATAR_ABS}:/workspace/src/AVATAR",
                         "-w", "/workspace/src/AVATAR",
                         "-e", "CUDA_VISIBLE_DEVICES=0",
                         DOCKER_IMAGE, "python", "-u", "-m", "avatar.generalization",
                         "-s", SYSTEM_NAME, "-sfx", suffix, "-j", str(jid),
                         "-pn", f"{name}.pnml", "-strategy", "naive",
                         "--csv-only", "--csv-path",
                         f"/workspace/src/AVATAR/data/avatar/variants/{SYSTEM_NAME}_relgan_{suffix}_j{jid}_naive_decoded.csv"]
            t0 = time.time()
            r = subprocess.run(gen_args, capture_output=True, text=True)
            elapsed = time.time() - t0
            score = -1
            if r.returncode != 0:
                print(f"  Gen j{jid} FAILED (exit={r.returncode})")
                stderr_lines = r.stderr.strip().split('\n')
                print('\n'.join(stderr_lines[-20:]))
            for line in r.stdout.split("\n"):
                if "AVATAR Generalization=" in line:
                    try:
                        score = float(line.split("=")[-1].strip())
                    except Exception:
                        pass
            scores.append(score)
            runtimes.append(elapsed)

        # Compute statistics
        valid = [s for s in scores if s >= 0]
        if not valid:
            print(f"  All jobs FAILED")
            continue
        n = len(valid)
        mean_val = sum(valid) / n
        std_val = (sum((s - mean_val)**2 for s in valid) / max(n - 1, 1))**0.5
        runtime_max = max(runtimes)

        notes = "multi-word activity fix (corrected CSV)"
        params = {"GAN": "RelGAN", "suffix": suffix, "strategy": "naive",
                  "npre_epochs": npre, "nadv_steps": nadv,
                  "n_samples": 10000, "n_runs": n,
                  "decoding": "greedy_longest_match"}
        cfg = {
            "dataset": dname, "miner": name, "method": "M5",
            "method_label": "AVATAR (RelGAN)",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "host": "docker", "seed": 42,
            "parameters": params,
            "results": {
                "mean": round(mean_val, 6),
                "std": round(std_val, 6),
                "raw_runs": [round(s, 6) for s in valid],
                "n_runs": n,
                "runtime_s": round(runtime_max, 3),
                "training_time": training_elapsed,
            },
            "notes": f"{n} evaluation run(s); {notes}.",
        }
        path = os.path.join(output_dir, f"{dname}__{name}__M5.json")
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)
        raw_str = ", ".join(f"{s:.4f}" for s in valid)
        print(f"{'✅' if mean_val>=0 else '❌'} mean={mean_val:.4f} ± {std_val:.4f}  raw=[{raw_str}]  ({runtime_max:.0f}s)")

    print(f"\nGeneralization done ({time.time()-t_gen:.0f}s) → {output_dir}/")


def main():
    ap = argparse.ArgumentParser(description="AVATAR (M5) via Docker")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--miners", nargs="*", default=None)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-runs", type=int, default=2,
                    help="Number of sampling runs (default=2)")
    args = ap.parse_args()

    import shutil, secrets
    from datetime import datetime as dt
    workdir = f"/tmp/benchmark_M5_{args.dataset}_{dt.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}/"
    output_dir = args.output or os.path.join(workdir, "results")
    os.makedirs(output_dir, exist_ok=True)

    run(args.dataset, workdir, output_dir, quick=args.quick,
        eval_only=args.eval_only, miners=args.miners, n_runs=args.n_runs)
    shutil.rmtree(workdir)
    print(f"  [clean] removed {workdir}")


if __name__ == "__main__":
    main()
