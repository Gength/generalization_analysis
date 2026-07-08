#!/usr/bin/env python3
"""M5 — AVATAR (RelGAN) via Docker
===================================
Provides run() for job wrappers. CLI via main().
"""
import subprocess, os, sys, json, time, glob, re, argparse
from datetime import datetime, timezone
from collections import defaultdict

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def docker_kill_all():
    for img in ["avatar-tf1", "avatar-tf2"]:
        try:
            out = subprocess.run(["docker", "ps", "-q", "--filter", f"ancestor={img}"],
                                 capture_output=True, text=True, timeout=10)
            ids = out.stdout.strip().split()
            if ids:
                subprocess.run(["docker", "kill"] + ids, capture_output=True, timeout=10)
        except Exception:
            pass


def run(dataset_key, workdir, output_dir, quick=False, tf2=False,
        eval_only=False, miners=None):
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

    DOCKER_IMAGE = "avatar-tf2" if tf2 else "avatar-tf1"
    AVATAR_DIR = "src/AVATAR_tf2" if tf2 else "src/AVATAR"
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

    # ── Training + Sampling ──────────────────────────────────────────────
    if eval_only:
        sdir = os.path.join(AVATAR_ABS, "data/avatar/variants")
        sf = glob.glob(os.path.join(sdir, f"{SYSTEM_NAME}_relgan_*_j0_naive.txt"))
        suffix = None
        for f in sf:
            m = re.search(r'_relgan_(\d+)_j0_naive\.txt$', f)
            if m:
                if glob.glob(os.path.join(AVATAR_ABS, f"data/avatar/sgans/{SYSTEM_NAME}/0/tf_logs/ckpt/*.meta")):
                    suffix = m.group(1)
                    break
        if not suffix:
            print("--eval-only failed: no checkpoint+sampling pair!")
            return
        print(f"--eval-only: suffix={suffix}")
    else:
        print(f"AVATAR {'QUICK' if quick else 'FULL'}: {npre} pre-epochs, {nadv} adv steps")
        docker_kill_all()
        t0 = time.time()
        r = subprocess.run(["docker", "run", "--rm", "--gpus", "all", "--ipc=host",
                            "-v", f"{AVATAR_ABS}:/workspace/src/AVATAR",
                            "-w", "/workspace/src/AVATAR",
                            "-e", f"AVATAR_NPRE_EPOCHS={npre}",
                            "-e", f"AVATAR_NADV_STEPS={nadv}",
                            "-e", f"AVATAR_BATCH_SIZE={os.environ.get('AVATAR_BATCH_SIZE', '8' if tf2 else '16')}", "-e", "CUDA_VISIBLE_DEVICES=0",
                            "-e", "TF_GPU_ALLOCATOR=cuda_malloc_async",
                            DOCKER_IMAGE, "python", "-u", "-m", "avatar.training",
                            "-s", SYSTEM_NAME, "-j", "0", "-gpu", "0", "-n", "10000"],
                           capture_output=True, text=True)
        elapsed = time.time() - t0
        training_elapsed = elapsed
        if r.returncode != 0:
            print(f"Training FAILED (exit={r.returncode})")
            # Print last 20 lines of stderr for diagnosis
            stderr_lines = r.stderr.strip().split('\n')
            print('\n'.join(stderr_lines[-20:]))
            # Don't continue to sampling
            return
        print(f"Training done ({elapsed:.0f}s)")

        ckpts = glob.glob(os.path.join(AVATAR_ABS, f"data/avatar/sgans/{SYSTEM_NAME}/0/tf_logs/ckpt/*.meta"))
        if not ckpts:
            print("No checkpoints!")
            return
        suffixes = [int(re.search(r'-(\d+)\.meta$', f).group(1)) for f in ckpts if re.search(r'-(\d+)\.meta$', f)]
        suffix = str(max(suffixes)) if suffixes else "5000"
        print(f"Checkpoint: suffix={suffix}")

        print("Sampling...")
        docker_kill_all()
        r2 = subprocess.run(["docker", "run", "--rm", "--gpus", "all", "--ipc=host",
                             "-v", f"{AVATAR_ABS}:/workspace/src/AVATAR",
                             "-w", "/workspace/src/AVATAR",
                             "-e", "CUDA_VISIBLE_DEVICES=0",
                             DOCKER_IMAGE, "python", "-u", "-m", "avatar.sampling",
                             "-s", SYSTEM_NAME, "-j", "0", "-sfx", suffix, "-gpu", "0",
                             "-strategy", "naive", "-n_n", "10000"],
                            capture_output=True, text=True)
        if r2.returncode != 0:
            print(f"Sampling FAILED (exit={r2.returncode})")
            stderr_lines = r2.stderr.strip().split('\n')
            print('\n'.join(stderr_lines[-20:]))
            return
        print("Sampling done")
        samp_path = os.path.join(AVATAR_ABS, f"data/avatar/variants/{SYSTEM_NAME}_relgan_{suffix}_j0_naive.txt")
        if not os.path.exists(samp_path):
            print(f"WARNING: sampling output not found at {samp_path}")

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

    os.makedirs(output_dir, exist_ok=True)
    tag = "_TF2" if tf2 else ""
    for name, fn in target:
        print(f"\n[{name}] ", end="", flush=True)
        try:
            net, im, fm = _flower(disc_log) if name == "Flower" else fn(disc_log)
            pm4py.write_pnml(net, im, fm, os.path.join(PNML_DIR, f"{name}.pnml"))
        except Exception as e:
            print(f"Discovery error: {e}")
            continue
        docker_kill_all()
        t0 = time.time()
        r = subprocess.run(["docker", "run", "--rm", "--gpus", "all", "--ipc=host",
                            "-v", f"{AVATAR_ABS}:/workspace/src/AVATAR",
                            "-w", "/workspace/src/AVATAR",
                            "-e", "CUDA_VISIBLE_DEVICES=0",
                            DOCKER_IMAGE, "python", "-u", "-m", "avatar.generalization",
                            "-s", SYSTEM_NAME, "-sfx", suffix, "-j", "0",
                            "-pn", f"{name}.pnml", "-strategy", "naive"],
                           capture_output=True, text=True)
        elapsed = time.time() - t0
        score = -1
        if r.returncode != 0:
            print(f"  Generalization FAILED (exit={r.returncode})")
            stderr_lines = r.stderr.strip().split('\n')
            print('\n'.join(stderr_lines[-20:]))
        for line in r.stdout.split("\n"):
            if "AVATAR Generalization=" in line:
                try:
                    score = float(line.split("=")[-1].strip())
                except Exception:
                    pass
        cfg = {
            "dataset": dname, "miner": name, "method": "M5",
            "method_label": "AVATAR (RelGAN)",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "host": "docker", "seed": 42,
            "parameters": {"GAN": "RelGAN", "suffix": suffix, "strategy": "naive",
                           "npre_epochs": npre, "nadv_steps": nadv},
            "results": {"score": score, "runtime_s": elapsed, "training_time": training_elapsed},
            "notes": "",
        }
        path = os.path.join(output_dir, f"{dname}__{name}__M5{tag}.json")
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"{'✅' if score>=0 else '❌'} score={score:.4f} ({elapsed:.0f}s)")

    print(f"\nDone → {output_dir}/")


def main():
    ap = argparse.ArgumentParser(description="AVATAR (M5) via Docker")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--miners", nargs="*", default=None)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tf2", action="store_true")
    args = ap.parse_args()

    import shutil, secrets
    from datetime import datetime as dt
    workdir = f"/tmp/benchmark_M5_{args.dataset}_{dt.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}/"
    output_dir = args.output or os.path.join(workdir, "results")
    os.makedirs(output_dir, exist_ok=True)

    run(args.dataset, workdir, output_dir, quick=args.quick, tf2=args.tf2,
        eval_only=args.eval_only, miners=args.miners)
    shutil.rmtree(workdir)
    print(f"  [clean] removed {workdir}")


if __name__ == "__main__":
    main()
