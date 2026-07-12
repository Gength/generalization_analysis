#!/usr/bin/env python3
"""M5 — AVATAR (RelGAN) via Docker
===================================
Provides run() for job wrappers. CLI via main().
"""
import subprocess, os, sys, json, time, glob, re, argparse, csv, shutil, secrets
from datetime import datetime, timezone
from collections import defaultdict

import pm4py
from pm4py.objects.log.obj import EventLog

from miners import MINERS
from datasets import DATASETS

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AVATAR_ABS = os.path.join(PROJ, "src", "AVATAR")
AVATAR_BEST_CFG = os.path.join(PROJ, "benchmark", "avatar_best_suffix.json")
MIN_SUFFIX = 2000  # skip early-training noise in fallback

DOCKER_IMAGE = "avatar-tf1"
DOCKER_GPU = ["--rm", "--gpus", "all", "--ipc=host", "-e", "CUDA_VISIBLE_DEVICES=0"]


# ═══════════════════════════════════════════════════════════════════════
# Docker helpers
# ═══════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════
# Suffix selection
# ═══════════════════════════════════════════════════════════════════════

def _get_best_suffix(system_name):
    """Pick best suffix. Priority: 1) centralized config, 2) ranks JSON,
    3) fallback: max checkpoint step (>= MIN_SUFFIX guard)."""
    # Priority 1: centralized config
    if os.path.exists(AVATAR_BEST_CFG):
        with open(AVATAR_BEST_CFG) as f:
            cfg = json.load(f)
        if system_name in cfg:
            best = str(cfg[system_name])
            print(f"  [best_suffix] from avatar_best_suffix.json: {best}")
            return best
    # Priority 2: ranks JSON (best_suffix field)
    ranks_path = os.path.join(AVATAR_ABS, "data", "avatar", "sgans",
                              system_name, "0", f"suffix_ranks_relgan_{system_name}_0.json")
    if os.path.exists(ranks_path):
        try:
            with open(ranks_path) as f:
                ranks = json.load(f)
            if "best_suffix" in ranks:
                best = str(ranks["best_suffix"])
                print(f"  [best_suffix] from suffix_ranks JSON: {best}")
                return best
        except Exception:
            pass
    # Priority 3: fallback — max checkpoint (skip early noise)
    ckpt_glob = os.path.join(AVATAR_ABS, "data", "avatar", "sgans",
                             system_name, "0", "tf_logs", "ckpt", "*.meta")
    ckpts = glob.glob(ckpt_glob)
    if ckpts:
        nums = [int(re.search(r'-(\d+)\.meta$', f).group(1))
                for f in ckpts if re.search(r'-(\d+)\.meta$', f)]
        valid = [n for n in nums if n >= MIN_SUFFIX]
        if valid:
            best = str(max(valid))
            print(f"  [best_suffix] fallback (max checkpoint >= {MIN_SUFFIX}): {best}")
            return best
        elif nums:
            best = str(max(nums))
            print(f"  [best_suffix] fallback (max checkpoint, no >={MIN_SUFFIX}): {best}")
            return best
    return None


def _has_checkpoints(system_name):
    """Check if any training checkpoints exist."""
    ckpt_glob = os.path.join(AVATAR_ABS, "data", "avatar", "sgans",
                             system_name, "0", "tf_logs", "ckpt", "*.meta")
    return bool(glob.glob(ckpt_glob))


# ═══════════════════════════════════════════════════════════════════════
# CSV decoding helpers
# ═══════════════════════════════════════════════════════════════════════

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
                if tuple(tokens[pos:pos + n]) == act_words:
                    decoded.append(vocab[act_words])
                    pos += n
                    matched = True
                    break
        if not matched:
            pos += 1
    return decoded


def _create_corrected_csv(samp_path, csv_path, vocab, sorted_keys, train_path):
    """Read generated samples, decode, merge with train variants (dedup), write CSV."""
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
            rows.append([act_name, str(case_id), str(datetime.fromtimestamp(ts))])
        case_id += 1
    # Add training variants (dedup)
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
                ts = 1000000
                for act_name in decoded:
                    ts += 1
                    rows.append([act_name, str(case_id), str(datetime.fromtimestamp(ts))])
                case_id += 1
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerows(rows)
    n_traces = len(rows) - 1
    print(f"  Corrected CSV: {case_id} traces, {n_traces} events → {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
# Core pipeline
# ═══════════════════════════════════════════════════════════════════════

def run(dataset_key, workdir, output_dir, quick=False,
        eval_only=False, miners=None, n_runs=2):
    """Run M5. Reads XES from workdir, uses AVATAR source for Docker, writes configs."""
    # ── Resolve dataset ────────────────────────────────────────────
    mp = os.path.join(workdir, "manifest.json")
    if os.path.exists(mp):
        with open(mp) as f:
            mf = json.load(f)
        dname, xes_path = mf["dataset"], mf["xes_file"]
    else:
        from job_prepare import prepare_workdir
        ctx = prepare_workdir(workdir, dataset_key, copy_xes=True)
        dname, xes_path = ctx["dataset_name"], ctx["xes_path"]

    SYSTEM_NAME = DATASETS[dataset_key]["system_name"]
    VARIANT_DIR = os.path.join(AVATAR_ABS, "data/variants")
    os.makedirs(VARIANT_DIR, exist_ok=True)
    vfile = os.path.join(VARIANT_DIR, f"{SYSTEM_NAME}_train.txt")

    npre = "3" if quick else "100"
    nadv = "100" if quick else "5000"

    # ── Read log once ──────────────────────────────────────────────
    log = pm4py.read_xes(xes_path)
    log = pm4py.convert_to_event_log(log)
    vm = defaultdict(list)
    for t in log:
        vm[tuple(e["concept:name"] for e in t)].append(t)
    vs = list(vm.keys())
    split = int(len(vs) * 0.8)

    # ── Create variant files (training only) ───────────────────────
    if not eval_only and not os.path.exists(vfile):
        print(f"Creating variant file: {vfile}")
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

    # ── Step 1: Resolve suffix ─────────────────────────────────────
    suffix = None
    if eval_only:
        suffix = _get_best_suffix(SYSTEM_NAME)
        if suffix is None:
            print("--eval-only failed: no checkpoint found!")
            return
        print(f"--eval-only: selected checkpoint suffix={suffix}")

    # ── Step 2: Training (skip when --eval-only) ───────────────────
    training_elapsed = 0.0
    if not eval_only:
        if _has_checkpoints(SYSTEM_NAME):
            print("Found existing checkpoints; will retrain and re-select best suffix")
        print(f"AVATAR {'QUICK' if quick else 'FULL'}: {npre} pre-epochs, {nadv} adv steps")
        docker_kill_all()
        t0 = time.time()
        r = subprocess.run(
            ["docker", "run"] + DOCKER_GPU
            + ["-v", f"{AVATAR_ABS}:/workspace/src/AVATAR",
               "-w", "/workspace/src/AVATAR",
               "-e", f"AVATAR_NPRE_EPOCHS={npre}",
               "-e", f"AVATAR_NADV_STEPS={nadv}",
               "-e", f"AVATAR_BATCH_SIZE={os.environ.get('AVATAR_BATCH_SIZE', '16')}",
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
        suffix = _get_best_suffix(SYSTEM_NAME)
        if suffix is None:
            print("No checkpoint after training!")
            return
        print(f"Best checkpoint: suffix={suffix}")

    # ── Step 3: Sampling (always, n_runs times) ────────────────────
    print(f"Sampling ({n_runs} run(s))...")
    for jid in range(n_runs):
        t_samp = time.time()
        docker_kill_all()
        docker_args = ["docker", "run"] + DOCKER_GPU \
            + ["-v", f"{AVATAR_ABS}:/workspace/src/AVATAR",
               "-w", "/workspace/src/AVATAR"]
        if jid > 0:
            host_ckpt = os.path.join(AVATAR_ABS, "data", "avatar", "sgans",
                                     SYSTEM_NAME, "0", "tf_logs", "ckpt")
            container_ckpt = f"/workspace/src/AVATAR/data/avatar/sgans/{SYSTEM_NAME}/{jid}/tf_logs/ckpt"
            docker_args += ["-v", f"{host_ckpt}:{container_ckpt}"]
        docker_args += [DOCKER_IMAGE, "python", "-u", "-m", "avatar.sampling",
                        "-s", SYSTEM_NAME, "-j", str(jid), "-sfx", suffix, "-gpu", "0",
                        "-strategy", "naive", "-n_n", "10000"]
        r2 = subprocess.run(docker_args, capture_output=True, text=True)
        if r2.returncode != 0:
            print(f"Sampling j{jid} FAILED (exit={r2.returncode})")
            stderr_lines = r2.stderr.strip().split('\n')
            print('\n'.join(stderr_lines[-20:]))
            return
        print(f"  j{jid} done ({time.time() - t_samp:.0f}s)")
        sp = os.path.join(AVATAR_ABS, "data", "avatar", "variants",
                          f"{SYSTEM_NAME}_relgan_{suffix}_j{jid}_naive.txt")
        if not os.path.exists(sp):
            print(f"WARNING: sampling output not found at {sp}")

    # ── Step 4: Generalization ─────────────────────────────────────
    disc_log = EventLog([t for v in vs[split:] for t in vm[v]])
    PNML_DIR = os.path.join(AVATAR_ABS, "data", "pns", SYSTEM_NAME)
    os.makedirs(PNML_DIR, exist_ok=True)
    target = [(n, f) for n, f in MINERS.items() if miners is None or n in miners]

    # Pre-compute vocab for CSV decoding (same log → same vocab)
    vocab, sorted_keys = _build_vocab(log)
    train_path = os.path.join(AVATAR_ABS, "data", "variants", f"{SYSTEM_NAME}_train.txt")

    # Create corrected CSV per job
    available_jobs = []
    for jid in range(n_runs):
        sp = os.path.join(AVATAR_ABS, "data", "avatar", "variants",
                          f"{SYSTEM_NAME}_relgan_{suffix}_j{jid}_naive.txt")
        cp = os.path.join(AVATAR_ABS, "data", "avatar", "variants",
                          f"{SYSTEM_NAME}_relgan_{suffix}_j{jid}_naive_decoded.csv")
        if os.path.exists(sp):
            _create_corrected_csv(sp, cp, vocab, sorted_keys, train_path)
            available_jobs.append(jid)
        else:
            print(f"  WARNING: no sample file at {sp} — job {jid} skipped")

    os.makedirs(output_dir, exist_ok=True)
    t_gen = time.time()
    for name, fn in target:
        print(f"\n[{name}] ", end="", flush=True)
        try:
            net, im, fm = fn(disc_log)
            pm4py.write_pnml(net, im, fm, os.path.join(PNML_DIR, f"{name}.pnml"))
        except Exception as e:
            print(f"Discovery error: {e}")
            continue

        scores, runtimes = [], []
        for jid in available_jobs:
            docker_kill_all()
            gen_args = ["docker", "run"] + DOCKER_GPU \
                + ["-v", f"{AVATAR_ABS}:/workspace/src/AVATAR",
                   "-w", "/workspace/src/AVATAR",
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

        valid = [s for s in scores if s >= 0]
        if not valid:
            print("  All jobs FAILED")
            continue
        n = len(valid)
        mean_val = sum(valid) / n
        std_val = (sum((s - mean_val) ** 2 for s in valid) / max(n - 1, 1)) ** 0.5
        runtime_max = max(runtimes)

        cfg = {
            "dataset": dname, "miner": name, "method": "M5",
            "method_label": "AVATAR (RelGAN)",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "host": "docker", "seed": 42,
            "parameters": {
                "GAN": "RelGAN", "suffix": suffix, "strategy": "naive",
                "npre_epochs": npre, "nadv_steps": nadv,
                "n_samples": 10000, "n_runs": n,
                "decoding": "greedy_longest_match",
            },
            "results": {
                "mean": round(mean_val, 6),
                "std": round(std_val, 6),
                "raw_runs": [round(s, 6) for s in valid],
                "n_runs": n,
                "runtime_s": round(runtime_max, 3),
                "training_time": training_elapsed,
            },
            "notes": f"{n} evaluation run(s); multi-word activity fix (corrected CSV).",
        }
        path = os.path.join(output_dir, f"{dname}__{name}__M5.json")
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)
        raw_str = ", ".join(f"{s:.4f}" for s in valid)
        print(f"{'✅' if mean_val >= 0 else '❌'} mean={mean_val:.4f} ± {std_val:.4f}  "
              f"raw=[{raw_str}]  ({runtime_max:.0f}s)")

    print(f"\nGeneralization done ({time.time() - t_gen:.0f}s) → {output_dir}/")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

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

    workdir = f"/tmp/benchmark_M5_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}/"
    output_dir = args.output or os.path.join(workdir, "results")
    os.makedirs(output_dir, exist_ok=True)

    run(args.dataset, workdir, output_dir, quick=args.quick,
        eval_only=args.eval_only, miners=args.miners, n_runs=args.n_runs)
    shutil.rmtree(workdir)
    print(f"  [clean] removed {workdir}")


if __name__ == "__main__":
    main()
