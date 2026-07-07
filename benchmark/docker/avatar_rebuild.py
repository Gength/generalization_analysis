"""AVATAR independent rebuild (2026-07-07) — prep step.

Regenerates the Sepsis variant files for the fresh ProminentLab/AVATAR clone
(src/AVATAR, gitignored) with the exact split logic of run_avatar.py: variants
in log-encounter order, 80/20 head/tail, pop/train/test files. Deterministic
given the committed log, so the retrained GAN sees the same training variants
as the original (deleted) M5 runs.

Usage:  uv run python benchmark/docker/avatar_rebuild.py
Then train (quick anchor example, published defaults are 100/5000):
  docker run --rm -e AVATAR_NPRE_EPOCHS=3 -e AVATAR_NADV_STEPS=100 \
    -v <repo>/src/AVATAR:/workspace/src/AVATAR -w /workspace/src/AVATAR \
    avatar-cpu python -u -m avatar.training -s sepsis -j 0 -gpu 0 -n 10000
"""
import os, sys
from collections import defaultdict

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJ, "benchmark"))
import pm4py
from datasets import DATASETS

AVATAR_ABS = os.path.join(PROJ, "src", "AVATAR")
SYSTEM = DATASETS["D1"]["system_name"]          # "sepsis"
LOG = os.path.join(PROJ, DATASETS["D1"]["log_path"])

VARIANT_DIR = os.path.join(AVATAR_ABS, "data", "variants")
os.makedirs(VARIANT_DIR, exist_ok=True)
for d in ("train_data", "sgans", "variants"):
    os.makedirs(os.path.join(AVATAR_ABS, "data", "avatar", d), exist_ok=True)

log = pm4py.convert_to_event_log(pm4py.read_xes(LOG))
vm = defaultdict(list)
for t in log:
    vm[tuple(e["concept:name"] for e in t)].append(t)
vs = list(vm.keys())
split = int(len(vs) * 0.8)

with open(os.path.join(VARIANT_DIR, f"{SYSTEM}_pop.txt"), "w") as f:
    for v in vs:
        f.write(" ".join(v) + "\n")
with open(os.path.join(VARIANT_DIR, f"{SYSTEM}_train.txt"), "w") as f:
    for v in vs[:split]:
        f.write(" ".join(v) + "\n")
with open(os.path.join(VARIANT_DIR, f"{SYSTEM}_test.txt"), "w") as f:
    for v in vs[split:]:
        f.write(" ".join(v) + "\n")

print(f"{SYSTEM}: {len(vs)} variants ({split} train, {len(vs)-split} test) -> {VARIANT_DIR}")
