"""
Canonical dataset definitions for the Generalization Benchmark (Methodology v2).

Single source of truth — all benchmark scripts SHOULD import from here
instead of defining their own DATASETS dicts.

See BenchmarkDesign.md for selection rationale (Methodology v2):
  D1   Sepsis           — smallest, ideal smoke test
  D2   BPI 2013 Inc.    — small but diverse (1.5K variants, 4 activities)
  D3   BPI 2017         — variant explosion (87% singletons) + deep traces
  D4   BPI 2018         — largest variant count (28K) + deepest traces (avg 57)
  D5   BPI 2019         — largest case count (251K), tests PM4Py scaling
"""

import os

# ── Canonical dataset registry ──────────────────────────────────────────────
# Every entry MUST have: name, log_path
# Optional extras for specific bridges: system_name (AVATAR), config_dir
# model_dir and manifest_path are derived (same for all datasets).

DATASETS = {
    "D1": {
        "name": "Sepsis",
        "log_path": "data/Sepsis Cases - Event Log_1_all/Sepsis Cases - Event Log.xes.gz",
        "system_name": "sepsis",
    },
    "D2": {
        "name": "BPI2013_Incidents",
        "log_path": "data/BPI-Challenge_2013/Incident_Management_Log.xes.gz",
        "system_name": "bpi2013_incidents",
    },
    "D3": {
        "name": "BPI2017",
        "log_path": "data/BPI-Challenge_2017/BPI Challenge 2017.xes.gz",
        "system_name": "bpi2017",
    },
    "D4": {
        "name": "BPI2018",
        "log_path": "data/BPI-Challenge_2018/BPI Challenge 2018.xes.gz",
        "system_name": "bpi2018",
    },
    "D5": {
        "name": "BPI2019",
        "log_path": "data/BPI-Challenge_2019/BPI_Challenge_2019.xes.gz",
        "system_name": "bpi2019",
    },
}

# ── Derived paths (dataset-independent) ─────────────────────────────────────
MODEL_DIR = "benchmark/models"
MANIFEST_PATH = os.path.join(MODEL_DIR, "manifest.json")
CONFIG_DIR_V2 = "benchmark/results/configs_v2"

# ── Convenience helpers ─────────────────────────────────────────────────────
def get_info(ds_key):
    """Return the dataset info dict for *ds_key* (e.g. 'D1')."""
    if ds_key not in DATASETS:
        raise KeyError(f"Unknown dataset '{ds_key}'. Available: {list(DATASETS.keys())}")
    info = dict(DATASETS[ds_key])
    info.setdefault("model_dir", MODEL_DIR)
    info.setdefault("manifest", MANIFEST_PATH)
    info.setdefault("config_dir", CONFIG_DIR_V2)
    return info

def get_log_path(ds_key):
    """Return the log path for *ds_key*."""
    return DATASETS[ds_key]["log_path"]
