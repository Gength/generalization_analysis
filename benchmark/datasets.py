"""
Canonical dataset definitions for the Generalization Benchmark (Methodology v2).

Single source of truth — all benchmark scripts SHOULD import from here
instead of defining their own DATASETS dicts.

See BenchmarkDesign.md for the full catalog and selection rationale.
  D1   Sepsis                — smallest, ideal smoke test
  D2   BPI 2013 Incidents    — small but diverse (1.5K variants, 4 activities)
  D3   BPI 2017              — variant explosion (87% singletons) + deep traces
  D4   BPI 2018              — largest variant count (28K) + deepest traces (avg 57)
  D5   BPI 2019              — largest case count (251K), tests PM4Py scaling
  D6   BPI 2013 Problem Open
  D7   BPI 2013 Problem Closed
  D8   BPI 2015 Municipality 2
  D9   BPI 2015 Municipality 4
  D10  BPI 2015 Municipality 1
  D11  BPI 2011 Hospital
  D12  BPI 2015 Municipality 5
  D13  BPI 2015 Municipality 3
  D14  BPI 2020 PrepaidTravel
  D15  BPI 2020 InternationalDecl.
  D16  BPI 2020 RequestForPayment
  D17  BPI 2020 PermitLog
  D18  BPI 2020 DomesticDecl.
  D19  BPI 2012
  D20  Hospital Billing
  D21  Road Traffic Fine
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
    "D6": {
        "name": "BPI2013_Problem_Open",
        "log_path": "data/BPI-Challenge_2013/Problem_Management_Log_Open_Problems.xes.gz",
        "system_name": "bpi2013_problem_open",
    },
    "D7": {
        "name": "BPI2013_Problem_Closed",
        "log_path": "data/BPI-Challenge_2013/Problem_Management_Log_Closed_Problems.xes.gz",
        "system_name": "bpi2013_problem_closed",
    },
    "D8": {
        "name": "BPI2015_Municipality_2",
        "log_path": "data/BPI-Challenge_2015/Municipality_log_2.xes",
        "system_name": "bpi2015_municipality_2",
    },
    "D9": {
        "name": "BPI2015_Municipality_4",
        "log_path": "data/BPI-Challenge_2015/Municipality_log_4.xes",
        "system_name": "bpi2015_municipality_4",
    },
    "D10": {
        "name": "BPI2015_Municipality_1",
        "log_path": "data/BPI-Challenge_2015/Municipality_log_1.xes",
        "system_name": "bpi2015_municipality_1",
    },
    "D11": {
        "name": "BPI2011_Hospital",
        "log_path": "data/BPI-Challenge_2011/Hospital_log.xes.gz",
        "system_name": "bpi2011_hospital",
    },
    "D12": {
        "name": "BPI2015_Municipality_5",
        "log_path": "data/BPI-Challenge_2015/Municipality_log_5.xes",
        "system_name": "bpi2015_municipality_5",
    },
    "D13": {
        "name": "BPI2015_Municipality_3",
        "log_path": "data/BPI-Challenge_2015/Municipality_log_3.xes",
        "system_name": "bpi2015_municipality_3",
    },
    "D14": {
        "name": "BPI2020_PrepaidTravel",
        "log_path": "data/BPI-Challenge_2020/PrepaidTravelCost.xes.gz",
        "system_name": "bpi2020_prepaid_travel",
    },
    "D15": {
        "name": "BPI2020_InternationalDecl",
        "log_path": "data/BPI-Challenge_2020/InternationalDeclarations.xes.gz",
        "system_name": "bpi2020_international_decl",
    },
    "D16": {
        "name": "BPI2020_RequestForPayment",
        "log_path": "data/BPI-Challenge_2020/RequestForPayment.xes.gz",
        "system_name": "bpi2020_request_payment",
    },
    "D17": {
        "name": "BPI2020_PermitLog",
        "log_path": "data/BPI-Challenge_2020/PermitLog.xes.gz",
        "system_name": "bpi2020_permit_log",
    },
    "D18": {
        "name": "BPI2020_DomesticDecl",
        "log_path": "data/BPI-Challenge_2020/DomesticDeclarations.xes.gz",
        "system_name": "bpi2020_domestic_decl",
    },
    "D19": {
        "name": "BPI2012",
        "log_path": "data/BPI-Challenge_2012/BPI_Challenge_2012.xes.gz",
        "system_name": "bpi2012",
    },
    "D20": {
        "name": "Hospital_Billing",
        "log_path": "data/Hospital Billing - Event Log_1_all/Hospital Billing - Event Log.xes.gz",
        "system_name": "hospital_billing",
    },
    "D21": {
        "name": "Road_Traffic_Fine",
        "log_path": "data/Road Traffic Fine Management Process_1_all/Road_Traffic_Fine_Management_Process.xes.gz",
        "system_name": "road_traffic_fine",
    },
}

# ── Derived paths (dataset-independent) ─────────────────────────────────────
MODEL_DIR = "benchmark/models"
MANIFEST_PATH = os.path.join(MODEL_DIR, "manifest.json")
CONFIG_DIR_V2 = "benchmark/results/configs"

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
