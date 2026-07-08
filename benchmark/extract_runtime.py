#!/usr/bin/env python3
"""
Runtime extractor for the Generalization Benchmark (Methodology v2).

Extracts `runtime_s` from benchmark config JSONs and prints aligned Markdown tables.

Usage:
    uv run python benchmark/extract_runtime.py --dataset D3
    uv run python benchmark/extract_runtime.py --dataset BPI2017
    uv run python benchmark/extract_runtime.py --all
    uv run python benchmark/extract_runtime.py --dataset D3 --results-dir benchmark/results/configs_v2

Output: Markdown table(s) to stdout.
"""
import argparse
import json
import sys
from pathlib import Path

from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmark.datasets import DATASETS

_DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results" / "configs"

METHOD_ORDER = [
    "M1a", "M1b", "M1c", "M1d", "M1e", "M1f", "M1g",
    "M2", "M3", "M4", "M5", "M6adapted", "M6original", "M7", "R1", "R2", "R3",
]
MINER_ORDER = [
    "Trace_Filtered", "Alpha", "Alpha+", "Heuristics", "Heuristics_Strict",
    "Inductive_Strict", "Inductive_Infrequent", "Flower",
]


def resolve_dataset(dataset_arg):
    """Return (name, ds_key_or_None) for the given CLI argument."""
    if dataset_arg.upper().startswith("D") and dataset_arg.upper() in DATASETS:
        ds_key = dataset_arg.upper()
        return DATASETS[ds_key]["name"], ds_key
    for ds_key, info in DATASETS.items():
        if info["name"].lower() == dataset_arg.lower():
            return info["name"], ds_key
    raise SystemExit(
        f"Unknown dataset '{dataset_arg}'. "
        f"Use a D-key (D1-D21) or a dataset name "
        f"({', '.join(v['name'] for v in DATASETS.values())})."
    )


def extract_dataset(results_dir, name, label=None):
    """Extract runtime_s for one dataset and print an aligned Markdown table."""
    heading = label or name
    files = sorted(results_dir.glob(f"{name}__*.json"))

    if not files:
        print(f"### {heading}")
        print()
        print("_No results found._")
        print()
        return

    data = {}
    for fpath in files:
        stem = fpath.stem
        parts = stem.split("__")
        if len(parts) != 3:
            continue
        _, miner, method = parts
        with open(fpath) as f:
            d = json.load(f)
        rt = d.get("results", {}).get("runtime_s")
        if rt is None or (isinstance(rt, (int, float)) and rt < 0):
            val = "-"
        elif isinstance(rt, (int, float)):
            val = round(rt, 1)
        else:
            val = "-"
        data[(miner, method)] = val

    headers = ["Miner"] + [f"{m}(s)" for m in METHOD_ORDER]
    rows = []
    for miner in MINER_ORDER:
        rows.append([miner] + [str(data.get((miner, m), "-")) for m in METHOD_ORDER])

    print(f"### {heading}")
    print()
    print(tabulate(rows, headers=headers, tablefmt="pipe", numalign="right", stralign="left"))
    print()


def main():
    parser = argparse.ArgumentParser(description="Extract benchmark runtime_s into Markdown tables.")
    parser.add_argument("--results-dir", type=str, default=str(_DEFAULT_RESULTS_DIR),
                        help=f"Results directory (default: {_DEFAULT_RESULTS_DIR})")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", type=str, help="Dataset key (e.g. D3) or name (e.g. BPI2017)")
    group.add_argument("--all", action="store_true", help="Extract runtime for every dataset with results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise SystemExit(f"Results directory not found: {results_dir}")

    def _extract(name, label=None):
        extract_dataset(results_dir, name, label=label)

    if args.all:
        seen = set()
        for fpath in results_dir.glob("*.json"):
            name = fpath.stem.split("__")[0]
            seen.add(name)
        printed = 0
        for ds_key in sorted(DATASETS.keys()):
            name = DATASETS[ds_key]["name"]
            if name in seen:
                _extract(name, label=f"{ds_key} ({name})")
                printed += 1
        if printed == 0:
            print("_No results found for any dataset._")
    else:
        name, ds_key = resolve_dataset(args.dataset)
        label = f"{ds_key} ({name})" if ds_key else name
        _extract(name, label=label)


if __name__ == "__main__":
    main()
