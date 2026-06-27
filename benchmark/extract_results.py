#!/usr/bin/env python3
"""
Universal result extractor for the Generalization Benchmark (Methodology v2).

Usage:
    uv run python benchmark/extract_results.py --dataset D3
    uv run python benchmark/extract_results.py --dataset BPI2017
    uv run python benchmark/extract_results.py --all

Output: Markdown table(s) to stdout.
"""
import argparse
import json
import sys
from pathlib import Path

# Import dataset registry
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmark.datasets import DATASETS

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "configs_v2"

METHOD_ORDER = [
    "M1a", "M1b", "M1c", "M1d", "M1e", "M1f", "M1g",
    "M2", "M3", "M4", "M5", "M6", "M7", "R1", "R2", "R3",
]
MINER_ORDER = [
    "Trace_Filtered", "Alpha", "Alpha+", "Heuristics", "Heuristics_Strict",
    "Inductive_Strict", "Inductive_Infrequent", "Flower",
]

# For each method, the key under results{} to extract.
# Methods not listed (e.g. M8) are always filled with "-".
KEY_MAP = {
    "M1a": "mean", "M1b": "mean", "M1c": "mean", "M1d": "mean",
    "M1e": "mean", "M1f": "mean", "M1g": "mean",
    "M2": "score",
    "M3": "entropic_relevance_raw",
    "M4": "gen_score",
    "M5": "mean",           # fallback to "score" in extraction logic
    "M6": "gen_score",
    "M7": "gen_score",
    "R1": "mean", "R2": "mean", "R3": "mean",
}


def resolve_dataset(dataset_arg):
    """Return (name, ds_key_or_None) for the given CLI argument.

    Accepts:
      - D-key like 'D3'  -> look up DATASETS dict
      - Name like 'BPI2017' -> find matching DATASETS entry by 'name' field
    """
    # Try as a D-key first
    if dataset_arg.upper().startswith("D") and dataset_arg.upper() in DATASETS:
        ds_key = dataset_arg.upper()
        info = DATASETS[ds_key]
        return info["name"], ds_key

    # Try matching by name (case-insensitive)
    for ds_key, info in DATASETS.items():
        if info["name"].lower() == dataset_arg.lower():
            return info["name"], ds_key

    raise SystemExit(
        f"Unknown dataset '{dataset_arg}'. "
        f"Use a D-key (D1-D21) or a dataset name "
        f"({', '.join(v['name'] for v in DATASETS.values())})."
    )


def extract_dataset(name, label=None):
    """Extract results for one dataset and print a Markdown table.

    Parameters
    ----------
    name : str
        Dataset 'name' field used for file matching (e.g. 'BPI2017').
    label : str or None
        Optional section heading (e.g. 'D3 (BPI2017)'). If None, uses *name*.
    """
    heading = label or name
    files = sorted(RESULTS_DIR.glob(f"{name}__*.json"))

    if not files:
        print(f"### {heading}")
        print()
        print("_No results found._")
        print()
        return

    # Build a lookup: (miner, method) -> formatted value
    data = {}
    for fpath in files:
        stem = fpath.stem
        parts = stem.split("__")
        if len(parts) != 3:
            continue
        _, miner, method = parts
        with open(fpath) as f:
            d = json.load(f)
        key = KEY_MAP.get(method)
        if key is None:
            # Unregistered method -> dash
            continue
        try:
            val = d["results"][key]
        except KeyError:
            # M5 fallback: some configs use "score" instead of "mean"
            if method == "M5" and key == "mean":
                try:
                    val = d["results"]["score"]
                except KeyError:
                    val = "-"
            else:
                val = "-"
        if isinstance(val, float):
            val = round(val, 4)
        data[(miner, method)] = val

    print(f"### {heading}")
    print()

    header = "| Miner | " + " | ".join(METHOD_ORDER) + " |"
    sep = "|---|" + "|".join(["---"] * len(METHOD_ORDER)) + "|"
    print(header)
    print(sep)
    for miner in MINER_ORDER:
        cells = [miner]
        for method in METHOD_ORDER:
            v = data.get((miner, method), "-")
            cells.append(str(v))
        print("| " + " | ".join(cells) + " |")
    print()


def main():
    parser = argparse.ArgumentParser(description="Extract benchmark results into Markdown tables.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", type=str, help="Dataset key (e.g. D3) or name (e.g. BPI2017)")
    group.add_argument("--all", action="store_true", help="Extract results for every dataset with results")
    args = parser.parse_args()

    if args.all:
        # Collect all dataset names that actually have result files
        seen = set()
        for fpath in RESULTS_DIR.glob("*.json"):
            name = fpath.stem.split("__")[0]
            seen.add(name)
        # Print in D-key order for consistency
        printed = 0
        for ds_key in sorted(DATASETS.keys()):
            name = DATASETS[ds_key]["name"]
            if name in seen:
                extract_dataset(name, label=f"{ds_key} ({name})")
                printed += 1
        if printed == 0:
            print("_No results found for any dataset._")
    else:
        name, ds_key = resolve_dataset(args.dataset)
        label = f"{ds_key} ({name})" if ds_key else name
        extract_dataset(name, label=label)


if __name__ == "__main__":
    main()
