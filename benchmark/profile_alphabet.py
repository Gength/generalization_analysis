"""Per-log alphabet size (distinct event-level activities) for the 21-log catalog,
cached to results/alphabet.json. Streaming XES parse (no pm4py): collect the
concept:name values that occur inside <event> elements. Fast and memory-flat, so
it runs on the multi-million-event logs (BPI2017/2018/2019) without a full parse.

Usage:
  python benchmark/profile_alphabet.py            # all 21 logs
  python benchmark/profile_alphabet.py D1 D2       # a subset (validation)
"""
import os
import sys
import json
import gzip
import time
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import benchmark.datasets as D

OUT = "benchmark/results/alphabet.json"


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def alphabet_size(path):
    opener = gzip.open if path.endswith(".gz") else open
    acts = set()
    in_event = False
    with opener(path, "rb") as f:
        for ev, el in ET.iterparse(f, events=("start", "end")):
            t = _local(el.tag)
            if ev == "start":
                if t == "event":
                    in_event = True
            else:
                if t == "event":
                    in_event = False
                    el.clear()
                elif t == "string" and in_event and el.attrib.get("key") == "concept:name":
                    v = el.attrib.get("value")
                    if v is not None:
                        acts.add(v)
    return len(acts)


def main():
    which = sys.argv[1:] or [f"D{i}" for i in range(1, 22)]
    out = {}
    if os.path.exists(OUT):
        out = json.load(open(OUT, encoding="utf-8"))
    for k in which:
        entry = D.DATASETS.get(k)
        if not entry:
            print(f"{k}: not in registry, skipped")
            continue
        path = entry.get("log_path", "")
        if not path or not os.path.exists(path):
            print(f"{k}: log missing ({path})")
            continue
        t0 = time.time()
        n = alphabet_size(path)
        out[k] = {"name": entry["name"], "n_activities": n}
        print(f"{k:4} {entry['name'][:26]:26} |A|={n:5}  ({time.time()-t0:.1f}s)")
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"written -> {OUT} ({len(out)} logs)")


if __name__ == "__main__":
    main()
