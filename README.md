# ShadowGen: Quantifying Process Model Generalization

Generalization, the ability of a discovered process model to accept future, valid
behavior absent from the recorded event log, is the least validated of the four
process model quality dimensions (fitness, precision, generalization, simplicity).

This repository contains:

- **ShadowGen**, a generative N-gram metric: it learns a variable-order Markov
  model of the log (Katz backoff, Good-Turing novelty), generates a synthetic
  *shadow log* of plausible-but-unseen traces, and scores a model by how well it
  replays them. Median cost: 3 seconds per model.
- **A cross-paradigm benchmark** that validates ShadowGen and eight published
  generalization metrics (M2-M9) against variant-based hold-out ground truth on
  five real-life logs. ShadowGen tracks the ground truth on every log (Pearson
  0.986-0.999); the widely used PM4Py metric is anti-correlated on four of five.

## Paper

Christian Daniel Krengel, Tianhao Geng, Gabriel Marques Tavares, Daniel Schuster.
*Quantifying Process Model Generalization: A Hold-out Validation of Eight
Paradigms and a Generative N-gram Metric.* International Conference on Process
Mining (ICPM), 2027.

The paper is the reference description of the metric and the benchmark. The
report in [`report/`](report/) (LNCS format, July 2026) predates the paper and
is superseded by it where the two differ.

> Naming: the Python package keeps the project's legacy name `HybridGen`; the
> metric itself is called ShadowGen throughout the paper. They are the same thing.

## Requirements

- Python 3.12+ with [uv](https://docs.astral.sh/uv/) (`uv sync` creates the venv)
- **Git LFS** (the event logs in `data/` are LFS objects; without
  `git lfs install` before cloning you get pointer files, not logs)

## Quick start

```bash
git lfs install
git clone https://github.com/Gength/generalization_analysis.git && cd generalization_analysis
uv sync
```

Score a model in three lines (discovers a model, then scores it):

```python
import pm4py
from shadowgen import gen_shadow

log = pm4py.convert_to_event_log(pm4py.read_xes("data/Sepsis Cases - Event Log_1_all/Sepsis Cases - Event Log.xes.gz"))
net, im, fm = pm4py.discover_petri_net_inductive(log)
print(gen_shadow(log, net, im, fm))   # graded generalization score in [0, 1], ~3 s
```

Or from the command line, if you already have a Petri net:

```bash
uv run python shadowgen.py LOG.xes MODEL.pnml            # single draw (the default)
uv run python shadowgen.py LOG.xes MODEL.pnml --iterations 5 --details   # adds an error bar
```

The shipped configuration (one draw, N=6, tau=5, MLE weighting, seed 42) is the
exact configuration validated in the paper; every parameter is overridable.

## Reproducing the benchmark

[`benchmark/README.md`](benchmark/README.md) documents the harness: one runner
per method, `shell/run_all.sh` as the full pipeline, one sidecar JSON per
(log, miner, method) cell.

- `benchmark/results/configs/` is the committed source of truth: 800 result
  files with exact parameters, raw scores, and runtimes. Every number in the
  paper regenerates from these.
- Figures regenerate via `benchmark/make_figures.py` (and `make_*_figure.py`
  for the supplementary validations).
- `benchmark/results/NEW_EXPERIMENTS.md` documents three independent
  validations beyond the main matrix: a temporal train/future split, a
  synthetic known-system study, and bootstrap confidence intervals.
- Models are re-discovered on first run. Discovery is seeded. Token replay on
  nets with duplicate labels or silent transitions depends on iteration order,
  which moves the trace model by at most 0.024 and Alpha+ by at most 0.006
  between processes; every other cell reproduces exactly.

## Data

The five logs of the paper, as published on 4TU.ResearchData:

| Paper | Log | DOI |
|-------|-----|-----|
| L1 | Sepsis Cases | https://doi.org/10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460 |
| L2 | BPI Challenge 2013, incidents | https://doi.org/10.4121/uuid:500573e6-accc-4b0c-9576-aa5468b10cee |
| L3 | BPI Challenge 2017 | https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b |
| L4 | BPI Challenge 2018 | https://doi.org/10.4121/uuid:3301445f-95e8-4ff0-98a4-901f1f204972 |
| L5 | BPI Challenge 2019 | https://doi.org/10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1 |

`data/` ships these and the 16 further logs of the catalog through Git LFS. If an
LFS download is refused, fetch the logs from the DOIs and place them at the paths
listed in `benchmark/datasets.py`.

## Repository layout

```
shadowgen.py            # the released metric: CLI + gen_shadow() API
HybridGen/              # metric package (frozen, versioned algorithm modules)
benchmark/              # harness, per-method bridges, miners, results
  results/configs/      # provenance: one JSON per benchmark cell
data/                   # event logs (Git LFS; L1-L5 + the 21-log catalog)
report/                 # earlier LNCS report (July 2026), superseded by the paper
presentation/           # slide deck of the study and its figures
archive/                # earlier exploratory work, kept for the record
Method_GenShadow.md     # metric specification
BenchmarkDesign.md      # benchmark methodology
BenchmarkGuide.md       # operations guide
```

## Algorithm versions

`HybridGen/algorithm/` keeps every historical version as a frozen module
(v1 through v2.6), so all numbers in the paper regenerate from the exact code that
produced them. The released metric is v2.6 with MLE weighting, reported as M1
(ShadowGen) in the paper; `shadowgen.py` wraps it behind one function.

## License and citation

The code is released under the MIT License (`LICENSE`). The event logs in
`data/` are redistributed from 4TU.ResearchData and remain under the terms of
their original records. To cite this work, use the paper above; `CITATION.cff`
carries the same entry.
