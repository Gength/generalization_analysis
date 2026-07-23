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

The full study is the report in [`report/`](report/): `main_v5_short.pdf` is the
submitted version; material cut for length is catalogued in
[`report/OMITTED_MATERIAL.md`](report/OMITTED_MATERIAL.md).

> Naming: the Python package keeps the project's legacy name `HybridGen`; the
> metric itself is called ShadowGen throughout the report. They are the same thing.

## Requirements

- Python 3.12+ with [uv](https://docs.astral.sh/uv/) (`uv sync` creates the venv)
- **Git LFS** (the event logs in `data/` are LFS objects; without
  `git lfs install` before cloning you get pointer files, not logs)

## Quick start

```bash
git lfs install
git clone <this repo> && cd generalization_analysis
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
exact configuration validated in the report; every parameter is overridable.

## Reproducing the benchmark

[`benchmark/README.md`](benchmark/README.md) documents the harness: one runner
per method, `shell/run_all.sh` as the full pipeline, one sidecar JSON per
(log, miner, method) cell.

- `benchmark/results/configs/` is the committed source of truth: 800 result
  files with exact parameters, raw scores, and runtimes. Every number in the
  report regenerates from these.
- Figures regenerate via `benchmark/make_figures.py` (and `make_*_figure.py`
  for the supplementary validations).
- `benchmark/results/NEW_EXPERIMENTS.md` documents three independent
  validations beyond the main matrix: a temporal train/future split, a
  synthetic known-system study, and bootstrap confidence intervals.
- Models are re-discovered on first run (discovery is seeded; the residual
  replay drift is quantified in the report's threats section).

## Repository layout

```
shadowgen.py            # the released metric: CLI + gen_shadow() API
HybridGen/              # metric package (frozen, versioned algorithm modules)
benchmark/              # harness, per-method bridges, miners, results
  results/configs/      # provenance: one JSON per benchmark cell
data/                   # event logs (Git LFS; L1-L5 + the 21-log catalog)
report/                 # LaTeX report + figures
presentation/           # final talk and defense notes
archive/                # earlier exploratory work, kept for the record
Method_GenShadow.md     # metric specification
BenchmarkDesign.md      # benchmark methodology
BenchmarkGuide.md       # operations guide
```

## Algorithm versions

`HybridGen/algorithm/` keeps every historical version as a frozen module
(v1 through v2.6), so all report numbers regenerate from the exact code that
produced them. The released metric is v2.6 with MLE weighting, reported as M1
(ShadowGen) in the report; `shadowgen.py` wraps it behind one function.
