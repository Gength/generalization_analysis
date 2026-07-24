# Report (LNCS)

M.Sc. report: *Quantifying Process Model Generalization: A Generative N-gram
Metric and a Cross-Paradigm Benchmark*.

## Files

- `final_report.tex` / `.pdf`: the submitted and only maintained version (the
  appendix holds only data exhibits); the long draft (`main_v5`) was removed on
  2026-07-23 and remains in git history
- `references.bib`: BibTeX references (style `splncs04`)
- `figures/`: all figures as vector PDFs; they regenerate from
  `benchmark/make_figures.py` and `benchmark/make_*_figure.py`

Earlier drafts (`main.tex`, `main_v2` to `main_v4`) are removed; they remain in
the git history.

## Structure (the supervisor's 7-section layout)

1. Introduction
2. Background (logs, models, token replay's two readings, hold-out, N-grams)
3. Related Work (paradigm table, groupings, the gap)
4. Method (construct + litmus, framework, generator, algorithm, example, scope)
5. Implementation (architecture, design decisions, baselines, deviations)
6. Evaluation (setup; feasibility, cross-paradigm agreement, bootstrap study,
   scale + ablation, acceptance, generator premise, runtime; discussion; threats)
7. Conclusion

## Build

```bash
cd report
pdflatex final_report && bibtex final_report && pdflatex final_report && pdflatex final_report
```

Uses `algorithm`/`algpseudocode`, `tikz`; packages auto-install on first MiKTeX
compile (add `-enable-installer` if prompted).

## Provenance

Every number derives from `benchmark/results/configs/` (seed 42, one sidecar
JSON per cell); the supplementary validations are documented in
`benchmark/results/NEW_EXPERIMENTS.md`. Re-running cells means regenerating the
tables.
