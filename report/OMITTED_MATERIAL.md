# Omitted material

`final_report.pdf` is the single, authoritative version of the report. A longer
draft (`main_v5`, 37 pages) existed until 2026-07-23; it was removed to keep one
canonical text, and its full source remains in git history. Earlier drafts
(`main`, `main_v2` to `main_v4`) were removed in commit `c08672a` and are also
in history. This file records what the submission does not include and where to
find it.

## In the long draft, cut from the submission

- **Raw cross-paradigm score table on L1** (appendix section "Raw cross-paradigm
  scores on L1", `tab:external`): every method's raw score per miner, side by
  side. The submission keeps the agreement statistics (Table 4) and the
  calibration scatter (Fig. 7); the raw values are recoverable from
  `benchmark/results/configs/`.
- **The bootstrap contamination study as a table** ("The bootstrap contamination
  reading", `tab:m6`): the four scoring readings of the same `bsgen` sample. The
  submission carries all its numbers in Sect. 6.2.3 prose.
- **Katz-mutation backoff diagram** (`fig:mutation`): a step-by-step picture of
  the mutation proposal; the running example (Fig. 4) covers the same mechanism.
- **Longer prose passages**: per-method feasibility forensics, baseline
  integration detail, the complexity derivation, and the Related Work
  paradigm-by-paradigm walk-through (submission: Table 1 plus the grouping
  discussion). The verdicts and key evidence survive in the submission body.

## Generated for this project but never in the report

- `figures/fig_discrimination.pdf`: score histograms with separation AUC
  (ShadowGen 0.85 vs PM4Py 0.59 on the synthetic systems; 1.00 vs 0.62 on the
  real logs). The resolving-power view of the report's Spread criterion.
- `figures/fig_temporal_holdout.pdf`, `figures/fig_synth_groundtruth.pdf`,
  `figures/fig_synth_vs_real_temporal.pdf`, `figures/fig_bootstrap_ci.pdf`:
  the three independent validations of Sect. 6.4 (temporal split, synthetic
  known-system study, bootstrap CIs) as figures; the submission reports them in
  prose only.
- A raw per-(log, miner) score table for ShadowGen vs the R1 ground truth:
  never typeset; recoverable from `benchmark/results/configs/`.
- Eleven-configuration agreement checks beyond L1 (`benchmark/exp4_miners.py`
  on L2 to L5) and full leave-one-variant-out R2 on L2: run 2026-07-23/24,
  results under `benchmark/results/`.

## Where to find things

- Long-version text: git history (removal commit of 2026-07-23).
- Every number in the report: the sidecar JSONs in `benchmark/results/configs/`.
- Figures regenerate from `benchmark/make_figures.py` and
  `benchmark/make_*_figure*.py`.
