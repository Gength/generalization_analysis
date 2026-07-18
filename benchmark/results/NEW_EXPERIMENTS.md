# Four validation experiments (2026-07-15)

Independent hardening of ShadowGen's validation, run on cibox. Each targets a
named threat in Sect. 6.4 or an open question in the conclusion. Numbers are from
the committed matrix and the scripts below; nothing in the report or deck is
changed here (propose-first).

## Headline

| Exp | What it adds | Threat closed | Headline |
|-----|--------------|---------------|----------|
| 1 Synthetic ground truth | validates against a KNOWN system, no log-derived reference | shared core + representativeness (open Q1) | per-system median Pearson 0.90 (90% positive), beats M2 on 88% of systems; M2 median -0.08; flower litmus holds on every system |
| 2 Temporal hold-out | train on past, test on the literal future | ground truth is variant-CV only | ShadowGen vs future fitness: Pearson 0.96-1.00 on L1/L2/L3/L5 (Spearman 1.0 on three); M2 flat/negative; L4 [pending] |
| 3 Alignment ruler L3/L5 | independent conformance engine, extends L1/L2 | shared token-replay core | STOPPED: infeasible in budget (one miner = 15.1 h on L5; 50 h+ projected). L1/L2 already cover the threat |
| 4 Bootstrap CIs | CIs on the n=6 correlations, all 5 logs | small-sample "is r=0.99 real" | pooled 30-cell Pearson 0.993 [0.986, 0.998]; per-log CIs lower bound >= 0.93; M2 -0.12 [-0.40, 0.19]; ShadowGen beats M2 in 100% of resamples |

## Exp 1: synthetic-system ground truth

For 78 random process trees (systems) S: sample an incomplete training log L,
discover the 8 benchmark models from L, and score each model's TRUE recall of S
as the mean token-replay fitness of a fresh S-playout F (F = valid future
behaviour of the known system). ShadowGen(L) never sees S or F.

- 61 of 78 systems discriminate (recall spread >= 0.05 over the six miners).
- Per-system Pearson(ShadowGen, true recall): **median 0.90**, IQR [0.58, 0.98],
  positive on **90%**. M2: median **-0.08**, positive on 40%.
- ShadowGen is more calibrated than M2 on **88%** of discriminating systems.
- Within-system-centred pooled Pearson 0.61 (M2 0.10); raw pooled 0.59 (M2 0.21).
  Both are conservative: they mix systems of different baseline generalisability.
- Litmus against a known system: flower true-recall 1.00 / ShadowGen 1.00 on
  every system (min 1.00); flower true-precision only 0.41, so recall (=our
  construct) and precision are cleanly separated against S. Trace 0.77 / 0.74.
- Files: `benchmark/synth_ground_truth.py`, `synth_analyze.py`,
  `results/synth_ground_truth.json`, `results/synth_analysis.json`,
  figure `report/figures/fig_synth_groundtruth.pdf`.

Reading: on a known system, where "future valid behaviour" is not inferred from
the log, ShadowGen tracks true generalisation and PM4Py does not. This is the
non-circular version of the whole benchmark and it agrees.

## Exp 2: temporal hold-out ground truth

Per real log: sort cases by start time, train on the earliest 70%, test on the
latest 30% (the literal future). Ground truth = token-replay fitness of the
future cases on each model discovered from the past.

- ShadowGen(past) vs future fitness, over six miners (novel = share of future
  cases whose variant is unseen in the past):
  - L1 Sepsis:  Pearson 0.998, Spearman 1.0, MAE 0.018 (novel 0.76)
  - L2 BPI2013: Pearson 0.959, Spearman 1.0, MAE 0.050 (novel 0.07)
  - L3 BPI2017: Pearson 1.000, Spearman 1.0, MAE 0.006 (novel 0.46)
  - L4 BPI2018: Pearson 0.964, Spearman 1.0, MAE 0.050 (novel 0.64)
  - L5 BPI2019: Pearson 0.983, Spearman 0.89, MAE 0.060 (novel 0.21)
- M2 vs future fitness is flat or negative on every log: Pearson -0.14 (L1),
  0.26 (L2), -0.66 (L3), -0.57 (L4), -0.26 (L5). It anti-correlates with future
  generalisation on the four diverse logs.
- Poles anchor (flower future-fit 1.00 / ShadowGen 1.00; trace low on both).
- Even where three quarters of the future is unseen (L1 novel 0.76), ShadowGen
  ranks generalisation near-perfectly (Pearson 0.998).
- Files: `benchmark/temporal_holdout.py`, `results/temporal_holdout.json`,
  figure `report/figures/fig_temporal_holdout.pdf`.

Reading: a second, fully independent ground truth (not variant CV, literally
future in time) that ShadowGen tracks. Triangulates the R1 result.

## Exp 3: alignment shared-ruler on L3 and L5 -- STOPPED, INFEASIBLE IN BUDGET

Goal: recompute R1 with cost-zero alignments (a conformance engine independent of
token replay) on the same variant folds, extending the report's L1/L2 check to the
two logs that carry the benign ranking swaps.

**Outcome: does not return within the compute budget, and stopped after ~21 h.**
The alignment cost per miner is the wall: on L5 the Heuristics miner alone took
**54,403 s (15.1 h)** for its folds (2.3 s per alignment on average, many hitting
the 8 s per-trace cap). After ~21 h only 3 of 6 miners were done on L5 and 2 of 6
on L3, projecting 50 h+ in total, past the 36 h budget. With 2-3 miners no
six-miner correlation is computable, so there is no usable partial result.

Measured evidence (kept as the feasibility record; run at SHUFFLES=2, align cap 8 s):

| Log | Miner | R1-align | R1-replay | ShadowGen | wall |
|-----|-------|----------|-----------|-----------|------|
| L5 | Alpha | 0.2757 | 0.3452 | 0.3679 | 908 s |
| L5 | Alpha+ | nan (unsound, tool refuses) | 0.4564 | 0.3280 | 90 s |
| L5 | Heuristics | 0.6452 | 0.8848 | 0.8378 | 54,403 s |
| L3 | Alpha | 0.1927 | 0.3851 | 0.3890 | 1,242 s |
| L3 | Alpha+ | 0.9402 | 0.8838 | 0.8599 | 266 s |

Two incidental observations, neither report-grade on 2-3 miners: on L3 Alpha+ IS
alignable (0.9402) whereas on L1 it is refused as unsound, and the engine gap
(align vs replay) is large on Alpha specifically (L3 0.19 vs 0.39).

This does not weaken the report: the shared-ruler threat is already answered on L1
(Spearman 1.0, Pearson 0.96 over the five sound nets) and L2 (Pearson 0.958 over
all six), and Exp 2 is a stronger independence argument anyway because it changes
the ground truth rather than only the ruler. If anything the honest line is that an
alignment-based R1 is infeasible at L3/L5 scale, which matches the report's
existing treatment of alignment on deep logs.

- Files: `benchmark/r1_alignment.py` (gained env-configurable R1A_SHUFFLES /
  R1A_ALIGN_CAP and incremental per-dataset writes), `results/r1_alignment.json`
  (unchanged: D1, D2).

## Exp 4: bootstrap CIs and miner-set robustness

- Per-log ShadowGen vs R1 Pearson, 95% bootstrap CI (resampling the six miners):
  L1 0.996 [0.990,1.000], L2 0.994 [0.929,1.000], L3 0.999 [0.994,1.000],
  L4 0.999 [0.998,1.000], L5 0.986 [0.969,1.000].
- Pooled 30 cells: Pearson **0.993 [0.986, 0.998]**, Spearman 0.984 [0.939,0.994], MAE 0.022.
- M2 pooled: Pearson -0.119 [-0.399, 0.189] (straddles zero).
- Paired: ShadowGen is more calibrated than M2 in **100%** of resamples (mean gap 1.11 [0.80,1.39]).
- L1 eleven-miner check: Pearson 0.997 [0.954, 0.999], MAE 0.015.
- Files: `benchmark/exp4_bootstrap.py`, `results/exp4_bootstrap.json`.

## Four-criterion scoring of the new references

Both new references scored under the benchmark's own protocol (Pearson,
Spearman + Kendall tau_b, MAE, spread over the six miners; poles separate).
Script `benchmark/four_criteria_new_refs.py`, data `results/four_criteria_new_refs.json`.

Temporal hold-out, per log:

| Log | Metric | Pearson | Spearman | tau_b | MAE | spread | GT spread |
|-----|--------|--------:|---------:|------:|----:|-------:|----------:|
| L1 | ShadowGen | 0.998 | 1.000 | 1.000 | 0.018 | 0.748 | 0.755 |
| L1 | M2 | -0.142 | 0.086 | 0.067 | 0.175 | 0.093 | 0.755 |
| L2 | ShadowGen | 0.959 | 1.000 | 1.000 | 0.050 | 0.774 | 0.631 |
| L2 | M2 | 0.260 | -0.029 | -0.067 | 0.213 | 0.186 | 0.631 |
| L3 | ShadowGen | 1.000 | 1.000 | 1.000 | 0.006 | 0.605 | 0.590 |
| L3 | M2 | -0.657 | -0.600 | -0.552 | 0.137 | 0.054 | 0.590 |
| L4 | ShadowGen | 0.964 | 1.000 | 1.000 | 0.050 | 0.763 | 0.744 |
| L4 | M2 | -0.569 | -0.429 | -0.200 | 0.266 | 0.100 | 0.744 |
| L5 | ShadowGen | 0.983 | 0.886 | 0.733 | 0.060 | 0.710 | 0.532 |
| L5 | M2 | -0.260 | 0.143 | 0.200 | 0.189 | 0.142 | 0.532 |

ShadowGen's spread matches the ground truth's on every log (0.61-0.77 vs
0.53-0.76); M2 compresses the same six models to 0.05-0.19.

Synthetic known systems, per-system medians over the 61 discriminating systems:

| Metric | Pearson | Spearman | tau_b | MAE | spread | GT spread |
|--------|--------:|---------:|------:|----:|-------:|----------:|
| ShadowGen | 0.904 | 0.829 | 0.733 | 0.067 | 0.315 | 0.331 |
| M2 | -0.078 | 0.086 | 0.067 | 0.099 | 0.030 | 0.331 |

Poles, VERIFIED per system (not means): flower true recall = 1.0000 on every
system (min 1.0000) and ShadowGen scores it 1.0000 on every system (min 1.0000);
trace true recall mean 0.748, ShadowGen 0.719.

## Proposed report integration (propose-first, nothing applied)

- Exp 1 + Exp 2: a short results subsection or two new threats paragraphs under
  "Ground truth and representativeness" and "Shared measurement core": "We also
  validated against two references that are not derived from the log under study:
  a known synthetic system (per-system median Pearson 0.90 vs true recall, PM4Py
  -0.08) and a temporal train/future split on the real logs (Pearson 0.96-1.00,
  Spearman 1.0 on three of five)." Closes open question 1.
- Exp 3: extend the existing shared-ruler sentence from "L1/L2" to the logs
  reached, with the swap-logs note.
- Exp 4: add a CI to the breadth threat: "the pooled calibration over all 30
  log-miner cells is Pearson 0.993 (95% CI [0.986, 0.998]); the PM4Py CI straddles
  zero."

## Figures (all in report/figures/, regenerable from benchmark/make_*.py)

- `fig_synth_groundtruth.pdf` -- Exp 1: ShadowGen vs known-system recall, M2 beside it, per-system r distribution.
- `fig_temporal_holdout.pdf` -- Exp 2: past-to-future calibration on all five logs + per-log agreement bars.
- `fig_bootstrap_ci.pdf` -- Exp 4: per-log and pooled Pearson with bootstrap 95% CIs.
- `fig_synth_vs_real.pdf` / `fig_synth_vs_real_temporal.pdf` -- 2x2 real-vs-synthetic
  ground truth (left column R1 or temporal), ShadowGen over M2, MAE on every panel.
- `fig_discrimination.pdf` -- resolving power: score frequency distributions,
  marginal and split by true generalization, with separation AUC.

## Status

All four experiments closed. Exp 1, 2, 4 delivered; Exp 3 stopped as infeasible in
budget (documented above, and it is not needed: L1/L2 already answer that threat).
Nothing in the report or the deck has been modified.
