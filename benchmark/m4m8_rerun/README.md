# M4 / M8 honest reruns

Reruns of the two "infeasible" baselines under the 1-hour protocol budget, D1-D5,
with per-cell instrumentation. Replaces the earlier placeholder sentinels and backs
the Sect. 6.2 + appendix (`tab:external`) M4/M8 claims in `report/main_v4.tex`.

## What

- **M4** = anti-alignment generalization (van Dongen, Carmona, Chatain 2016), the real
  `org.processmining.antialignments` implementation bundled in Reissner's
  AutomataConformance repo (not the lost custom `m4bridge`).
- **M8** = pattern-based generalization (Reissner, Armas-Cervantes, La Rosa 2022),
  `au.unimelb.patternBasedGeneralization.PatternGeneralizationCommandLineTool` from the
  same repo.

## Setup (Linux with a display)

1. `git clone https://github.com/reissnda/AutomataConformance ~/m8_attempt`
   (ships precompiled classes under `out/production/`; no build needed).
2. Java 8 (`~/jdk8`), a display (`DISPLAY=:0` via WSLg, or `Xvfb`), and 64-bit lpsolve
   for the anti-alignment ILP: from `lp_solve_5.5.2.11_dev_ux64.tar.gz` +
   `lp_solve_5.5.2.11_java.zip`, place `liblpsolve55.so` and `lib/ux64/liblpsolve55j.so`
   in `~/m8_attempt/lp/`. The Java driver `Libraries/lpsolve55j.jar` ships with the repo.
3. `M4Progress.java` (here) is the instrumented driver: it runs the real
   `HeuristicAntiAlignmentAlgorithm` with timestamped stage logging and a live per-trace
   progress percentage (through a logging ProM `Progress`), so an OS-level 1h kill shows
   exactly where a cell stopped. Compile it into the classpath:
   `~/jdk8/bin/javac -cp "out/production/AutomataConformance:Libraries/*" -d out/production/AutomataConformance M4Progress.java`

## Run scripts

- `run_m4_d1.sh` / `run_d1.sh` -- D1 M4 / M8, 8 cells in parallel, `timeout 3660` per cell.
- `run_all_d2d5.sh` -- D2-D5, both methods, per-dataset heap and concurrency (D4 gets
  36 GB / 2-way because its log is ~2 GB unzipped), 1h cap.
- `run_m8_bigheap.sh` -- re-run of D1's three M8 cells that OOM'd at 4 GB, now at `-Xmx16g`
  (they time out rather than OOM, confirming 0/8 within budget).
  Each cell writes `out.txt` and appends `___EXIT=<code> WALL=<seconds>`.

## Consolidation

`python3 consolidate.py` (run on the box holding the raw output) reduces every
`out.txt` to `{outcome, value?, progress_pct?, error?, wall_s}` and writes
`m4m8_rerun.json` (copied to `benchmark/results/`). Re-run to refresh after D5 finishes.

## Results (D1-D4 final; D5 finishing at time of writing)

- **M4**: 3/8 on D1 (Alpha, Heuristics-strict, and the trace pole at 0.0), 3/8 on D2,
  0 on D3/D4 (does not finish the alignment step on the 2.5M-event D4 log). The flower
  never returns; Inductive-infrequent aborts on an internal ~2^16 ILP index overflow.
- **M8**: 0/8 on D1, 3/8 on D2 (Heuristics 0.99, Heuristics-strict 1.00, trace 0.90 =
  wrong pole), 0 on D3/D4.
- Neither completes all six real miners on any log, so neither can be placed on the
  four-criteria agreement.

The AutomataConformance repo and the Entropia jar (used for the Exp5 proper-M3 SDFA, see
`benchmark/pn_to_sdfa.py`) are external and not vendored here.
