# AVATAR full D1 run (CHARLES) - launch package

Status: PREPARED (2026-07-08), DEFERRED by user decision; user launches when ready. Expected wall time ~3 to 4 days.

## Purpose
Replace the projected Fig 8 / feasibility anchor (~3-4 days per log, over 400 CPU-hours, projected from the measured 2026-07-07 quick anchor) with one honest end-to-end measurement, and produce a properly trained D1 scorecard measured start to finish by us. The conclusion will not move; the number becomes a measurement instead of a projection. If pretraining pushes past 4 days, the sentinel verdict only strengthens.

## Where and what
- CHARLES, docker image `avatar-cpu` (TensorFlow 1.15.5, CPU), repo at `/workspace/src/AVATAR`.
- Full published config is the STOCK `avatar/training.py`: npre_epochs=100, nadv_steps=5000 (hardcoded at its lines ~104-105), job 0 (temperature beta 100), RelGAN/RSGAN, rmc_vanilla, batch 64.
- The 2026-07-07 quick anchor was the same pipeline reduced to 3 pre-epochs / 100 adv steps (log: `benchmark/results/avatar_rebuild/quick_anchor.log`, timing: `quick_anchor_timing.json`). Mirror that invocation exactly, minus the reductions: if the quick anchor ran a locally modified training.py, restore 100/5000 (stock values).

## Launch (inside the container, from /workspace/src/AVATAR)
    LOG=/workspace/src/AVATAR/full_run_d1.log
    echo "START $(date -Is)" | tee "$LOG"
    nohup python -u avatar/training.py -s sepsis -j 0 -gpu -1 -n 10000 >> "$LOG" 2>&1 &
    # on exit, append: echo "END $(date -Is) exit=$?" >> "$LOG"
    # simplest: wrap in a small script so END/exit is captured; or run under tmux.

Notes:
- `-gpu`: use the same value the quick anchor used (CPU image; the flag only sets CUDA_VISIBLE_DEVICES).
- Disk: sampling writes per-checkpoint samples (~260 s per checkpoint at anchor rate); check free space in the data dir first.
- Do NOT run on cibox (no Docker there).

## Expected timeline (from quick_anchor_timing.json)
- Adversarial phase: 5000 x 45.11 s/step = 62.6 h (rigorous, linear in steps).
- Sampling phase: ~18 h (estimate).
- Pretraining 100 epochs: unmeasured; the anchor ran only 3.

## When it lands
1. Copy `full_run_d1.log` to `benchmark/results/avatar_rebuild/` and record wall time + exit code in a `full_run_d1_timing.json` next to it.
2. report/main_v4.tex feasibility para (~line 382): change "projects the full published configuration to" to the measured number; keep the projection as the cross-check.
3. benchmark/make_figures.py: set AVATAR_ANCHOR_S to the measured seconds; regenerate fig_pareto_scale.pdf, fig_pareto.pdf, fig_runtime.pdf (repo .venv has matplotlib).
4. Fig 8 caption (~line 579): "projected CPU cost" becomes "measured CPU cost".
5. If the trained model scores D1, record the scorecard next to the June off-protocol ones; it stays off-protocol for the benchmark (budget rule) but hardens the litmus reading.

## Open observation (for the report, propose first)
The rebuild derived/used seq_len=182 on sepsis (see quick_anchor.log: "seq_len: 182, vocab_size: 21"), while main_v4.tex ~382 says AVATAR has a "published 20-event sequence cap" that would truncate D3/D4. The cap is a configuration, not an architecture limit; the honest phrasing is that the published configuration caps sequences at 20 events and raising the cap raises the (already prohibitive) cost. Proposed wording change is pending user approval.
