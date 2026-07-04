# Runbook: D3–D5 gap fixes (2026-07-02)

Closes the four gaps in the D3–D5 results: missing `gen_accept` (runner bug),
missing construct-faithful M6, missing R1-accept ground truth at scale, and
the SpeciAL/BPI2019 hole. Run everything **from the repo root on the benchmark
runner** (needs `src/` vendor code and Linux; `run_m1_family.py` uses fork).

**When invoking python directly (not via the shell scripts), prepend
`PYTHONHASHSEED=0`** — all `benchmark/shell/*.sh` now export it. Discovery
(pm4py Heuristics in particular) is hash-randomization-sensitive; the in-code
`setdefault` comes too late for the main process.

## New infrastructure (verified end-to-end on a 32-core test box)

- **Model cache**: discovered PNMLs persist under `benchmark/models/<dataset_key>/`.
  Discovery now runs once per dataset EVER; every job (M1 Phase 1 and all
  `prepare_workdir` bridges) copies from the cache, so all methods score
  byte-identical models and discovery time is excluded from method runtimes by
  construction. First run seeds the cache; delete the folder to force
  rediscovery. Disable per run with `--no-model-cache` (M1 only).
- **Per-cell metric budget (protocol: 3600 s, references exempt)**:
  `--cell-timeout <seconds>` on `job_m1.py`, `job_m6.py`, `job_m6_adapted.py`,
  `job_m7.py` (and the bridges' own CLIs), **default 3600** (1 h). Counts
  METRIC time only; model discovery is excluded (cached and shared). A
  timed-out cell is written as a `-1` sentinel with note "exceeds budget".
  `--cell-timeout 0` = unlimited. **R1/R2 (and r1_accept) have no timeout by
  design: they are the ground truth and must complete.** Sanity: the worst
  known method cell (M1g on D4 Inductive-strict, 39 min) fits inside the
  budget; AVATAR training does not, consistent with its exclusion from D3 up.
- **`-bgen` sentinel fix**: failed/timed-out Entropia cells now write
  `gen_score: -1` (previously `0.0`, which read as a valid low score — this is
  why the D4 timeouts looked like zeros).

## Reproducibility caveat (measured, not hypothetical)

pm4py token replay is order-sensitive on nets with duplicate labels or silent
transitions: with identical models and seeds, **Trace_Filtered varies by ~±0.02
and Heuristics by ~±0.005 across processes** (element iteration order is not
pinned by any seed). Alpha, Alpha+, both Inductive nets, Flower, and the
gen_accept values reproduce exactly (verified: Ind-S D1 gen_accept 0.7444
matches the published value bit-for-bit). Consequence for re-runs: expect the
Heuristics and Trace columns to move within those bands relative to published
numbers; everything else should match to 4 decimals.

## What changed in the code

| File | Change |
|---|---|
| `benchmark/run_m1_family.py` | Bug fix: configs again include `gen_accept`, `gen_accept_std`, regular/mutated split, `duplicates_kept`, `truncated_traces`, `max_trace_length_used` (whenever the algorithm version reports them). Output format verified identical to the old D1/D2 configs. |
| `benchmark/bridges/run_m6_adapted.py` (new) | The construct-faithful M6 (bsgen breeding + PM4Py token replay), ported from `archive/Tianhao/benchmark/bridges/run_m6.py` into the job architecture. Same parameters and scoring as the D1 run (seed 42, 10 replicates, 10 generations, k=2, p=1.0, n=200). One deliberate change: RNG re-seeded per miner (cells order-independent). Needs `src/bsgen/bsgen_eval.py`. |
| `benchmark/job_m6_adapted.py` + `benchmark/shell/m6_adapted.sh` (new) | Job wrapper + SLURM script, same pattern as the other methods. |
| `benchmark/bridges/run_m7.py` | Sentinel pre-write: every target miner gets a `-1` config before evaluation starts, overwritten on completion. A killed job now leaves "-1 did not complete" cells instead of holes (the BPI2019 problem). |
| `benchmark/r1_accept.py` | Works for all datasets (D1–D21 via `datasets.py`), writes provenance configs (`<Name>__<miner>__R1accept.json`), and skips the M1f/M1g comparison with a clear message when `gen_accept` is missing. Computation unchanged (verified: reproduces the D1 numbers). |

## Output policy: benchmark/results/configs_second_try/

**Nothing in `configs/` or `configs_v2/` is overwritten.** All new and fixed
results go to `benchmark/results/configs_second_try/`, which was seeded
(`benchmark/seed_second_try.py`) as the combined, complete dataset: every
currently-valid config copied in, the Entropia `-bgen` M6 renamed to method id
`M6bgen` (the id `M6` now means the token-replay adaptation, matching the
report), the 8 broken D4 `-bgen` files excluded pending an honest re-run, and
explicit `-1` sentinels generated for every silent hole (M4/M8 beyond D1, M5
beyond D2). `benchmark/audit_configs.py` reports the matrix state; the goal is
zero MISS and zero BAD.

Pass `--output benchmark/results/configs_second_try` to every job below
(r1_accept.py already defaults to it).

## Run order

### 1. Re-run M1 v2.5/v2.6 on D4 and D5 (restores acceptance + counters)

```bash
sbatch benchmark/shell/m1.sh --dataset D5 --output benchmark/results/configs_second_try --methods M1e M1f M1g --workers 16
sbatch benchmark/shell/m1.sh --dataset D4 --output benchmark/results/configs_second_try --methods M1e M1f M1g --workers 16
```

- Replaces the seeded copies inside configs_second_try only. Expected: D5 ≈
  10-20 min; D4 ≈ 1-1.5 h (three ~39 min Inductive-strict cells run in parallel).
- Reproducibility: Heuristics cells ±~0.005, Trace ±~0.02 vs published (replay
  order effect); everything else and gen_accept exact.

### 2. M6 adaptation at scale + D2 + the missing D1 Trace cell (partner box)

```bash
sbatch benchmark/shell/m6_adapted.sh --dataset D1 --output benchmark/results/configs_second_try   # only Trace_Filtered missing; subset via MINERS array
sbatch benchmark/shell/m6_adapted.sh --dataset D2 --output benchmark/results/configs_second_try
sbatch benchmark/shell/m6_adapted.sh --dataset D3 --output benchmark/results/configs_second_try
sbatch benchmark/shell/m6_adapted.sh --dataset D5 --output benchmark/results/configs_second_try
sbatch benchmark/shell/m6_adapted.sh --dataset D4 --output benchmark/results/configs_second_try   # 1 h/cell budget applies
```

- Needs `src/bsgen/bsgen_eval.py`. D2 ≈ 10 min; D3/D5 ≈ 20-40 min each; D4 real
  scores or earned sentinels.

### 3. Re-run Entropia -bgen on D4 under the real budget (partner box)

```bash
sbatch benchmark/shell/m6.sh --dataset D4 --output benchmark/results/configs_second_try
```

- The old 600 s timeouts do NOT justify "exceeds the 1 h budget"; this run does.
  NOTE: writes `<ds>__<miner>__M6.json` names — after it finishes, rename those
  files to `M6bgen` (rerun `seed_second_try.py` logic or by hand) so they do not
  collide with the adapted M6. Same honesty item applies to M8's 120 s timeout
  on D1 if you ever want that claim under the 1 h protocol.

### 4. R1-accept ground truth, all remaining logs (parallel, budget-exempt)

```bash
PYTHONHASHSEED=0 python benchmark/r1_accept.py D2 --workers 16
PYTHONHASHSEED=0 python benchmark/r1_accept.py D3 --workers 32
PYTHONHASHSEED=0 python benchmark/r1_accept.py D5 --workers 32
PYTHONHASHSEED=0 python benchmark/r1_accept.py D4 --workers 32
```

- Fold-level parallel (120 tasks); verified to reproduce the serial numbers
  exactly on D1. D2 ≈ minutes; D3/D5 ≈ 5-15 min; D4 ≈ 30-60 min (was ~9 h serial).
- Run after step 1 so the M1f/M1g comparison tail has gen_accept.

### 5. Retry SpeciAL on BPI2019, missing miners only (partner box)

```bash
sbatch benchmark/shell/m7.sh --dataset D5 --output benchmark/results/configs_second_try
```

- Edit the MINERS array to the 7 missing (keep Trace_Filtered's existing score).
  Sentinel pre-write + 1 h/cell budget: a crash or timeout leaves earned `-1`s.

### 6. Afterwards

```bash
python benchmark/audit_configs.py                     # goal: 0 BAD, 0 MISS
uv run python benchmark/make_figures.py               # once tooling points at configs_second_try
```

Remaining partner-box optional items: R2's D4 Inductive-strict cell (reference,
no budget, ~10 h), AVATAR's 3rd run.
