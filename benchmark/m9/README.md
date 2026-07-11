# M9 — Negative-Event Generalization

Benchmarked result for the weighted artificial negative-event generalization metric
(vanden Broucke, De Weerdt, Vanthienen, Baesens; TKDE 2014), as shipped in CoBeFra.
Reported in the paper's feasibility section (Sect. 6.2); construct-drift, so it stays
out of the four-criteria agreement tables (like M2 PM4Py and M3 entropic relevance).

## Metric
`generalization = AG / (AG + DG)` — allowed vs disallowed generalizations accumulated by
`org.processmining.plugins.neconformance.metrics.impl.BehavioralWeightedGeneralizationMetric`
over the log with weighted artificial negative events induced per position.

## Configuration (CoBeFra shipped default)
`replayer=0` (token replay), `inducer=0` (log-tree weighted), `useWeighted=true`,
`useBothRatios=false`, `useCutOff=false`, `negWindow=-1`, `genWindow=-1`,
`unmapped{Recall,Precision,Generalization}=true`, single-threaded.
(The authors' alignment-based config — `replayer=1`, Arya alignments, windows 20 — hangs
into disk-swap on our real-life logs, so we use the shipped token-replay default.)

## Files
- `M9Runner.java` — headless driver: loads a PNML net + XES log, builds the standard
  `PetrinetLogMapper`, calls the negative-event generalization metric, prints `gen` + runtime.
- `m9_batch.py` — runs the full 5-log x 8-config matrix, up to 5 runs per cell for
  variance within a 1-hour budget, sentinels on timeout/OOM, writes
  `../results/m9_negative_events.jsonl` (resumable, incremental).

## Reproduction
Built on the `neconformance` classes bundled in the AutomataConformance project
(Java 8, `DISPLAY=:0` for ProM init). Classpath:
`out/production/AutomataConformance:Libraries/*`. Logs are the canonical
`benchmark/datasets.py` paths; models are `benchmark/models/D*/*.pnml`.

## Result (`../results/m9_negative_events.jsonl`, 40 cells)
Does not track held-out generalization on any log: D1 anti-correlated with R1
(Pearson -0.55, MAE 0.60); D2 and D5 degenerate (every model, including the trace pole,
scores 1.0); D3 only 2 of 6 real miners return within the hour, D4 none. Passes the flower
litmus on the small logs (1.0) but mis-anchors the trace pole (0.48 on D1, 1.0 on D2/D5).
Non-deterministic on silent-transition nets (D1 Inductive-infrequent, std 0.07).
