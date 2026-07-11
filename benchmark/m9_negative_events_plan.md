# M9: Negative Event Generalization (vanden Broucke et al.) - port plan

Status: PREPARED (2026-07-08), not started. Estimated effort: about half a day plus validation runs.

## What it is
Weighted artificial negative events generalization (vanden Broucke, De Weerdt, Vanthienen, Baesens; TKDE 2014), shipped as "Negative Event Generalization" in CoBeFra. The report currently cites the paradigm in related work and positions it out ("ProM-era and unmaintained"); porting it makes that an empirical entry instead. Construct: replay plus induced negative events, fitness/precision-adjacent. Like M2/M3 it sits OUT of the agreement tables; it enters the flower litmus and the narrative.

When M9 lands, the report changes: related-work positioning sentence (main_v4.tex ~127), and "seven published baselines" becomes eight (lines ~67, ~82, ~322).

## The formula (from decompiled source, exact)
`BehavioralWeightedGeneralizationMetric.getCalculatedValue()`:

    gen = AG / (AG + DG),  and 1.0 if AG = DG = 0

AG = allowedGeneralizations, DG = disallowedGeneralizations, accumulated per replay position by `AbstractBehavioralPetrinetMetric` over the log with negative events induced per position. Exact accumulation semantics: read `AbstractBehavioralPetrinetMetric.java` (217 lines) during the port.

## Sources (all local)
- Tool: `../cobefra` (clone of github.com/Macuyiko/cobefra, 2026-07-08). Thin wrapper class: `src/be/kuleuven/econ/cbf/metrics/generalization/NegativeEventGeneralizationMetric.java` delegates to the ProM package below with metric key "generalization".
- Algorithm jar: `../cobefra/kulib/neconformance.jar` (plus `kutoolbox.jar`).
- Decompiled source: `../cobefra/neconformance-src/` (CFR 0.152 at `../cobefra/cfr.jar`, zero .java shipped in jar, no public source repo found).
- Port surface (non-UI core, ~650 lines of decompiled Java):
  - `metrics/impl/BehavioralWeightedGeneralizationMetric.java` (27)
  - `metrics/impl/AbstractBehavioralPetrinetMetric.java` (217)
  - `negativeevents/AbstractNegativeEventInducer.java` (89)
  - `negativeevents/impl/LogBagWeightedNegativeEventInducer.java` (126, default inducer)
  - `models/impl/PetrinetReplayModel.java` (180) plus `models/ProcessReplayModel` interface
  - trees/* (suffix tree) only needed for the LogTree inducer, NOT for the default

## CoBeFra defaults (from NegativeEventConformanceMetric ctor; use these)
replayer=0, inducer=0 (log-bag weighted), useWeighted=true, useBothRatios=false, useCutOff=false, negWindow=-1, genWindow=-1, unmappedRecall/Precision/Generalization=true, multiThreaded=false.

## Recipe (mirror the M3 pattern)
1. Reference values: headless Java runner calling `org.processmining.plugins.neconformance.plugins.PetrinetEvaluatorPlugin.getMetricValue(...)` with the defaults above. Local Java 8 verified (1.8.0_491). Classpath: `../cobefra/kulib/*;../cobefra/lib/*;../cobefra/promlib/*;../cobefra/packagelib/*`. Run on the 8 D1 nets.
2. Python reimplementation `benchmark/m9_negative_events.py` from the decompiled source (pm4py for XES + PNML io and replay primitives; the negative-event induction and AG/DG accounting are ours).
3. Validate: 3-decimal agreement against the Java tool on every D1 cell (same bar as M3).
4. Run the full 40-cell matrix under the 1 h per-cell budget; sentinel with evidence where exceeded.

## Inputs
- XES logs: in-repo under `./data/` (all five).
- Canonical discovered models (PNML): NOT in this repo; they live on cibox at `~/genbench/benchmark/models/<D*>/`. scp them down first (40 small files). Do not rediscover locally (version/seed drift); do not import cibox repo modules (that checkout is behind).

## Risks
- Decompiled code is exact but unannotated; the TKDE paper is the semantic reference where CFR output is opaque.
- ProM-era XES/PNML parsing quirks in the Java reference runner (label mapping via `PetrinetLogMapper`); if the headless classpath fights back, fall back to running the reference inside CoBeFra's own harness once per cell.
