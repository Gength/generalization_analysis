# Hybrid Generative-Structural Generalization Evaluation

In order to comprehensively assess a process model's ability to handle unseen, future behavior without falling into the trap of severe overfitting (i.e. Trace Models) or underfitting (i.e. Flower Models), we introduce a hybrid evaluation framework. This approach evaluates generalization through two complementary approaches: **Generative Behavioral Analysis** (Gen_shadow) and **Structural Frequency Analysis** (Gen_struct).

## 2.1. Generative Behavioral Analysis (Gen_shadow)
Gen_shadow measures the model's flexibility by simulating potential future event logs, essentially acting as a probabilistic stress test.

Local Marking and Probability Estimation:
The algorithm relies on Local Variance (the token distribution of immediate input places). We employ the Good-Turing frequency estimation to calculate the mutation probability (P_{unseen}) of these local states.
Dynamic Trace Generation:
For predictable states (high historical frequency, low variance), the algorithm applies a low mutation rate, limiting the generation of illogical traces. However, for unpredictable states (low frequency, high variance), it explores new, potentially logically valid variations.
Replay Evaluation:
A synthetic "shadow log" is generated through this stochastic random walk. Gen_shadow is formally defined as the replay fitness of this newly generated synthetic log evaluated against the discovered model.

## 2.2. Structural Frequency Analysis (Gen_struct)
A purely generative approach risks falsely rewarding "Flower Models" that permit all possible behaviors and generating/speculating beyond the reality of the given Scenario. In order to counteract this, we introduce Gen_struct as a strict reality-based mathematical constraint.

Overfitting Penalty:
This component replays the original event log on the discovered model to analyze the usage frequency of its internal structure. If certain structural paths or transitions are visited very rarely, it is penalized. This identifies and downgrades models that overfit by constructing specific isolated branches solely to memorize rare outlier traces (i.e. Spaghetti-model)

## 2.3. The Hybrid Synthesis
The final generalization metric is computed as a weighted combination of the generative and structural scores.

Gen_Total = w*Gen_shadow + (1 - w)*Gen_struct

The parameter w $\in$ [0, 1] allows for the calibration of the evaluation focus. By adjusting w, the metric can balance the reward for probabilistic flexibility against the penalty for structural "bloat".
Alternatively, the default w value of 0.5 would be used.

# Version History
## V1
Basic Version
## V2
### Mutated Path Generation:
Introduce N-gram-based path generation to capture local dependencies in the process model, allowing for more realistic trace generation while still maintaining a balance between exploration and exploitation.
## V2.1
### Predictable Path Generation:
replace frequency counts with ln(count+1) to reduce the dominance of extremely common paths and increase variability in the generated traces, while still respecting historical likelihoods.
## V2.2
add Gen_struct_v2 from `./analysis/Structure/StructMetricAnalysis.md`.

## V2.3
### Structural Analysis Removed (Gen_struct Deprecated):
Gen_struct is removed from this version. The evaluation relies solely on the Gen_shadow component, making the weight parameter `w` in `Gen_Total = w*Gen_shadow + (1-w)*Gen_struct` effectively ignored (Gen_total = Gen_shadow). This decision was made to eliminate the confounding influence of the structural penalty on the generalization score, allowing a pure measurement of the generative shadow log's replay fitness.

### Trace Deduplication:
V2.1 could generate shadow traces that are exact copies of original event log traces (empirically ~1.9% in the Sepsis dataset with 1000 shadow traces). V2.3 adds a check in `generate_shadow_log` that compares each generated trace against the set of all original trace sequences. If a match is found, the trace is discarded and regenerated (up to a safety cap of 100 retries). This ensures the shadow log contains only genuinely synthetic traces, preventing inflated fitness scores from replicated original behavior.

### Context-Aware Termination via Katz Backoff:
In V2.1, trace termination used only the current activity name (`current_local`) to compute `P_end`, regardless of where in the trace that activity appeared. Activities like `Release A` had a flat ~59% termination probability whether they occurred at position 3 or position 30 — but in the original log they always appeared in the last 20% of traces. This caused the DFS walker to terminate prematurely when "ending activities" were generated early, systematically shortening shadow traces.
V2.3 fixes this by applying the same Katz backoff strategy used for next-activity selection to the termination decision. It now considers the last N activities (from N=max_n down to safe N=1) to compute a context-dependent `P_end(state)`. The same pre-computed N-gram termination statistics (`ngram_termination_ends` and `ngram_termination_totals`) support this backoff, ensuring termination probabilities are grounded in the historical ending behavior of the specific context, not just the isolated activity name.

## V2.5 (v25)
### Katz-Backoff Mutation Proposal:
v25 completes the Katz backoff idea introduced in V2.3, extending it from the mutation rate (how often to insert a never-seen continuation) to the mutation proposal (which activity to insert when a mutation fires). Previously, the inserted activity was drawn uniformly from the whole alphabet (`random.choice(alphabet)`), which became a binding constraint on sparse logs (e.g. ~50% of shadow traces on Sepsis carry at least one mutation vs. ~4% on BPI 2017). In v25, the new activity is drawn from the next-shorter context's continuation distribution, restricted to activities never seen after the current context — mirroring how Katz backoff redistributes unseen probability mass in language models. Injected events are now plausible-but-novel by construction.

### Transparency Counters (no behavior change):
- `duplicates_kept`: reports if the dedup retry cap (100) was ever exhausted.
- `truncated_traces`: reports walks cut by the length cap.

## V2.6 (v26)
### Acceptance Rate (`gen_accept`):
Besides mean replay fitness (which grants partial credit), v26 reports the fraction of shadow traces the model replays *perfectly*. This provides interpretable bounds: the Trace Model scores exactly 0.0, the Flower Model exactly 1.0. Reported alongside the existing fitness score — nothing is replaced.

### Data-Driven Length Cap:
Replaces the fixed 100-trace length cap with `2 × longest observed trace` (critical for datasets like Sepsis, which has traces up to length 185).

### `successor_weighting` Switch:
Two sampling modes for successor selection:
- `'log'` (default, v2.1 behavior): ln-damped sampling that up-weights rare paths, retained as the stress-test mode.
- `'mle'`: samples proportionally to raw frequencies, so the shadow log represents the expected future using the same statistics as the Good–Turing estimate.

Benchmarked against the R1 variant-based 5-fold CV ground truth, v26-mle achieves **MAE 0.024** (Sepsis) and **0.021** (BPI 2013) — roughly one-third of v24's calibration error — and resolves the ranking disagreement on BPI 2013 (rank correlation 1.000). Full results in [WhatChanged_v25_v26.md](WhatChanged_v25_v26.md).
