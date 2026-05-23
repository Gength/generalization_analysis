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
add Gen_struct_v2 from `StructMetricAnalysis.md`.