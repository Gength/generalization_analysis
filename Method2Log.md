# 2. Hybrid Generative-Structural Generalization Evaluation

To comprehensively assess a process model's ability to handle unseen, future behavior without falling into the trap of severe overfitting (e.g., Trace Models) or underfitting (e.g., Flower Models), we introduce a hybrid evaluation framework. This approach evaluates generalization through two complementary lenses: **Generative Behavioral Analysis** and **Structural Frequency Analysis**.

## 2.1. Generative Behavioral Analysis ($Gen\_shadow$)
The first component, $Gen\_shadow$, measures the model's flexibility by simulating potential future event logs. Instead of relying purely on existing data, it acts as a probabilistic stress test.

* **Local Marking and Probability Estimation:** To avoid the state-space explosion typically associated with global Petri net markings, the algorithm relies on **Local Markings** (the token distribution of immediate input places). We employ the **Good-Turing frequency estimation** to calculate the mutation probability ($P_{unseen}$) of these local states. 
* **Dynamic Trace Generation (Play-out):** For predictable states (high historical frequency, low variance), the algorithm applies a low mutation rate, restricting the generation of illogical traces. Conversely, for unpredictable states (low frequency, high variance), it actively explores new, logically valid variations.
* **Replay Evaluation:** A synthetic "shadow" log is generated through this stochastic random walk. $Gen\_shadow$ is formally defined as the replay fitness of this newly generated synthetic log evaluated against the discovered model.

## 2.2. Structural Frequency Analysis ($Gen\_struct$)
A purely generative approach risks falsely rewarding "Flower Models" that permit all possible behaviors. To counteract this, we introduce $Gen\_struct$ as a strict, reality-based mathematical constraint.

* **Overfitting Penalty:** This component replays the *original* event log on the discovered model to analyze the usage frequency of its internal structure. If certain structural paths or transitions are visited exceptionally rarely, the algorithm penalizes the score. This effectively identifies and downgrades models that overfit by constructing specific, isolated branches solely to memorize rare outlier traces.

## 2.3. The Hybrid Synthesis
The final generalization metric is computed as a weighted combination of the generative and structural scores. This guarantees that the model is neither strictly constrained by historical data nor overly permissive of random noise.

$$Gen\_Total = w \times Gen\_shadow + (1 - w) \times Gen\_struct$$

The parameter $w \in [0, 1]$ allows for the calibration of the evaluation focus. By adjusting $w$, the metric can dynamically balance the reward for probabilistic flexibility against the penalty for structural bloat, providing a holistic measurement of process model generalization.

