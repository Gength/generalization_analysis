# 1. Pick one out Approach
## 1.1 Core Mathematical Formula Design

The core of this metric lies in the **Weighting Function**. We cannot rely on pure linear frequencies, otherwise, the contribution of "long-tail" variants will be completely drowned out. We typically introduce logarithms to smooth out frequency differences.

Assume your event log contains a set $V$ with a total of $n$ unique variants. For any given variant $v_i$, its absolute frequency in the log is $f(v_i)$. 

We can construct the following generalization score formula:

$$Generalization = \frac{\sum_{i=1}^{n} w(v_i) \cdot Score(v_i, M_{\setminus v_i})}{\sum_{i=1}^{n} w(v_i)}$$

### Formula Breakdown
* $M_{\setminus v_i}$: Represents the process model discovered using the remaining log after **removing** variant $v_i$.
* $Score(v_i, M_{\setminus v_i})$: Represents the **replay score** of the removed variant $v_i$ on the newly generated model. In PM4Py, it is highly recommended to use the **Alignment Score**, which yields a precise continuous value between $0$ and $1$, rather than a rigid binary $0$ (fail) or $1$ (success).
* $w(v_i)$: This is the core of your "frequency weighting." To prevent extremely high-frequency "happy paths" from dominating the score, a logarithmic weight is recommended, such as $w(v_i) = \ln(f(v_i) + 1)$. Adding $1$ prevents the logarithm of a variant with a frequency of $1$ from evaluating to $0$.


## 1.2 The Joint Weighting Formula: State-Aware Fine-Grained Variant Weighting

To accurately assess a process model's generalization capability without being skewed by random noise or heavily concurrent logs, we define a joint weighting function $w(v_i)$ for each variant. This formula synergizes two crucial perspectives: the **macroscopic frequency** of the variant itself and the **microscopic average frequency** of its internal state transitions.

The formula is defined as follows:

$$w(v_i) = \underbrace{\ln(f(v_i) + 1)}_{\text{Variant's Own Frequency}} \times \underbrace{\ln \left( \frac{1}{|E_{v_i}|} \sum_{e \in E_{v_i}} GlobalFreq(e) + 1 \right)}_{\text{Average Frequency of Internal State Transitions}}$$

### Formula Breakdown:
* **$w(v_i)$**: The final calculated weight for variant $v_i$.
* **$f(v_i)$**: The absolute occurrence frequency of variant $v_i$ in the event log.
* **$E_{v_i}$**: The set of all directly-follows relationships contained within variant $v_i$.
* **$|E_{v_i}|$**: The total number of state transitions in variant $v_i$.
* **$GlobalFreq(e)$**: The absolute frequency of a specific state transition $e$ across the *entire* event log (which can be easily extracted from a global Directly-Follows Graph).
* ***Note on Logarithms:** The natural logarithm ($\ln$) is applied to both terms to smooth out extreme frequency differences (the long-tail effect). We add $+1$ to prevent the result from evaluating to zero for instances that only appear once.*

---

### The Rationale: The Multiplier Effect

The brilliance of this formula lies in its mathematical multiplication, which automatically categorizes and appropriately weights three distinct types of variants:

**1. True Low-Frequency Noise (Low $\times$ Low $\rightarrow$ Extremely Low Weight)**
If a variant is rarely seen in the log AND consists of activities or state transitions that are globally rare (e.g., a system crash or a unique manual intervention), both sides of the multiplication yield a small value. The resulting weight is practically negligible. If the discovered model cannot replay this variant, the overall generalization score will barely be penalized.

**2. Concurrency-Induced Rarity (Low $\times$ High $\rightarrow$ Moderate to High Weight)**
This is the core problem the formula solves. A variant might appear only once in the entire log due to a rare interleaving of highly concurrent activities. While its macroscopic frequency $f(v_i)$ is low, its internal state transitions are globally very frequent. The high average state transition frequency on the right side of the equation significantly boosts the overall weight. A good process model *should* be able to generalize and support these recombined frequent state transitions. If it fails to replay this variant, the metric will rightfully penalize the model.

**3. Standard Main-Path Processes (High $\times$ High $\rightarrow$ Maximum Weight)**
For standard, highly frequent business processes, both the variant frequency and the internal state transition frequencies are high. This yields the highest possible weight, ensuring that the model's ability to support the core operational pathways remains the dominant factor in the generalization score.

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

# Experiment Strategy

## Overview

This section outlines a comprehensive experimental framework to evaluate, compare, and validate our two proposed generalization metrics — **Method 1 (Pick-One-Out with Joint Variant–Transition Weighting)** and **Method 2 (Hybrid Generative–Structural Evaluation)**. The experiments are organized into three tiers: (1) deep-dive single-dataset analysis, (2) cross-method benchmarking and consistency analysis, and (3) stress-testing via noise injection, extreme model confrontation, data-characteristic sensitivity, and scalability profiling.

---

## Tier I: Single-Dataset Deep Analysis

In this tier, we select one richly characterized real-world event log (e.g., a subset of **BPI Challenge 2012** or **BPI Challenge 2017**, chosen for moderate concurrency and natural noise) and run **3–4 process discovery algorithms** with fundamentally different characteristics: **Inductive Miner (IM)** , **Heuristics Miner (HM)** , **Alpha Miner (AM)** , and optionally **Split Miner (SM)** . Each model is evaluated using both Method 1 and Method 2.

### 1.1 Ranking Attribution & Micro-Variant Tracing

**Objective:** Demonstrate that our metrics are not black boxes — every score difference can be traced back to concrete, interpretable behavioral phenomena in the model.

**Procedure:**
1. Rank all discovered models by their generalization scores under each method.
2. Identify the top-ranked and bottom-ranked models. When a significant gap exists (e.g., IM scores substantially higher than AM), isolate **1–2 specific long-tail or unseen variants** that drive the divergence.
3. For each selected variant, perform a microscopic replay analysis:
   - Show **why** the weaker model fails to replay the variant (e.g., a missing directly-follows relation in AM, or an overly restrictive cut in HM).
   - Show **why** the stronger model succeeds (e.g., IM introduces a silent transition that bridges the gap).
   - Quantify the variant's weight contribution using Method 1's joint weighting formula — highlight whether the variant's $GlobalFreq(e)$ is high (concurrency-induced rarity) or low (true noise), and explain how this weight amplified or dampened the score gap.

**Deliverables:** A table mapping each variant to its replay outcome, weight, and per-method score contribution; a narrative walk-through of at least one "hero" variant that epitomizes the metric's discriminative power.

### 1.2 Scatter-Plot Correlation Against Baseline Metrics

**Objective:** Show that our metrics capture generalization dimensions that PM4Py's default generalization metric misses.

**Procedure:**
1. Generate a pool of **10–20 models** by varying miner parameters (e.g., noise thresholds for HM, IM variants).
2. Compute two scores for each model: (a) PM4Py's built-in generalization metric, and (b) our proposed metric (Method 1 or Method 2).
3. Plot a **2D scatter plot** with PM4Py on the x-axis and our metric on the y-axis. Each point represents one model.

**Analysis:**
- Identify **off-diagonal outliers** — models where PM4Py gives a high score (e.g., 0.9) but our metric gives a low score (e.g., 0.4).
- Deep-dive into these outlier models. Diagnose the blind spots of the baseline metric:
  - Does PM4Py fail to penalize structural overfitting (e.g., a model with many dead, rarely-used branches)?
  - Does PM4Py lack the local Markov probability awareness that our metrics capture?
  - Does PM4Py over-reward permissiveness without checking structural parsimony?
- Argue why our score is the more human-intuitive and practically useful evaluation.

**Deliverables:** Scatter plot with annotated outliers; case-study analysis of at least two outlier models.

### 1.3 Internal Ablation Study of Sub-Components

**Objective:** Prove that every term in our formulas is necessary — removing any component degrades evaluation accuracy.

**Ablation for Method 1 (Pick-One-Out):**
- **Variant A:** Pure variant-frequency weighting only: $w(v_i) = \ln(f(v_i) + 1)$.
- **Variant B:** Full joint weighting: $w(v_i) = \ln(f(v_i) + 1) \times \ln(\text{AvgGlobalFreq} + 1)$.
- Compare the model rankings produced by Variant A vs. Variant B on the same dataset and miners.
- Demonstrate specific cases where Variant B correctly down-weights true noise variants (low $GlobalFreq$) and up-weights concurrency-induced rare variants (high $GlobalFreq$), leading to a more sensible ranking.

**Ablation for Method 2 (Hybrid Generative–Structural):**
- Fix the dataset and miner, then sweep the fusion weight $w$ from **0.0 to 1.0 in steps of 0.1**.
- At $w = 0.0$, only $Gen\_struct$ (structural penalty) is active.
- At $w = 1.0$, only $Gen\_shadow$ (generative flexibility) is active.
- Observe how the ranking of different miners changes as $w$ varies.
- Identify the $w$ region where the ranking stabilizes and best aligns with domain knowledge.

**Analysis:** Show that removing either the generative or structural component leads to pathological rankings — $w=0$ over-penalizes flexible but slightly bloated models; $w=1$ over-rewards Flower-like models.

**Deliverables:** Ablation comparison tables; a line chart of model scores as a function of $w$.

---

## Tier II: Cross-Method Benchmarking & Consistency Analysis

### 2.1 Method 1 vs. Method 2 Benchmark

**Objective:** Establish whether our two independently designed metrics produce consistent rankings, and analyze discrepancies to understand their complementary strengths.

**Procedure:**
1. On a single dataset, run **3–4 miners** and compute generalization scores using both Method 1 and Method 2.
2. Produce a **rank correlation matrix** (Spearman's $\rho$ or Kendall's $\tau$) between the two methods.
3. For cases where the two methods disagree significantly on a model's rank, perform a root-cause analysis:
   - Does Method 1 penalize the model because a specific removed variant could not be replayed?
   - Does Method 2 reward the model because the shadow log generation found many plausible unseen traces?
   - Is the disagreement due to a fundamental philosophical difference (empirical leave-one-out vs. probabilistic simulation)?

**Analysis:** Map the scenarios where each method excels. Method 1 is expected to be more sensitive to variant-level structural completeness; Method 2 is expected to be more sensitive to local-state flexibility and the model's ability to absorb unseen behavior.

**Deliverables:** Rank correlation table; qualitative analysis of the top-3 disagreement cases.

---

## Tier III: Stress-Testing & Boundary Analysis

### 3.1 Extreme Model Confrontation: Trace Model vs. Flower Model

**Objective:** Verify that our metrics correctly identify and severely penalize the two well-known pathological extremes in process mining — the **Trace Model** (extreme overfitting) and the **Flower Model** (extreme underfitting).

**Procedure:**
1. Take the original event log and construct two degenerate models:
   - **Trace Model:** Every trace is a dedicated sequential path; zero generalization.
   - **Flower Model:** All activities are connected in a fully connected Petri net; permits any arbitrary sequence.
2. Compute both Method 1 and Method 2 scores for these two models.

**Expected Results & Analysis:**
- **Trace Model:** Both methods should assign an extremely low score. Method 1's leave-one-out replay will fail catastrophically for any removed variant. Method 2's $Gen\_struct$ will heavily penalize the explosion of rarely-used structural paths.
- **Flower Model:** Method 1 may assign a moderate-to-high score (since all variants replay perfectly after removal). However, Method 2's $Gen\_struct$ penalty should drag the score down significantly, exposing the model's lack of meaningful structure. Analyze whether Method 2's hybrid formula successfully "immune" itself against the Flower Model's cheating.

**Deliverables:** Score comparison table; discussion of each method's immunity to extreme model pathologies.

### 3.2 Data-Characteristic Sensitivity Analysis

**Objective:** No metric is universally optimal. This experiment characterizes the scenarios under which each method demonstrates dominant performance.

**Procedure:**
1. Select **2–3 public event logs** with radically different structural properties:
   - **High-Concurrency, Low-Long-Tail Log:** Many parallel activities, few rare variants (e.g., a log with heavy parallel approvals).
   - **Highly Sequential, Heavy-Long-Tail Log:** A mostly linear process with many infrequent error/rework branches.
   - **Balanced Log:** Moderate concurrency and moderate long-tail (the dataset from Tier I can serve this role).
2. Run the same set of miners on each dataset and compute both Method 1 and Method 2 scores.

**Analysis:**
- On the high-concurrency log, Method 2's Good-Turing-based local marking and shadow log generation should exhibit strong discriminative power, as it can reason about unseen interleavings probabilistically.
- On the heavy-long-tail log, Method 1's joint weighting formula (multiplying variant frequency by internal transition frequency) should excel at noise suppression, correctly down-weighting true outliers while up-weighting structurally meaningful rare variants.
- Discuss the **regime of dominance** for each metric and provide practical guidance on which metric to choose given dataset characteristics.

**Deliverables:** Radar chart or grouped bar chart comparing scores across datasets; a decision table mapping data characteristics to recommended metric.

### 3.3 Noise Injection & Robustness Analysis

**Objective:** Measure how gracefully each metric degrades under controlled, escalating levels of random noise — a hallmark of a well-designed generalization measure.

**Procedure:**
1. Start with a clean, structurally well-defined event log (this can be a synthetic log with known ground-truth behavior or a carefully filtered real log).
2. Inject random noise at increasing proportions: **0% (baseline), 5%, 10%, 15%, 20%, 25%** .
   - *Noise Type A — Activity Swaps:* Randomly swap the positions of two adjacent activities in a trace.
   - *Noise Type B — Activity Deletions:* Randomly delete individual events from traces.
   - *Noise Type C — Activity Insertions:* Randomly insert spurious activities into traces.
3. For each noise level, re-discover the process model (using a fixed miner) and compute both Method 1 and Method 2 scores.
4. Plot **noise–score decay curves** for each method.

**Analysis:**
- A robust metric should exhibit **gradual, near-linear decay** rather than a cliff-edge drop at low noise levels.
- Compare the decay slopes: which method is more resilient to low-level noise?
- Analyze whether Method 1's joint weighting naturally absorbs noise (low-$GlobalFreq$ noise variants get negligible weight), and whether Method 2's Good-Turing estimation provides statistical robustness against unseen mutations.

**Deliverables:** Noise–score decay line charts; a robustness ranking of the two methods per noise type.

### 3.4 Scalability & Computational Cost Analysis

**Objective:** Characterize the runtime profiles of both methods, providing practical guidance for industrial adoption.

**Procedure:**
1. Prepare event logs at three scales: **1,000 traces, 10,000 traces, 100,000 traces** (all derived from the same process to control for structural complexity).
2. For each log size, run both Method 1 and Method 2 end-to-end (including model discovery) and record:
   - Total wall-clock execution time.
   - Breakdown by phase (discovery, replay/alignment, weighting computation, shadow log generation).
3. Plot **execution time vs. log size** for both methods.

**Analysis:**
- Method 1 (Pick-One-Out) is expected to be lightweight: it relies on pre-computed DFG statistics and global frequencies, with alignment computation scaling linearly with the number of unique variants.
- Method 2 (Hybrid Generative–Structural) is expected to be more expensive due to shadow log generation via stochastic play-out and Good-Turing estimation.
- Frame the positioning: **Method 1 = lightweight, concurrency-friendly, industry-grade metric**; **Method 2 = deep, theoretically rigorous, expert-grade metric**.
- Discuss the trade-off between computational cost and evaluative depth.

**Deliverables:** Runtime comparison table and log-scale line chart; a cost–benefit positioning matrix.

---

## Summary of Experimental Contributions

| Experiment | Key Question Answered |
|---|---|
| 1.1 Micro-Variant Tracing | Can we explain every score difference in business terms? |
| 1.2 Baseline Correlation | Do our metrics see what PM4Py misses? |
| 1.3 Ablation Study | Is every formula component necessary? |
| 2.1 Cross-Method Benchmark | Do Method 1 and Method 2 agree? If not, why? |
| 3.1 Extreme Models | Can we detect and penalize Trace/Flower pathologies? |
| 3.2 Data Sensitivity | Which metric dominates under which data characteristics? |
| 3.3 Noise Robustness | How gracefully do our metrics degrade under noise? |
| 3.4 Scalability | What is the computational cost of each method? |
