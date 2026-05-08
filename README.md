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

This section outlines a comprehensive experimental framework to evaluate, compare, and validate our two proposed generalization metrics — **Method 1 (Pick-One-Out with Joint Variant–Transition Weighting)** and **Method 2 (Hybrid Generative–Structural Evaluation)**. The experiments are organized into three tiers: (1) deep-dive single-dataset analysis, (2) cross-method benchmarking and consistency analysis, and (3) stress-testing via model morphology confrontation, noise injection, data-characteristic sensitivity, and scalability profiling.

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

### 3.1 Model Morphology Confrontation: From Theoretical Extremes to Real-World Archetypes

**Objective:** Go beyond the two theoretical boundary extremes (Trace Model and Flower Model) and evaluate our metrics against the full spectrum of model morphologies encountered in real-world process mining — characterized by Wil van der Aalst's "Italian Food" taxonomy. This experiment verifies that our metrics correctly rank models along the generalization spectrum and uniquely identify the "Lasagna" ideal.

**Background — The Generalization Spectrum:**
Trace Model (absolute overfitting) and Flower Model (absolute underfitting) are theoretical boundary cases. Real process mining on authentic event logs produces models inhabiting the continuum between these extremes, each with distinct structural and behavioral signatures. We adopt the classic taxonomy:

#### 3.1.1 The Model Morphology Catalog

**A. Trace Model (Theoretical Extreme — Absolute Overfitting)**
- *Morphology:* Every trace in the log is encoded as a dedicated sequential path with zero structural sharing. No generalization whatsoever.
- *Our Metrics' Expected Behavior:* Both methods should assign extremely low scores. Method 1's leave-one-out replay fails catastrophically for any removed variant. Method 2's $Gen\_struct$ heavily penalizes the explosion of single-use structural paths.

**B. Spaghetti Model (Real-World Overfitting — "The Nightmare")**
- *Morphology:* Extremely chaotic — nearly all activities are connected with crisscrossing arcs, like a tangled plate of spaghetti. The main process backbone is unrecognizable. This is the most common outcome when mining unstructured real-world logs (e.g., hospital records, unclassified customer service tickets) without adequate frequency filtering.
- *Root Cause:* The discovery algorithm (e.g., Heuristics Miner with no frequency threshold, or Alpha Miner on noisy data) attempts to accommodate every low-frequency, long-tail anomaly, drawing arcs for every coincidental co-occurrence.
- *Expected Behavior Under Our Metrics:*
  - **Method 1 (Pick-One-Out):** May assign a deceptively moderate-to-high score because the model's excessive permissiveness allows it to replay many removed variants successfully.
  - **Method 2 ($Gen\_struct$):** Will **brutally penalize** the Spaghetti Model. The structural penalty term identifies the massive redundancy — countless arcs visited only once globally — and drives the composite score down sharply. This is a key differentiator: Method 2's hybrid formula successfully exposes the Spaghetti Model's lack of meaningful structure.
- *How to Generate:* Run Heuristics Miner with dependency threshold = 0.0 and no frequency filtering on a noisy real-world log (e.g., BPI Challenge 2012 without preprocessing). Alternatively, use Alpha Miner on the same unfiltered log.

**C. Flower Model (Theoretical Extreme — Absolute Underfitting)**
- *Morphology:* A fully connected Petri net where all activities are placed in a single concurrent block, permitting any arbitrary activity sequence.
- *Our Metrics' Expected Behavior:* Method 1 may assign a moderate-to-high score (all variants replay perfectly). Method 2's $Gen\_struct$ penalty should drag the score down significantly, exposing the model's vacuous structure.

**D. Causal / Heuristics Net (Probability-Driven Pragmatism)**
- *Morphology:* Occupies the middle ground between Spaghetti and Lasagna. The model does not guarantee logical perfection (potential deadlocks are tolerated), but it exhibits strong immunity to low-frequency noise through probability-driven arc filtering. Arcs are annotated with dependency/confidence measures derived from directly-follows frequencies.
- *Representative Algorithm:* **Heuristics Miner** — its core logic (using directly-follows frequency and dependency metrics to prune arcs) shares a philosophical kinship with Method 1's joint weighting formula.
- *Expected Behavior Under Our Metrics:* A well-tuned Heuristics Miner (with appropriate dependency/confidence thresholds) should rank second only to Inductive Miner on our leaderboard. Its probabilistic arc pruning naturally aligns with the frequency-weighted reasoning in both methods.
- *How to Generate:* Run Heuristics Miner with dependency threshold $\approx 0.9$ and relative-to-best threshold $\approx 0.1$–$0.3$.

**E. Strict Block-Structured Model (Algorithmic Discipline)**
- *Morphology:* The model is constructed like building blocks — every split gateway (e.g., AND-Split) is guaranteed to have a corresponding join gateway (AND-Join) within a well-nested scope. No unstructured cross-level arcs are permitted. This structural discipline is a direct product of the discovery algorithm's design philosophy.
- *Representative Algorithm:* **Inductive Miner (IM)** — inherently builds models by recursively identifying "cuts" (sequence, parallel, exclusive choice, loop) in the directly-follows graph, guaranteeing block-structured output.
- *Expected Behavior Under Our Metrics:* Block-structured models exhibit an innate structural restraint that prevents overfitting. IM will never produce a Flower Model or a Spaghetti Model. In experiments pitting IM against Alpha Miner, IM consistently produces the most structurally parsimonious models that still capture the core process logic.
- *How to Generate:* Run Inductive Miner (IM or IMf) directly — no parameter tuning needed for the block-structure guarantee.

**F. Lasagna Model (The "Holy Grail")**
- *Morphology:* The ideal process model — structurally crisp with clearly stratified layers. The high-frequency "Happy Path" backbone runs through the center like the main pasta layers of a lasagna, while rare exception branches are elegantly encapsulated in adjacent concurrent or choice constructs without tangling the main flow. Exhibits both structural clarity and behavioral flexibility.
- *Root Cause:* Produced either from highly normative data (e.g., automated manufacturing pipelines) or by applying a well-tuned discovery algorithm (e.g., Inductive Miner with noise filtering) on well-preprocessed logs.
- *Expected Behavior Under Our Metrics:* This should be the **absolute top-scoring model** under both methods. It retains reasonable flexibility ($Gen\_shadow$ scores high) while avoiding spurious arcs ($Gen\_struct$ penalty is minimal). Method 1's joint weighting correctly identifies that all meaningful variants are replayable, and Method 2's hybrid formula confirms that the model's structure is both flexible and parsimonious.
- *How to Generate:* Run Inductive Miner with noise threshold $\approx 0.2$–$0.4$ on a well-preprocessed log, or use Split Miner with appropriate filtering.

#### 3.1.2 Experimental Procedure

1. **Construct the Model Gallery:** Using a single richly characterized real-world event log (e.g., BPI Challenge 2012 or 2017), generate all six model archetypes:
   - **Trace Model** — manually construct from the variant list.
   - **Spaghetti Model** — Heuristics Miner with dependency threshold = 0.0, or Alpha Miner on unfiltered log.
   - **Causal Net** — Heuristics Miner with tuned thresholds (dependency $\approx 0.9$).
   - **Strict Block-Structured Model** — Inductive Miner (default).
   - **Lasagna Model** — Inductive Miner with noise filtering, or Split Miner.
   - **Flower Model** — manually construct a fully connected Petri net.

2. **Compute Scores:** Evaluate all six models using both Method 1 and Method 2. Record $Gen\_Total$, $Gen\_shadow$, and $Gen\_struct$ separately for Method 2.

3. **Quadrant Visualization:** Plot each model on a **2D quadrant diagram**:
   - **X-axis: Structural Complexity** (from Minimal / Parsimonious → Chaotic / Over-parameterized).
   - **Y-axis: Behavioral Permissiveness** (from Rigid / Strict → Permissive / Anything-Goes).
   - Annotate each model's position with its $Gen\_Total$ score from Method 2.
   - The ideal "Lasagna Zone" occupies the center-right region (moderate complexity, moderate permissiveness).

   ```
   Behavioral Permissiveness
        ↑
   1.0  │  Flower Model          │
        │  (Gen_Total ≈ 0.2)     │  Spaghetti Model
        │                        │  (Gen_Total ≈ 0.3–0.5)
        │                        │
        │         Lasagna Model  │
        │         (Gen_Total ≈   │
        │          0.85–0.95)    │
        │                        │
        │  Strict Block-Struct.  │  Causal/Heuristics Net
        │  (Gen_Total ≈ 0.7–0.8) │  (Gen_Total ≈ 0.6–0.8)
        │                        │
   0.0  │  Trace Model           │
        │  (Gen_Total ≈ 0.0–0.1) │
        └────────────────────────┘
        0.0   Structural Complexity →   1.0
   ```

4. **Score Trajectory Analysis:** Trace how $Gen\_Total$ evolves as we move from Trace Model → Spaghetti → Causal Net → Lasagna → Flower Model. The score should peak at Lasagna and decline toward both extremes, forming an inverted-U shape — demonstrating that our metric correctly identifies the sweet spot between overfitting and underfitting.

5. **Decomposition Analysis (Method 2 Only):** For each model archetype, examine the individual contributions of $Gen\_shadow$ and $Gen\_struct$:
   - Spaghetti Model: $Gen\_shadow$ moderate-to-high, but $Gen\_struct$ very low → composite score pulled down.
   - Flower Model: $Gen\_shadow$ very high, but $Gen\_struct$ near zero → composite score pulled down.
   - Lasagna Model: Both $Gen\_shadow$ and $Gen\_struct$ are high → composite score maximized.
   - This decomposition proves that neither sub-component alone suffices — only the hybrid formula correctly identifies the Lasagna ideal.

**Deliverables:**
- Model gallery table with morphological descriptions, generation parameters, and per-method scores.
- Quadrant diagram with $Gen\_Total$ annotations.
- Score trajectory chart (inverted-U curve) across the six archetypes.
- Decomposition bar chart ($Gen\_shadow$ vs. $Gen\_struct$) for each model.
- Narrative analysis: "How our metrics guide the search toward Lasagna."

**PM4Py Feasibility Note:** All required miners are natively available in PM4Py:
- `pm4py.discover_petri_net_inductive()` for Inductive Miner (block-structured / Lasagna).
- `pm4py.discover_petri_net_heuristics()` for Heuristics Miner (Spaghetti with low thresholds, Causal Net with high thresholds).
- `pm4py.discover_petri_net_alpha()` for Alpha Miner (can produce Spaghetti on noisy logs).
- Trace and Flower Models can be constructed manually via `pm4py.objects.petri_net` primitives.

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
| 3.1 Model Morphology Confrontation | Can our metrics navigate the full spectrum from Trace → Spaghetti → Lasagna → Flower, and uniquely identify the Lasagna ideal? |
| 3.2 Data Sensitivity | Which metric dominates under which data characteristics? |
| 3.3 Noise Robustness | How gracefully do our metrics degrade under noise? |
| 3.4 Scalability | What is the computational cost of each method? |
