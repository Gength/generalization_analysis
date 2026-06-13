# Experiment Strategy

## Overview

This section outlines a comprehensive experimental framework to evaluate and validate our proposed generalization metric — **Method 2 (Hybrid Generative–Structural Evaluation)**. The experiments are organized into two tiers: (1) deep-dive single-dataset analysis, and (2) stress-testing via model morphology confrontation, noise injection, data-characteristic sensitivity, and scalability profiling.

---

## Tier I: Single-Dataset Deep Analysis

In this tier, we select one richly characterized real-world event log (e.g., a subset of **BPI Challenge 2012** or **BPI Challenge 2017**, chosen for moderate concurrency and natural noise) and run **3–4 process discovery algorithms** with fundamentally different characteristics: **Inductive Miner (IM)** , **Heuristics Miner (HM)** , **Alpha Miner (AM)** , and optionally **Split Miner (SM)** . Each model is evaluated using Method 2.

### 1.1 Ranking Attribution & Decomposition Analysis

**Objective:** Demonstrate that our metric is not a black box — every score difference can be traced back to concrete, interpretable Gen_Shadow vs. Gen_Struct contributions.

**Procedure:**
1. Rank all discovered models by their Gen_Total scores.
2. Identify the top-ranked and bottom-ranked models. When a significant gap exists (e.g., IM scores substantially higher than AM), decompose Gen_Total into Gen_Shadow and Gen_Struct:
   - Show **why** the weaker model suffers — is it penalized by Gen_Struct (bloated/spaghetti structure) or does it score low on Gen_Shadow (poor behavioral flexibility)?
   - Show **why** the stronger model succeeds — does it strike a balance between structural parsimony and generative flexibility?
3. For Gen_Struct, further decompose into ArcFlow, Reach, and Cyclo to pinpoint the exact structural driver.

**Deliverables:** A decomposition table mapping each model to its Gen_Shadow, Gen_Struct, and sub-dimension scores; a narrative walk-through of at least one "hero" case that epitomizes the hybrid metric's discriminative power.

### 1.2 Scatter-Plot Correlation Against Baseline Metrics

**Objective:** Show that our metric captures generalization dimensions that PM4Py's default generalization metric misses.

**Procedure:**
1. Generate a pool of **10–20 models** by varying miner parameters (e.g., noise thresholds for HM, IM variants).
2. Compute two scores for each model: (a) PM4Py's built-in generalization metric, and (b) our proposed Method 2 metric (Gen_Total).
3. Plot a **2D scatter plot** with PM4Py on the x-axis and our metric on the y-axis. Each point represents one model.

**Analysis:**
- Identify **off-diagonal outliers** — models where PM4Py gives a high score (e.g., 0.9) but our metric gives a low score (e.g., 0.4).
- Deep-dive into these outlier models. Diagnose the blind spots of the baseline metric:
  - Does PM4Py fail to penalize structural overfitting (e.g., a model with many dead, rarely-used branches)?
  - Does PM4Py lack the local Markov probability awareness that our metric captures?
  - Does PM4Py over-reward permissiveness without checking structural parsimony?
- Argue why our score is the more human-intuitive and practically useful evaluation.

**Deliverables:** Scatter plot with annotated outliers; case-study analysis of at least two outlier models.

### 1.3 Internal Ablation Study of Sub-Components

**Objective:** Prove that every component in our formula is necessary — removing any sub-dimension degrades evaluation accuracy.

**Ablation 1 — Fusion Weight Sweep:**
- Fix the dataset and miner, then sweep the fusion weight $w$ from **0.0 to 1.0 in steps of 0.1**.
- At $w = 0.0$, only $Gen\_struct$ (structural penalty) is active.
- At $w = 1.0$, only $Gen\_shadow$ (generative flexibility) is active.
- Observe how the ranking of different miners changes as $w$ varies.
- Identify the $w$ region where the ranking stabilizes and best aligns with domain knowledge.

**Deliverables:** Ablation comparison tables; a line chart of Gen_Total as a function of $w$; sub-dimension ablation bar chart.

---

## Tier II: Stress-Testing & Boundary Analysis

### 2.1 Model Morphology Confrontation: From Theoretical Extremes to Real-World Archetypes

**Objective:** Go beyond the two theoretical boundary extremes (Trace Model and Flower Model) and evaluate our metric against the full spectrum of model morphologies encountered in real-world process mining — characterized by Wil van der Aalst's "Italian Food" taxonomy. This experiment verifies that our metric correctly ranks models along the generalization spectrum and uniquely identifies the "Lasagna" ideal.

**Background — The Generalization Spectrum:**
Trace Model (absolute overfitting) and Flower Model (absolute underfitting) are theoretical boundary cases. Real process mining on authentic event logs produces models inhabiting the continuum between these extremes, each with distinct structural and behavioral signatures. We adopt the classic taxonomy:

#### 2.1.1 The Model Morphology Catalog

**A. Trace Model (Theoretical Extreme — Absolute Overfitting)**
- *Morphology:* Every trace in the log is encoded as a dedicated sequential path with zero structural sharing. No generalization whatsoever.
- *Our Metric's Expected Behavior:* Method 2 should assign extremely low scores. $Gen\_struct$ heavily penalizes the explosion of single-use structural paths, and $Gen\_shadow$ scores low because the shadow log generates traces that the rigid Trace Model cannot replay.

**B. Spaghetti Model (Real-World Overfitting — "The Nightmare")**
- *Morphology:* Extremely chaotic — nearly all activities are connected with crisscrossing arcs, like a tangled plate of spaghetti. The main process backbone is unrecognizable. This is the most common outcome when mining unstructured real-world logs (e.g., hospital records, unclassified customer service tickets) without adequate frequency filtering.
- *Root Cause:* The discovery algorithm (e.g., Heuristics Miner with no frequency threshold, or Alpha Miner on noisy data) attempts to accommodate every low-frequency, long-tail anomaly, drawing arcs for every coincidental co-occurrence.
- *Expected Behavior Under Our Metric:*
  - **Method 2 ($Gen\_struct$):** Will **brutally penalize** the Spaghetti Model. The ArcFlow dimension identifies the massive redundancy — countless arcs visited only once globally — and drives the composite score down sharply. This is a key differentiator: the hybrid formula successfully exposes the Spaghetti Model's lack of meaningful structure.
- *How to Generate:* Run Heuristics Miner with dependency threshold = 0.0 and no frequency filtering on a noisy real-world log (e.g., BPI Challenge 2012 without preprocessing). Alternatively, use Alpha Miner on the same unfiltered log.

**C. Flower Model (Theoretical Extreme — Absolute Underfitting)**
- *Morphology:* A fully connected Petri net where all activities are placed in a single concurrent block, permitting any arbitrary activity sequence.
- *Our Metric's Expected Behavior:* Method 2's $Gen\_struct$ penalty should drag the score down significantly (low ArcFlow usage density, poor Cyclo score due to excessive arcs), exposing the model's vacuous structure as an underfitting extreme.

**D. Causal / Heuristics Net (Probability-Driven Pragmatism)**
- *Morphology:* Occupies the middle ground between Spaghetti and Lasagna. The model does not guarantee logical perfection (potential deadlocks are tolerated), but it exhibits strong immunity to low-frequency noise through probability-driven arc filtering. Arcs are annotated with dependency/confidence measures derived from directly-follows frequencies.
- *Representative Algorithm:* **Heuristics Miner** — its core logic uses directly-follows frequency and dependency metrics to prune arcs.
- *Expected Behavior Under Our Metric:* A well-tuned Heuristics Miner (with appropriate dependency/confidence thresholds) should rank second only to Inductive Miner on our leaderboard. Its probabilistic arc pruning produces models with decent structural parsimony.
- *How to Generate:* Run Heuristics Miner with dependency threshold $\approx 0.9$ and relative-to-best threshold $\approx 0.1$–$0.3$.

**E. Strict Block-Structured Model (Algorithmic Discipline)**
- *Morphology:* The model is constructed like building blocks — every split gateway (e.g., AND-Split) is guaranteed to have a corresponding join gateway (AND-Join) within a well-nested scope. No unstructured cross-level arcs are permitted. This structural discipline is a direct product of the discovery algorithm's design philosophy.
- *Representative Algorithm:* **Inductive Miner (IM)** — inherently builds models by recursively identifying "cuts" (sequence, parallel, exclusive choice, loop) in the directly-follows graph, guaranteeing block-structured output.
- *Expected Behavior Under Our Metric:* Block-structured models exhibit an innate structural restraint that prevents overfitting. IM will never produce a Flower Model or a Spaghetti Model. In experiments pitting IM against Alpha Miner, IM consistently produces the most structurally parsimonious models that still capture the core process logic.
- *How to Generate:* Run Inductive Miner (IM or IMf) directly — no parameter tuning needed for the block-structure guarantee.

**F. Lasagna Model (The "Holy Grail")**
- *Morphology:* The ideal process model — structurally crisp with clearly stratified layers. The high-frequency "Happy Path" backbone runs through the center like the main pasta layers of a lasagna, while rare exception branches are elegantly encapsulated in adjacent concurrent or choice constructs without tangling the main flow. Exhibits both structural clarity and behavioral flexibility.
- *Root Cause:* Produced either from highly normative data (e.g., automated manufacturing pipelines) or by applying a well-tuned discovery algorithm (e.g., Inductive Miner with noise filtering) on well-preprocessed logs.
- *Expected Behavior Under Our Metric:* This should be the **absolute top-scoring model**. It retains reasonable flexibility ($Gen\_shadow$ scores high) while avoiding spurious arcs ($Gen\_struct$ penalty is minimal). The hybrid formula confirms that the model's structure is both flexible and parsimonious.
- *How to Generate:* Run Inductive Miner with noise threshold $\approx 0.2$–$0.4$ on a well-preprocessed log, or use Split Miner with appropriate filtering.

#### 2.1.2 Experimental Procedure

1. **Construct the Model Gallery:** Using a single richly characterized real-world event log (e.g., BPI Challenge 2012 or 2017), generate all six model archetypes:
   - **Trace Model** — manually construct from the variant list.
   - **Spaghetti Model** — Heuristics Miner with dependency threshold = 0.0, or Alpha Miner on unfiltered log.
   - **Causal Net** — Heuristics Miner with tuned thresholds (dependency $\approx 0.9$).
   - **Strict Block-Structured Model** — Inductive Miner (default).
   - **Lasagna Model** — Inductive Miner with noise filtering, or Split Miner.
   - **Flower Model** — manually construct a fully connected Petri net.

2. **Compute Scores:** Evaluate all six models using Method 2. Record $Gen\_Total$, $Gen\_shadow$, and $Gen\_struct$ (with ArcFlow, Reach, Cyclo sub-dimensions).

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
- Model gallery table with morphological descriptions, generation parameters, and per-metric scores.
- Quadrant diagram with $Gen\_Total$ annotations.
- Score trajectory chart (inverted-U curve) across the six archetypes.
- Decomposition bar chart ($Gen\_shadow$ vs. $Gen\_struct$) for each model.
- Narrative analysis: "How our metric guides the search toward Lasagna."

**PM4Py Feasibility Note:** All required miners are natively available in PM4Py:
- `pm4py.discover_petri_net_inductive()` for Inductive Miner (block-structured / Lasagna).
- `pm4py.discover_petri_net_heuristics()` for Heuristics Miner (Spaghetti with low thresholds, Causal Net with high thresholds).
- `pm4py.discover_petri_net_alpha()` for Alpha Miner (can produce Spaghetti on noisy logs).
- Trace and Flower Models can be constructed manually via `pm4py.objects.petri_net` primitives.

### 2.2 Data-Characteristic Sensitivity Analysis

**Objective:** Understand how Method 2's behavior varies across datasets with radically different structural properties.

**Procedure:**
1. Select **2–3 public event logs** with radically different structural properties:
   - **High-Concurrency, Low-Long-Tail Log:** Many parallel activities, few rare variants (e.g., a log with heavy parallel approvals).
   - **Highly Sequential, Heavy-Long-Tail Log:** A mostly linear process with many infrequent error/rework branches.
   - **Balanced Log:** Moderate concurrency and moderate long-tail (the dataset from Tier I can serve this role).
2. Run the same set of miners on each dataset and compute Method 2 Gen_Total.

**Analysis:**
- On the high-concurrency log, Gen_Shadow's Good-Turing-based local marking should exhibit strong discriminative power, as it can reason about unseen interleavings probabilistically.
- On the heavy-long-tail log, Gen_Struct's ArcFlow dimension should help penalize overfitted structures.
- Discuss how Gen_Shadow vs. Gen_Struct contributions shift across data characteristics.

**Deliverables:** Radar chart or grouped bar chart comparing scores across datasets; decomposition table showing Gen_Shadow/Gen_Struct balance per dataset type.

### 2.3 Noise Injection & Robustness Analysis

**Objective:** Measure how gracefully Method 2 degrades under controlled, escalating levels of random noise — a hallmark of a well-designed generalization measure.

**Procedure:**
1. Start with a clean, structurally well-defined event log (this can be a synthetic log with known ground-truth behavior or a carefully filtered real log).
2. Inject random noise at increasing proportions: **0% (baseline), 5%, 10%, 15%, 20%, 25%** .
   - *Noise Type A — Activity Swaps:* Randomly swap the positions of two adjacent activities in a trace.
   - *Noise Type B — Activity Deletions:* Randomly delete individual events from traces.
   - *Noise Type C — Activity Insertions:* Randomly insert spurious activities into traces.
3. For each noise level, re-discover the process model (using a fixed miner) and compute Method 2 scores.
4. Plot **noise–score decay curves** for Gen_Total, Gen_Shadow, and Gen_Struct separately.

**Analysis:**
- A robust metric should exhibit **gradual, near-linear decay** rather than a cliff-edge drop at low noise levels.
- Does Gen_Shadow's Good-Turing estimation provide statistical robustness against unseen mutations?
- Does Gen_Struct's ArcFlow dimension get hit harder by noise than the Reach/Cyclo dimensions?

**Deliverables:** Noise–score decay line charts; per-noise-type robustness analysis.

### 2.4 Scalability & Computational Cost Analysis

**Objective:** Characterize the runtime profile of Method 2, providing practical guidance for industrial adoption.

**Procedure:**
1. Prepare event logs at three scales: **1,000 traces, 10,000 traces, 100,000 traces** (all derived from the same process to control for structural complexity).
2. For each log size, run Method 2 end-to-end (including model discovery) and record:
   - Total wall-clock execution time.
   - Breakdown by phase (discovery, Gen_Shadow — shadow log generation + replay, Gen_Struct — token replay + BFS).
3. Plot **execution time vs. log size**.

**Analysis:**
- Gen_Shadow is expected to be the dominant cost due to shadow log generation via stochastic play-out and Good-Turing estimation.
- Gen_Struct is cheap: one token replay + BFS + O(1) cyclomatic computation.
- Compare the cost ratio between the generative and structural components.

**Deliverables:** Runtime breakdown table and log-scale line chart.

---

## Summary of Experimental Contributions

| Experiment | Key Question Answered |
|---|---|
| 1.1 Decomposition Analysis | Can we explain every score difference via Gen_Shadow vs. Gen_Struct contributions? |
| 1.2 Baseline Correlation | Does our metric see what PM4Py misses? |
| 1.3 Ablation Study | Is every component (fusion weight, Gen_Struct sub-dimension) necessary? |
| 2.1 Model Morphology Confrontation | Can our metric navigate the full spectrum from Trace → Spaghetti → Lasagna → Flower, and uniquely identify the Lasagna ideal? |
| 2.2 Data Sensitivity | How does Method 2 behave across different data characteristics? |
| 2.3 Noise Robustness | How gracefully does Method 2 degrade under noise? |
| 2.4 Scalability | What is the computational cost of Method 2? |
