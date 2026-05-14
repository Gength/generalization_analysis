# Pick-One-Out Generalization Metric — Iterative Design Log

> **Architecture note:** This document maintains a versioned history of the joint weighting formula. Each version documents what was tried, why it failed (or succeeded), and what replaced it. Future versions (v3, v4, …) will be appended at the top of this section, keeping the design evolution traceable.

---

## v2: Harmonic-Mean Normalized Joint Weighting *(current)*

### Motivation — Why v1 Failed

v1's formula $w(v_i) = \ln(f(v_i) + 1) \times \ln(AvgGlobalFreq + 1)$ produced a **near-constant multiplier** across all variants. Empirically, on the event log, $\ln(AvgGlobalFreq + 1)$ converged to $\approx 10.7$ for nearly every variant, regardless of whether it was a high-frequency happy path or low-frequency noise. The mathematical and structural reasons for this failure are threefold:

1. **The Dilution Effect of Arithmetic Means:** Most long-tail variants are not entirely composed of rare behavior; they typically consist of a long sequence of frequent "main-path" edges followed by a single rare deviation. When calculating the arithmetic mean ($Avg$), the massive frequencies of the normal edges completely swallow and dilute the tiny frequency of the rare edge, making the average artificially high.
2. **Logarithmic Over-compression:** Even if a slight difference remains after averaging (e.g., 30,000 vs. 50,000), the natural logarithm violently compresses this variance. A linear difference of 20,000 collapses to a mere $\approx 0.5$ difference in log scale.
3. **Weighted-Average Cancellation:** Because the dilution and compression force the right-hand term to converge to an approximate constant $k$, it simply factors out of both the numerator and denominator in the final generalization score. 

$$Generalization = \frac{\sum (w_{pure} \cdot k) \cdot Score_i}{\sum (w_{pure} \cdot k)} = \frac{k \cdot \sum w_{pure} \cdot Score_i}{k \cdot \sum w_{pure}} = \frac{\sum w_{pure} \cdot Score_i}{\sum w_{pure}}$$

**Consequence:** The joint weighting was mathematically neutralized. Pure and Joint scores differed by only $0.0001$–$0.0003$, rendering the internal transition frequency component functionally bankrupt.

### v2 Formula

To break the constant-ratio degeneracy and solve the **dilution effect**, v2 replaces the logarithmic average with a **linear, normalized ratio** based on the **Harmonic Mean**. The Harmonic Mean is mathematically dominated by its smallest elements, perfectly capturing the 'weakest link' of a process variant **and translating it into a severe penalty for the variant's overall weight.**

$$w_{v2}(v_i) = \ln\big(f(v_i) + 1\big) \;\times\; \frac{HMean(E_{v_i})}{MaxGlobalFreq(E_{log})}$$

where the Harmonic Mean of the edge frequencies is defined as:

$$HMean(E_{v_i}) = \frac{|E_{v_i}|}{\sum_{e \in E_{v_i}} \frac{1}{GlobalFreq(e)}}$$

| Symbol | Meaning |
|---|---|
| $f(v_i)$ | Absolute frequency of variant $v_i$ in the log |
| $E_{v_i}$ | Set of directly-follows edges within variant $v_i$ |
| $HMean(E_{v_i})$ | Harmonic Mean of global frequencies of all edges in $E_{v_i}$ |
| $MaxGlobalFreq(E_{log})$ | Frequency of the single most-traversed edge in the entire log |

### Why v2 Works

1. **True multiplicative penalty for noise variants (The "Weakest Link").** Unlike arithmetic averages, the harmonic mean cannot be artificially inflated by normal edges. If a variant has 9 edges with a frequency of 50,000 and just 1 edge with a frequency of 1, the $HMean$ plummets to $\approx 10$. The ratio $\frac{HMean}{Max}$ becomes $0.0002$ — its weight is brutally slashed by $99.9\%$.
2. **Happy-path preservation.** Core-process variants whose edges are *all* near the global maximum keep a harmonic mean close to the maximum, preserving a ratio near $1.0$ and maintaining their dominant voice in the final score.
3. **Bounded and interpretable.** The ratio is always in $(0, 1]$. Multiplication by $\ln(f+1)$ keeps the frequency-smoothing benefit of the original design while the linear ratio supplies genuine discriminative power.
4. **No more cancellation.** The ratio varies substantially across variants (from near $0$ to $1.0$), so it no longer factors out of the weighted average — the Joint score will finally diverge meaningfully from the Pure score.

### Expected Impact

- Pure vs. Joint score gap should widen from $\sim 0.0001$ to at least $0.05$–$0.10$.
- Noise variants (characterized by at least one extremely low-frequency edge pulling down the $HMean$) will have their contribution to the final score proportionally suppressed.
- Concurrency-induced rare variants (high $HMean$ because *all* internal edges are frequent globally, but low $f$ locally) will retain meaningful weight — perfectly preserving the original design goal.

---

## v1: Double-Log Joint Weighting *(deprecated)*


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

### Implementation Record
```python
def compute_joint_weight(variant_tuple, variant_freq, global_dfg):
    """
    Joint weighting formula (Variant B):
      w(v_i) = ln(f(v_i) + 1) × ln(avg_global_transition_freq + 1)

    Where avg_global_transition_freq is the average global DFG frequency
    of all directly-follows edges within this variant.
    """
    freq_component = log(variant_freq + 1)

    if len(variant_tuple) < 2:
        transition_component = log(1 + 1)  # single-event trace
    else:
        edge_freqs = []
        for i in range(len(variant_tuple) - 1):
            edge = (variant_tuple[i], variant_tuple[i + 1])
            edge_freqs.append(global_dfg.get(edge, 0))
        avg_edge_freq = sum(edge_freqs) / len(edge_freqs)
        transition_component = log(avg_edge_freq + 1)

    return freq_component * transition_component

def compute_pure_freq_weight(variant_freq):
    """Pure frequency weighting (Variant A): w(v_i) = ln(f(v_i) + 1)."""
    return log(variant_freq + 1)

```