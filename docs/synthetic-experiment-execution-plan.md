Here’s a structured **execution plan** for the **synthetic tasks experiment**, where we measure survival both by **steps (composition depth)** and **tokens**.

---

# **Execution Plan: Synthetic Survival Experiment**

## 1. **Task Design**

* **Task type:** function composition (e.g., nested arithmetic, pointer-chasing, synthetic boolean function composition).
* **Definition of a step:** one layer of composition (e.g., applying one function to the output of the previous function).
* **Input generation:** randomly generate functions and arguments for depths $L = 1 \dots N$ (e.g., $N=50$).
* **Output evaluation:** ground truth can be computed deterministically.

---

## 2. **Metrics**

* **Step survival (composition survival):**

  * Time axis = number of composition steps $L$.
  * Event = first incorrect output at step $L$.
  * Survival = $P(\text{model correct up to depth } L)$.

* **Token survival (context survival):**

  * Time axis = number of tokens in the input/output sequence (including function definitions + reasoning).
  * Event = first incorrect output at token count $T$.
  * Token counts logged with the tokenizer of the model used (e.g., GPT-2’s BPE).

---

## 3. **Experiment Conditions**

* **Baseline condition:** terse representation (minimal tokens per step).
* **Redundant condition(s):**

  * Verbose function names.
  * Step-by-step descriptions / explicit parentheses.
  * Chain-of-thought hints.
* This yields different **token densities** for the same step depth.

---

## 4. **Data Collection**

1. For each depth $L$:

   * Generate input-output pairs.
   * Query the model.
   * Record correctness.
   * Record total tokens used.

2. Repeat for multiple seeds (to reduce variance).

3. Store results as:

   * $(\text{depth } L, \text{tokens } T, \text{correct?})$.

---

## 5. **Modeling**

* **Step survival (KM):**

  * Fit Kaplan–Meier survival curve with $L$ as time, event = failure at depth.

* **Token survival (KM):**

  * Fit Kaplan–Meier survival curve with $T$ as time, event = failure at token count.
  * Compare curves across token-density conditions.

* **Comparative analysis:**

  * Overlay step-based and token-based curves.
  * Show that in synthetic tasks, token-based survival is *not more forgiving* (no redundancy benefit).
  * Contrast with expectation in natural tasks (redundancy buys survival).

---

## 6. **Statistical Tests**

* **Log-rank test:** compare survival curves across densities (terse vs verbose).
* **Cox regression:** test effect of verbosity/token density as covariate.
* **Hazard shape analysis:** fit parametric models (Weibull, log-normal) to characterize fragility.

---

## 7. **Deliverables**

* **Plots:**

  * KM survival curve by steps (all conditions).
  * KM survival curve by tokens (all conditions).
  * Overlay to show redundancy effect (or lack thereof).
* **Tables:** hazard ratios for verbosity conditions.
* **Interpretation:**

  * Synthetic → collapse tied to depth, not token count.
  * Establishes **lower bound** for natural dialog survival.

---

✅ This plan cleanly separates **step survival** (pure sequential depth) from **token survival** (depth + verbosity), and shows why redundancy doesn’t help synthetic tasks but does in dialog/software. That becomes a critical contrast in Paper 1.

---

Would you like me to also **mock up a pseudocode pipeline** (data generation → model query → survival analysis) so you can see exactly how you’d implement this experiment in Python?
