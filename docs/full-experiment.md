Here’s the **updated Paper 1 summary** with the token/turn redundancy dimension integrated:

---

# **Paper 1: Survival Modeling of LLM Context Degradation**

## **Motivation**

Large language models (LLMs) degrade as dialog grows, manifesting as forgetting, hallucination, or collapse into uninformative responses.

* In **synthetic tasks**, failure is easy to measure but comes quickly (low redundancy).
* In **software and dialog tasks**, failures emerge later, with redundancy and clarifications buffering degradation.
* The core hypothesis: **survival dynamics learned from synthetic tasks can proxy natural dialog survival**, and redundancy explains why real dialogs survive longer.
* This motivates **summarization/recontextualization cutovers**: survival models can identify where in the curve cutovers preserve performance.

---

## **Methods**

1. **Survival analysis framework**

   * **Kaplan–Meier (KM)** survival curves in synthetic tasks (single risk = first wrong answer).
   * **Cox proportional hazards / parametric models** to quantify effects of task “noisiness” or verbosity.
   * **Fine–Gray competing risks** introduced as the natural extension for multi-risk settings (e.g., dialog collapse vs win vs lose).

2. **Three levels of evidence**

   * **Synthetic (verifiable):** pointer-chasing/arithmetic tasks. Failure = first wrong answer.
   * **Software (semi-objective):** incremental coding/spec tasks, failure = first unmet test/spec.
   * **Dialog (qualitative):** multi-turn Q\&A, failure = judged degradation (hallucination, contradiction, forgetting).

3. **Turns vs Tokens**

   * **Turns = sequential depth.** Cleanest survival clock for synthetic tasks.
   * **Tokens = mixed signal.** Captures both sequential depth and redundancy/context support.
   * Synthetic tasks: *tokens don’t help* (no redundancy), so collapse can occur early despite low token counts.
   * Dialog/software tasks: *tokens help* (redundancy buys depth), so collapse occurs later in token space.
   * The **gap between turn-based and token-based curves** is itself informative — it shows redundancy’s contribution to survival.

---

## **Experiment Design**

### Level 1: Synthetic

* Generate pointer chains/arithmetic with varying depths.
* Fit KM survival curves (turns and tokens).
* Show sharp collapse in both, but token curves don’t confer extra survival.

### Level 2: Software

* Use iterative coding/spec tasks (HumanEval+constraints, SWE-bench).
* Track failures by turns and tokens.
* Show that token-based survival curves flatten relative to turn curves, reflecting redundancy from clarifications/tests.

### Level 3: Dialog

* Collect Q\&A/chat logs, mark first hallucination/contradiction.
* Fit survival curves by turns and tokens.
* Show that verbosity increases token counts but also buys robustness, flattening hazard.

### Comparative Analysis

* Demonstrate that synthetic KM curves = *lower bounds* (fragility with no redundancy).
* Software and dialog curves show redundancy moderates hazard, visible in token-based survival.
* Establish statistical continuity: synthetic → software → dialog as an evidentiary ladder toward Fine–Gray cutover.

---

## **Positioning**

* **Paper 1 (this one):** Develops survival modeling as a general framework, compares turns vs tokens, and shows redundancy as a moderator of survival hazard.
* **Paper 2 (future):** Applies Fine–Gray concretely to GPT-2 in 20Q, where the event of interest is *strategic collapse into low-IG questions*.

---

✅ This version makes explicit how **token survival curves matter not as “the clock,” but as a diagnostic lens for redundancy** — reinforcing the argument that synthetic results are lower bounds on real dialog survival.

---

Would you like me to now **draft the abstract-style summary** (≈200 words) for Paper 1, with this redundancy/turns-tokens insight baked in, so you can reuse it as a seed for a conference submission?
