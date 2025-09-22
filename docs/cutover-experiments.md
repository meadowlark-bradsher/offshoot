---

# Experiment 1 — KM-quantile cutover (no early warning)

## What you need as input

From your synthetic runs (per *condition*: terse / verbose / redundant):

* A survival table by **step**:

  * `t` = composition step (1, 2, …)
  * `n_at_risk[t]`, `n_events[t]`, `n_censored[t]`
  * KM estimate `S_hat[t]`
  * 95% CI for `S_hat[t]` using **log-log CI** (recommended):

    $$
    \text{CI}_{95\%}(t) = \Big[\,S_{\text{low}}(t),\ S_{\text{high}}(t)\,\Big]
    $$

    where
    $se\{\log(-\log S)\} \approx \frac{\sqrt{\widehat{\text{Var}}(S)}}{S\,\left|\log S\right|}$,
    $\widehat{\text{Var}}(S)$ from Greenwood,
    then invert with log-log to get bounds in probability space.

> In code you’ll just use the CI that `lifelines` gives for KM with `alpha=0.05` and `ci_show=True`.

## Cutover rule (target reliability $R=0.95$)

1. **Find the largest step $t^\*$** such that **both**:

   * $\widehat{S}(t^\*) \ge R$ (point estimate meets reliability), and
   * $S_{\text{low}}(t^\*) \ge R$ (lower 95% CI also at/above $R$).
2. If **no step** satisfies the CI condition (common when aiming high reliability), **fall back to Weibull** (below), then **round down** to the nearest observed step (floor).

> Using the **lower CI** in the rule bakes your “with CI” requirement directly into the threshold. It’s conservative and reproducible.

## Weibull fallback (smooth inversion)

Fit a Weibull $S(t)=\exp\{-(t/\lambda)^k\}$ to the step times (right-censor as usual). Then:

$$
t^\*_{\text{Weibull}}(R)=\lambda\,[-\ln R]^{1/k}
$$

* **Operational threshold:** $t^\* \leftarrow \big\lfloor t^\*_{\text{Weibull}}(R)\big\rfloor$.
* (Optional) CI for $t^\*$: use parameter covariance from the fit or bootstrap 200×; report the **lower** bound to be safe.

## How to apply at runtime

* Keep a per-condition threshold table:

  ```
  condition, t_star_steps
  terse,     …
  verbose,   …
  redundant, …
  ```
* During generation, **summarize / recontextualize** once the current step count reaches `t_star_steps` (or sooner if you adopt experiment 2’s early warning).

## Minimal code you can drop in (KM + fallback)

Create `src/survival/thresholds.py`:

```python
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class CutoverThreshold:
    t_star_km: int | None
    t_star_km_ci_safe: int | None
    t_star_weibull: int | None
    chosen_t_star: int

def km_cutover_threshold(km_df: pd.DataFrame, R: float = 0.95) -> tuple[int | None, int | None]:
    """
    km_df columns: t, S_hat, S_low, S_high  (per condition)
    Returns:
      - t_star (point-est. only): max t with S_hat >= R (or None)
      - t_star_ci (CI-safe): max t with S_low >= R (or None)
    """
    ok_point = km_df[km_df["S_hat"] >= R]
    t_star = int(ok_point["t"].max()) if not ok_point.empty else None

    ok_ci = km_df[km_df["S_low"] >= R]
    t_star_ci = int(ok_ci["t"].max()) if not ok_ci.empty else None
    return t_star, t_star_ci

def weibull_cutover_threshold(lambda_hat: float, k_hat: float, R: float = 0.95) -> int:
    t = lambda_hat * ((-np.log(R)) ** (1.0 / k_hat))
    return int(np.floor(t))
```

In your pipeline (after fitting KM per condition and, if needed, Weibull):

```python
t_star, t_star_ci = km_cutover_threshold(km_table, R=0.95)
if t_star_ci is not None:
    chosen = t_star_ci
elif t_star is not None:
    # optional: if you accept point-estimate when CI misses by a hair
    chosen = t_star
else:
    # fit Weibull -> lambda_hat, k_hat
    chosen = weibull_cutover_threshold(lambda_hat, k_hat, R=0.95)
```

---

# Experiment 2 — Transfer to **qwen-math** with an early-warning override

You keep the **same KM 95% thresholding** as the backbone, but add a **lightweight early-warning** that triggers a cutover **before** $t^\*$ when the model starts **emulating tool calls** (your observed stress behavior).

## Early-warning signal (example)

Define a simple detector on each turn’s text:

* **Tool-call emulation regex** (tune to what you see in qwen-math logs):

  * Lines like `>>>`, `CALL(`, `tool:`, `# Tool`, fenced blocks that look like function calls, or JSON “args/func” stubs.
* **Firing rule:** the signal is “on” if **any** of the last `w=2` turns match the pattern, **and** the current turn matches too (i.e., persistent onset, not a one-off).

## Combined policy

Cutover time (in steps) is:

$$
\text{cutover} \;=\; \min\left(\; t^\*_{\text{KM/Weibull}},\; t_{\text{first persistent early-warning}} \;\right).
$$

## How to select the early-warning parameters

* Hold out \~20–30 dialogs (or synthetic chains run with qwen-math).
* Sweep `w ∈ {1,2,3}` and optionally require ≥2 matches in the last 3 turns (to reduce false alarms).
* Choose the setting that **maximizes reliability at the chosen target** (e.g., 95%) with the smallest median cutover loss (i.e., you don’t summarize too early).

> Since this is a clean function-composition challenge, you won’t have IG or repeat-question features; the tool-call emulation is your sole signal.

---

# Good practice notes (both experiments)

* **Pick one clock for Paper-1:** you said “time = steps,” so keep tokens out of the decision rule for this experiment to avoid conflating with redundancy. You can still report token curves descriptively.
* **Report the conservative choice:** always cite the **CI-safe** KM threshold; if you used Weibull, say so and include the parametric CI or a bootstrap band.
* **One threshold per condition & per model family:** keep a small CSV that your runtime reads.
* **Hysteresis:** even in Exp-2, add a short cooldown after a cutover (e.g., don’t allow another summarize for 3–5 steps) to avoid thrash.

---

# What to drop into your repo

* `src/survival/thresholds.py` (above)
* `src/pipeline/run_synthetic.py`

  * produce KM tables (`t, S_hat, S_low, S_high`) per condition
  * fit Weibull as needed
  * emit `results/synthetic/thresholds_steps_R95.csv`
* `src/pipeline/run_qwen_math.py` (Exp-2)

  * same as above **plus** early-warning detector
  * emit thresholds + an ablation table showing reliability gain from early-warning

Example output CSV:

```
model,condition,R,t_star_km,t_star_km_ci,t_star_weibull,chosen_t_star
qwen2.5,terse,0.95,7,6,6,6
qwen2.5,verbose,0.95,8,7,7,7
qwen-math,terse,0.95,5,4,4, min(4, ew=3)->3
...
```

---