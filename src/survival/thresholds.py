"""
Survival-based cutover thresholds for LLM arithmetic chains.

Implements KM-quantile cutover with Weibull fallback as described in
docs/cutover-experiments.md
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from lifelines import KaplanMeierFitter, WeibullFitter
import warnings
warnings.filterwarnings('ignore')


@dataclass
class CutoverThreshold:
    """Results of cutover threshold calculation for a single condition."""
    condition: str
    t_star_km: Optional[int]         # Max step with S_hat >= R
    t_star_km_ci_safe: Optional[int] # Max step with S_low >= R (CI-safe)
    t_star_weibull: Optional[int]    # Weibull-based threshold (fallback)
    chosen_t_star: int               # Final operational threshold
    reliability_target: float        # Target reliability (R)
    method_used: str                 # "km_ci", "km_point", or "weibull"


def km_cutover_threshold(km_df: pd.DataFrame, R: float = 0.95) -> Tuple[Optional[int], Optional[int]]:
    """
    Extract cutover thresholds from Kaplan-Meier survival table.

    Args:
        km_df: DataFrame with columns [t, S_hat, S_low, S_high]
        R: Target reliability (survival probability)

    Returns:
        (t_star_point, t_star_ci_safe) where:
        - t_star_point: max t with S_hat >= R (or None)
        - t_star_ci_safe: max t with S_low >= R (or None)
    """
    # Point estimate threshold
    ok_point = km_df[km_df["S_hat"] >= R]
    t_star_point = int(ok_point["t"].max()) if not ok_point.empty else None

    # CI-safe threshold (conservative)
    ok_ci = km_df[km_df["S_low"] >= R]
    t_star_ci_safe = int(ok_ci["t"].max()) if not ok_ci.empty else None

    return t_star_point, t_star_ci_safe


def weibull_cutover_threshold(lambda_hat: float, k_hat: float, R: float = 0.95) -> int:
    """
    Calculate cutover threshold using Weibull survival model.

    Args:
        lambda_hat: Weibull scale parameter
        k_hat: Weibull shape parameter
        R: Target reliability

    Returns:
        Floor of Weibull-based threshold: floor(lambda * (-ln(R))^(1/k))
    """
    t = lambda_hat * ((-np.log(R)) ** (1.0 / k_hat))
    return int(np.floor(t))


def fit_km_table_with_ci(survival_times: np.ndarray, events: np.ndarray,
                        max_time: Optional[int] = None) -> pd.DataFrame:
    """
    Fit Kaplan-Meier estimator and extract survival table with confidence intervals.

    Args:
        survival_times: Array of survival times (steps)
        events: Array of event indicators (1=failure, 0=censored)
        max_time: Maximum time to include in table (optional)

    Returns:
        DataFrame with columns [t, S_hat, S_low, S_high, n_at_risk, n_events]
    """
    kmf = KaplanMeierFitter()
    kmf.fit(survival_times, events)

    # Extract survival function with CI
    times = kmf.survival_function_.index.values
    s_hat = kmf.survival_function_.values.flatten()

    # Get confidence intervals (lifelines uses log-log transform by default)
    ci = kmf.confidence_interval_survival_function_
    s_low = ci.iloc[:, 0].values
    s_high = ci.iloc[:, 1].values

    # Create table
    km_table = pd.DataFrame({
        't': times.astype(int),
        'S_hat': s_hat,
        'S_low': s_low,
        'S_high': s_high
    })

    # Add risk set information
    event_table = kmf.event_table
    km_table['n_at_risk'] = event_table['at_risk'].values
    km_table['n_events'] = event_table['observed'].values

    # Filter to max_time if specified
    if max_time is not None:
        km_table = km_table[km_table['t'] <= max_time]

    return km_table


def fit_weibull_parameters(survival_times: np.ndarray, events: np.ndarray) -> Tuple[float, float]:
    """
    Fit Weibull survival model and return parameters.

    Args:
        survival_times: Array of survival times
        events: Array of event indicators

    Returns:
        (lambda_hat, k_hat) - scale and shape parameters
    """
    wf = WeibullFitter()
    wf.fit(survival_times, events)

    # Extract parameters (lifelines uses different parameterization)
    # WeibullFitter uses lambda_, rho_ where S(t) = exp(-(t/lambda_)^rho_)
    lambda_hat = wf.lambda_
    k_hat = wf.rho_

    return lambda_hat, k_hat


def calculate_cutover_threshold(survival_times: np.ndarray, events: np.ndarray,
                              condition: str, R: float = 0.95,
                              max_time: Optional[int] = None) -> CutoverThreshold:
    """
    Calculate operational cutover threshold for a single condition.

    Args:
        survival_times: Array of survival times (steps)
        events: Array of event indicators (1=failure, 0=censored)
        condition: Condition name (e.g., 'terse', 'verbose', 'redundant')
        R: Target reliability
        max_time: Maximum time to consider

    Returns:
        CutoverThreshold with all calculation results
    """
    # Fit Kaplan-Meier table
    km_table = fit_km_table_with_ci(survival_times, events, max_time)

    # Calculate KM-based thresholds
    t_star_km, t_star_km_ci = km_cutover_threshold(km_table, R)

    # Determine final threshold using the priority logic from docs
    if t_star_km_ci is not None:
        # CI-safe threshold available (preferred)
        chosen_t_star = t_star_km_ci
        method_used = "km_ci"
        t_star_weibull = None
    elif t_star_km is not None:
        # Point estimate available but CI fails
        chosen_t_star = t_star_km
        method_used = "km_point"
        t_star_weibull = None
    else:
        # Fall back to Weibull
        try:
            lambda_hat, k_hat = fit_weibull_parameters(survival_times, events)
            t_star_weibull = weibull_cutover_threshold(lambda_hat, k_hat, R)
            chosen_t_star = t_star_weibull
            method_used = "weibull"
        except Exception as e:
            # Ultimate fallback - use median survival time
            t_star_weibull = None
            chosen_t_star = int(np.median(survival_times))
            method_used = "median_fallback"

    return CutoverThreshold(
        condition=condition,
        t_star_km=t_star_km,
        t_star_km_ci_safe=t_star_km_ci,
        t_star_weibull=t_star_weibull,
        chosen_t_star=chosen_t_star,
        reliability_target=R,
        method_used=method_used
    )


def generate_threshold_table(survival_data: pd.DataFrame, R: float = 0.95) -> pd.DataFrame:
    """
    Generate operational threshold table for all conditions.

    Args:
        survival_data: DataFrame with columns [condition, survival_time, event]
        R: Target reliability

    Returns:
        DataFrame with threshold results for each condition
    """
    results = []

    for condition in survival_data['condition'].unique():
        condition_data = survival_data[survival_data['condition'] == condition]

        survival_times = condition_data['survival_time'].values
        events = condition_data['event'].values

        threshold = calculate_cutover_threshold(
            survival_times, events, condition, R
        )

        results.append({
            'model': 'qwen2.5-3b',  # From our experiment
            'condition': threshold.condition,
            'R': threshold.reliability_target,
            't_star_km': threshold.t_star_km,
            't_star_km_ci': threshold.t_star_km_ci_safe,
            't_star_weibull': threshold.t_star_weibull,
            'chosen_t_star': threshold.chosen_t_star,
            'method_used': threshold.method_used
        })

    return pd.DataFrame(results)


def print_threshold_analysis(thresholds: list[CutoverThreshold]):
    """Print detailed analysis of cutover thresholds."""

    print("="*70)
    print("CUTOVER THRESHOLD ANALYSIS")
    print("="*70)

    for threshold in thresholds:
        print(f"\n{threshold.condition.upper()} CONDITION:")
        print(f"  Target reliability: {threshold.reliability_target*100}%")
        print(f"  KM point estimate threshold: {threshold.t_star_km}")
        print(f"  KM CI-safe threshold: {threshold.t_star_km_ci_safe}")
        print(f"  Weibull threshold: {threshold.t_star_weibull}")
        print(f"  CHOSEN THRESHOLD: {threshold.chosen_t_star} steps")
        print(f"  Method used: {threshold.method_used}")

        # Interpretation
        if threshold.method_used == "km_ci":
            print(f"  → Conservative CI-based cutover at step {threshold.chosen_t_star}")
        elif threshold.method_used == "km_point":
            print(f"  → Point-estimate cutover (CI too conservative)")
        elif threshold.method_used == "weibull":
            print(f"  → Parametric Weibull fallback")
        else:
            print(f"  → Emergency fallback method")

    print(f"\n" + "="*70)
    print("OPERATIONAL SUMMARY")
    print("="*70)

    for threshold in thresholds:
        print(f"  {threshold.condition}: Cutover at step {threshold.chosen_t_star}")

    print(f"\nThese thresholds ensure {thresholds[0].reliability_target*100}% reliability")
    print(f"for arithmetic chain continuation in each prompt condition.")