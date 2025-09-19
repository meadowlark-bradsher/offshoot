"""Prepare survival analysis data from raw experimental logs."""

import pandas as pd
from typing import List, Dict, Any, Optional


def prepare_survival_data(
    raw_results: List[Dict[str, Any]],
    time_var: str = "step"
) -> pd.DataFrame:
    """
    Convert raw experimental results to survival analysis format.

    Args:
        raw_results: List of experimental records
        time_var: Either "step" or "tokens_total"

    Returns:
        DataFrame with columns: time_var, event, condition, task_family, seed, model
    """
    df = pd.DataFrame(raw_results)

    if time_var == "step":
        time_col = "depth_step"
    elif time_var == "tokens_total":
        time_col = "tokens_total"
    else:
        raise ValueError(f"Unknown time_var: {time_var}")

    survival_records = []

    for (instance_id, condition, seed, model), group in df.groupby([
        "instance_id", "condition", "seed", "model"
    ]):
        group = group.sort_values(time_col)

        max_time = group[time_col].iloc[-1]
        last_correct = group["correct"].iloc[-1]

        event = 0 if last_correct else 1

        survival_records.append({
            "instance_id": instance_id,
            time_var: max_time,
            "event": event,
            "condition": condition,
            "task_family": group["task_family"].iloc[0],
            "seed": seed,
            "model": model,
            "max_depth": group["depth_max"].iloc[0],
            "failure_reason": group["failure_reason"].iloc[-1] if event else "none",
        })

    return pd.DataFrame(survival_records)


def prepare_by_both_time_vars(
    raw_results: List[Dict[str, Any]]
) -> Dict[str, pd.DataFrame]:
    """Prepare survival data for both step and token time variables."""
    return {
        "step": prepare_survival_data(raw_results, "step"),
        "tokens_total": prepare_survival_data(raw_results, "tokens_total"),
    }


def filter_survival_data(
    df: pd.DataFrame,
    min_time: Optional[int] = None,
    max_time: Optional[int] = None,
    conditions: Optional[List[str]] = None
) -> pd.DataFrame:
    """Filter survival data by time range and conditions."""
    filtered = df.copy()

    if min_time is not None:
        time_col = "step" if "step" in df.columns else "tokens_total"
        filtered = filtered[filtered[time_col] >= min_time]

    if max_time is not None:
        time_col = "step" if "step" in df.columns else "tokens_total"
        filtered = filtered[filtered[time_col] <= max_time]

    if conditions is not None:
        filtered = filtered[filtered["condition"].isin(conditions)]

    return filtered