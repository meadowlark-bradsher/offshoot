"""Kaplan-Meier survival analysis utilities."""

import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from typing import Dict, Any, List, Optional, Tuple


class KMAnalyzer:
    """Kaplan-Meier survival analysis for experimental data."""

    def __init__(self):
        self.fitted_models = {}
        self.survival_data = {}

    def fit_survival_curves(
        self,
        df: pd.DataFrame,
        time_col: str,
        event_col: str = "event",
        group_col: str = "condition"
    ) -> Dict[str, KaplanMeierFitter]:
        """
        Fit Kaplan-Meier curves for each group.

        Returns dictionary of fitted KM models by group.
        """
        fitted_models = {}

        for group_name, group_data in df.groupby(group_col):
            km = KaplanMeierFitter()
            km.fit(
                durations=group_data[time_col],
                event_observed=group_data[event_col],
                label=group_name
            )
            fitted_models[group_name] = km

        self.fitted_models[group_col] = fitted_models
        return fitted_models

    def get_survival_table(
        self,
        group_col: str = "condition",
        time_points: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Get survival probabilities at specified time points."""
        if group_col not in self.fitted_models:
            raise ValueError(f"No fitted models for group_col: {group_col}")

        models = self.fitted_models[group_col]
        records = []

        for group_name, km in models.items():
            if time_points is None:
                survival_probs = km.survival_function_
                for time_idx, row in survival_probs.iterrows():
                    records.append({
                        "group": group_name,
                        "time": time_idx,
                        "survival_prob": row[group_name],
                        "at_risk": km.durations.index[km.durations >= time_idx].size
                    })
            else:
                for time_point in time_points:
                    try:
                        survival_prob = km.survival_function_at_times(time_point).iloc[0]
                        at_risk = (km.durations >= time_point).sum()
                        records.append({
                            "group": group_name,
                            "time": time_point,
                            "survival_prob": survival_prob,
                            "at_risk": at_risk
                        })
                    except:
                        pass

        return pd.DataFrame(records)

    def compare_groups(
        self,
        df: pd.DataFrame,
        time_col: str,
        event_col: str = "event",
        group_col: str = "condition",
        groups: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform log-rank test to compare survival curves between groups.
        """
        if groups is None:
            groups = df[group_col].unique().tolist()

        if len(groups) != 2:
            raise ValueError("Currently only supports pairwise comparisons")

        group1_data = df[df[group_col] == groups[0]]
        group2_data = df[df[group_col] == groups[1]]

        results = logrank_test(
            durations_A=group1_data[time_col],
            durations_B=group2_data[time_col],
            event_observed_A=group1_data[event_col],
            event_observed_B=group2_data[event_col]
        )

        return {
            "test_statistic": results.test_statistic,
            "p_value": results.p_value,
            "groups_compared": groups,
            "null_hypothesis": "No difference in survival curves",
            "significant": results.p_value < 0.05
        }

    def get_median_survival(
        self,
        group_col: str = "condition"
    ) -> Dict[str, float]:
        """Get median survival time for each group."""
        if group_col not in self.fitted_models:
            raise ValueError(f"No fitted models for group_col: {group_col}")

        median_survivals = {}
        for group_name, km in self.fitted_models[group_col].items():
            try:
                median_survivals[group_name] = km.median_survival_time_
            except:
                median_survivals[group_name] = np.inf

        return median_survivals