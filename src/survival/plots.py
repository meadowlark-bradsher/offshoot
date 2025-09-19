"""Plotting utilities for survival analysis."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from lifelines import KaplanMeierFitter

plt.style.use('default')
sns.set_palette("husl")


def plot_km_curves(
    fitted_models: Dict[str, KaplanMeierFitter],
    title: str = "Kaplan-Meier Survival Curves",
    xlabel: str = "Time",
    ylabel: str = "Survival Probability",
    confidence_intervals: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot Kaplan-Meier survival curves for multiple groups."""
    fig, ax = plt.subplots(figsize=figsize)

    for group_name, km in fitted_models.items():
        km.plot_survival_function(
            ax=ax,
            ci_show=confidence_intervals,
            label=group_name
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_survival_comparison(
    survival_tables: Dict[str, pd.DataFrame],
    title: str = "Survival Comparison: Steps vs Tokens",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot side-by-side comparison of survival curves by different time variables."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    time_vars = list(survival_tables.keys())

    for idx, (time_var, table) in enumerate(survival_tables.items()):
        ax = axes[idx]

        for group in table['group'].unique():
            group_data = table[table['group'] == group]
            ax.plot(
                group_data['time'],
                group_data['survival_prob'],
                label=group,
                marker='o',
                markersize=3
            )

        ax.set_title(f"Survival by {time_var.title()}")
        ax.set_xlabel(time_var.title())
        ax.set_ylabel("Survival Probability")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_hazard_analysis(
    df: pd.DataFrame,
    time_col: str,
    event_col: str = "event",
    group_col: str = "condition",
    title: str = "Event Analysis",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot hazard and event distribution analysis."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    axes[0, 0].hist(
        [df[df[group_col] == group][time_col] for group in df[group_col].unique()],
        bins=20,
        alpha=0.7,
        label=df[group_col].unique()
    )
    axes[0, 0].set_title("Distribution of Event Times")
    axes[0, 0].set_xlabel(time_col)
    axes[0, 0].legend()

    event_counts = df.groupby([group_col, event_col]).size().unstack(fill_value=0)
    event_counts.plot(kind='bar', ax=axes[0, 1])
    axes[0, 1].set_title("Event Counts by Group")
    axes[0, 1].set_xlabel("Group")
    axes[0, 1].legend(["Censored", "Event"])

    median_times = df.groupby(group_col)[time_col].median()
    median_times.plot(kind='bar', ax=axes[1, 0])
    axes[1, 0].set_title("Median Time by Group")
    axes[1, 0].set_xlabel("Group")

    df.boxplot(column=time_col, by=group_col, ax=axes[1, 1])
    axes[1, 1].set_title("Time Distribution by Group")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def create_summary_plot(
    raw_data: List[Dict[str, Any]],
    survival_tables: Dict[str, pd.DataFrame],
    output_dir: str
) -> None:
    """Create comprehensive summary plots for an experiment."""
    df = pd.DataFrame(raw_data)

    plot_hazard_analysis(
        survival_tables["step"],
        time_col="step",
        save_path=f"{output_dir}/hazard_analysis_steps.png"
    )

    plot_hazard_analysis(
        survival_tables["tokens_total"],
        time_col="tokens_total",
        save_path=f"{output_dir}/hazard_analysis_tokens.png"
    )

    plot_survival_comparison(
        survival_tables,
        save_path=f"{output_dir}/survival_comparison.png"
    )