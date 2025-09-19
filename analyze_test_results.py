#!/usr/bin/env python3
"""Analyze experimental results with configurable input."""

import argparse
from pathlib import Path
from src.common.io import load_jsonl
from src.survival.prepare import prepare_by_both_time_vars
from src.survival.km import KMAnalyzer
import pandas as pd

def analyze_results(results_path: str):
    """Analyze experimental results from a given path."""
    results_file = Path(results_path)

    if results_file.is_dir():
        # If directory provided, look for results.jsonl in raw/ subfolder
        results_file = results_file / "raw" / "results.jsonl"

    if not results_file.exists():
        print(f"Error: Results file not found at {results_file}")
        return

    print(f"Loading experimental results from: {results_file}")
    raw_results = load_jsonl(results_file)
    print(f"Loaded {len(raw_results)} records")

    if len(raw_results) == 0:
        print("No results to analyze")
        return

    print("\nRaw results summary:")
    df = pd.DataFrame(raw_results)

    # Basic statistics
    print(f"Unique instances: {df['instance_id'].nunique()}")
    print(f"Conditions: {df['condition'].unique()}")
    print(f"Models: {df['model'].unique()}")
    print(f"Max depth reached: {df['depth_step'].max()}")

    print("\nFailure reasons:")
    print(df['failure_reason'].value_counts())

    print("\nDepth reached per instance:")
    depth_stats = df.groupby('instance_id')['depth_step'].max()
    print(depth_stats.value_counts().sort_index())

    print("\nCorrect answers by depth:")
    correct_by_depth = df.groupby('depth_step')['correct'].sum()
    total_by_depth = df.groupby('depth_step').size()
    accuracy_by_depth = correct_by_depth / total_by_depth

    accuracy_df = pd.DataFrame({
        'correct': correct_by_depth,
        'total': total_by_depth,
        'accuracy': accuracy_by_depth
    })
    print(accuracy_df)

    # Show sample responses
    print("\nSample successful responses (if any):")
    successful = df[df['correct'] == True]
    if len(successful) > 0:
        for _, row in successful.head(3).iterrows():
            print(f"Depth {row['depth_step']}: {row['raw_response'][:100]}...")
    else:
        print("No successful responses found")

    print("\nSample failed responses:")
    failed = df[df['correct'] == False]
    if len(failed) > 0:
        for _, row in failed.head(2).iterrows():
            print(f"Depth {row['depth_step']} ({row['failure_reason']}): {row['raw_response'][:100]}...")

    # Survival analysis if we have varying depths
    if df['depth_step'].nunique() > 1:
        print("\nPreparing survival data...")
        survival_data = prepare_by_both_time_vars(raw_results)

        print("\nStep-based survival data:")
        step_data = survival_data["step"]
        print(f"Events: {step_data['event'].sum()}/{len(step_data)}")
        print(f"Median survival time (steps): {step_data[step_data['event']==1]['step'].median()}")

        print("\nToken-based survival data:")
        token_data = survival_data["tokens_total"]
        print(f"Events: {token_data['event'].sum()}/{len(token_data)}")
        print(f"Median survival time (tokens): {token_data[token_data['event']==1]['tokens_total'].median()}")
    else:
        print("\nAll instances failed at the same depth - no survival curve possible")

def main():
    parser = argparse.ArgumentParser(description="Analyze experimental results")
    parser.add_argument(
        "results_path",
        help="Path to results file (.jsonl) or experiment directory"
    )
    args = parser.parse_args()

    analyze_results(args.results_path)

if __name__ == "__main__":
    main()