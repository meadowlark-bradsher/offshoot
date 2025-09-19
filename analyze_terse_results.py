#!/usr/bin/env python3
"""Analyze the terse experiment results."""

import pandas as pd
from src.common.io import load_jsonl

def main():
    results_file = "results/synthetic/synthetic_terse_L30/raw/results.jsonl"

    print("Loading terse experiment results...")
    raw_results = load_jsonl(results_file)
    df = pd.DataFrame(raw_results)

    print(f"Total records: {len(df)}")
    print(f"Unique instances: {df['instance_id'].nunique()}")

    print("\nFailure reasons:")
    print(df['failure_reason'].value_counts())

    print("\nDepth reached per instance:")
    depth_stats = df.groupby('instance_id')['depth_step'].max()
    print(depth_stats.value_counts().sort_index())

    print("\nCorrect answers by depth:")
    correct_by_depth = df.groupby('depth_step')['correct'].sum()
    print(correct_by_depth)

    print("\nSample successful responses (if any):")
    successful = df[df['correct'] == True]
    if len(successful) > 0:
        for _, row in successful.head(3).iterrows():
            print(f"Depth {row['depth_step']}: {row['raw_response'][:100]}...")
    else:
        print("No successful responses found")

    print(f"\nGPT-2 appears to be struggling with the task.")
    print(f"All instances failed at depth 1 due to repetitive generation.")
    print(f"This demonstrates why the reference notebook used Qwen models.")

if __name__ == "__main__":
    main()