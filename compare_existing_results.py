#!/usr/bin/env python3
"""Compare existing experimental results to show survival analysis patterns."""

import pandas as pd
from src.common.io import load_jsonl
from src.survival.prepare import prepare_by_both_time_vars
from src.survival.km import KMAnalyzer

def analyze_experiment(exp_name, description):
    """Analyze a single experiment."""
    try:
        results_file = f"results/synthetic/{exp_name}/raw/results.jsonl"
        raw_results = load_jsonl(results_file)
        df = pd.DataFrame(raw_results)

        print(f"\n{'='*60}")
        print(f"📊 {description}")
        print(f"{'='*60}")
        print(f"Records: {len(df)}")
        print(f"Instances: {df['instance_id'].nunique()}")
        print(f"Model: {df['model'].iloc[0]}")
        print(f"Max depth reached: {df['depth_step'].max()}")

        # Show failure patterns
        print(f"\nFailure reasons:")
        print(df['failure_reason'].value_counts())

        # Show depth distribution
        print(f"\nDepth reached per instance:")
        depth_stats = df.groupby('instance_id')['depth_step'].max()
        print(depth_stats.value_counts().sort_index())

        # Accuracy by depth
        print(f"\nAccuracy by depth:")
        accuracy_by_depth = df.groupby('depth_step')['correct'].agg(['sum', 'count', 'mean'])
        print(accuracy_by_depth)

        # Token growth
        if len(df) > 1:
            print(f"\nToken usage patterns:")
            token_stats = df.groupby('depth_step')['tokens_total'].agg(['min', 'max', 'mean'])
            print(token_stats.head())

        return df

    except Exception as e:
        print(f"❌ {description}: {e}")
        return None

def main():
    print("🔍 Comparing Existing Experimental Results")
    print("This shows the patterns we're looking for in survival analysis")

    experiments = [
        ("qwen_quick_test", "Qwen Math 1.5B - Depth 4 (Perfect Performance)"),
        ("qwen_terse_small", "Qwen Math 1.5B - Depth 8 (Perfect Performance)"),
        ("synthetic_qwen_micro", "Qwen Math 1.5B - Depth 5 (Limited by tokens)"),
        ("synthetic_terse_L30", "GPT-2 - Depth 30 (Immediate failure)"),
    ]

    results = {}
    for exp_name, description in experiments:
        df = analyze_experiment(exp_name, description)
        if df is not None:
            results[exp_name] = df

    print(f"\n{'='*60}")
    print("🎯 KEY INSIGHTS FOR SURVIVAL ANALYSIS")
    print(f"{'='*60}")

    print("1. 📈 WHAT WE WANT TO SEE:")
    print("   - Instances failing at DIFFERENT depths (creating survival curves)")
    print("   - Models pushed until they give wrong answers (not token limits)")
    print("   - Clear relationship between steps and tokens")

    print("\n2. 🔍 CURRENT STATUS:")
    print("   - GPT-2: Fails immediately (too weak)")
    print("   - Qwen Math: Perfect up to depth 8 (too strong for shallow tests)")
    print("   - Need: Generalist models with deeper chains (16-30) to find failure points")

    print("\n3. 🚀 NEXT: Running deeper experiments to generate real survival data!")

if __name__ == "__main__":
    main()