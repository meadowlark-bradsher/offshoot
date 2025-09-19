#!/usr/bin/env python3
"""Monitor running experiments by checking log files and results."""

import os
import time
from pathlib import Path

def monitor_experiment(experiment_dir):
    """Monitor a single experiment directory."""
    experiment_path = Path(f"results/synthetic/{experiment_dir}")

    if not experiment_path.exists():
        return f"{experiment_dir}: Directory not found"

    log_file = experiment_path / "experiment.log"
    results_file = experiment_path / "raw" / "results.jsonl"

    status = f"{experiment_dir}:"

    # Check log file
    if log_file.exists():
        with open(log_file, 'r') as f:
            lines = f.readlines()
            if lines:
                last_line = lines[-1].strip()
                if "completed successfully" in last_line:
                    status += " ✅ COMPLETED"
                elif "Starting synthetic experiment" in last_line:
                    status += " 🔄 RUNNING"
                else:
                    status += f" 📝 LOG: {last_line[-50:]}"

    # Check results file
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                record_count = sum(1 for _ in f)
            status += f" ({record_count} records)"
        except:
            status += " (results file exists but couldn't read)"

    return status

def main():
    """Monitor all experiment directories."""
    print("🔍 Monitoring experiments...")
    print("=" * 60)

    # Look for experiment directories
    results_dir = Path("results/synthetic")
    if not results_dir.exists():
        print("No results directory found")
        return

    experiment_dirs = [d.name for d in results_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

    if not experiment_dirs:
        print("No experiment directories found")
        return

    for exp_dir in sorted(experiment_dirs):
        status = monitor_experiment(exp_dir)
        print(status)

    print("=" * 60)

if __name__ == "__main__":
    main()