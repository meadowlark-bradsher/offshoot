#!/usr/bin/env python3
"""Test script to verify the repository setup works."""

from src.tasks.synthetic.gen_arith import generate_chain_task, calculate_chain_answer
from src.common.rng import SeededRNG

def test_basic_functionality():
    """Test basic functionality without requiring model loading."""
    print("Testing arithmetic chain generation...")

    # Test chain generation
    prompt = generate_chain_task(initial_value=1, depth=3, condition="terse")
    expected = calculate_chain_answer(initial_value=1, depth=3)

    print(f"Generated prompt: {prompt}")
    print(f"Expected answer: {expected}")

    # Test different conditions
    for condition in ["terse", "verbose", "redundant"]:
        prompt = generate_chain_task(initial_value=5, depth=2, condition=condition)
        print(f"\n{condition.upper()} condition:")
        print(f"Prompt: {prompt[:100]}...")

    # Test RNG
    rng = SeededRNG(42)
    print(f"\nRNG test: {rng.randint(1, 10)}")

    print("\n✓ Basic setup verification complete!")

if __name__ == "__main__":
    test_basic_functionality()