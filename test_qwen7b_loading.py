#!/usr/bin/env python3
"""Test if Qwen2.5-7B loads properly on this machine."""

import torch
from src.modeling.llm import LLMManager
from src.modeling.run import GenerationRunner
import time

def test_qwen7b_loading():
    print("Testing Qwen2.5-7B model loading...")
    print(f"MPS available: {torch.backends.mps.is_available()}")

    try:
        print("Loading LLM manager...")
        llm_manager = LLMManager(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            device_map="mps"
        )

        print("Loading model and tokenizer...")
        start_time = time.time()
        llm_manager.load_model()
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")

        print("Creating generation runner...")
        runner = GenerationRunner(llm_manager)

        print("Testing arithmetic generation...")
        start_time = time.time()
        response, tokens, info = runner.generate(
            "Evaluate step by step: Start with 5, then add 3, then multiply by 2. Put the final result inside \\boxed{}.",
            max_new_tokens=256,
            temperature=0.0
        )
        gen_time = time.time() - start_time

        print(f"Generation completed in {gen_time:.2f} seconds")
        print(f"Response: {response}")
        print(f"New tokens: {tokens}")

        print("Testing coding generation...")
        start_time = time.time()
        response, tokens, info = runner.generate(
            "Write a Python function that calculates factorial:",
            max_new_tokens=256,
            temperature=0.0
        )
        gen_time = time.time() - start_time

        print(f"Coding generation completed in {gen_time:.2f} seconds")
        print(f"Response: {response[:200]}...")

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = test_qwen7b_loading()
    if success:
        print("\n✓ Qwen2.5-7B loading test successful!")
    else:
        print("\n✗ Qwen2.5-7B loading test failed!")