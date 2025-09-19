#!/usr/bin/env python3
"""Test if Qwen model loads properly on this machine."""

import torch
from src.modeling.llm import LLMManager
from src.modeling.run import GenerationRunner
import time

def test_qwen_loading():
    print("Testing Qwen model loading...")
    print(f"MPS available: {torch.backends.mps.is_available()}")

    try:
        print("Loading LLM manager...")
        llm_manager = LLMManager(
            model_name="Qwen/Qwen2.5-Math-1.5B",
            device_map="mps"
        )

        print("Loading model and tokenizer...")
        start_time = time.time()
        llm_manager.load_model()
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")

        print("Creating generation runner...")
        runner = GenerationRunner(llm_manager)

        print("Testing simple generation...")
        start_time = time.time()
        response, tokens, info = runner.generate(
            "What is 2 + 2?",
            max_new_tokens=32,
            temperature=0.0
        )
        gen_time = time.time() - start_time

        print(f"Generation completed in {gen_time:.2f} seconds")
        print(f"Response: {response}")
        print(f"New tokens: {tokens}")

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = test_qwen_loading()
    if success:
        print("\n✓ Qwen model loading test successful!")
    else:
        print("\n✗ Qwen model loading test failed!")