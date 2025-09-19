"""Unified generation API with streaming token capture."""

import torch
from typing import Tuple, Dict, Any, Optional
from .llm import LLMManager


class GenerationRunner:
    """Handles model generation with token counting."""

    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        if self.llm_manager.model is None or self.llm_manager.tokenizer is None:
            self.llm_manager.load_model()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        do_sample: Optional[bool] = None,
        **generation_kwargs
    ) -> Tuple[str, int, Dict[str, Any]]:
        """
        Generate text with token counting.

        Returns:
            Tuple of (generated_text, new_token_count, generation_info)
        """
        if do_sample is None:
            do_sample = temperature > 0.0

        inputs = self.llm_manager.tokenizer(
            prompt, return_tensors="pt"
        ).to(self.llm_manager.model.device)

        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output = self.llm_manager.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                **generation_kwargs
            )

        full_text = self.llm_manager.tokenizer.decode(
            output[0], skip_special_tokens=True
        )

        total_len = output[0].shape[0]
        new_tokens = total_len - prompt_len

        generation_info = {
            "prompt_length": prompt_len,
            "total_length": total_len,
            "new_tokens": new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
            "max_new_tokens": max_new_tokens,
        }

        return full_text, new_tokens, generation_info

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the model's tokenizer."""
        return len(self.llm_manager.tokenizer.encode(text))