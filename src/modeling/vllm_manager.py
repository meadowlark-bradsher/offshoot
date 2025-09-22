"""vLLM-based LLM manager for high-performance inference."""

import time
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class GenerationResult:
    """Single generation result with timing info."""
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    generation_time: float


class VLLMManager:
    """High-performance LLM manager using vLLM for batch inference."""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
        **sampling_kwargs
    ):
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.sampling_kwargs = sampling_kwargs

        self.llm = None
        self.tokenizer = None

    def load_model(self) -> None:
        """Load vLLM model and tokenizer."""
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer

            print(f"Loading {self.model_name} with vLLM...")
            start_time = time.time()

            # Initialize vLLM
            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_memory_utilization,
                trust_remote_code=True,
            )

            # Load tokenizer separately for token counting
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            load_time = time.time() - start_time
            print(f"Model loaded in {load_time:.2f} seconds")

        except ImportError:
            raise ImportError(
                "vLLM not installed. Run: pip install vllm"
            )

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        **generation_kwargs
    ) -> List[GenerationResult]:
        """
        Generate completions for a batch of prompts.

        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter

        Returns:
            List of GenerationResult objects
        """
        if self.llm is None:
            self.load_model()

        from vllm import SamplingParams

        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            **generation_kwargs
        )

        # Count input tokens
        prompt_token_counts = [
            len(self.tokenizer.encode(prompt))
            for prompt in prompts
        ]

        # Generate
        start_time = time.time()
        outputs = self.llm.generate(prompts, sampling_params)
        generation_time = time.time() - start_time

        # Process results
        results = []
        for i, output in enumerate(outputs):
            completion = output.outputs[0].text
            completion_tokens = len(output.outputs[0].token_ids)

            result = GenerationResult(
                text=prompts[i] + completion,
                prompt_tokens=prompt_token_counts[i],
                completion_tokens=completion_tokens,
                total_tokens=prompt_token_counts[i] + completion_tokens,
                generation_time=generation_time / len(prompts)  # Average per prompt
            )
            results.append(result)

        return results

    def generate_single(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        **generation_kwargs
    ) -> Tuple[str, int, Dict[str, Any]]:
        """
        Generate completion for a single prompt (compatible with existing API).

        Returns:
            Tuple of (full_text, new_token_count, generation_info)
        """
        results = self.generate_batch(
            [prompt],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **generation_kwargs
        )

        result = results[0]
        generation_info = {
            "prompt_length": result.prompt_tokens,
            "total_length": result.total_tokens,
            "new_tokens": result.completion_tokens,
            "generation_time": result.generation_time,
            "temperature": temperature,
        }

        return result.text, result.completion_tokens, generation_info

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer is None:
            self.load_model()
        return len(self.tokenizer.encode(text))

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging."""
        return {
            "model_name": self.model_name,
            "backend": "vllm",
            "tensor_parallel_size": self.tensor_parallel_size,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
        }


class VLLMGenerationRunner:
    """Generation runner compatible with existing API but using vLLM."""

    def __init__(self, model_name: str, **vllm_kwargs):
        self.vllm_manager = VLLMManager(model_name, **vllm_kwargs)
        self.model_name = model_name

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        **generation_kwargs
    ) -> Tuple[str, int, Dict[str, Any]]:
        """Generate text (compatible with existing GenerationRunner API)."""
        try:
            return self.vllm_manager.generate_single(
                prompt, max_new_tokens, temperature, **generation_kwargs
            )
        except ImportError:
            # If vLLM is not available, raise an error that the pipeline can catch
            raise ImportError("vLLM not available, falling back to HuggingFace backend")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        try:
            if self.vllm_manager.tokenizer is None:
                self.vllm_manager.load_model()
            return self.vllm_manager.count_tokens(text)
        except ImportError:
            # Fallback to rough estimation if vLLM not available
            return int(len(text.split()) * 1.3)

    @property
    def llm_manager(self):
        """Compatibility property for existing code."""
        return self.vllm_manager