"""LLM loading and management."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Dict, Any


class LLMManager:
    """Manages LLM model and tokenizer loading."""

    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        **model_kwargs
    ):
        self.model_name = model_name
        self.device_map = device_map
        self.torch_dtype = torch_dtype or torch.float16
        self.model_kwargs = model_kwargs

        self.tokenizer = None
        self.model = None

    def load_model(self) -> None:
        """Load tokenizer and model."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=self.torch_dtype,
            device_map=self.device_map,
            **self.model_kwargs
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging."""
        return {
            "model_name": self.model_name,
            "torch_dtype": str(self.torch_dtype),
            "device_map": self.device_map,
            "model_kwargs": self.model_kwargs,
        }