"""Tokenization utilities."""

from transformers import AutoTokenizer
from typing import List, Tuple


class TokenCounter:
    """Token counting utility using HuggingFace tokenizers."""

    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))

    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """Count tokens for a batch of texts."""
        return [self.count_tokens(text) for text in texts]

    def encode_decode_pair(self, text: str) -> Tuple[List[int], str]:
        """Encode text to tokens and decode back for verification."""
        tokens = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return tokens, decoded