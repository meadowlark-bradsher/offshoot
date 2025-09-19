"""Random number generation utilities."""

import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: Optional[int] = None) -> int:
    """Set random seed for reproducibility across libraries."""
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return seed


class SeededRNG:
    """Seeded random number generator for reproducible experiments."""

    def __init__(self, seed: int):
        self.seed = seed
        self.rng = np.random.Generator(np.random.PCG64(seed))

    def randint(self, low: int, high: int) -> int:
        """Generate random integer in [low, high)."""
        return self.rng.integers(low, high)

    def choice(self, options: list):
        """Choose random element from list."""
        return self.rng.choice(options)

    def shuffle(self, items: list) -> list:
        """Return shuffled copy of list."""
        items_copy = items.copy()
        self.rng.shuffle(items_copy)
        return items_copy