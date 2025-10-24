import math
import random
from typing import Any, Literal

from .base import Sampler


class RandomSampler(Sampler):
    """Random sampler that draws uniform random samples."""

    def __init__(self, seed: int | None = None) -> None:
        """Initialize the random sampler.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = random.Random(seed)
        self.record: dict[str, Any] = {}

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        low_inclusive: bool,
        high_inclusive: bool,
        distribution: Literal["log", "uniform"],
    ) -> int:
        """Suggest a random integer."""
        # Adjust bounds based on inclusivity
        if not low_inclusive:
            low = low + 1
        if not high_inclusive:
            high = high - 1

        if distribution == "uniform":
            self.record[name] = self.rng.randint(low, high)
            return self.record[name]
        elif distribution == "log":
            # Log-uniform sampling
            assert low > 0
            log_low = math.log(low)
            log_high = math.log(high)
            log_value = self.rng.uniform(log_low, log_high)
            self.record[name] = int(round(math.exp(log_value)))
            return self.record[name]
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        low_inclusive: bool,
        high_inclusive: bool,
        distribution: Literal["log", "uniform"],
    ) -> float:
        """Suggest a random float."""
        if not low_inclusive:
            low = low + 1e-10
        if not high_inclusive:
            high = high - 1e-10

        if distribution == "uniform":
            self.record[name] = self.rng.uniform(low, high)
            return self.record[name]
        elif distribution == "log":
            # Log-uniform sampling
            assert low > 0
            log_low = math.log(low)
            log_high = math.log(high)
            log_value = self.rng.uniform(log_low, log_high)
            self.record[name] = math.exp(log_value)
            return self.record[name]
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

    def suggest_categorical(
        self,
        name: str,
        choices: list[Any],
        weights: list[float],
    ) -> Any:
        """Suggest a random categorical choice."""
        self.record[name] = self.rng.choices(choices, weights=weights, k=1)[0]
        return self.record[name]
