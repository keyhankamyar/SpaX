"""Random sampler for uniform and log-uniform parameter sampling.

This module provides RandomSampler, which generates parameter values using
Python's random module. It supports both uniform and log-uniform distributions.
"""

import math
import random
from typing import Any, Literal

from .base import Sampler


class RandomSampler(Sampler):
    """Sampler that generates random parameter values.

    RandomSampler uses Python's random module to generate parameter values
    with uniform or log-uniform distributions. It maintains an internal
    random number generator that can be seeded for reproducibility.

    This is the default sampler used by Config.random() and is suitable
    for basic hyperparameter search and exploration.

    Examples:
        >>> import spax as sp
        >>>
        >>> # Random sampling with seed for reproducibility
        >>> config = MyConfig.random(seed=42)
        >>>
        >>> # Multiple samples with same seed
        >>> sampler = RandomSampler(seed=42)
        >>> value1 = sampler.suggest_float("lr", 1e-5, 1e-1, True, True, "log")
        >>> value2 = sampler.suggest_int("layers", 1, 10, True, True, "uniform")
    """

    def __init__(self, seed: int | None = None) -> None:
        """Initialize a RandomSampler.

        Args:
            seed: Random seed for reproducibility. If None, uses system time.
        """
        self._rng = random.Random(seed)
        self._record: dict[str, Any] = {}

    @property
    def record(self) -> dict[str, Any]:
        """Get a copy of all sampled parameter values.

        Returns:
            Dictionary mapping parameter names to their sampled values.
        """
        return self._record.copy()

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        low_inclusive: bool,
        high_inclusive: bool,
        distribution: Literal["log", "uniform"],
    ) -> int:
        """Sample an integer value from a range.

        Args:
            name: Parameter name for tracking.
            low: Lower bound of the range.
            high: Upper bound of the range.
            low_inclusive: Whether the lower bound is inclusive.
            high_inclusive: Whether the upper bound is inclusive.
            distribution: Sampling distribution ('uniform' or 'log').

        Returns:
            Sampled integer value within the specified range.

        Raises:
            ValueError: If distribution is unknown.
        """
        # Adjust bounds based on inclusivity
        if not low_inclusive:
            low = low + 1
        if not high_inclusive:
            high = high - 1

        if distribution == "uniform":
            value = self._rng.randint(low, high)
            self._record[name] = value
            return value
        elif distribution == "log":
            # Log-uniform sampling for integers
            assert low > 0, "Low must be positive for log distribution"
            log_low = math.log(low)
            log_high = math.log(high)
            log_value = self._rng.uniform(log_low, log_high)
            value = int(round(math.exp(log_value)))
            self._record[name] = value
            return value
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
        """Sample a float value from a range.

        Args:
            name: Parameter name for tracking.
            low: Lower bound of the range.
            high: Upper bound of the range.
            low_inclusive: Whether the lower bound is inclusive.
            high_inclusive: Whether the upper bound is inclusive.
            distribution: Sampling distribution ('uniform' or 'log').

        Returns:
            Sampled float value within the specified range.

        Raises:
            ValueError: If distribution is unknown.
        """
        # Adjust bounds based on inclusivity (small epsilon for floats)
        if not low_inclusive:
            low = low + 1e-10
        if not high_inclusive:
            high = high - 1e-10

        if distribution == "uniform":
            value = self._rng.uniform(low, high)
            self._record[name] = value
            return value
        elif distribution == "log":
            # Log-uniform sampling for floats
            assert low > 0, "Low must be positive for log distribution"
            log_low = math.log(low)
            log_high = math.log(high)
            log_value = self._rng.uniform(log_low, log_high)
            value = math.exp(log_value)
            self._record[name] = value
            return value
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

    def suggest_categorical(
        self,
        name: str,
        choices: list[Any],
        weights: list[float],
    ) -> Any:
        """Sample a categorical value from choices.

        Args:
            name: Parameter name for tracking.
            choices: List of possible values to choose from.
            weights: List of weights for each choice. Higher weights
                increase selection probability.

        Returns:
            One of the choices, selected according to the weights.
        """
        self._record[name] = self._rng.choices(choices, weights=weights, k=1)[0]
        return self._record[name]
