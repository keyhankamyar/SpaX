"""
Distribution classes for sampling values from search spaces.

This module provides abstract and concrete distribution implementations
for numeric and categorical sampling used in hyperparameter optimization.
"""

from abc import ABC, abstractmethod
import math
import random
from typing import Any


class Distribution(ABC):
    """
    Abstract base class for all probability distributions.

    Distributions define how values are sampled from search spaces,
    supporting both direct random sampling and integration with HPO libraries.
    """

    @abstractmethod
    def sample(self, *args: Any, **kwargs: Any) -> Any:
        """
        Sample a value from this distribution.

        Args:
            *args: Positional arguments specific to the distribution type.
            **kwargs: Keyword arguments specific to the distribution type.

        Returns:
            A sampled value whose type depends on the distribution.
        """
        pass


class NumberDistribution(Distribution):
    """
    Abstract base class for numeric distributions.

    Used for sampling continuous or discrete numeric values within
    a specified range [low, high].
    """

    @abstractmethod
    def sample(self, low: float, high: float) -> float:
        """
        Sample a numeric value from the range [low, high].

        Args:
            low: Lower bound of the sampling range.
            high: Upper bound of the sampling range.

        Returns:
            A sampled numeric value.

        Raises:
            ValueError: If the range constraints are violated.
        """
        pass


class UniformDistribution(NumberDistribution):
    """
    Uniform distribution for numeric sampling.

    Samples values uniformly at random from the range [low, high],
    where all values have equal probability.
    """

    def sample(self, low: float, high: float) -> float:
        """
        Sample uniformly from [low, high].

        Args:
            low: Lower bound (inclusive).
            high: Upper bound (inclusive).

        Returns:
            A uniformly sampled float value.

        Raises:
            ValueError: If low >= high.
        """
        if low >= high:
            raise ValueError(
                f"Invalid range: low ({low}) must be less than high ({high})"
            )
        return random.uniform(low, high)


class LogDistribution(NumberDistribution):
    """
    Logarithmic (log-uniform) distribution for numeric sampling.

    Samples values uniformly in log-space, making it suitable for
    hyperparameters that span multiple orders of magnitude
    (e.g., learning rates: 1e-5 to 1e-1).
    """

    def sample(self, low: float, high: float) -> float:
        """
        Sample uniformly from log-space over [low, high].

        Args:
            low: Lower bound (must be positive).
            high: Upper bound (must be greater than low).

        Returns:
            A log-uniformly sampled float value.

        Raises:
            ValueError: If low <= 0 or low >= high.
        """
        if low <= 0:
            raise ValueError(f"Log distribution requires low > 0, got low={low}")
        if low >= high:
            raise ValueError(
                f"Invalid range: low ({low}) must be less than high ({high})"
            )

        log_low = math.log(low)
        log_high = math.log(high)
        return math.exp(random.uniform(log_low, log_high))


class CategoricalDistribution(Distribution):
    """
    Categorical distribution for discrete choice sampling.

    Samples from a discrete set of choices with optional weights,
    supporting both uniform and weighted sampling.
    """

    def sample(self, choices: list[Any], weights: list[float]) -> Any:
        """
        Sample one choice from the provided options using weights.

        Args:
            choices: List of possible values to choose from.
            weights: Probability weights for each choice (must sum to 1.0).

        Returns:
            One randomly selected choice.

        Raises:
            ValueError: If choices is empty or weights length doesn't match choices.
            ValueError: If any weight is negative or all weights sum to 0.
        """
        if not choices:
            raise ValueError("Cannot sample from empty choices list")

        if len(choices) != len(weights):
            raise ValueError(
                f"Choices length ({len(choices)}) must match weights length ({len(weights)})"
            )

        if any(w < 0 for w in weights):
            raise ValueError("All weights must be non-negative")

        if sum(weights) == 0:
            raise ValueError("Weights cannot all be zero")

        return random.choices(choices, weights=weights, k=1)[0]


# Singleton instances for convenience
UNIFORM = UniformDistribution()
LOG = LogDistribution()
