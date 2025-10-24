"""Base classes for parameter samplers.

This module defines the abstract base class for samplers that generate
parameter values from search spaces. Samplers provide a unified interface
for different sampling strategies (random, Optuna, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any, Literal


class Sampler(ABC):
    """Abstract base class for all samplers.

    A Sampler generates parameter values from search spaces. It provides
    methods for sampling integers, floats, and categorical values, and
    maintains a record of all sampled values.

    Samplers are used internally by Config.random() and can be implemented
    for different sampling strategies (random, Bayesian optimization, etc.).

    Attributes:
        record: Dictionary mapping parameter names to their sampled values.
    """

    @property
    @abstractmethod
    def record(self) -> dict[str, Any]:
        """Get a copy of all sampled parameter values.

        Returns:
            Dictionary mapping parameter names to their sampled values.
        """
        pass

    @abstractmethod
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
        """
        pass

    @abstractmethod
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
        """
        pass

    @abstractmethod
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
            weights: List of weights for each choice (must sum to > 0).
                Higher weights increase selection probability.

        Returns:
            One of the choices, selected according to the weights.
        """
        pass
