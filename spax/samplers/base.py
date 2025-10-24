from abc import ABC, abstractmethod
from typing import Any, Literal


class Sampler(ABC):
    """Abstract base class for samplers that suggest parameter values."""

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
        """Suggest an integer value.

        Args:
            name: Parameter name
            low: Lower bound
            high: Upper bound
            low_inclusive: Whether lower bound is inclusive
            high_inclusive: Whether upper bound is inclusive
            distribution: Distribution type

        Returns:
            Suggested integer value
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
        """Suggest a float value.

        Args:
            name: Parameter name
            low: Lower bound
            high: Upper bound
            low_inclusive: Whether lower bound is inclusive
            high_inclusive: Whether upper bound is inclusive
            distribution: Distribution type

        Returns:
            Suggested float value
        """
        pass

    @abstractmethod
    def suggest_categorical(
        self,
        name: str,
        choices: list[Any],
        weights: list[float],
    ) -> Any:
        """Suggest a categorical choice.

        Args:
            name: Parameter name
            choices: List of possible choices
            weights: Probability weights for each choice

        Returns:
            Suggested choice from the list
        """
        pass
