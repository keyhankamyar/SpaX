"""
Categorical search space for discrete choice parameters.

This module provides categorical spaces that sample from a discrete
set of options with optional weighting.
"""

from typing import Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from spax.distributions import CategoricalDistribution

from .base import UNSET, Space, _Unset


class Choice:
    """
    Represents a weighted choice in a categorical search space.

    Choices allow you to specify relative probabilities for different
    options in a categorical distribution.

    Example:
        >>> choices = [
        ...     Choice("relu", weight=1.0),
        ...     Choice("gelu", weight=2.0),  # 2x more likely than relu
        ...     "silu"  # Defaults to weight=1.0
        ... ]
    """

    def __init__(self, value: Any, weight: float = 1.0) -> None:
        """
        Initialize a weighted choice.

        Args:
            value: The value this choice represents.
            weight: Relative probability weight (must be positive).

        Raises:
            AssertionError: If weight is not numeric or is non-positive.
        """
        if not isinstance(weight, (float, int)):
            raise TypeError(f"weight must be numeric, got {type(weight).__name__}")

        weight = float(weight)

        if weight <= 0:
            raise ValueError(f"weight must be positive, got {weight}")

        self.value = value
        self.weight = weight

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"Choice(value={self.value!r}, weight={self.weight})"

    def __eq__(self, other: object) -> bool:
        """Check equality based on value and weight."""
        if not isinstance(other, Choice):
            return NotImplemented
        return self.value == other.value and self.weight == other.weight


class CategoricalSpace(Space[Any]):
    """
    Search space for categorical (discrete choice) parameters.

    Supports both uniform and weighted sampling from a discrete set
    of options. Choices can be simple values or nested Config types.

    Example:
        >>> activation: str = Categorical(["relu", "gelu", "silu"])
        >>> optimizer: Any = Categorical([
        ...     Choice("adam", weight=2),
        ...     Choice("sgd", weight=1),
        ... ])
    """

    def __init__(
        self,
        choices: list[Any | Choice],
        default: Any | _Unset = UNSET,
        description: str | None = None,
    ) -> None:
        """
        Initialize a categorical space.

        Args:
            choices: List of possible values or Choice objects.
            default: Default value to use when not specified. Must be one of the choices.
            description: Human-readable description of this parameter.

        Raises:
            ValueError: If choices list is empty.
            ValueError: If any choice value is not comparable or hashable.
        """
        from spax.config import Config

        if not choices:
            raise ValueError("Categorical space must have at least one choice")

        self.raw_choices = choices
        self.choices: list[Any] = []
        self.weights: list[float] = []

        # Process and validate choices
        for choice in choices:
            if isinstance(choice, Choice):
                value = choice.value
                weight = choice.weight
            else:
                value = choice
                weight = 1.0

            # Validate that value is comparable (has __eq__)
            # or is a Config type (BaseModel subclass)
            if not (
                hasattr(value, "__eq__")
                or (isinstance(value, type) and issubclass(value, Config))
            ):
                raise ValueError(
                    f"Choice value {value!r} must be comparable or a Config type"
                )

            self.choices.append(value)
            self.weights.append(weight)

        # Normalize weights to probabilities
        total_weight = sum(self.weights)
        if total_weight == 0:
            raise ValueError("Total weight cannot be zero")
        self.probs = [w / total_weight for w in self.weights]

        self.distribution = CategoricalDistribution()

        # Call parent __init__ with default and description
        super().__init__(default=default, description=description)

    def validate(self, value: Any) -> Any:
        """
        Validate that a value is one of the allowed choices.

        For nested Config types, checks if value is an instance of that type.
        For regular values, checks equality.

        Args:
            value: The value to validate.

        Returns:
            The validated value.

        Raises:
            ValueError: If value is not in the allowed choices.
        """
        from spax.config import Config

        field = self.field_name or "value"

        # Check each choice
        for choice in self.choices:
            # For Config types, check if value is an instance
            if isinstance(choice, type) and issubclass(choice, Config):
                if isinstance(value, choice):
                    return value
            # For regular values, check equality
            else:
                if not hasattr(value, "__eq__"):
                    raise TypeError(
                        f"{field}: Value {value!r} must be comparable (have __eq__ method)"
                    )
                if value == choice:
                    return value

        raise ValueError(
            f"{field}: Value {value!r} not in allowed choices {self.choices}"
        )

    def sample(self) -> Any:
        """
        Sample a random choice based on weights.

        Returns:
            A randomly selected choice from the categorical distribution.
        """
        return self.distribution.sample(choices=self.choices, weights=self.probs)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Provide Pydantic schema for any-type validation."""
        return core_schema.no_info_after_validator_function(
            lambda x: x, core_schema.any_schema()
        )

    def __repr__(self) -> str:
        """Return a string representation."""
        parts = [f"choices={self.choices}"]

        # Only show probs if they're not all equal (i.e., weights were specified)
        if len(set(self.probs)) > 1:
            parts.append(f"probs={self.probs}")

        if self.default is not UNSET:
            parts.append(f"default={self.default!r}")
        if self.description is not None:
            parts.append(f"description={self.description!r}")

        return f"CategoricalSpace({', '.join(parts)})"


def Categorical(
    choices: list[Any | Choice],
    default: Any | _Unset = UNSET,
    description: str | None = None,
) -> Any:
    """
    Create a categorical search space (type-checker friendly).

    This function returns Any to satisfy type checkers when used as:
        activation: str = Categorical(["relu", "gelu"])

    Args:
        choices: List of possible values or Choice objects.
        default: Default value to use when not specified. Must be one of the choices.
        description: Human-readable description of this parameter.

    Returns:
        A CategoricalSpace instance.
    """
    return CategoricalSpace(choices, default, description)
