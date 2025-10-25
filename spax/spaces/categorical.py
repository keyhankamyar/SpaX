"""Categorical search spaces for discrete choice parameters.

This module provides CategoricalSpace for defining parameters that can take
one of a discrete set of values, with optional weights for non-uniform sampling.
"""

from typing import Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from spax.utils import is_comparable

from .base import UNSET, Space, _Unset


class Choice:
    """A weighted choice option for categorical spaces.

    Choice allows specifying both a value and its sampling weight/probability.
    Higher weights make a choice more likely to be sampled.

    Attributes:
        value: The actual value of this choice.
        weight: Sampling weight (must be positive).

    Examples:
        >>> Choice("adam", weight=2.0)  # Twice as likely as weight=1.0
        >>> Choice("sgd", weight=1.0)
    """

    def __init__(self, value: Any, weight: float = 1.0) -> None:
        """Initialize a Choice with a value and weight.

        Args:
            value: The value this choice represents.
            weight: Sampling weight (must be positive, default=1.0).

        Raises:
            TypeError: If weight is not numeric.
            ValueError: If weight is not positive.
        """
        if not isinstance(weight, (float, int)):
            raise TypeError(f"weight must be numeric, got {type(weight).__name__}")

        weight = float(weight)
        if weight <= 0:
            raise ValueError(f"weight must be positive, got {weight}")

        self._value = value
        self._weight = weight

    @property
    def value(self) -> Any:
        """The value of this choice."""
        return self._value

    @property
    def weight(self) -> float:
        """The sampling weight of this choice."""
        return self._weight

    def __repr__(self) -> str:
        """Return a string representation of this choice."""
        return f"Choice(value={self._value!r}, weight={self._weight})"

    def __eq__(self, other: object) -> bool:
        """Check equality with another Choice.

        Args:
            other: Another object to compare with.

        Returns:
            True if both value and weight are equal.
        """
        if not isinstance(other, Choice):
            return NotImplemented
        return self._value == other._value and self._weight == other._weight


class CategoricalSpace(Space[Any]):
    """Search space for categorical (discrete choice) parameters.

    CategoricalSpace defines a parameter that can take one of a fixed set
    of values. Values can be primitives, Config types, or any comparable object.
    Optional weights allow non-uniform sampling probabilities.

    Attributes:
        choices: List of possible values (read-only).
        weights: List of sampling weights (read-only).
        probs: List of normalized probabilities (read-only).
        raw_choices: Original choices list including Choice objects (read-only).

    Examples:
        >>> import spax as sp
        >>>
        >>> # Simple categorical with equal probabilities
        >>> optimizer = sp.CategoricalSpace(["adam", "sgd", "rmsprop"])

        >>> # Weighted categorical (adam is twice as likely)
        >>> optimizer = sp.CategoricalSpace([
        ...     sp.Choice("adam", weight=2.0),
        ...     sp.Choice("sgd", weight=1.0),
        ...     sp.Choice("rmsprop", weight=1.0)
        ... ])

        >>> # Categorical with Config types
        >>> model = sp.CategoricalSpace([ResNet, VGG, Transformer])
    """

    def __init__(
        self,
        choices: list[Any | Choice],
        *,
        default: Any | _Unset = UNSET,
        description: str | None = None,
    ) -> None:
        """Initialize a CategoricalSpace with choices.

        Args:
            choices: List of possible values. Can be raw values or Choice objects
                with weights. All values must be comparable or Config types.
            default: Default choice value.
            description: Human-readable description.

        Raises:
            ValueError: If choices is empty or values are not comparable.
        """
        from spax.config import Config

        if not choices:
            raise ValueError("Categorical space must have at least one choice")

        self._raw_choices = choices
        self._choices: list[Any] = []
        self._weights: list[float] = []

        # Process and validate choices
        for choice in choices:
            if isinstance(choice, Choice):
                value = choice.value
                weight = choice.weight
            else:
                value = choice
                weight = 1.0

            # Validate that value is comparable or is a Config type
            if not (
                is_comparable(value)
                or (isinstance(value, type) and issubclass(value, Config))
            ):
                raise ValueError(
                    f"Choice value {value!r} must be comparable or a Config type"
                )

            self._choices.append(value)
            self._weights.append(weight)

        # Normalize weights to probabilities
        total_weight = sum(self._weights)
        if total_weight == 0:
            raise ValueError("Total weight cannot be zero")

        self._probs = [w / total_weight for w in self._weights]

        # Call parent __init__ with default and description
        super().__init__(default=default, description=description)

    @property
    def raw_choices(self) -> list[Any | Choice]:
        """Original choices list including Choice objects."""
        return self._raw_choices.copy()

    @property
    def choices(self) -> list[Any]:
        """List of possible values (extracted from Choice objects)."""
        return self._choices.copy()

    @property
    def weights(self) -> list[float]:
        """List of sampling weights for each choice."""
        return self._weights.copy()

    @property
    def probs(self) -> list[float]:
        """List of normalized probabilities for each choice."""
        return self._probs.copy()

    def contains(self, other: Any) -> bool:
        """Check if another space is contained within this space.

        Args:
            other: Another space to check.

        Returns:
            True if all choices in other are present in this space.
        """
        if not isinstance(other, Space):
            return False
        if not isinstance(other, CategoricalSpace):
            return False

        # Check if all choices in other are in self
        return all(choice in self._choices for choice in other._choices)

    def validate(self, value: Any) -> Any:
        """Validate that a value is one of the allowed choices.

        Args:
            value: The value to validate.

        Returns:
            The validated value.

        Raises:
            ValueError: If the value is not in the allowed choices.
            TypeError: If the value is not comparable.
        """
        from spax.config import Config

        field = self.field_name or "value"

        # Check each choice
        for choice in self._choices:
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
            f"{field}: Value {value!r} not in allowed choices {self._choices}"
        )

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Provide Pydantic schema for categorical validation."""
        return core_schema.no_info_after_validator_function(
            lambda x: x, core_schema.any_schema()
        )

    def __repr__(self) -> str:
        """Return a string representation of this space."""
        parts = [f"choices={self._choices}"]

        # Only show probs if they're not all equal (i.e., weights were specified)
        if len(set(self._probs)) > 1:
            parts.append(f"probs={self._probs}")

        if self.default is not UNSET:
            parts.append(f"default={self.default!r}")
        if self.description is not None:
            parts.append(f"description={self.description!r}")

        return f"CategoricalSpace({', '.join(parts)})"


def Categorical(
    choices: list[Any | Choice],
    *,
    default: Any | _Unset = UNSET,
    description: str | None = None,
) -> Any:
    """Factory function for creating a CategoricalSpace.

    This is the primary user-facing API for defining categorical parameter spaces.

    Args:
        choices: List of possible values or Choice objects with weights.
        default: Default choice value.
        description: Parameter description.

    Returns:
        A CategoricalSpace instance.

    Examples:
        >>> import spax as sp
        >>>
        >>> # Simple choices
        >>> optimizer = sp.Categorical(["adam", "sgd", "rmsprop"])

        >>> # Weighted choices
        >>> optimizer = sp.Categorical([
        ...     sp.Choice("adam", weight=2.0),
        ...     "sgd",  # Implicitly weight=1.0
        ...     "rmsprop"
        ... ])

        >>> # With default
        >>> activation = sp.Categorical(["relu", "gelu", "silu"], default="relu")
    """
    return CategoricalSpace(choices=choices, default=default, description=description)
