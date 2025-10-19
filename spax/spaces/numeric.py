"""
Numeric search spaces for continuous and discrete parameters.

This module provides Float and Int spaces with support for
different distributions and boundary conditions.
"""

from typing import Any, Literal, TypeAlias

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from spax.distributions import LOG, UNIFORM, NumberDistribution

from .base import Space

BoundsType: TypeAlias = Literal["none", "low", "high", "both"]
DistributionType: TypeAlias = NumberDistribution | Literal["uniform", "log"]


class NumberSpace(Space[float]):
    """
    Abstract base class for numeric (float/int) search spaces.

    Provides common functionality for bounded numeric ranges with
    different sampling distributions and boundary inclusion rules.
    """

    def __init__(
        self,
        low: float,
        high: float,
        distribution: DistributionType,
        bounds: BoundsType,
    ) -> None:
        """
        Initialize a numeric space.

        Args:
            low: Lower bound of the numeric range.
            high: Upper bound of the numeric range.
            distribution: Sampling distribution ("uniform", "log", or Distribution instance).
            bounds: Which boundaries are inclusive ("both", "low", "high", "none").

        Raises:
            AssertionError: If arguments have invalid types.
            ValueError: If distribution string is unrecognized.
        """
        super().__init__()

        assert isinstance(low, (int, float)), f"low must be numeric, got {type(low)}"
        assert isinstance(high, (int, float)), f"high must be numeric, got {type(high)}"
        assert low < high, f"low ({low}) must be less than high ({high})"
        assert bounds in ["none", "low", "high", "both"], f"Invalid bounds: {bounds}"

        self.low = float(low)
        self.high = float(high)
        self.bounds = bounds

        # Handle distribution specification
        if isinstance(distribution, str):
            if distribution == "uniform":
                self.distribution = UNIFORM
            elif distribution == "log":
                self.distribution = LOG
            else:
                raise ValueError(
                    f"Unknown distribution string '{distribution}'. "
                    "Expected 'uniform' or 'log'."
                )
        elif isinstance(distribution, NumberDistribution):
            self.distribution = distribution
        else:
            raise ValueError(
                f"distribution must be a NumberDistribution or string, "
                f"got {type(distribution).__name__}"
            )

    def _check_bounds(self, value: float) -> None:
        """
        Check if a value satisfies the boundary conditions.

        Args:
            value: The value to check.

        Raises:
            ValueError: If the value violates boundary constraints.
        """
        if self.field_name is None:
            raise RuntimeError(
                "Space field_name is None. This should not happen if the Space "
                "is properly attached to a Config class via __set_name__."
            )

        field = self.field_name

        match self.bounds:
            case "both":
                if not (self.low <= value <= self.high):
                    raise ValueError(
                        f"{field}: Value {value} must be in [{self.low}, {self.high}]"
                    )
            case "low":
                if not (self.low <= value < self.high):
                    raise ValueError(
                        f"{field}: Value {value} must be in [{self.low}, {self.high})"
                    )
            case "high":
                if not (self.low < value <= self.high):
                    raise ValueError(
                        f"{field}: Value {value} must be in ({self.low}, {self.high}]"
                    )
            case "none":
                if not (self.low < value < self.high):
                    raise ValueError(
                        f"{field}: Value {value} must be in ({self.low}, {self.high})"
                    )

    def sample(self) -> float:
        """
        Sample a random value from this numeric space.

        Returns:
            A float sampled according to the distribution.
        """
        return self.distribution.sample(self.low, self.high)

    def __repr__(self) -> str:
        """Return a detailed string representation."""
        return (
            f"{self.__class__.__name__}("
            f"low={self.low}, high={self.high}, "
            f"distribution={self.distribution.__class__.__name__}, "
            f"bounds='{self.bounds}')"
        )


class FloatSpace(NumberSpace):
    """
    Search space for continuous floating-point parameters.

    Supports uniform and logarithmic distributions with flexible
    boundary inclusion rules.

    Example:
        >>> learning_rate: float = Float(1e-5, 1e-1, "log", "both")
    """

    def validate(self, value: Any) -> float:
        """
        Validate and coerce a value to float.

        Args:
            value: The value to validate.

        Returns:
            The validated float value.

        Raises:
            ValueError: If value is not numeric or violates bounds.
        """
        field = self.field_name or "value"

        if not isinstance(value, (int, float)):
            raise ValueError(
                f"{field}: Expected numeric value, got {type(value).__name__}"
            )

        value = float(value)
        self._check_bounds(value)
        return value

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Provide Pydantic schema for float validation."""
        return core_schema.no_info_after_validator_function(
            lambda x: x, core_schema.float_schema()
        )


class IntSpace(NumberSpace):
    """
    Search space for discrete integer parameters.

    Supports uniform and logarithmic distributions (sampled then rounded)
    with flexible boundary inclusion rules.

    Example:
        >>> num_layers: int = Int(1, 10, "uniform", "both")
    """

    def __init__(
        self,
        low: int,
        high: int,
        distribution: DistributionType = "uniform",
        bounds: BoundsType = "both",
    ) -> None:
        """
        Initialize an integer space.

        Args:
            low: Lower bound (must be integer).
            high: Upper bound (must be integer).
            distribution: Sampling distribution.
            bounds: Which boundaries are inclusive.

        Raises:
            AssertionError: If low or high are not integers.
        """
        assert isinstance(low, int) and not isinstance(low, bool), (
            f"low must be int, got {type(low)}"
        )
        assert isinstance(high, int) and not isinstance(high, bool), (
            f"high must be int, got {type(high)}"
        )

        super().__init__(low, high, distribution, bounds)

        # Store as integers for cleaner representation
        self.low = low
        self.high = high

    def validate(self, value: Any) -> int:
        """
        Validate and coerce a value to integer.

        Args:
            value: The value to validate.

        Returns:
            The validated integer value.

        Raises:
            ValueError: If value is not an integer or violates bounds.
        """
        field = self.field_name or "value"

        # Check if it's an integer (or a float that represents an integer)
        if isinstance(value, int):
            int_value = value
        elif isinstance(value, float):
            if not value.is_integer():
                raise ValueError(f"{field}: Expected integer value, got float {value}")
            int_value = int(value)
        else:
            raise ValueError(f"{field}: Expected int, got {type(value).__name__}")

        self._check_bounds(float(int_value))
        return int_value

    def sample(self) -> int:
        """
        Sample a random integer from this space.

        Returns:
            An integer sampled according to the distribution and then rounded.
        """
        value = self.distribution.sample(self.low, self.high)
        return int(round(value))

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Provide Pydantic schema for integer validation."""
        return core_schema.no_info_after_validator_function(
            lambda x: x, core_schema.int_schema()
        )


def Float(
    low: float,
    high: float,
    distribution: DistributionType = "uniform",
    bounds: BoundsType = "both",
) -> Any:
    """
    Create a float search space (type-checker friendly).

    This function returns Any to satisfy type checkers when used as:
        learning_rate: float = Float(0.001, 0.1)

    Args:
        low: Lower bound of the range.
        high: Upper bound of the range.
        distribution: "uniform", "log", or a Distribution instance.
        bounds: Which boundaries to include ("both", "low", "high", "none").

    Returns:
        A FloatSpace instance.
    """
    return FloatSpace(low, high, distribution, bounds)


def Int(
    low: int,
    high: int,
    distribution: DistributionType = "uniform",
    bounds: BoundsType = "both",
) -> Any:
    """
    Create an integer search space (type-checker friendly).

    This function returns Any to satisfy type checkers when used as:
        num_layers: int = Int(1, 10)

    Args:
        low: Lower bound of the range.
        high: Upper bound of the range.
        distribution: "uniform", "log", or a Distribution instance.
        bounds: Which boundaries to include ("both", "low", "high", "none").

    Returns:
        An IntSpace instance.
    """
    return IntSpace(low, high, distribution, bounds)
