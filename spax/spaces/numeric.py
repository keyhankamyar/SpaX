"""
Numeric search spaces for continuous and discrete parameters.

This module provides Float and Int spaces with support for
different distributions and boundary conditions.
"""

from typing import Any, Literal, TypeAlias

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from spax.distributions import LOG, UNIFORM, NumberDistribution

from .base import UNSET, Space, _Unset

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
        default: float | int | _Unset = UNSET,
        gt: float | None = None,
        ge: float | None = None,
        lt: float | None = None,
        le: float | None = None,
        distribution: DistributionType = "uniform",
        description: str | None = None,
    ) -> None:
        """
        Initialize a numeric space.

        Args:
            default: Default value to use when not specified.
            gt: Greater than (exclusive lower bound).
            ge: Greater than or equal (inclusive lower bound).
            lt: Less than (exclusive upper bound).
            le: Less than or equal (inclusive upper bound).
            distribution: Sampling distribution ("uniform", "log", or Distribution instance).
            description: Human-readable description of this parameter.

        Raises:
            ValueError: If bounds are not properly specified or invalid.
        """
        # Validate that exactly one lower bound is specified
        lower_bounds = [gt, ge]
        if sum(b is not None for b in lower_bounds) != 1:
            raise ValueError(
                "Exactly one of 'gt' (greater than) or 'ge' (greater than or equal) must be specified"
            )

        # Validate that exactly one upper bound is specified
        upper_bounds = [lt, le]
        if sum(b is not None for b in upper_bounds) != 1:
            raise ValueError(
                "Exactly one of 'lt' (less than) or 'le' (less than or equal) must be specified"
            )

        # Determine low, high, and bounds type
        if gt is not None:
            low = gt
            low_inclusive = False
        else:
            low = ge  # type: ignore
            low_inclusive = True

        if lt is not None:
            high = lt
            high_inclusive = False
        else:
            high = le  # type: ignore
            high_inclusive = True

        # Validate range
        assert isinstance(low, (int, float)), (
            f"lower bound must be numeric, got {type(low)}"
        )
        assert isinstance(high, (int, float)), (
            f"upper bound must be numeric, got {type(high)}"
        )
        assert low < high, f"lower bound ({low}) must be less than upper bound ({high})"

        self.low = float(low)
        self.high = float(high)
        self.low_inclusive = low_inclusive
        self.high_inclusive = high_inclusive

        # Store original bound specifications for repr
        self.gt = gt
        self.ge = ge
        self.lt = lt
        self.le = le

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

        # Call parent __init__ with default and description
        super().__init__(default=default, description=description)

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

        # Check lower bound
        if self.low_inclusive:
            if value < self.low:
                raise ValueError(f"{field}: Value {value} must be >= {self.low}")
        else:
            if value <= self.low:
                raise ValueError(f"{field}: Value {value} must be > {self.low}")

        # Check upper bound
        if self.high_inclusive:
            if value > self.high:
                raise ValueError(f"{field}: Value {value} must be <= {self.high}")
        else:
            if value >= self.high:
                raise ValueError(f"{field}: Value {value} must be < {self.high}")

    def sample(self) -> float:
        """
        Sample a random value from this numeric space.

        Returns:
            A float sampled according to the distribution.
        """
        return self.distribution.sample(self.low, self.high)

    def __repr__(self) -> str:
        """Return a detailed string representation."""
        parts = []

        # Add bounds
        if self.gt is not None:
            parts.append(f"gt={self.gt}")
        if self.ge is not None:
            parts.append(f"ge={self.ge}")
        if self.lt is not None:
            parts.append(f"lt={self.lt}")
        if self.le is not None:
            parts.append(f"le={self.le}")

        # Add distribution
        parts.append(f"distribution='{self.distribution.__class__.__name__}'")

        # Add default and description from parent
        if self.default is not UNSET:
            parts.append(f"default={self.default!r}")
        if self.description is not None:
            parts.append(f"description={self.description!r}")

        return f"{self.__class__.__name__}({', '.join(parts)})"


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
        >>> num_layers: int = Int(ge=1, le=10)
    """

    def __init__(
        self,
        default: int | _Unset = UNSET,
        gt: int | None = None,
        ge: int | None = None,
        lt: int | None = None,
        le: int | None = None,
        distribution: DistributionType = "uniform",
        description: str | None = None,
    ) -> None:
        """
        Initialize an integer space.

        Args:
            default: Default value to use when not specified.
            gt: Greater than (exclusive lower bound, must be integer).
            ge: Greater than or equal (inclusive lower bound, must be integer).
            lt: Less than (exclusive upper bound, must be integer).
            le: Less than or equal (inclusive upper bound, must be integer).
            distribution: Sampling distribution.
            description: Human-readable description of this parameter.

        Raises:
            TypeError: If bounds are not integers.
        """
        # Validate that bounds are integers
        for name, value in [("gt", gt), ("ge", ge), ("lt", lt), ("le", le)]:
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool)
            ):
                raise TypeError(
                    f"{name} must be an integer, got {type(value).__name__}"
                )

        # Validate default is integer if provided
        if default is not UNSET and (
            not isinstance(default, int) or isinstance(default, bool)
        ):
            raise TypeError(f"default must be an integer, got {type(default).__name__}")

        super().__init__(default, gt, ge, lt, le, distribution, description)

        # Store as integers for cleaner representation
        self.low = int(self.low)
        self.high = int(self.high)

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
    default: float | _Unset = UNSET,
    gt: float | None = None,
    ge: float | None = None,
    lt: float | None = None,
    le: float | None = None,
    distribution: DistributionType = "uniform",
    description: str | None = None,
) -> Any:
    """
    Create a float search space (type-checker friendly).

    This function returns Any to satisfy type checkers when used as:
        learning_rate: float = Float(ge=0.001, lt=0.1)

    Args:
        default: Default value to use when not specified.
        gt: Greater than (exclusive lower bound).
        ge: Greater than or equal (inclusive lower bound).
        lt: Less than (exclusive upper bound).
        le: Less than or equal (inclusive upper bound).
        distribution: "uniform", "log", or a Distribution instance.
        description: Human-readable description of this parameter.

    Returns:
        A FloatSpace instance.
    """
    return FloatSpace(default, gt, ge, lt, le, distribution, description)


def Int(
    default: int | _Unset = UNSET,
    gt: int | None = None,
    ge: int | None = None,
    lt: int | None = None,
    le: int | None = None,
    distribution: DistributionType = "uniform",
    description: str | None = None,
) -> Any:
    """
    Create an integer search space (type-checker friendly).

    This function returns Any to satisfy type checkers when used as:
        num_layers: int = Int(ge=1, le=10)

    Args:
        default: Default value to use when not specified.
        gt: Greater than (exclusive lower bound).
        ge: Greater than or equal (inclusive lower bound).
        lt: Less than (exclusive upper bound).
        le: Less than or equal (inclusive upper bound).
        distribution: "uniform", "log", or a Distribution instance.
        description: Human-readable description of this parameter.

    Returns:
        An IntSpace instance.
    """
    return IntSpace(default, gt, ge, lt, le, distribution, description)
