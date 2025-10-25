"""Numeric search spaces for continuous and discrete parameters.

This module provides FloatSpace and IntSpace for defining numeric parameter
ranges with inclusive/exclusive bounds and different sampling distributions.
"""

from typing import Any, Literal

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from .base import UNSET, Space, _Unset


class NumberSpace(Space[float]):
    """Abstract base class for numeric search spaces.

    NumberSpace handles the common logic for both float and integer spaces,
    including bound validation, inclusivity handling, and distribution types.

    Attributes:
        low: Lower bound of the range.
        high: Upper bound of the range.
        low_inclusive: Whether the lower bound is inclusive.
        high_inclusive: Whether the upper bound is inclusive.
        distribution: Sampling distribution ('uniform' or 'log').
        gt: Greater than bound (if specified).
        ge: Greater than or equal bound (if specified).
        lt: Less than bound (if specified).
        le: Less than or equal bound (if specified).
    """

    def __init__(
        self,
        *,
        gt: float | None = None,
        ge: float | None = None,
        lt: float | None = None,
        le: float | None = None,
        distribution: Literal["uniform", "log"] = "uniform",
        default: float | int | _Unset = UNSET,
        description: str | None = None,
    ) -> None:
        """Initialize a NumberSpace with bounds and distribution.

        Args:
            gt: Greater than (exclusive lower bound).
            ge: Greater than or equal (inclusive lower bound).
            lt: Less than (exclusive upper bound).
            le: Less than or equal (inclusive upper bound).
            distribution: Sampling distribution - 'uniform' for linear sampling,
                'log' for logarithmic sampling (useful for learning rates, etc.).
            default: Default value for this space.
            description: Human-readable description.

        Raises:
            ValueError: If bounds are invalid or inconsistent.
            TypeError: If distribution is not 'uniform' or 'log'.
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

        # Store as private attributes
        self._low = float(low)
        self._high = float(high)
        self._low_inclusive = low_inclusive
        self._high_inclusive = high_inclusive

        # Store original bound specifications for repr (private)
        self._gt = gt
        self._ge = ge
        self._lt = lt
        self._le = le

        # Handle distribution specification
        if not isinstance(distribution, str):
            raise TypeError(
                f"Expected distribution to be string, got "
                f"'{distribution}' which is {type(distribution).__name__}"
            )
        if distribution not in ["uniform", "log"]:
            raise TypeError(
                f"Unknown distribution '{distribution}'. Expected 'uniform' or 'log'."
            )
        if distribution == "log" and (
            self._low < 0 or (self._low_inclusive and self._low == 0)
        ):
            raise ValueError(
                f"Low must be larger than 0 when using log distribution. Got {self._low}"
            )

        self._distribution: Literal["uniform", "log"] = distribution

        # Call parent __init__ with default and description
        super().__init__(default=default, description=description)

    # Public read-only properties for encapsulation
    @property
    def low(self) -> float:
        """Lower bound of the numeric range."""
        return self._low

    @property
    def high(self) -> float:
        """Upper bound of the numeric range."""
        return self._high

    @property
    def low_inclusive(self) -> bool:
        """Whether the lower bound is inclusive."""
        return self._low_inclusive

    @property
    def high_inclusive(self) -> bool:
        """Whether the upper bound is inclusive."""
        return self._high_inclusive

    @property
    def distribution(self) -> Literal["uniform", "log"]:
        """Sampling distribution ('uniform' or 'log')."""
        return self._distribution

    @property
    def gt(self) -> float | None:
        """Greater than (exclusive) lower bound, if specified."""
        return self._gt

    @property
    def ge(self) -> float | None:
        """Greater than or equal (inclusive) lower bound, if specified."""
        return self._ge

    @property
    def lt(self) -> float | None:
        """Less than (exclusive) upper bound, if specified."""
        return self._lt

    @property
    def le(self) -> float | None:
        """Less than or equal (inclusive) upper bound, if specified."""
        return self._le

    def contains(self, other: Any) -> bool:
        """Check if another space is contained within this space.

        Args:
            other: Another space to check.

        Returns:
            True if the other space's range is within this space's range.
        """
        if not isinstance(other, Space):
            return False
        if type(self) is not type(other):
            return False
        if not isinstance(other, NumberSpace):
            return False

        # Must have same distribution
        if self._distribution != other._distribution:
            return False

        # Check if other's range is within our range
        if other._low < self._low:
            return False
        if other._high > self._high:
            return False

        # Handle boundary inclusivity edge cases
        if (other._low == self._low) and (
            other._low_inclusive and not self._low_inclusive
        ):
            return False

        return not (
            other._high == self._high
            and (other._high_inclusive and not self._high_inclusive)
        )

    def _check_bounds(self, value: float) -> None:
        """Validate that a value is within the space's bounds.

        Args:
            value: The value to check.

        Raises:
            ValueError: If the value is outside the bounds.
            RuntimeError: If field_name is not set.
        """
        if self.field_name is None:
            raise RuntimeError(
                "Space field_name is None. This should not happen if the Space "
                "is properly attached to a Config class via __set_name__."
            )

        field = self.field_name

        # Check lower bound
        if self._low_inclusive:
            if value < self._low:
                raise ValueError(f"{field}: Value {value} must be >= {self._low}")
        else:
            if value <= self._low:
                raise ValueError(f"{field}: Value {value} must be > {self._low}")

        # Check upper bound
        if self._high_inclusive:
            if value > self._high:
                raise ValueError(f"{field}: Value {value} must be <= {self._high}")
        else:
            if value >= self._high:
                raise ValueError(f"{field}: Value {value} must be < {self._high}")

    def __repr__(self) -> str:
        """Return a string representation of this space."""
        parts = []

        # Add bounds using original specifications
        if self._gt is not None:
            parts.append(f"gt={self._gt}")
        if self._ge is not None:
            parts.append(f"ge={self._ge}")
        if self._lt is not None:
            parts.append(f"lt={self._lt}")
        if self._le is not None:
            parts.append(f"le={self._le}")

        # Add distribution
        parts.append(f"distribution='{self._distribution}'")

        # Add default and description from parent
        if self.default is not UNSET:
            parts.append(f"default={self.default!r}")
        if self.description is not None:
            parts.append(f"description={self.description!r}")

        return f"{self.__class__.__name__}({', '.join(parts)})"


class FloatSpace(NumberSpace):
    """Search space for floating-point parameters.

    FloatSpace defines a continuous range of floating-point values with
    configurable bounds and sampling distribution.

    Examples:
        >>> space = FloatSpace(ge=0.0, le=1.0)  # Range [0.0, 1.0]
        >>> space = FloatSpace(gt=0.0, lt=1.0)  # Range (0.0, 1.0)
        >>> space = FloatSpace(ge=1e-5, le=1.0, distribution='log')  # Log scale
    """

    def validate(self, value: Any) -> float:
        """Validate that a value is a valid float within bounds.

        Args:
            value: The value to validate.

        Returns:
            The value as a float.

        Raises:
            ValueError: If the value is not numeric or outside bounds.
        """
        field = self.field_name or "value"

        if not isinstance(value, (int, float)):
            raise ValueError(
                f"{field}: Expected numeric value, got {type(value).__name__}"
            )

        value = float(value)
        self._check_bounds(value)
        assert isinstance(value, float)
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
    """Search space for integer parameters.

    IntSpace defines a discrete range of integer values with configurable
    bounds and sampling distribution.

    Examples:
        >>> space = IntSpace(ge=1, le=10)  # Range [1, 10]
        >>> space = IntSpace(gt=0, lt=100)  # Range (0, 100) = [1, 99]
        >>> space = IntSpace(ge=1, le=1000, distribution='log')  # Log scale
    """

    def __init__(
        self,
        *,
        gt: int | None = None,
        ge: int | None = None,
        lt: int | None = None,
        le: int | None = None,
        distribution: Literal["uniform", "log"] = "uniform",
        default: int | _Unset = UNSET,
        description: str | None = None,
    ) -> None:
        """Initialize an IntSpace with integer bounds.

        Args:
            gt: Greater than (exclusive lower bound).
            ge: Greater than or equal (inclusive lower bound).
            lt: Less than (exclusive upper bound).
            le: Less than or equal (inclusive upper bound).
            distribution: Sampling distribution.
            default: Default integer value.
            description: Human-readable description.

        Raises:
            TypeError: If bounds or default are not integers.
            ValueError: If bounds are invalid.
        """
        # Validate that bounds are integers
        for name, value in [("gt", gt), ("ge", ge), ("lt", lt), ("le", le)]:
            if (
                value is not None
                and (not isinstance(value, int))
                or isinstance(value, bool)
            ):
                raise TypeError(
                    f"{name} must be an integer, got {type(value).__name__}"
                )

        # Validate default is integer if provided
        if default is not UNSET and (not isinstance(default, int)):
            raise TypeError(f"default must be an integer, got {type(default).__name__}")

        super().__init__(
            gt=gt,
            ge=ge,
            lt=lt,
            le=le,
            distribution=distribution,
            default=default,
            description=description,
        )

        # Store as integers for cleaner representation
        self._low = int(self._low)
        self._high = int(self._high)

    def validate(self, value: Any) -> int:
        """Validate that a value is a valid integer within bounds.

        Args:
            value: The value to validate.

        Returns:
            The value as an integer.

        Raises:
            ValueError: If the value is not an integer or outside bounds.
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

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Provide Pydantic schema for integer validation."""
        return core_schema.no_info_after_validator_function(
            lambda x: x, core_schema.int_schema()
        )


def Float(
    *,
    gt: float | None = None,
    ge: float | None = None,
    lt: float | None = None,
    le: float | None = None,
    distribution: Literal["uniform", "log"] = "uniform",
    default: float | _Unset = UNSET,
    description: str | None = None,
) -> Any:
    """Factory function for creating a FloatSpace.

    This is the primary user-facing API for defining float parameter spaces.

    Args:
        gt: Greater than (exclusive lower bound).
        ge: Greater than or equal (inclusive lower bound).
        lt: Less than (exclusive upper bound).
        le: Less than or equal (inclusive upper bound).
        distribution: Sampling distribution ('uniform' or 'log').
        default: Default value.
        description: Parameter description.

    Returns:
        A FloatSpace instance.

    Examples:
        >>> import spax as sp
        >>>
        >>> learning_rate = sp.Float(ge=1e-5, le=1e-1, distribution='log')
        >>> temperature = sp.Float(gt=0.0, lt=2.0, default=1.0)
    """
    return FloatSpace(
        gt=gt,
        ge=ge,
        lt=lt,
        le=le,
        distribution=distribution,
        default=default,
        description=description,
    )


def Int(
    *,
    gt: int | None = None,
    ge: int | None = None,
    lt: int | None = None,
    le: int | None = None,
    distribution: Literal["uniform", "log"] = "uniform",
    default: int | _Unset = UNSET,
    description: str | None = None,
) -> Any:
    """Factory function for creating an IntSpace.

    This is the primary user-facing API for defining integer parameter spaces.

    Args:
        gt: Greater than (exclusive lower bound).
        ge: Greater than or equal (inclusive lower bound).
        lt: Less than (exclusive upper bound).
        le: Less than or equal (inclusive upper bound).
        distribution: Sampling distribution ('uniform' or 'log').
        default: Default integer value.
        description: Parameter description.

    Returns:
        An IntSpace instance.

    Examples:
        >>> import spax as sp
        >>>
        >>> num_layers = sp.Int(ge=1, le=10, default=3)
        >>> batch_size = sp.Int(gt=0, le=512, distribution='log')
    """
    return IntSpace(
        gt=gt,
        ge=ge,
        lt=lt,
        le=le,
        distribution=distribution,
        default=default,
        description=description,
    )
