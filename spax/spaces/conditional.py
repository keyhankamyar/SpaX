"""
Conditional search spaces for dependent parameter relationships.

This module provides conditional spaces where the valid range or choice
depends on the value of other fields in the configuration.
"""

from typing import Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from .base import UNSET, Space, _Unset
from .conditions import Condition


class ConditionalSpace(Space[Any]):
    """
    Search space that switches between options based on a condition.

    The condition is evaluated against the configuration object during
    validation and sampling. This enables complex dependent parameter
    relationships.

    Example:
        >>> # Different range based on another field
        >>> learning_rate_multiplier: float = Conditional(
        ...     condition=FieldCondition("optimizer", EqualsTo("sgd")),
        ...     true=Float(0.1, 10.0),      # Higher for SGD
        ...     false=Float(0.01, 1.0)       # Lower for Adam
        ... )
        >>>
        >>> # Nested conditionals
        >>> depth: int = Conditional(
        ...     condition=FieldCondition("model_type", EqualsTo("transformer")),
        ...     true=Conditional(
        ...         condition=FieldCondition("size", EqualsTo("large")),
        ...         true=Int(12, 24),
        ...         false=Int(6, 12)
        ...     ),
        ...     false=Int(1, 6)
        ... )
    """

    def __init__(
        self,
        condition: Condition,
        true: Space[Any] | Any,
        false: Space[Any] | Any,
        default: Any | _Unset = UNSET,
        description: str | None = None,
    ) -> None:
        """
        Initialize a conditional space.

        Args:
            condition: Condition to evaluate.
            true: Space or fixed value to use when condition is True.
            false: Space or fixed value to use when condition is False.
            default: Default value to use when not specified.
            description: Human-readable description of this parameter.
        """
        self.condition = condition
        self.true_branch = true
        self.false_branch = false

        # Store whether branches are spaces or fixed values
        self.true_is_space = isinstance(true, Space)
        self.false_is_space = isinstance(false, Space)

        # Call parent __init__ with default and description
        super().__init__(default=default, description=description)

    def __set_name__(self, owner: type, name: str) -> None:
        """
        Called when the space is assigned to a class attribute.
        Propagates the field name to nested spaces in branches.
        """
        super().__set_name__(owner, name)

        # Propagate field_name to nested spaces
        if self.true_is_space:
            self.true_branch.field_name = name
        if self.false_is_space:
            self.false_branch.field_name = name

    def _get_active_branch(self, config: Any) -> Space[Any] | Any:
        """
        Determine which branch to use based on the condition.

        Args:
            config: The configuration object to evaluate against.

        Returns:
            Either true_branch or false_branch depending on condition.
        """
        if self.condition(config):
            return self.true_branch
        else:
            return self.false_branch

    def validate(self, value: Any) -> Any:
        """
        Validate a value against the appropriate branch.

        Note: This method signature matches the base Space class.
        The config object is accessed via the descriptor protocol's __set__.

        Args:
            value: The value to validate.

        Returns:
            The validated value.

        Raises:
            ValueError: If validation fails.
        """
        # This will be called from __set__ with the config object available
        # We need to get the config from the call context
        # This is handled in the Config class validator
        return value

    def validate_with_config(self, value: Any, config: Any) -> Any:
        """
        Validate a value against the appropriate branch with explicit config.

        Args:
            value: The value to validate.
            config: The configuration object (needed to evaluate condition).

        Returns:
            The validated value.

        Raises:
            ValueError: If validation fails.
            RuntimeError: If the conditional cannot be evaluated.
        """
        try:
            active_branch = self._get_active_branch(config)
        except Exception as e:
            raise RuntimeError(
                f"Failed to evaluate condition for field '{self.field_name}': {e}"
            ) from e

        # If the active branch is a Space, validate through it
        if isinstance(active_branch, Space):
            # Ensure nested space has a field_name for error messages
            if active_branch.field_name is None and self.field_name is not None:
                active_branch.field_name = self.field_name

            # For nested Conditionals, pass config through
            if isinstance(active_branch, ConditionalSpace):
                return active_branch.validate_with_config(value, config)
            else:
                return active_branch.validate(value)
        else:
            # Fixed value - check equality
            if value != active_branch:
                raise ValueError(
                    f"{self.field_name}: Expected fixed value {active_branch!r}, "
                    f"got {value!r}"
                )
            return value

    def sample(self) -> Any:
        """
        Sample from this space.

        Note: Conditional spaces cannot be sampled independently.
        They require a config object with dependency values already set.
        Use Config.random() instead.

        Raises:
            NotImplementedError: Always, as conditionals need config context.
        """
        raise NotImplementedError(
            f"ConditionalSpace '{self.field_name}' cannot be sampled independently. "
            "Use Config.random() to sample the entire configuration with proper "
            "dependency ordering."
        )

    def sample_with_config(self, config: Any) -> Any:
        """
        Sample from the appropriate branch with explicit config.

        Args:
            config: The configuration object (needed to evaluate condition).
                   Must contain values for all fields this conditional depends on.

        Returns:
            A sampled value from the active branch.

        Raises:
            RuntimeError: If condition evaluation fails.
        """
        try:
            active_branch = self._get_active_branch(config)
        except Exception as e:
            raise RuntimeError(
                f"Failed to evaluate condition for field '{self.field_name}': {e}"
            ) from e

        # If the active branch is a Space, sample from it
        if isinstance(active_branch, Space):
            # For nested Conditionals, pass config through
            if isinstance(active_branch, ConditionalSpace):
                return active_branch.sample_with_config(config)
            else:
                return active_branch.sample()
        else:
            # Fixed value
            return active_branch

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
        parts = [
            f"condition={self.condition!r}",
            f"true={self.true_branch!r}",
            f"false={self.false_branch!r}",
        ]

        if self.default is not UNSET:
            parts.append(f"default={self.default!r}")
        if self.description is not None:
            parts.append(f"description={self.description!r}")

        return f"ConditionalSpace({', '.join(parts)})"


def Conditional(
    condition: Condition,
    true: Space[Any] | Any,
    false: Space[Any] | Any,
    default: Any | _Unset = UNSET,
    description: str | None = None,
) -> Any:
    """
    Create a conditional search space (type-checker friendly).

    This function returns Any to satisfy type checkers when used as:
        my_param: float = Conditional(...)

    Args:
        condition: Condition to evaluate.
        true: Space or value to use when condition is True.
        false: Space or value to use when condition is False.
        default: Default value to use when not specified.
        description: Human-readable description of this parameter.

    Returns:
        A ConditionalSpace instance.

    Example:
        >>> class MyConfig(Config):
        ...     optimizer: str = Categorical(["adam", "sgd"])
        ...     learning_rate: float = Conditional(
        ...         condition=FieldCondition("optimizer", EqualsTo("sgd")),
        ...         true=Float(ge=0.01, le=1.0),
        ...         false=Float(ge=0.0001, le=0.01),
        ...         default=0.001,
        ...         description="Learning rate for training"
        ...     )
    """
    return ConditionalSpace(condition, true, false, default, description)
