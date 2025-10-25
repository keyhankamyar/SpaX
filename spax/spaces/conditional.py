"""Conditional spaces for parameter dependencies.

This module provides ConditionalSpace, which represents parameters whose
possible values depend on the values of other parameters in the configuration.

Conditional spaces have:
- A condition (must be an AttributeCondition at top level)
- A true branch (active when condition evaluates to True)
- A false branch (active when condition evaluates to False)

Condition Requirements:
----------------------
The top-level condition must be an AttributeCondition to enable proper
dependency tracking and ordered sampling. This includes:
- FieldCondition
- MultiFieldLambdaCondition
- Composite conditions (And, Or, Not) containing AttributeConditions

ObjectConditions (EqualsTo, In, etc.) can only be used INSIDE AttributeConditions,
not as the top-level condition.

Valid Top-Level Conditions:
    ✓ sp.FieldCondition("use_dropout", sp.EqualsTo(True))
    ✓ sp.And([
          sp.FieldCondition("use_l2", sp.EqualsTo(True)),
          sp.FieldCondition("use_dropout", sp.EqualsTo(True))
      ])
    ✓ sp.Not(sp.FieldCondition("use_batch_norm", sp.EqualsTo(True)))

Invalid Top-Level Conditions:
    ✗ sp.EqualsTo(True)  # No field dependency
    ✗ sp.In([1, 2, 3])   # No field dependency
    ✗ sp.And([sp.EqualsTo(True), sp.EqualsTo(False)])  # ObjectConditions in composite

Examples:
    >>> import spax as sp
    >>>
    >>> # Basic conditional: dropout rate depends on use_dropout
    >>> class MyConfig(sp.Config):
    ...     use_dropout: bool = sp.Categorical([True, False])
    ...     dropout_rate: float = sp.Conditional(
    ...         sp.FieldCondition("use_dropout", sp.EqualsTo(True)),
    ...         true=sp.Float(gt=0.0, lt=0.5),
    ...         false=0.0
    ...     )
    >>>
    >>> # Composite condition: parameter depends on multiple fields
    >>> class MyConfig(sp.Config):
    ...     use_l2: bool = sp.Categorical([True, False])
    ...     use_dropout: bool = sp.Categorical([True, False])
    ...     strong_reg: bool = sp.Conditional(
    ...         sp.And([
    ...             sp.FieldCondition("use_l2", sp.EqualsTo(True)),
    ...             sp.FieldCondition("use_dropout", sp.EqualsTo(True))
    ...         ]),
    ...         true=sp.Categorical([True, False]),
    ...         false=False
    ...     )
"""

from typing import Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from .base import Space
from .conditions import AttributeCondition


class ConditionalSpace(Space[Any]):
    """Search space with branches that depend on other parameter values.

    ConditionalSpace allows parameters to be conditional on the values of
    other parameters. The condition evaluates to True or False, determining
    which branch (true or false) is active. Each branch can be a Space
    (for sampling) or a fixed value.

    ConditionalSpaces require AttributeCondition at the top level to enable
    proper dependency tracking and ordered sampling/validation. This includes:
    - FieldCondition (single field dependency)
    - MultiFieldLambdaCondition (multiple field dependencies)
    - Composite conditions: And, Or, Not (with AttributeCondition children)

    Attributes:
        condition: The AttributeCondition that determines which branch is active.
        true_branch: The space or value used when condition is True.
        false_branch: The space or value used when condition is False.
        true_is_space: Whether true_branch is a Space (vs. fixed value).
        false_is_space: Whether false_branch is a Space (vs. fixed value).

    Examples:
        >>> import spax as sp
        >>>
        >>> # Simple field-based conditional
        >>> class MyConfig(sp.Config):
        ...     use_dropout: bool
        ...     dropout_rate: float = sp.Conditional(
        ...         sp.FieldCondition("use_dropout", sp.EqualsTo(True)),
        ...         true=sp.Float(gt=0.0, lt=0.5),
        ...         false=0.0
        ...     )
        >>>
        >>> # Composite condition (And)
        >>> class TrainingConfig(sp.Config):
        ...     use_l2: bool
        ...     use_dropout: bool
        ...     strong_reg: bool = sp.Conditional(
        ...         sp.And([
        ...             sp.FieldCondition("use_l2", sp.EqualsTo(True)),
        ...             sp.FieldCondition("use_dropout", sp.EqualsTo(True))
        ...         ]),
        ...         true=sp.Categorical([True, False]),
        ...         false=False
        ...     )
        >>>
        >>> # Multi-field conditional
        >>> class OptConfig(sp.Config):
        ...     batch_size: int = sp.Int(ge=1, le=128)
        ...     grad_accum: int = sp.Int(ge=1, le=32)
        ...     optimizer: str = sp.Conditional(
        ...         sp.MultiFieldLambdaCondition(
        ...             ["batch_size", "grad_accum"],
        ...             lambda bs, ga: bs * ga > 64
        ...         ),
        ...         true="adam",
        ...         false="sgd"
        ...     )
        >>>
        >>> # Nested conditionals
        >>> class ModelConfig(sp.Config):
        ...     model_type: str = sp.Categorical(["small", "large"])
        ...     hidden_size: int = sp.Conditional(
        ...         sp.FieldCondition("model_type", sp.EqualsTo("large")),
        ...         true=sp.Int(ge=512, le=2048),
        ...         false=sp.Int(ge=64, le=256)
        ...     )
    """

    def __init__(
        self,
        condition: AttributeCondition,
        *,
        true: Space[Any] | Any,
        false: Space[Any] | Any,
        description: str | None = None,
    ) -> None:
        """Initialize a ConditionalSpace.

        Args:
            condition: AttributeCondition that determines which branch is active.
                Must be FieldCondition, MultiFieldLambdaCondition, or a composite
                condition (And/Or/Not) containing only AttributeConditions for
                proper dependency tracking.
            true: Space or fixed value to use when condition is True.
            false: Space or fixed value to use when condition is False.
            description: Human-readable description.

        Raises:
            TypeError: If condition is not an AttributeCondition at top level,
                or if composite conditions contain ObjectConditions.
        """
        if not isinstance(condition, AttributeCondition):
            raise TypeError(
                f"ConditionalSpace requires an AttributeCondition (FieldCondition, "
                f"MultiFieldLambdaCondition, or composite conditions like And/Or/Not "
                f"with AttributeCondition children) at the top level, got {type(condition).__name__}. "
                f"AttributeConditions are required for proper dependency tracking and ordered sampling. "
                f"Hint: ObjectConditions (EqualsTo, In, etc.) can only be used INSIDE "
                f"AttributeConditions, not as the top-level condition."
            )

        self._condition = condition
        self._true_branch = true
        self._false_branch = false

        # Store whether branches are spaces or fixed values
        self._true_is_space = isinstance(true, Space)
        self._false_is_space = isinstance(false, Space)

        # Call parent __init__ (no default for ConditionalSpace)
        super().__init__(description=description)

    @property
    def condition(self) -> AttributeCondition:
        """The condition that determines which branch is active."""
        return self._condition

    @property
    def true_branch(self) -> Space[Any] | Any:
        """The space or value used when condition is True."""
        return self._true_branch

    @property
    def false_branch(self) -> Space[Any] | Any:
        """The space or value used when condition is False."""
        return self._false_branch

    @property
    def true_is_space(self) -> bool:
        """Whether true_branch is a Space (vs. fixed value)."""
        return self._true_is_space

    @property
    def false_is_space(self) -> bool:
        """Whether false_branch is a Space (vs. fixed value)."""
        return self._false_is_space

    def __set_name__(self, owner: type, name: str) -> None:
        """Set the field name and propagate to nested spaces.

        Args:
            owner: The class that owns this descriptor.
            name: The name of the attribute.
        """
        super().__set_name__(owner, name)

        # Propagate field_name to nested spaces
        if self._true_is_space:
            self._true_branch.field_name = name
        if self._false_is_space:
            self._false_branch.field_name = name

    def _get_active_branch(self, config: Any) -> Space[Any] | Any:
        """Get the active branch based on the condition.

        Args:
            config: Config object to evaluate the condition on.

        Returns:
            The active branch (true or false).
        """
        if self._condition(config):
            return self._true_branch
        else:
            return self._false_branch

    def contains(self, other: Space) -> bool:
        """Check if another space is contained within this space.

        Not implemented for ConditionalSpace due to complexity.

        Raises:
            NotImplementedError: Always raised.
        """
        raise NotImplementedError

    def validate(self, value: Any) -> Any:
        """Validate a value (requires config context).

        This method is called from __set__ but needs the config object
        for condition evaluation. The actual validation is handled by
        validate_with_config() which is called from the Config class validator.

        Args:
            value: The value to validate.

        Returns:
            The value (validation happens in validate_with_config).
        """
        # This will be called from __set__ with the config object available
        # The actual validation is handled in the Config class validator
        return value

    def validate_with_config(self, value: Any, config: Any) -> Any:
        """Validate a value using the config object for condition evaluation.

        Args:
            value: The value to validate.
            config: The config object to evaluate the condition on.

        Returns:
            The validated value.

        Raises:
            ValueError: If the value is invalid for the active branch.
            RuntimeError: If condition evaluation fails.
        """
        from spax.config import Config

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
            # Fixed value - check equality or type match
            if isinstance(active_branch, type) and issubclass(active_branch, Config):
                if not isinstance(value, active_branch):
                    raise ValueError(
                        f"{self.field_name}: Expected {active_branch.__name__} instance, "
                        f"got {value!r}"
                    )
            # Fixed value - check equality
            elif value != active_branch:
                raise ValueError(
                    f"{self.field_name}: Expected fixed value {active_branch!r}, "
                    f"got {value!r}"
                )
            return value

    def sample(self) -> Any:
        """Sample a value (not supported without config context).

        ConditionalSpace cannot be sampled independently because it needs
        the config object to evaluate the condition. Use Config.random()
        instead, which samples the entire configuration with proper
        dependency ordering.

        Raises:
            NotImplementedError: Always raised.
        """
        raise NotImplementedError(
            f"ConditionalSpace '{self.field_name}' cannot be sampled independently. "
            "Use Config.random() to sample the entire configuration with proper "
            "dependency ordering."
        )

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Provide Pydantic schema for conditional validation."""
        return core_schema.no_info_after_validator_function(
            lambda x: x, core_schema.any_schema()
        )

    def __repr__(self) -> str:
        """Return a string representation of this space."""
        parts = [
            f"condition={self._condition!r}",
            f"true={self._true_branch!r}",
            f"false={self._false_branch!r}",
        ]
        if self.description is not None:
            parts.append(f"description={self.description!r}")

        return f"ConditionalSpace({', '.join(parts)})"


def Conditional(
    condition: AttributeCondition,
    *,
    true: Space[Any] | Any,
    false: Space[Any] | Any,
    description: str | None = None,
) -> Any:
    """Factory function for creating a ConditionalSpace.

    This is the primary user-facing API for defining conditional parameter spaces.

    Args:
        condition: AttributeCondition that determines which branch is active.
        true: Space or fixed value to use when condition is True.
        false: Space or fixed value to use when condition is False.
        description: Parameter description.

    Returns:
        A ConditionalSpace instance.

    Examples:
        >>> import spax as sp
        >>>
        >>> # Conditional with fixed false value
        >>> dropout_rate = sp.Conditional(
        ...     sp.FieldCondition("use_dropout", sp.EqualsTo(True)),
        ...     true=sp.Float(gt=0.0, lt=0.5),
        ...     false=0.0
        ... )
        >>>
        >>> # Conditional with both branches as spaces
        >>> learning_rate = sp.Conditional(
        ...     sp.FieldCondition("optimizer", sp.EqualsTo("adam")),
        ...     true=sp.Float(ge=1e-4, le=1e-2, distribution="log"),
        ...     false=sp.Float(ge=1e-3, le=1e-1, distribution="log")
        ... )
    """
    return ConditionalSpace(
        condition=condition,
        true=true,
        false=false,
        description=description,
    )
