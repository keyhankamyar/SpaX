"""
Conditional spaces that activate different branches based on configuration state.

ConditionalSpace allows defining parameters whose type, range, or value depends
on other configuration fields. The top-level condition must be an AttributeCondition
to enable proper dependency tracking and ordered sampling.
"""

from typing import Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from .base import UNSET, Space, _Unset
from .conditions import AttributeCondition


class ConditionalSpace(Space[Any]):
    """
    A space that conditionally activates one of two branches based on config state.

    The condition must be an AttributeCondition (FieldCondition or MultiFieldLambda)
    at the top level to ensure proper dependency tracking. The branches can contain
    any Space types, fixed values, or nested ConditionalSpaces.

    Examples:
        >>> import spax as sp
        >>>
        >>> # Simple field-based conditional
        >>> class MyConfig(sp.Config):
        ...     use_dropout: bool
        ...     dropout_rate: float = sp.Conditional(
        ...         sp.FieldCondition("use_dropout", sp.EqualsTo(True)),
        ...         true=sp.Float(gt=0, lt=0.5),
        ...         false=0.0
        ...     )
        >>>
        >>> # Multi-field conditional
        >>> class TrainingConfig(sp.Config):
        ...     batch_size: int = sp.Int(ge=1, le=128)
        ...     grad_accumulation_steps: int = sp.Int(ge=1, le=32)
        ...     optimizer: str = sp.Conditional(
        ...         sp.MultiFieldLambda(
        ...             ["batch_size", "grad_accumulation_steps"],
        ...             lambda batch_size, grad_accumulation_steps:
        ...                 batch_size * grad_accumulation_steps > 64
        ...         ),
        ...         true="adam",
        ...         false="sgd"
        ...     )
        >>>
        >>> # Nested conditionals with different space types
        >>> class ModelConfig(sp.Config):
        ...     model_type: str = sp.Categorical(["transformer", "cnn", "rnn"])
        ...     hidden_size: int = sp.Conditional(
        ...         sp.FieldCondition("model_type", sp.EqualsTo("transformer")),
        ...         true=sp.Conditional(
        ...             sp.FieldCondition("model_type", sp.EqualsTo("transformer")),
        ...             true=sp.Int(ge=512, le=2048),
        ...             false=sp.Int(ge=128, le=512)
        ...         ),
        ...         false=sp.Int(ge=64, le=256)
        ...     )
    """

    def __init__(
        self,
        condition: AttributeCondition,
        *,
        true: Space[Any] | Any,
        false: Space[Any] | Any,
        default: Any | _Unset = UNSET,
        description: str | None = None,
    ) -> None:
        """
        Initialize a conditional space.

        Args:
            condition: AttributeCondition that determines which branch to activate.
                Must be FieldCondition or MultiFieldLambda at the top level.
            true: Space or fixed value used when condition evaluates to True
            false: Space or fixed value used when condition evaluates to False
            default: Default value for the space
            description: Description of this conditional parameter

        Raises:
            TypeError: If condition is not an AttributeCondition
        """
        if not isinstance(condition, AttributeCondition):
            raise TypeError(
                f"ConditionalSpace requires an AttributeCondition "
                f"(FieldCondition or MultiFieldLambda) at the top level, "
                f"got {type(condition).__name__}. AttributeConditions are required "
                f"for proper dependency tracking and ordered sampling."
            )

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
    condition: AttributeCondition,
    *,
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
        condition: AttributeCondition determining which branch to activate
        true: Space or value when condition is True
        false: Space or value when condition is False
        default: Default value
        description: Description of the parameter

    Returns:
        A ConditionalSpace instance.

    Examples:
        >>> import spax as sp
        >>>
        >>> # Simple field-based conditional
        >>> class MyConfig(sp.Config):
        ...     use_dropout: bool
        ...     dropout_rate: float = sp.Conditional(
        ...         sp.FieldCondition("use_dropout", sp.EqualsTo(True)),
        ...         true=sp.Float(gt=0, lt=0.5),
        ...         false=0.0
        ...     )
        >>>
        >>> # Multi-field conditional
        >>> class TrainingConfig(sp.Config):
        ...     batch_size: int = sp.Int(ge=1, le=128)
        ...     grad_accumulation_steps: int = sp.Int(ge=1, le=32)
        ...     optimizer: str = sp.Conditional(
        ...         sp.MultiFieldLambda(
        ...             ["batch_size", "grad_accumulation_steps"],
        ...             lambda batch_size, grad_accumulation_steps:
        ...                 batch_size * grad_accumulation_steps > 64
        ...         ),
        ...         true="adam",
        ...         false="sgd"
        ...     )
        >>>
        >>> # Nested conditionals with different space types
        >>> class ModelConfig(sp.Config):
        ...     model_type: str = sp.Categorical(["transformer", "cnn", "rnn"])
        ...     hidden_size: int = sp.Conditional(
        ...         sp.FieldCondition("model_type", sp.EqualsTo("transformer")),
        ...         true=sp.Conditional(
        ...             sp.FieldCondition("model_type", sp.EqualsTo("transformer")),
        ...             true=sp.Int(ge=512, le=2048),
        ...             false=sp.Int(ge=128, le=512)
        ...         ),
        ...         false=sp.Int(ge=64, le=256)
        ...     )
    """
    return ConditionalSpace(
        condition=condition,
        true=true,
        false=false,
        default=default,
        description=description,
    )
