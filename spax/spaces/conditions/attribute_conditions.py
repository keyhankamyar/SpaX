"""
Attribute-based conditions that depend on specific config fields.

These conditions can be used at the top level of ConditionalSpace because they
explicitly declare which fields they depend on, enabling proper dependency
tracking and ordered sampling.
"""

from abc import abstractmethod
from collections.abc import Callable, Iterable
import inspect
from typing import Any

from .base import Condition


class AttributeCondition(Condition):
    """
    Base class for conditions that depend on specific configuration fields.

    AttributeConditions can be used at the top level of ConditionalSpace because
    they explicitly declare their field dependencies via get_required_fields().
    This enables proper dependency tracking and ensures fields are sampled in
    the correct order.
    """

    @abstractmethod
    def get_required_fields(self) -> set[str]:
        """
        Return list of field names this condition depends on.

        Returns:
            Set of field names that must be available to evaluate this condition
        """
        pass


class FieldCondition(AttributeCondition):
    """
    Condition that evaluates another condition on a specific field's value.

    Examples:
        >>> import spax as sp
        >>>
        >>> class MyConfig(sp.Config):
        ...     learning_rate: float = sp.Float(gt=0, lt=1)
        ...     optimizer: str = sp.Conditional(
        ...         sp.FieldCondition("learning_rate", sp.LargerThan(0.01)),
        ...         true="adam",
        ...         false="sgd"
        ...     )
        >>>
        >>> # When learning_rate > 0.01, optimizer will be "adam"
        >>> config = MyConfig(learning_rate=0.1, optimizer="adam")
        >>>
        >>> # When learning_rate <= 0.01, optimizer must be "sgd"
        >>> config = MyConfig(learning_rate=0.001, optimizer="sgd")
    """

    def __init__(self, field_name: str, condition: Condition) -> None:
        """
        Initialize a field condition.

        Args:
            field_name: Name of the field to check
            condition: Condition to apply to the field's value
        """
        if not isinstance(field_name, str):
            raise TypeError(f"field_name must be str, got {type(field_name).__name__}")
        if not isinstance(condition, Condition):
            raise TypeError(
                f"condition must be a Condition instance, got {type(condition).__name__}"
            )

        self.field_name = field_name
        self.condition = condition

    def get_required_fields(self) -> set[str]:
        """Return the single field this condition depends on."""
        return {self.field_name}

    def __call__(self, config: Any) -> bool:
        """
        Evaluate the condition on the specified field.

        Args:
            config: Configuration object with the required field

        Returns:
            Result of applying the condition to the field's value
        """
        if not hasattr(config, self.field_name):
            raise AttributeError(
                f"Configuration object has no field '{self.field_name}'"
            )
        field_value = getattr(config, self.field_name)
        return self.condition(field_value)

    def __repr__(self) -> str:
        return (
            f"FieldCondition(field='{self.field_name}', condition={self.condition!r})"
        )


class MultiFieldLambdaCondition(AttributeCondition):
    """
    Condition based on a lambda function applied to multiple fields.

    The lambda function receives field values as keyword arguments and must
    return a boolean.

    Example:
        >>> import spax as sp
        >>>
        >>> class TrainingConfig(sp.Config):
        ...     grad_accum_steps: int = sp.Int(ge=1, le=32)
        ...     batch_size: int = sp.Int(ge=1, le=128)
        ...     epochs: int = sp.Int(ge=1, le=100)
        ...     use_mixed_precision: bool = sp.Conditional(
        ...         sp.MultiFieldLambdaCondition(
        ...             ["grad_accum_steps", "batch_size"],
        ...             lambda grad_accum_steps, batch_size:
        ...                 grad_accum_steps * batch_size > 64
        ...         ),
        ...         true=True,
        ...         false=False
        ...     )
        >>>
        >>> # Effective batch size = 8 * 16 = 128 > 64, so mixed precision is enabled
        >>> config = TrainingConfig(
        ...     grad_accum_steps=8,
        ...     batch_size=16,
        ...     epochs=10,
        ...     use_mixed_precision=True
        ... )
        >>>
        >>> # More complex example with multiple conditions
        >>> class AdvancedConfig(sp.Config):
        ...     grad_accum_steps: int = sp.Int(ge=1, le=32)
        ...     num_batches: int = sp.Int(ge=10, le=1000)
        ...     epochs: int = sp.Int(ge=1, le=100)
        ...     lr_schedule: str = sp.Conditional(
        ...         sp.MultiFieldLambdaCondition(
        ...             ["grad_accum_steps", "num_batches", "epochs"],
        ...             lambda grad_accum_steps, num_batches, epochs:
        ...                 grad_accum_steps * num_batches > 100 and epochs < 20
        ...         ),
        ...         true="cosine",
        ...         false="constant"
        ...     )
    """

    def __init__(self, field_names: Iterable[str], func: Callable[..., bool]) -> None:
        """
        Initialize a multi-field lambda condition.

        Args:
            field_names: Iterable of field names the function depends on.
                        Must contain non-duplicated strings
            func: Callable that takes field values as kwargs and returns bool
        """
        if not isinstance(field_names, Iterable) or isinstance(field_names, str):
            raise TypeError(
                f"Expected an iterable for field_names, got {type(field_names).__name__}"
            )

        unique_field_names = set(field_names)
        if not unique_field_names:
            raise ValueError("field_names cannot be empty")

        for name in unique_field_names:
            if not isinstance(name, str):
                raise TypeError(
                    f"All field names must be strings, got {type(name).__name__}"
                )

        if len(unique_field_names) != len(list(field_names)):
            raise ValueError("field_names cannot contain duplicates")

        field_names = unique_field_names

        if not callable(func):
            raise TypeError(f"func must be callable, got {type(func).__name__}")

        # Validate function signature matches field names
        try:
            sig = inspect.signature(func)
            params = set(sig.parameters.keys())

            if params != field_names:
                raise ValueError(
                    f"Function parameters {params} must match field_names {field_names} "
                )
        except (ValueError, TypeError) as e:
            raise TypeError(f"Could not validate function signature: {e}") from e

        self.field_names = field_names
        self.func = func

    def get_required_fields(self) -> set[str]:
        """Return the set of fields this condition depends on."""
        return self.field_names.copy()

    def __call__(self, config: Any) -> bool:
        """
        Evaluate the lambda function with field values from config.

        Args:
            config: Configuration object with the required fields

        Returns:
            Result of the lambda function
        """
        # Extract field values
        kwargs = {}
        for field_name in self.field_names:
            if not hasattr(config, field_name):
                raise AttributeError(
                    f"Configuration object has no field '{field_name}'"
                )
            kwargs[field_name] = getattr(config, field_name)

        # Call function with kwargs
        result = self.func(**kwargs)

        if not isinstance(result, bool):
            raise TypeError(
                f"Lambda function must return bool, got {type(result).__name__}"
            )

        return result

    def __repr__(self) -> str:
        try:
            sig = inspect.signature(self.func)
            func_name = getattr(self.func, "__name__", "<lambda>")
            return f"MultiFieldLambdaCondition(fields={self.field_names}, func={func_name}{sig})"
        except (ValueError, TypeError):
            return f"MultiFieldLambdaCondition(fields={self.field_names})"
