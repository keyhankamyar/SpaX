"""Attribute conditions for evaluating config object fields.

This module provides conditions that evaluate fields of configuration objects,
enabling conditional parameters that depend on other parameter values.
AttributeConditions are used at the top level of ConditionalSpace to ensure
proper dependency tracking and ordered sampling.
"""

from abc import abstractmethod
from collections.abc import Callable, Iterable
import inspect
from typing import Any

from .base import Condition


class AttributeCondition(Condition):
    """Base class for conditions that evaluate config object attributes.

    AttributeConditions are used in ConditionalSpace to make parameters
    conditional on the values of other parameters. They track which fields
    they depend on to enable proper ordering during sampling and validation.

    Unlike ObjectConditions which evaluate single values, AttributeConditions
    receive the entire config object and extract the relevant field(s) to check.
    """

    @abstractmethod
    def get_required_fields(self) -> set[str]:
        """Get the set of field names this condition depends on.

        This is used for dependency tracking and topological sorting to ensure
        that fields are sampled/validated in the correct order.

        Returns:
            Set of field names that this condition requires.
        """
        pass


class FieldCondition(AttributeCondition):
    """Condition that evaluates a single field of a config object.

    FieldCondition wraps an ObjectCondition and applies it to a specific
    field of the config object. This is the primary way to make parameters
    conditional on other parameters.

    Attributes:
        field_name: The name of the field to evaluate.
        condition: The ObjectCondition to apply to the field value.

    Examples:
        >>> import spax as sp
        >>>
        >>> # Make learning_rate conditional on optimizer choice
        >>> sp.FieldCondition("optimizer", EqualsTo("adam"))

        >>> # Make dropout conditional on model size
        >>> sp.FieldCondition("num_layers", LargerThan(5, or_equals=True))
    """

    def __init__(self, field_name: str, condition: Condition) -> None:
        """Initialize a FieldCondition.

        Args:
            field_name: Name of the field to evaluate.
            condition: ObjectCondition to apply to the field value.

        Raises:
            TypeError: If field_name is not a string or condition is not a Condition.
        """
        if not isinstance(field_name, str):
            raise TypeError(f"field_name must be str, got {type(field_name).__name__}")
        if not isinstance(condition, Condition):
            raise TypeError(
                f"condition must be a Condition instance, got {type(condition).__name__}"
            )

        self._field_name = field_name
        self._condition = condition

    @property
    def field_name(self) -> str:
        """The name of the field to evaluate."""
        return self._field_name

    @property
    def condition(self) -> Condition:
        """The condition to apply to the field value."""
        return self._condition

    def get_required_fields(self) -> set[str]:
        """Get the field this condition depends on.

        Returns:
            Set containing the single field name.
        """
        return {self._field_name}

    def __call__(self, config: Any) -> bool:
        """Evaluate the condition on a config object's field.

        Args:
            config: The config object to evaluate.

        Returns:
            True if the condition is satisfied for the field value.

        Raises:
            AttributeError: If the config object doesn't have the specified field.
        """
        if not hasattr(config, self._field_name):
            raise AttributeError(
                f"Configuration object has no field '{self._field_name}'"
            )

        field_value = getattr(config, self._field_name)
        return self._condition(field_value)

    def __repr__(self) -> str:
        return (
            f"FieldCondition(field='{self._field_name}', condition={self._condition!r})"
        )


class MultiFieldLambdaCondition(AttributeCondition):
    """Condition that evaluates multiple fields using a custom function.

    MultiFieldLambdaCondition allows complex conditions that depend on
    multiple parameter values. The function receives the field values as
    keyword arguments and returns a boolean.

    Attributes:
        field_names: Set of field names this condition depends on.
        func: The function that evaluates the condition.

    Examples:
        >>> import spax as sp
        >>>
        >>> # Condition based on two fields
        >>> sp.MultiFieldLambdaCondition(
        ...     ["batch_size", "num_layers"],
        ...     lambda batch_size, num_layers: batch_size * num_layers < 1000
        ... )

        >>> # Condition with three fields
        >>> sp.MultiFieldLambdaCondition(
        ...     ["optimizer", "learning_rate", "weight_decay"],
        ...     lambda optimizer, learning_rate, weight_decay:
        ...         optimizer == "adam" and learning_rate > 0.001
        ... )
    """

    def __init__(self, field_names: Iterable[str], func: Callable[..., bool]) -> None:
        """Initialize a MultiFieldLambdaCondition.

        Args:
            field_names: Iterable of field names the function depends on.
            func: Callable that takes field values as kwargs and returns bool.
                The function signature must exactly match field_names.

        Raises:
            TypeError: If field_names is not iterable, contains non-strings,
                or func is not callable.
            ValueError: If field_names is empty, contains duplicates, or
                function signature doesn't match field_names.
        """
        # Validate field_names is iterable but not a string
        if not isinstance(field_names, Iterable) or isinstance(field_names, str):
            raise TypeError(
                f"Expected an iterable for field_names, got {type(field_names).__name__}"
            )

        unique_field_names = set(field_names)

        if not unique_field_names:
            raise ValueError("field_names cannot be empty")

        # Validate all field names are strings
        for name in unique_field_names:
            if not isinstance(name, str):
                raise TypeError(
                    f"All field names must be strings, got {type(name).__name__}"
                )

        # Check for duplicates
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

        self._field_names = field_names
        self._func = func

    @property
    def field_names(self) -> set[str]:
        """The set of field names this condition depends on."""
        return self._field_names.copy()

    @property
    def func(self) -> Callable[..., bool]:
        """The function that evaluates the condition."""
        return self._func

    def get_required_fields(self) -> set[str]:
        """Get the fields this condition depends on.

        Returns:
            Set of all field names used by this condition.
        """
        return self._field_names.copy()

    def __call__(self, config: Any) -> bool:
        """Evaluate the condition on a config object.

        Args:
            config: The config object to evaluate.

        Returns:
            True if the condition is satisfied.

        Raises:
            AttributeError: If the config object is missing any required field.
            TypeError: If the function doesn't return a bool.
        """
        # Extract field values
        kwargs = {}
        for field_name in self._field_names:
            if not hasattr(config, field_name):
                raise AttributeError(
                    f"Configuration object has no field '{field_name}'"
                )
            kwargs[field_name] = getattr(config, field_name)

        # Call function with kwargs
        result = self._func(**kwargs)
        if not isinstance(result, bool):
            raise TypeError(
                f"Lambda function must return bool, got {type(result).__name__}"
            )
        return result

    def __repr__(self) -> str:
        """Return a string representation with function signature."""
        try:
            sig = inspect.signature(self._func)
            func_name = getattr(self._func, "__name__", "<lambda>")
            return f"MultiFieldLambdaCondition(fields={self._field_names}, func={func_name}{sig})"
        except (ValueError, TypeError):
            return f"MultiFieldLambdaCondition(fields={self._field_names})"
