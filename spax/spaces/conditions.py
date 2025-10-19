"""
Conditions used in conditional search spaces for dependent parameter relationships.
"""

from collections.abc import Callable, Iterable
from typing import Any


class Condition:
    """
    Base class for conditions used in conditional spaces.

    Conditions are callable objects that evaluate to True or False
    based on the current configuration state.
    """

    def __call__(self, object: Any) -> bool:
        """
        Evaluate the condition.

        Args:
            object: The object to evaluate against.

        Returns:
            True if the condition is met, False otherwise.
        """
        raise NotImplementedError("Subclasses must implement __call__")

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"{self.__class__.__name__}()"


class FieldCondition(Condition):
    """
    Condition that checks a specific field's value.

    This enables nested conditions by allowing you to check a field
    and then apply another condition to that field's value.

    Example:
        >>> # Check if activation == "relu"
        >>> condition = FieldCondition("activation", EqualsTo("relu"))
        >>>
        >>> # Nested: check if nested_config.num_layers < 5
        >>> condition = FieldCondition(
        ...     "nested_config",
        ...     FieldCondition("num_layers", SmallerThan(5))
        ... )
    """

    def __init__(self, field_name: str, condition: Condition) -> None:
        """
        Initialize a field condition.

        Args:
            field_name: Name of the field to check.
            condition: Condition to apply to the field's value.
        """
        self.field_name = field_name
        self.condition = condition

    def __call__(self, config: Any) -> bool:
        """
        Evaluate by checking the specified field.

        Args:
            config: The configuration object.

        Returns:
            True if the field exists and satisfies the nested condition.
        """
        if not hasattr(config, self.field_name):
            raise AttributeError(
                f"Configuration object has no field '{self.field_name}'"
            )

        field_value = getattr(config, self.field_name)
        return self.condition(field_value)

    def __repr__(self) -> str:
        """Return a string representation."""
        return (
            f"FieldCondition(field='{self.field_name}', condition={self.condition!r})"
        )


class EqualsTo(Condition):
    """
    Condition that checks equality.

    Example:
        >>> condition = EqualsTo("relu")
        >>> condition("relu")  # True
        >>> condition("gelu")  # False
    """

    def __init__(self, value: Any) -> None:
        """
        Initialize with a value to compare against.

        Args:
            value: The value to check for equality.
        """
        if not hasattr(value, "__eq__"):
            raise TypeError(
                f"Value must be comparable (have __eq__ method), "
                f"got {type(value).__name__}"
            )
        self.value = value

    def __call__(self, value: Any) -> bool:
        """Check if the given value equals the stored value."""
        if not hasattr(value, "__eq__"):
            raise TypeError(
                f"Value must be comparable (have __eq__ method), "
                f"got {type(value).__name__}"
            )
        return value == self.value

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"EqualsTo({self.value!r})"


class NotEqualsTo(Condition):
    """
    Condition that checks inequality.

    Example:
        >>> condition = NotEqualsTo("none")
        >>> condition("relu")  # True
        >>> condition("none")  # False
    """

    def __init__(self, value: Any) -> None:
        """
        Initialize with a value to compare against.

        Args:
            value: The value to check for inequality.
        """
        if not hasattr(value, "__eq__"):
            raise TypeError(
                f"Value must be comparable (have __eq__ method), "
                f"got {type(value).__name__}"
            )
        self.value = value

    def __call__(self, value: Any) -> bool:
        """Check if the given value does not equal the stored value."""
        if not hasattr(value, "__eq__"):
            raise TypeError(
                f"Value must be comparable (have __eq__ method), "
                f"got {type(value).__name__}"
            )
        return value != self.value

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"NotEqualsTo({self.value!r})"


class In(Condition):
    """
    Condition that checks membership in a collection.

    Example:
        >>> condition = In(["relu", "gelu", "silu"])
        >>> condition("relu")  # True
        >>> condition("tanh")  # False
    """

    def __init__(self, values: list[Any] | set[Any] | tuple[Any, ...]) -> None:
        """
        Initialize with a collection of valid values.

        Args:
            values: Collection of values to check membership against.
        """
        for value in values:
            if not hasattr(value, "__eq__"):
                raise TypeError(
                    f"All values must be comparable (have __eq__ method), "
                    f"got {type(value).__name__}"
                )
        self.values = values

    def __call__(self, value: Any) -> bool:
        """Check if the given value is in the collection."""
        if not hasattr(value, "__eq__"):
            raise TypeError(
                f"Value must be comparable (have __eq__ method), "
                f"got {type(value).__name__}"
            )
        return value in self.values

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"In({self.values!r})"


class NotIn(Condition):
    """
    Condition that checks non-membership in a collection.

    Example:
        >>> condition = NotIn(["deprecated_opt1", "deprecated_opt2"])
        >>> condition("modern_opt")  # True
        >>> condition("deprecated_opt1")  # False
    """

    def __init__(self, values: list[Any] | set[Any] | tuple[Any, ...]) -> None:
        """
        Initialize with a collection of excluded values.

        Args:
            values: Collection of values to check non-membership against.
        """
        for value in values:
            if not hasattr(value, "__eq__"):
                raise TypeError(
                    f"All values must be comparable (have __eq__ method), "
                    f"got {type(value).__name__}"
                )
        self.values = values

    def __call__(self, value: Any) -> bool:
        """Check if the given value is not in the collection."""
        if not hasattr(value, "__eq__"):
            raise TypeError(
                f"Value must be comparable (have __eq__ method), "
                f"got {type(value).__name__}"
            )
        return value not in self.values

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"NotIn({self.values!r})"


class SmallerThan(Condition):
    """
    Condition that checks if a value is smaller than a threshold.

    Example:
        >>> condition = SmallerThan(10, or_equals=True)
        >>> condition(5)   # True
        >>> condition(10)  # True
        >>> condition(15)  # False
    """

    def __init__(self, value: float | int, or_equals: bool = False) -> None:
        """
        Initialize with a threshold.

        Args:
            value: The threshold to compare against.
            or_equals: If True, allow equality (<=). If False, strict (<).
        """
        if not isinstance(value, (int, float)):
            raise TypeError(f"Expected numeric value, got {type(value).__name__}")
        if not isinstance(or_equals, bool):
            raise TypeError(f"or_equals must be bool, got {type(or_equals).__name__}")
        self.value = value
        self.or_equals = or_equals

    def __call__(self, value: Any) -> bool:
        """Check if the given value is smaller than the threshold."""
        if not isinstance(value, (int, float)):
            raise TypeError(f"Expected numeric value, got {type(value).__name__}")

        if self.or_equals:
            return value <= self.value
        else:
            return value < self.value

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"SmallerThan({self.value}, or_equals={self.or_equals})"


class LargerThan(Condition):
    """
    Condition that checks if a value is larger than a threshold.

    Example:
        >>> condition = LargerThan(0, or_equals=True)
        >>> condition(5)   # True
        >>> condition(0)   # True
        >>> condition(-5)  # False
    """

    def __init__(self, value: float | int, or_equals: bool = False) -> None:
        """
        Initialize with a threshold.

        Args:
            value: The threshold to compare against.
            or_equals: If True, allow equality (>=). If False, strict (>).
        """
        if not isinstance(value, (int, float)):
            raise TypeError(f"Expected numeric value, got {type(value).__name__}")
        if not isinstance(or_equals, bool):
            raise TypeError(f"or_equals must be bool, got {type(or_equals).__name__}")
        self.value = value
        self.or_equals = or_equals

    def __call__(self, value: Any) -> bool:
        """Check if the given value is larger than the threshold."""
        if not isinstance(value, (int, float)):
            raise TypeError(f"Expected numeric value, got {type(value).__name__}")

        if self.or_equals:
            return value >= self.value
        else:
            return value > self.value

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"LargerThan({self.value}, or_equals={self.or_equals})"


class IsInstance(Condition):
    """
    Condition that checks if a value is an instance of a type.

    Example:
        >>> condition = IsInstance(int)
        >>> condition(5)    # True
        >>> condition(5.0)  # False
    """

    def __init__(self, class_or_tuple: type | tuple[type, ...]) -> None:
        """
        Initialize with a type or tuple of types.

        Args:
            class_or_tuple: Type(s) to check against.
        """
        if isinstance(class_or_tuple, tuple):
            for v in class_or_tuple:
                if not isinstance(v, type):
                    raise TypeError(
                        f"All values in tuple must be types, got {type(v).__name__}"
                    )
        else:
            if not isinstance(class_or_tuple, type):
                raise TypeError(
                    f"Expected type or tuple of types, got {type(class_or_tuple).__name__}"
                )
        self.class_or_tuple = class_or_tuple

    def __call__(self, value: Any) -> bool:
        """Check if the given value is an instance of the type(s)."""
        return isinstance(value, self.class_or_tuple)

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"IsInstance({self.class_or_tuple!r})"


class And(Condition):
    """
    Condition that combines multiple conditions with logical AND.

    All sub-conditions must be True for this to be True.

    Example:
        >>> condition = And([
        ...     LargerThan(0),
        ...     SmallerThan(100)
        ... ])
        >>> condition(50)   # True
        >>> condition(150)  # False
    """

    def __init__(self, conditions: Iterable[Condition]) -> None:
        """
        Initialize with an iterable of conditions.

        Args:
            conditions: Iterable of conditions that must all be True.

        Raises:
            TypeError: If conditions is not iterable or contains non-Condition objects.
            ValueError: If conditions is empty.
        """
        # Check if iterable
        try:
            conditions_list = list(conditions)
        except TypeError:
            raise TypeError(
                f"conditions must be iterable, got {type(conditions).__name__}"
            ) from None

        if not conditions_list:
            raise ValueError("And requires at least one condition")

        # Validate all are Condition instances
        for i, cond in enumerate(conditions_list):
            if not isinstance(cond, Condition):
                raise TypeError(
                    f"All conditions must be Condition instances, "
                    f"got {type(cond).__name__} at index {i}"
                )

        self.conditions = conditions_list

    def __call__(self, value: Any) -> bool:
        """Check if all conditions are satisfied."""
        return all(condition(value) for condition in self.conditions)

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"And({self.conditions!r})"


class Or(Condition):
    """
    Condition that combines multiple conditions with logical OR.

    At least one sub-condition must be True for this to be True.

    Example:
        >>> condition = Or([
        ...     EqualsTo("relu"),
        ...     EqualsTo("gelu")
        ... ])
        >>> condition("relu")  # True
        >>> condition("tanh")  # False
    """

    def __init__(self, conditions: Iterable[Condition]) -> None:
        """
        Initialize with an iterable of conditions.

        Args:
            conditions: Iterable of Condition objects where at least one must be True.

        Raises:
            TypeError: If conditions is not iterable or contains non-Condition objects.
            ValueError: If conditions is empty.
        """
        # Check if iterable
        try:
            conditions_list = list(conditions)
        except TypeError:
            raise TypeError(
                f"conditions must be iterable, got {type(conditions).__name__}"
            ) from None

        if not conditions_list:
            raise ValueError("Or requires at least one condition")

        # Validate all are Condition instances
        for i, cond in enumerate(conditions_list):
            if not isinstance(cond, Condition):
                raise TypeError(
                    f"All conditions must be Condition instances, "
                    f"got {type(cond).__name__} at index {i}"
                )

        self.conditions = conditions_list

    def __call__(self, value: Any) -> bool:
        """Check if any condition is satisfied."""
        return any(condition(value) for condition in self.conditions)

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"Or({self.conditions!r})"


class Not(Condition):
    """
    Condition that negates another condition.

    Example:
        >>> condition = Not(EqualsTo("none"))
        >>> condition("relu")  # True
        >>> condition("none")  # False
    """

    def __init__(self, condition: Condition) -> None:
        """
        Initialize with a condition to negate.

        Args:
            condition: The condition to negate.

        Raises:
            TypeError: If condition is not a Condition instance.
        """
        if not isinstance(condition, Condition):
            raise TypeError(
                f"condition must be a Condition instance, "
                f"got {type(condition).__name__}"
            )
        self.condition = condition

    def __call__(self, value: Any) -> bool:
        """Check if the negated condition is satisfied."""
        return not self.condition(value)

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"Not({self.condition!r})"


class Lambda(Condition):
    """
    Condition from a custom callable.

    This allows you to define arbitrary conditions using functions.

    Example:
        >>> condition = Lambda(lambda x: x % 2 == 0)
        >>> condition(4)  # True
        >>> condition(5)  # False
    """

    def __init__(self, func: Callable[[Any], bool]) -> None:
        """
        Initialize with a callable.

        Args:
            func: Function that takes a value and returns a boolean.

        Raises:
            TypeError: If func is not callable.
        """
        if not callable(func):
            raise TypeError(f"func must be callable, got {type(func).__name__}")
        self.func = func

    def __call__(self, value: Any) -> bool:
        """Apply the custom function."""
        result = self.func(value)
        if not isinstance(result, bool):
            raise TypeError(
                f"Lambda condition function must return bool, got {type(result).__name__}"
            )
        return result

    def __repr__(self) -> str:
        """Return a string representation."""
        import inspect

        # Try to get useful information about the function
        func_name = getattr(self.func, "__name__", "<lambda>")

        # Try to get signature
        try:
            sig = inspect.signature(self.func)
            return f"Lambda({func_name}{sig})"
        except (ValueError, TypeError):
            # Fallback if signature extraction fails
            return f"Lambda({func_name})"
