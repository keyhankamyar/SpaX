"""Object conditions for evaluating single values.

This module provides conditions that operate on single values, such as
equality checks, membership tests, comparisons, and logical combinations.
These are used in ConditionalSpace to make parameters conditional on
the value of another parameter.
"""

from collections.abc import Callable, Iterable
from typing import Any

from spax.utils import is_comparable

from .base import Condition


class ObjectCondition(Condition):
    """Base class for conditions that evaluate single values.

    ObjectConditions take a single value as input (not a config object)
    and return True or False. They are typically used with FieldCondition
    to evaluate specific fields of a config.

    Examples:
        >>> FieldCondition("optimizer", EqualsTo("adam"))
        >>> FieldCondition("learning_rate", LargerThan(0.001))
    """

    pass


class EqualsTo(ObjectCondition):
    """Condition that checks if a value equals a target value.

    Attributes:
        value: The target value to compare against.

    Examples:
        >>> import spax as sp
        >>>
        >>> cond = sp.EqualsTo("adam")
        >>> cond("adam")  # True
        >>> cond("sgd")   # False
    """

    def __init__(self, value: Any) -> None:
        """Initialize an EqualsTo condition.

        Args:
            value: The target value to compare against. Must be comparable.

        Raises:
            TypeError: If value is not comparable.
        """
        if not is_comparable(value):
            raise TypeError(f"Value must be comparable, got {type(value).__name__}")
        self._value = value

    @property
    def value(self) -> Any:
        """The target value to compare against."""
        return self._value

    def __call__(self, value: Any) -> bool:
        """Check if value equals the target value.

        Args:
            value: The value to check.

        Returns:
            True if value equals the target value.

        Raises:
            TypeError: If value is not comparable.
        """
        if not is_comparable(value):
            raise TypeError(f"Value must be comparable, got {type(value).__name__}")
        return value == self._value

    def __repr__(self) -> str:
        return f"EqualsTo({self._value!r})"


class NotEqualsTo(ObjectCondition):
    """Condition that checks if a value does not equal a target value.

    Attributes:
        value: The target value to compare against.

    Examples:
        >>> import spax as sp
        >>>
        >>> cond = sp.NotEqualsTo("adam")
        >>> cond("sgd")   # True
        >>> cond("adam")  # False
    """

    def __init__(self, value: Any) -> None:
        """Initialize a NotEqualsTo condition.

        Args:
            value: The target value to compare against. Must be comparable.

        Raises:
            TypeError: If value is not comparable.
        """
        if not is_comparable(value):
            raise TypeError(f"Value must be comparable, got {type(value).__name__}")
        self._value = value

    @property
    def value(self) -> Any:
        """The target value to compare against."""
        return self._value

    def __call__(self, value: Any) -> bool:
        """Check if value does not equal the target value.

        Args:
            value: The value to check.

        Returns:
            True if value does not equal the target value.

        Raises:
            TypeError: If value is not comparable.
        """
        if not is_comparable(value):
            raise TypeError(f"Value must be comparable, got {type(value).__name__}")
        return value != self._value

    def __repr__(self) -> str:
        return f"NotEqualsTo({self._value!r})"


class In(ObjectCondition):
    """Condition that checks if a value is in a set of allowed values.

    Attributes:
        values: The set of allowed values.

    Examples:
        >>> import spax as sp
        >>>
        >>> cond = sp.In(["adam", "sgd", "rmsprop"])
        >>> cond("adam")  # True
        >>> cond("adagrad")  # False
    """

    def __init__(self, values: list[Any] | set[Any] | tuple[Any, ...]) -> None:
        """Initialize an In condition.

        Args:
            values: Collection of allowed values. All must be comparable.

        Raises:
            TypeError: If any value is not comparable.
        """
        for value in values:
            if not is_comparable(value):
                raise TypeError(f"Value must be comparable, got {type(value).__name__}")
        self._values = values

    @property
    def values(self) -> list[Any] | set[Any] | tuple[Any, ...]:
        """The set of allowed values."""
        return self._values

    def __call__(self, value: Any) -> bool:
        """Check if value is in the allowed values.

        Args:
            value: The value to check.

        Returns:
            True if value is in the allowed values.

        Raises:
            TypeError: If value is not comparable.
        """
        if not is_comparable(value):
            raise TypeError(f"Value must be comparable, got {type(value).__name__}")
        return value in self._values

    def __repr__(self) -> str:
        return f"In({self._values!r})"


class NotIn(ObjectCondition):
    """Condition that checks if a value is not in a set of disallowed values.

    Attributes:
        values: The set of disallowed values.

    Examples:
        >>> import spax as sp
        >>>
        >>> cond = sp.NotIn(["deprecated_optimizer", "old_optimizer"])
        >>> cond("adam")  # True
        >>> cond("deprecated_optimizer")  # False
    """

    def __init__(self, values: list[Any] | set[Any] | tuple[Any, ...]) -> None:
        """Initialize a NotIn condition.

        Args:
            values: Collection of disallowed values. All must be comparable.

        Raises:
            TypeError: If any value is not comparable.
        """
        for value in values:
            if not is_comparable(value):
                raise TypeError(f"Value must be comparable, got {type(value).__name__}")
        self._values = values

    @property
    def values(self) -> list[Any] | set[Any] | tuple[Any, ...]:
        """The set of disallowed values."""
        return self._values

    def __call__(self, value: Any) -> bool:
        """Check if value is not in the disallowed values.

        Args:
            value: The value to check.

        Returns:
            True if value is not in the disallowed values.

        Raises:
            TypeError: If value is not comparable.
        """
        if not is_comparable(value):
            raise TypeError(f"Value must be comparable, got {type(value).__name__}")
        return value not in self._values

    def __repr__(self) -> str:
        return f"NotIn({self._values!r})"


class SmallerThan(ObjectCondition):
    """Condition that checks if a value is smaller than a threshold.

    Attributes:
        value: The threshold value.
        or_equals: Whether to include equality (<=) or not (<).

    Examples:
        >>> import spax as sp
        >>>
        >>> cond = sp.SmallerThan(0.5)
        >>> cond(0.3)  # True
        >>> cond(0.5)  # False
        >>> cond_eq = sp.SmallerThan(0.5, or_equals=True)
        >>> cond_eq(0.5)  # True
    """

    def __init__(self, value: float | int, or_equals: bool = False) -> None:
        """Initialize a SmallerThan condition.

        Args:
            value: The threshold value to compare against.
            or_equals: If True, use <= instead of <.

        Raises:
            TypeError: If value is not numeric or or_equals is not bool.
        """
        if not isinstance(value, (int, float)):
            raise TypeError(f"Expected numeric value, got {type(value).__name__}")
        if not isinstance(or_equals, bool):
            raise TypeError(f"or_equals must be bool, got {type(or_equals).__name__}")

        self._value = value
        self._or_equals = or_equals

    @property
    def value(self) -> float | int:
        """The threshold value."""
        return self._value

    @property
    def or_equals(self) -> bool:
        """Whether equality is included (<=) or not (<)."""
        return self._or_equals

    def __call__(self, value: Any) -> bool:
        """Check if value is smaller than the threshold.

        Args:
            value: The value to check. Must be numeric.

        Returns:
            True if value < threshold (or <= if or_equals=True).

        Raises:
            TypeError: If value is not numeric.
        """
        if not isinstance(value, (int, float)):
            raise TypeError(f"Expected numeric value, got {type(value).__name__}")

        if self._or_equals:
            return value <= self._value
        else:
            return value < self._value

    def __repr__(self) -> str:
        return f"SmallerThan({self._value}, or_equals={self._or_equals})"


class LargerThan(ObjectCondition):
    """Condition that checks if a value is larger than a threshold.

    Attributes:
        value: The threshold value.
        or_equals: Whether to include equality (>=) or not (>).

    Examples:
        >>> import spax as sp
        >>>
        >>> cond = sp.LargerThan(0.5)
        >>> cond(0.7)  # True
        >>> cond(0.5)  # False
        >>> cond_eq = sp.LargerThan(0.5, or_equals=True)
        >>> cond_eq(0.5)  # True
    """

    def __init__(self, value: float | int, or_equals: bool = False) -> None:
        """Initialize a LargerThan condition.

        Args:
            value: The threshold value to compare against.
            or_equals: If True, use >= instead of >.

        Raises:
            TypeError: If value is not numeric or or_equals is not bool.
        """
        if not isinstance(value, (int, float)):
            raise TypeError(f"Expected numeric value, got {type(value).__name__}")
        if not isinstance(or_equals, bool):
            raise TypeError(f"or_equals must be bool, got {type(or_equals).__name__}")

        self._value = value
        self._or_equals = or_equals

    @property
    def value(self) -> float | int:
        """The threshold value."""
        return self._value

    @property
    def or_equals(self) -> bool:
        """Whether equality is included (>=) or not (>)."""
        return self._or_equals

    def __call__(self, value: Any) -> bool:
        """Check if value is larger than the threshold.

        Args:
            value: The value to check. Must be numeric.

        Returns:
            True if value > threshold (or >= if or_equals=True).

        Raises:
            TypeError: If value is not numeric.
        """
        if not isinstance(value, (int, float)):
            raise TypeError(f"Expected numeric value, got {type(value).__name__}")

        if self._or_equals:
            return value >= self._value
        else:
            return value > self._value

    def __repr__(self) -> str:
        return f"LargerThan({self._value}, or_equals={self._or_equals})"


class IsInstance(ObjectCondition):
    """Condition that checks if a value is an instance of a type.

    Attributes:
        class_or_tuple: The type(s) to check against.

    Examples:
        >>> import spax as sp
        >>>
        >>> cond = sp.IsInstance(str)
        >>> cond("hello")  # True
        >>> cond(123)      # False
        >>> cond = sp.IsInstance((int, float))
        >>> cond(123)      # True
        >>> cond(1.5)      # True
    """

    def __init__(self, class_or_tuple: type | tuple[type, ...]) -> None:
        """Initialize an IsInstance condition.

        Args:
            class_or_tuple: A type or tuple of types to check against.

        Raises:
            TypeError: If argument is not a type or tuple of types.
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

        self._class_or_tuple = class_or_tuple

    @property
    def class_or_tuple(self) -> type | tuple[type, ...]:
        """The type(s) to check against."""
        return self._class_or_tuple

    def __call__(self, value: Any) -> bool:
        """Check if value is an instance of the type(s).

        Args:
            value: The value to check.

        Returns:
            True if value is an instance of the specified type(s).
        """
        return isinstance(value, self._class_or_tuple)

    def __repr__(self) -> str:
        return f"IsInstance({self._class_or_tuple!r})"


class And(ObjectCondition):
    """Condition that requires all sub-conditions to be True.

    Attributes:
        conditions: List of conditions that must all be satisfied.

    Examples:
        >>> import spax as sp
        >>>
        >>> cond = sp.And([sp.LargerThan(0), sp.SmallerThan(1)])
        >>> cond(0.5)  # True (0 < 0.5 < 1)
        >>> cond(1.5)  # False (not < 1)
    """

    def __init__(self, conditions: Iterable[Condition]) -> None:
        """Initialize an And condition.

        Args:
            conditions: Iterable of Condition objects that must all be True.

        Raises:
            TypeError: If conditions is not iterable or contains non-Conditions.
            ValueError: If no conditions are provided.
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

        self._conditions = conditions_list

    @property
    def conditions(self) -> list[Condition]:
        """List of conditions that must all be satisfied."""
        return self._conditions.copy()

    def __call__(self, value: Any) -> bool:
        """Check if all sub-conditions are satisfied.

        Args:
            value: The value to check against all conditions.

        Returns:
            True if all conditions are True, False otherwise.
        """
        return all(condition(value) for condition in self._conditions)

    def __repr__(self) -> str:
        return f"And({self._conditions!r})"


class Or(ObjectCondition):
    """Condition that requires at least one sub-condition to be True.

    Attributes:
        conditions: List of conditions where at least one must be satisfied.

    Examples:
        >>> import spax as sp
        >>>
        >>> cond = sp.Or([sp.EqualsTo("adam"), sp.EqualsTo("sgd")])
        >>> cond("adam")     # True
        >>> cond("rmsprop")  # False
    """

    def __init__(self, conditions: Iterable[Condition]) -> None:
        """Initialize an Or condition.

        Args:
            conditions: Iterable of Condition objects where at least one must be True.

        Raises:
            TypeError: If conditions is not iterable or contains non-Conditions.
            ValueError: If no conditions are provided.
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

        self._conditions = conditions_list

    @property
    def conditions(self) -> list[Condition]:
        """List of conditions where at least one must be satisfied."""
        return self._conditions.copy()

    def __call__(self, value: Any) -> bool:
        """Check if at least one sub-condition is satisfied.

        Args:
            value: The value to check against all conditions.

        Returns:
            True if any condition is True, False otherwise.
        """
        return any(condition(value) for condition in self._conditions)

    def __repr__(self) -> str:
        return f"Or({self._conditions!r})"


class Not(ObjectCondition):
    """Condition that negates another condition.

    Attributes:
        condition: The condition to negate.

    Examples:
        >>> import spax as sp
        >>>
        >>> cond = sp.Not(sp.EqualsTo("adam"))
        >>> cond("sgd")   # True
        >>> cond("adam")  # False
    """

    def __init__(self, condition: Condition) -> None:
        """Initialize a Not condition.

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
        self._condition = condition

    @property
    def condition(self) -> Condition:
        """The condition to negate."""
        return self._condition

    def __call__(self, value: Any) -> bool:
        """Check if the sub-condition is False.

        Args:
            value: The value to check.

        Returns:
            True if the sub-condition is False, False otherwise.
        """
        return not self._condition(value)

    def __repr__(self) -> str:
        return f"Not({self._condition!r})"


class Lambda(ObjectCondition):
    """Condition defined by a custom function.

    Attributes:
        func: The function that evaluates the condition.

    Examples:
        >>> import spax as sp
        >>>
        >>> cond = sp.Lambda(lambda x: x % 2 == 0)  # Even numbers
        >>> cond(4)  # True
        >>> cond(5)  # False
    """

    def __init__(self, func: Callable[[Any], bool]) -> None:
        """Initialize a Lambda condition.

        Args:
            func: A callable that takes a value and returns a bool.

        Raises:
            TypeError: If func is not callable.
        """
        if not callable(func):
            raise TypeError(f"func must be callable, got {type(func).__name__}")
        self._func = func

    @property
    def func(self) -> Callable[[Any], bool]:
        """The function that evaluates the condition."""
        return self._func

    def __call__(self, value: Any) -> bool:
        """Evaluate the lambda function on the value.

        Args:
            value: The value to check.

        Returns:
            The result of calling func(value).

        Raises:
            TypeError: If the function does not return a bool.
        """
        result = self._func(value)
        if not isinstance(result, bool):
            raise TypeError(
                f"Lambda condition function must return bool, got {type(result).__name__}"
            )
        return result

    def __repr__(self) -> str:
        """Return a string representation with function name and signature."""
        import inspect

        func_name = getattr(self._func, "__name__", "<lambda>")
        try:
            sig = inspect.signature(self._func)
            return f"Lambda({func_name}{sig})"
        except (ValueError, TypeError):
            return f"Lambda({func_name})"
