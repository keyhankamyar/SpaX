"""
Object-based conditions that evaluate entire configuration objects.

These conditions receive a full object/value and evaluate it. They cannot be
used at the top level of ConditionalSpace (only AttributeConditions can), but
can be used within FieldCondition or as nested conditions in ConditionalSpace
branches.
"""

from collections.abc import Callable, Iterable
from typing import Any

from spax.utils import is_comparable

from .base import Condition


class ObjectCondition(Condition):
    """
    Base class for conditions that evaluate objects or values directly.

    ObjectConditions cannot be used at the top level of ConditionalSpace because
    they don't declare their dependencies. They are used within FieldCondition
    or as nested conditions.
    """

    pass


class EqualsTo(ObjectCondition):
    """
    Condition that checks if a value equals a specific target value.

    Examples:
        >>> import spax as sp
        >>>
        >>> class MyConfig(sp.Config):
        ...     optimizer: str = sp.Categorical(["adam", "sgd", "rmsprop"])
        ...     learning_rate: float = sp.Conditional(
        ...         sp.FieldCondition("optimizer", sp.EqualsTo("adam")),
        ...         true=sp.Float(gt=0.0001, lt=0.01),
        ...         false=sp.Float(gt=0.001, lt=0.1)
        ...     )
    """

    def __init__(self, value: Any) -> None:
        """
        Initialize an equality condition.

        Args:
            value: The target value to compare against
        """
        if not is_comparable(value):
            raise TypeError(f"Value must be comparable, got {type(value).__name__}")
        self.value = value

    def __call__(self, value: Any) -> bool:
        """Check if value equals the target."""
        if not is_comparable(value):
            raise TypeError(f"Value must be comparable, got {type(value).__name__}")
        return value == self.value

    def __repr__(self) -> str:
        return f"EqualsTo({self.value!r})"


class NotEqualsTo(ObjectCondition):
    """
    Condition that checks if a value does not equal a specific target value.

    Examples:
        >>> import spax as sp
        >>>
        >>> class MyConfig(sp.Config):
        ...     activation: str = sp.Categorical(["relu", "gelu", "tanh"])
        ...     use_bias: bool = sp.Conditional(
        ...         sp.FieldCondition("activation", sp.NotEqualsTo("relu")),
        ...         true=True,
        ...         false=False
        ...     )
    """

    def __init__(self, value: Any) -> None:
        """
        Initialize a not-equals condition.

        Args:
            value: The target value to compare against
        """
        if not is_comparable(value):
            raise TypeError(f"Value must be comparable, got {type(value).__name__}")
        self.value = value

    def __call__(self, value: Any) -> bool:
        """Check if value does not equal the target."""
        if not is_comparable(value):
            raise TypeError(f"Value must be comparable, got {type(value).__name__}")
        return value != self.value

    def __repr__(self) -> str:
        return f"NotEqualsTo({self.value!r})"


class In(ObjectCondition):
    """
    Condition that checks if a value is in a collection.

    Examples:
        >>> import spax as sp
        >>>
        >>> class MyConfig(sp.Config):
        ...     model_size: str = sp.Categorical(["small", "medium", "large", "xlarge"])
        ...     use_gradient_checkpointing: bool = sp.Conditional(
        ...         sp.FieldCondition("model_size", sp.In(["large", "xlarge"])),
        ...         true=True,
        ...         false=False
        ...     )
    """

    def __init__(self, values: list[Any] | set[Any] | tuple[Any, ...]) -> None:
        """
        Initialize an 'in' condition.

        Args:
            values: Collection of allowed values
        """
        for value in values:
            if not is_comparable(value):
                raise TypeError(f"Value must be comparable, got {type(value).__name__}")
        self.values = values

    def __call__(self, value: Any) -> bool:
        """Check if value is in the collection."""
        if not is_comparable(value):
            raise TypeError(f"Value must be comparable, got {type(value).__name__}")
        return value in self.values

    def __repr__(self) -> str:
        return f"In({self.values!r})"


class NotIn(ObjectCondition):
    """
    Condition that checks if a value is not in a collection.

    Examples:
        >>> import spax as sp
        >>>
        >>> class MyConfig(sp.Config):
        ...     pooling: str = sp.Categorical(["max", "avg", "attention", "none"])
        ...     dropout: float = sp.Conditional(
        ...         sp.FieldCondition("pooling", sp.NotIn(["none"])),
        ...         true=sp.Float(gt=0, lt=0.5),
        ...         false=0.0
        ...     )
    """

    def __init__(self, values: list[Any] | set[Any] | tuple[Any, ...]) -> None:
        """
        Initialize a 'not in' condition.

        Args:
            values: Collection of disallowed values
        """
        for value in values:
            if not is_comparable(value):
                raise TypeError(f"Value must be comparable, got {type(value).__name__}")
        self.values = values

    def __call__(self, value: Any) -> bool:
        """Check if value is not in the collection."""
        if not is_comparable(value):
            raise TypeError(f"Value must be comparable, got {type(value).__name__}")
        return value not in self.values

    def __repr__(self) -> str:
        return f"NotIn({self.values!r})"


class SmallerThan(ObjectCondition):
    """
    Condition that checks if a numeric value is smaller than a threshold.

    Examples:
        >>> import spax as sp
        >>>
        >>> class MyConfig(sp.Config):
        ...     temperature: float = sp.Float(gt=0, lt=2)
        ...     top_k: int = sp.Conditional(
        ...         sp.FieldCondition("temperature", sp.SmallerThan(0.5)),
        ...         true=sp.Int(ge=1, le=10),
        ...         false=sp.Int(ge=10, le=100)
        ...     )
        >>>
        >>> # With or_equals=True
        >>> class MyConfig2(sp.Config):
        ...     batch_size: int = sp.Int(ge=1, le=128)
        ...     num_workers: int = sp.Conditional(
        ...         sp.FieldCondition("batch_size", sp.SmallerThan(32, or_equals=True)),
        ...         true=2,
        ...         false=8
        ...     )
    """

    def __init__(self, value: float | int, or_equals: bool = False) -> None:
        """
        Initialize a smaller-than condition.

        Args:
            value: Threshold value to compare against
            or_equals: If True, allows equality (<=), otherwise strict (<)
        """
        if not isinstance(value, (int, float)):
            raise TypeError(f"Expected numeric value, got {type(value).__name__}")
        if not isinstance(or_equals, bool):
            raise TypeError(f"or_equals must be bool, got {type(or_equals).__name__}")
        self.value = value
        self.or_equals = or_equals

    def __call__(self, value: Any) -> bool:
        """Check if value is smaller than (or equal to) the threshold."""
        if not isinstance(value, (int, float)):
            raise TypeError(f"Expected numeric value, got {type(value).__name__}")
        if self.or_equals:
            return value <= self.value
        else:
            return value < self.value

    def __repr__(self) -> str:
        return f"SmallerThan({self.value}, or_equals={self.or_equals})"


class LargerThan(ObjectCondition):
    """
    Condition that checks if a numeric value is larger than a threshold.

    Examples:
        >>> import spax as sp
        >>>
        >>> class MyConfig(sp.Config):
        ...     learning_rate: float = sp.Float(gt=0, lt=1)
        ...     warmup_steps: int = sp.Conditional(
        ...         sp.FieldCondition("learning_rate", sp.LargerThan(0.01)),
        ...         true=sp.Int(ge=100, le=1000),
        ...         false=sp.Int(ge=0, le=100)
        ...     )
        >>>
        >>> # With or_equals=True
        >>> class MyConfig2(sp.Config):
        ...     num_layers: int = sp.Int(ge=1, le=48)
        ...     use_layer_norm: bool = sp.Conditional(
        ...         sp.FieldCondition("num_layers", sp.LargerThan(12, or_equals=True)),
        ...         true=True,
        ...         false=False
        ...     )
    """

    def __init__(self, value: float | int, or_equals: bool = False) -> None:
        """
        Initialize a larger-than condition.

        Args:
            value: Threshold value to compare against
            or_equals: If True, allows equality (>=), otherwise strict (>)
        """
        if not isinstance(value, (int, float)):
            raise TypeError(f"Expected numeric value, got {type(value).__name__}")
        if not isinstance(or_equals, bool):
            raise TypeError(f"or_equals must be bool, got {type(or_equals).__name__}")
        self.value = value
        self.or_equals = or_equals

    def __call__(self, value: Any) -> bool:
        """Check if value is larger than (or equal to) the threshold."""
        if not isinstance(value, (int, float)):
            raise TypeError(f"Expected numeric value, got {type(value).__name__}")
        if self.or_equals:
            return value >= self.value
        else:
            return value > self.value

    def __repr__(self) -> str:
        return f"LargerThan({self.value}, or_equals={self.or_equals})"


class IsInstance(ObjectCondition):
    """
    Condition that checks if a value is an instance of specific type(s).

    Examples:
        >>> import spax as sp
        >>>
        >>> class InnerConfig1(sp.Config):
        ...     param: int = sp.Int(ge=0, le=100)
        >>>
        >>> class InnerConfig2(sp.Config):
        ...     param: float = sp.Float(gt=0, lt=1)
        >>>
        >>> class MyConfig(sp.Config):
        ...     inner: InnerConfig1 | InnerConfig2
        ...     extra_param: int = sp.Conditional(
        ...         sp.FieldCondition("inner", sp.IsInstance(InnerConfig1)),
        ...         true=sp.Int(ge=0, le=10),
        ...         false=sp.Int(ge=100, le=1000)
        ...     )
    """

    def __init__(self, class_or_tuple: type | tuple[type, ...]) -> None:
        """
        Initialize an isinstance condition.

        Args:
            class_or_tuple: Type or tuple of types to check against
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
        """Check if value is an instance of the specified type(s)."""
        return isinstance(value, self.class_or_tuple)

    def __repr__(self) -> str:
        return f"IsInstance({self.class_or_tuple!r})"


class And(ObjectCondition):
    """
    Condition that requires all nested conditions to be True.

    Examples:
        >>> import spax as sp
        >>>
        >>> class MyConfig(sp.Config):
        ...     value: float = sp.Float(gt=0, lt=100)
        ...     category: str = sp.Conditional(
        ...         sp.FieldCondition(
        ...             "value",
        ...             sp.And([sp.LargerThan(10), sp.SmallerThan(50)])
        ...         ),
        ...         true="medium",
        ...         false="other"
        ...     )
    """

    def __init__(self, conditions: Iterable[Condition]) -> None:
        """
        Initialize an AND condition.

        Args:
            conditions: Iterable of conditions that must all be True
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
        """Check if all conditions are True."""
        return all(condition(value) for condition in self.conditions)

    def __repr__(self) -> str:
        return f"And({self.conditions!r})"


class Or(ObjectCondition):
    """
    Condition that requires at least one nested condition to be True.

    Examples:
        >>> import spax as sp
        >>>
        >>> class MyConfig(sp.Config):
        ...     value: float = sp.Float(gt=0, lt=100)
        ...     special_handling: bool = sp.Conditional(
        ...         sp.FieldCondition(
        ...             "value",
        ...             sp.Or([sp.SmallerThan(5), sp.LargerThan(95)])
        ...         ),
        ...         true=True,
        ...         false=False
        ...     )
    """

    def __init__(self, conditions: Iterable[Condition]) -> None:
        """
        Initialize an OR condition.

        Args:
            conditions: Iterable of conditions where at least one must be True
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
        """Check if any condition is True."""
        return any(condition(value) for condition in self.conditions)

    def __repr__(self) -> str:
        return f"Or({self.conditions!r})"


class Not(ObjectCondition):
    """
    Condition that negates another condition.

    Examples:
        >>> import spax as sp
        >>>
        >>> class MyConfig(sp.Config):
        ...     optimizer: str = sp.Categorical(["adam", "sgd", "rmsprop"])
        ...     use_weight_decay: bool = sp.Conditional(
        ...         sp.FieldCondition("optimizer", sp.Not(sp.EqualsTo("adam"))),
        ...         true=True,
        ...         false=False
        ...     )
    """

    def __init__(self, condition: Condition) -> None:
        """
        Initialize a NOT condition.

        Args:
            condition: Condition to negate
        """
        if not isinstance(condition, Condition):
            raise TypeError(
                f"condition must be a Condition instance, "
                f"got {type(condition).__name__}"
            )
        self.condition = condition

    def __call__(self, value: Any) -> bool:
        """Check if the condition is False."""
        return not self.condition(value)

    def __repr__(self) -> str:
        return f"Not({self.condition!r})"


class Lambda(ObjectCondition):
    """
    Condition based on a custom lambda function.

    Note: For conditions that depend on multiple config fields, use
    MultiFieldLambda instead, as it properly declares field dependencies.

    Examples:
        >>> import spax as sp
        >>>
        >>> class MyConfig(sp.Config):
        ...     name: str = sp.Categorical(["model_a", "model_b", "model_c"])
        ...     use_cache: bool = sp.Conditional(
        ...         sp.FieldCondition("name", sp.Lambda(lambda x: x.startswith("model_"))),
        ...         true=True,
        ...         false=False
        ...     )
        >>>
        >>> # More complex example
        >>> class MyConfig2(sp.Config):
        ...     score: float = sp.Float(gt=0, lt=100)
        ...     grade: str = sp.Conditional(
        ...         sp.FieldCondition(
        ...             "score",
        ...             sp.Lambda(lambda x: 90 <= x <= 100)
        ...         ),
        ...         true="A",
        ...         false="B"
        ...     )
    """

    def __init__(self, func: Callable[[Any], bool]) -> None:
        """
        Initialize a lambda condition.

        Args:
            func: Callable that takes a value and returns bool
        """
        if not callable(func):
            raise TypeError(f"func must be callable, got {type(func).__name__}")
        self.func = func

    def __call__(self, value: Any) -> bool:
        """Evaluate the lambda function."""
        result = self.func(value)
        if not isinstance(result, bool):
            raise TypeError(
                f"Lambda condition function must return bool, got {type(result).__name__}"
            )
        return result

    def __repr__(self) -> str:
        import inspect

        func_name = getattr(self.func, "__name__", "<lambda>")
        try:
            sig = inspect.signature(self.func)
            return f"Lambda({func_name}{sig})"
        except (ValueError, TypeError):
            return f"Lambda({func_name})"
