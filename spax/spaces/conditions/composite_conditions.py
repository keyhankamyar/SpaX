"""Composite conditions for combining and modifying conditions.

This module provides logical operators (And, Or, Not) that can combine or
modify other conditions. These are special because they inherit from
AttributeCondition, allowing them to be used as top-level conditions in
ConditionalSpace, but they can contain any Condition type.

Key Design:
----------
- CompositeConditions inherit from AttributeCondition for dependency tracking
- Can contain any Condition type (AttributeCondition or ObjectCondition)
- get_required_fields() validates that all children are AttributeConditions
- This ensures proper dependency tracking when used in ConditionalSpace
- Can still be used as inner conditions with ObjectConditions

Usage Rules:
-----------
1. As top-level in ConditionalSpace: Must contain only AttributeConditions
2. As inner conditions: Can contain any Condition type
3. Inside other AttributeConditions: Can contain any Condition type

Examples:
    >>> # Valid: As top-level with AttributeConditions
    >>> condition = sp.And([
    ...     sp.FieldCondition("use_dropout", sp.EqualsTo(True)),
    ...     sp.FieldCondition("use_l2", sp.EqualsTo(True))
    ... ])
    >>> param = sp.Conditional(condition, true=..., false=...)
    >>>
    >>> # Valid: As inner condition with ObjectConditions
    >>> condition = sp.FieldCondition(
    ...     "optimizer",
    ...     sp.And([sp.EqualsTo("adam"), sp.Not(sp.EqualsTo("sgd"))])
    ... )
    >>>
    >>> # Invalid: As top-level with ObjectConditions
    >>> condition = sp.And([sp.EqualsTo(True), sp.EqualsTo(False)])
    >>> param = sp.Conditional(condition, true=..., false=...)  # Error!
"""

from collections.abc import Iterable
from typing import Any

from .attribute_conditions import AttributeCondition
from .base import Condition


class CompositeCondition(AttributeCondition):
    """Base class for composite conditions (And, Or, Not).

    CompositeConditions inherit from AttributeCondition to enable their use
    as top-level conditions in ConditionalSpace. They can contain any type
    of Condition, but dependency tracking (get_required_fields) requires
    all children to be AttributeConditions.

    This design allows:
    - Using composites as top-level conditions (with AttributeCondition children)
    - Using composites as inner conditions (with any Condition children)
    - Proper dependency tracking for ConditionalSpace
    """

    pass


class And(CompositeCondition):
    """Logical AND of multiple conditions.

    Returns True only if all child conditions evaluate to True.

    When used as a top-level condition in ConditionalSpace, all child
    conditions must be AttributeConditions to enable dependency tracking.
    When used as an inner condition, children can be any Condition type.

    Args:
        conditions: Iterable of Condition instances to combine with AND.

    Raises:
        TypeError: If conditions is not iterable or contains non-Condition items.
        ValueError: If fewer than 2 conditions provided.

    Examples:
        >>> # Both conditions must be True
        >>> condition = sp.And([
        ...     sp.FieldCondition("use_dropout", sp.EqualsTo(True)),
        ...     sp.FieldCondition("use_l2", sp.EqualsTo(True))
        ... ])
        >>>
        >>> # Used in ConditionalSpace
        >>> param = sp.Conditional(
        ...     condition,
        ...     true=sp.Float(ge=0.0, le=1.0),
        ...     false=0.0
        ... )
    """

    def __init__(self, conditions: Iterable[Condition]) -> None:
        """Initialize an And condition.

        Args:
            conditions: Iterable of at least 2 Condition instances.

        Raises:
            TypeError: If conditions is not iterable or contains non-Conditions.
            ValueError: If fewer than 2 conditions provided.
        """
        try:
            conditions_list = list(conditions)
        except TypeError:
            raise TypeError(
                f"conditions must be iterable, got {type(conditions).__name__}"
            ) from None

        if len(conditions_list) < 2:
            raise ValueError("And requires at least two conditions")

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
        """Get a copy of the child conditions."""
        return self._conditions.copy()

    def get_required_fields(self) -> set[str]:
        """Get all fields required by child conditions.

        This method validates that all children are AttributeConditions
        (not ObjectConditions), ensuring proper dependency tracking.

        Returns:
            Set of field names required by all child conditions.

        Raises:
            TypeError: If any child is not an AttributeCondition.
        """
        required_fields: set[str] = set()

        for i, cond in enumerate(self._conditions):
            if not isinstance(cond, AttributeCondition):
                raise TypeError(
                    f"Cannot get required fields: condition at index {i} is not an "
                    f"AttributeCondition. When using And as a top-level condition in "
                    f"ConditionalSpace, all children must be AttributeConditions. "
                    f"Got: {type(cond).__name__}. "
                    f"Hint: ObjectConditions can only be used inside AttributeConditions, "
                    f"not at the top level."
                )
            required_fields.update(cond.get_required_fields())

        return required_fields

    def __call__(self, value: Any) -> bool:
        """Evaluate all child conditions with AND logic.

        Args:
            value: The value to evaluate (typically a config object).

        Returns:
            True if all child conditions are True, False otherwise.
        """
        return all(condition(value) for condition in self._conditions)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"And({self._conditions!r})"


class Or(CompositeCondition):
    """Logical OR of multiple conditions.

    Returns True if at least one child condition evaluates to True.

    When used as a top-level condition in ConditionalSpace, all child
    conditions must be AttributeConditions to enable dependency tracking.
    When used as an inner condition, children can be any Condition type.

    Args:
        conditions: Iterable of Condition instances to combine with OR.

    Raises:
        TypeError: If conditions is not iterable or contains non-Condition items.
        ValueError: If fewer than 2 conditions provided.

    Examples:
        >>> # At least one condition must be True
        >>> condition = sp.Or([
        ...     sp.FieldCondition("use_dropout", sp.EqualsTo(True)),
        ...     sp.FieldCondition("use_l2", sp.EqualsTo(True))
        ... ])
        >>>
        >>> # Used in ConditionalSpace
        >>> param = sp.Conditional(
        ...     condition,
        ...     true=sp.Float(ge=0.0, le=1.0),
        ...     false=0.0
        ... )
    """

    def __init__(self, conditions: Iterable[Condition]) -> None:
        """Initialize an Or condition.

        Args:
            conditions: Iterable of at least 2 Condition instances.

        Raises:
            TypeError: If conditions is not iterable or contains non-Conditions.
            ValueError: If fewer than 2 conditions provided.
        """
        try:
            conditions_list = list(conditions)
        except TypeError:
            raise TypeError(
                f"conditions must be iterable, got {type(conditions).__name__}"
            ) from None

        if len(conditions_list) < 2:
            raise ValueError("Or requires at least two conditions")

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
        """Get a copy of the child conditions."""
        return self._conditions.copy()

    def get_required_fields(self) -> set[str]:
        """Get all fields required by child conditions.

        This method validates that all children are AttributeConditions
        (not ObjectConditions), ensuring proper dependency tracking.

        Returns:
            Set of field names required by all child conditions.

        Raises:
            TypeError: If any child is not an AttributeCondition.
        """
        required_fields: set[str] = set()

        for i, cond in enumerate(self._conditions):
            if not isinstance(cond, AttributeCondition):
                raise TypeError(
                    f"Cannot get required fields: condition at index {i} is not an "
                    f"AttributeCondition. When using Or as a top-level condition in "
                    f"ConditionalSpace, all children must be AttributeConditions. "
                    f"Got: {type(cond).__name__}. "
                    f"Hint: ObjectConditions can only be used inside AttributeConditions, "
                    f"not at the top level."
                )
            required_fields.update(cond.get_required_fields())

        return required_fields

    def __call__(self, value: Any) -> bool:
        """Evaluate all child conditions with OR logic.

        Args:
            value: The value to evaluate (typically a config object).

        Returns:
            True if at least one child condition is True, False otherwise.
        """
        return any(condition(value) for condition in self._conditions)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"Or({self._conditions!r})"


class Not(CompositeCondition):
    """Logical NOT (negation) of a condition.

    Returns True if the child condition evaluates to False, and vice versa.

    When used as a top-level condition in ConditionalSpace, the child
    condition must be an AttributeCondition to enable dependency tracking.
    When used as an inner condition, the child can be any Condition type.

    Args:
        condition: The Condition instance to negate.

    Raises:
        TypeError: If condition is not a Condition instance.

    Examples:
        >>> # Negates the condition
        >>> condition = sp.Not(
        ...     sp.FieldCondition("use_dropout", sp.EqualsTo(True))
        ... )
        >>> # Equivalent to: use_dropout != True (i.e., use_dropout == False)
        >>>
        >>> # Used in ConditionalSpace
        >>> param = sp.Conditional(
        ...     condition,
        ...     true=sp.Float(ge=0.0, le=1.0),
        ...     false=0.0
        ... )
    """

    def __init__(self, condition: Condition) -> None:
        """Initialize a Not condition.

        Args:
            condition: The Condition instance to negate.

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
        """Get the child condition being negated."""
        return self._condition

    def get_required_fields(self) -> set[str]:
        """Get all fields required by the child condition.

        This method validates that the child is an AttributeCondition
        (not an ObjectCondition), ensuring proper dependency tracking.

        Returns:
            Set of field names required by the child condition.

        Raises:
            TypeError: If child is not an AttributeCondition.
        """
        if not isinstance(self._condition, AttributeCondition):
            raise TypeError(
                f"Cannot get required fields: condition is not an "
                f"AttributeCondition. When using Not as a top-level condition in "
                f"ConditionalSpace, the child must be an AttributeCondition. "
                f"Got: {type(self._condition).__name__}. "
                f"Hint: ObjectConditions can only be used inside AttributeConditions, "
                f"not at the top level."
            )

        return self._condition.get_required_fields()

    def __call__(self, value: Any) -> bool:
        """Evaluate the child condition and negate the result.

        Args:
            value: The value to evaluate (typically a config object).

        Returns:
            True if child condition is False, False if child is True.
        """
        return not self._condition(value)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"Not({self._condition!r})"
