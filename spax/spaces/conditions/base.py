"""Base classes for conditions used in conditional search spaces.

This module defines the abstract base class that all conditions must implement.
Conditions are used to make parameters conditional on the values of other parameters.
"""

from abc import ABC, abstractmethod
from typing import Any


class _NotGiven:
    """
    Sentinel class to represent an unset or undefined field name.
    """

    def __repr__(self) -> str:
        return "NotGiven"


NotGiven = _NotGiven()


class Condition(ABC):
    """Abstract base class for all conditions.

    A Condition is a predicate that evaluates to True or False based on
    some input value(s). Conditions are used in ConditionalSpace to determine
    which branch of the space is active.

    There are two main types of conditions:
    - ObjectCondition: Evaluates a single value (e.g., EqualsTo, In, LargerThan)
    - AttributeCondition: Evaluates config object attributes (e.g., FieldCondition)
    """

    @abstractmethod
    def __call__(self, value: Any) -> bool:
        """Evaluate the condition on a value.

        Args:
            value: The value to evaluate. Can be a single value for ObjectConditions
                or a config object for AttributeConditions.

        Returns:
            True if the condition is satisfied, False otherwise.
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """Return a string representation of this condition.

        Returns:
            A string describing this condition for debugging and logging.
        """
        pass

    @abstractmethod
    def render(self, field_name: str) -> str:
        """Generate a human-readable string representation of this condition.

        Used internally for debugging, logging, and displaying conditional
        dependencies in tree visualizations.

        Args:
            field_name: The field name this condition is evaluating.

        Returns:
            String representation like "field_name == 'value'" or "field_name > 5".
        """
        pass
