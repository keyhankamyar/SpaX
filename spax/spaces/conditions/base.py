"""
Base classes for all condition types in SpaX.

Conditions are predicates that evaluate to True or False based on configuration
values. They are used in ConditionalSpace to determine which branch to activate.
"""

from abc import ABC, abstractmethod
from typing import Any


class Condition(ABC):
    """
    Abstract base class for all conditions.

    Conditions are callable objects that evaluate configuration values or objects
    and return a boolean result.
    """

    @abstractmethod
    def __call__(self, value: Any) -> bool:
        """
        Evaluate the condition.

        Args:
            value: The value or object to evaluate

        Returns:
            True if condition is satisfied, False otherwise
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """Return string representation of the condition."""
        pass
