"""
Base classes for conditions in conditional spaces.

Conditions are divided into two categories:
- AttributeCondition: Depends on object attributes (used for sampling/instantiation)
- ObjectCondition: Works on values directly (used for validation only)
"""

from abc import ABC, abstractmethod
from typing import Any


class Condition(ABC):
    """Base class for all conditions."""

    @abstractmethod
    def __call__(self, value: Any) -> bool:
        """Evaluate the condition."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class AttributeCondition(Condition):
    """
    Base class for conditions that depend on object attributes.

    These conditions can be used at the top level of ConditionalSpace
    because they explicitly declare which fields they depend on.
    """

    @abstractmethod
    def get_required_fields(self) -> list[str]:
        """Return list of field names this condition depends on."""
        pass


class ObjectCondition(Condition):
    """
    Base class for conditions that work on values directly.

    These conditions cannot be used at the top level of ConditionalSpace
    because they don't declare field dependencies. They must be wrapped
    in an AttributeCondition (like FieldCondition).
    """

    pass
