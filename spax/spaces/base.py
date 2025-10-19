"""
Base class and protocols for search space definitions.

This module provides the abstract Space class that all concrete
space types (Float, Int, Categorical, etc.) inherit from.
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema


class _Unset:
    """Sentinel value to distinguish 'no default provided' from 'default is None'."""

    def __repr__(self) -> str:
        return "UNSET"


UNSET = _Unset()

T = TypeVar("T")


class Space(ABC, Generic[T]):
    """
    Abstract base class for all searchable parameter spaces.

    A Space defines:
    - The valid range/choices for a parameter
    - How to validate parameter values
    - How to sample random values
    - How to integrate with Pydantic for type validation

    Spaces act as Python descriptors, intercepting attribute access
    to provide automatic validation on assignment.

    Type Parameters:
        T: The type of values this space produces (e.g., float, int, Any).
    """

    def __init__(
        self, default: T | _Unset = UNSET, description: str | None = None
    ) -> None:
        """
        Initialize a space with optional default value and description.

        Args:
            default: Default value to use when not specified. Must be valid for this space.
                    If UNSET, no default is provided.
            description: Human-readable description of this parameter.
        """
        self.field_name: str | None = None
        self.default = default
        self.description = description

        # Validate default if provided
        if default is not UNSET:
            # Temporarily set field_name for validation error messages
            original_field_name = self.field_name
            self.field_name = "default"
            try:
                self.validate(default)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid default value {default!r}: {e}") from e
            finally:
                # Restore original field_name
                self.field_name = original_field_name

    def __set_name__(self, owner: type, name: str) -> None:
        """
        Called automatically when the space is assigned to a class attribute.

        This is part of the descriptor protocol and allows the space to
        know which field name it's associated with for error messages.

        Args:
            owner: The class that owns this descriptor.
            name: The attribute name this space was assigned to.
        """
        self.field_name = name

    @abstractmethod
    def validate(self, value: Any) -> T:
        """
        Validate and potentially transform a value for this space.

        Args:
            value: The value to validate.

        Returns:
            The validated (and possibly coerced) value.

        Raises:
            ValueError: If the value is invalid for this space.
        """
        pass

    @abstractmethod
    def sample(self) -> T:
        """
        Sample a random valid value from this space.

        Returns:
            A randomly sampled value that satisfies this space's constraints.
        """
        pass

    @classmethod
    @abstractmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """
        Provide Pydantic with schema information for validation.

        This method is called by Pydantic during model creation to
        understand how to validate values of this type.

        Args:
            source_type: The type annotation from the model field.
            handler: Pydantic's schema generation handler.

        Returns:
            A Pydantic CoreSchema describing validation rules.
        """
        pass

    def __get__(self, obj: Any, objtype: type | None = None) -> T | "Space[T]":
        """
        Descriptor protocol: retrieve the value from an instance.

        When accessed from the class (obj is None), returns the Space itself.
        When accessed from an instance, returns the stored value.

        Args:
            obj: The instance being accessed, or None if accessed from class.
            objtype: The type of the owner class.

        Returns:
            The Space descriptor itself (if class access) or the field value (if instance access).
        """
        if obj is None:
            # Accessing from class returns the descriptor itself
            return self
        # Accessing from instance returns the stored value
        return obj.__dict__.get(self.field_name)

    def __set__(self, obj: Any, value: T) -> None:
        """
        Descriptor protocol: set and validate the value on an instance.

        This is called whenever the attribute is assigned to, and ensures
        that validation always occurs.

        Args:
            obj: The instance being modified.
            value: The value being assigned.

        Raises:
            ValueError: If validation fails.
        """
        validated_value = self.validate(value)
        obj.__dict__[self.field_name] = validated_value

    def __repr__(self) -> str:
        """Return a string representation of this space."""
        parts = [f"field_name={self.field_name!r}"]
        if self.default is not UNSET:
            parts.append(f"default={self.default!r}")
        if self.description is not None:
            parts.append(f"description={self.description!r}")
        return f"{self.__class__.__name__}({', '.join(parts)})"
