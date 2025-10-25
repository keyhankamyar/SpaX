"""Base classes and types for search spaces.

This module defines the fundamental abstract base class for all search spaces
in SpaX, along with utility types for handling unset/undefined values.
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema


class _Unset:
    """Sentinel class to represent an unset or undefined value.

    This is used instead of None to distinguish between "no value provided"
    and "None was explicitly provided as a value".
    """

    def __repr__(self) -> str:
        return "UNSET"


# Singleton instance representing an unset value
UNSET = _Unset()


T = TypeVar("T")


class Space(ABC, Generic[T]):
    """Abstract base class for all search spaces.

    A Space defines a set of possible values that a parameter can take,
    along with validation logic and metadata like defaults and descriptions.
    Spaces can be numeric ranges, categorical choices, or conditional spaces
    that depend on other parameters.

    This class implements the descriptor protocol, allowing spaces to be
    used as class attributes that validate values on assignment.

    Attributes:
        field_name: Name of the field this space is attached to (set by __set_name__).
        default: Default value for this space, or UNSET if no default.
        description: Optional human-readable description of this parameter.
    """

    def __init__(
        self, *, default: T | _Unset = UNSET, description: str | None = None
    ) -> None:
        """Initialize a Space.

        Args:
            default: Default value for this space. Must be valid according to
                the space's constraints if provided.
            description: Optional description of what this parameter controls.

        Raises:
            ValueError: If the default value is invalid for this space.
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
        """Called when the space is assigned to a class attribute.

        This is part of the descriptor protocol and allows the space to know
        its field name for better error messages and parameter tracking.

        Args:
            owner: The class that owns this descriptor.
            name: The name of the attribute this descriptor is assigned to.
        """
        self.field_name = name

    @abstractmethod
    def contains(self, other: Any) -> bool:
        """Check if another space is a subset of this space.

        This is used for validating overrides - an override space must be
        contained within the original space.

        Args:
            other: Another space to check.

        Returns:
            True if the other space's possible values are a subset of this
            space's possible values.
        """
        pass

    @abstractmethod
    def validate(self, value: Any) -> T:
        """Validate and potentially transform a value for this space.

        Args:
            value: The value to validate.

        Returns:
            The validated (and possibly transformed) value.

        Raises:
            ValueError: If the value is not valid for this space.
            TypeError: If the value is of the wrong type.
        """
        pass

    @classmethod
    @abstractmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Provide Pydantic schema for this space type.

        This is part of Pydantic v2's schema generation and allows spaces
        to integrate with Pydantic's validation system.

        Args:
            source_type: The source type being validated.
            handler: Pydantic's schema handler.

        Returns:
            A Pydantic core schema for validation.
        """
        pass

    def __get__(self, obj: Any, objtype: type | None = None) -> T | "Space[T]":
        """Descriptor protocol: get the value from an instance.

        Args:
            obj: The instance to get the value from, or None if accessing from class.
            objtype: The type of the instance.

        Returns:
            The space itself if accessed from the class, or the stored value
            if accessed from an instance.
        """
        if obj is None:
            # Accessing from class returns the descriptor itself
            return self
        # Accessing from instance returns the stored value
        return obj.__dict__.get(self.field_name)  # type: ignore

    def __set__(self, obj: Any, value: T) -> None:
        """Descriptor protocol: set and validate the value on an instance.

        Args:
            obj: The instance to set the value on.
            value: The value to set (will be validated first).

        Raises:
            ValueError: If the value is invalid for this space.
        """
        validated_value = self.validate(value)
        obj.__dict__[self.field_name] = validated_value

    def __repr__(self) -> str:
        """Return a string representation of this space.

        Returns:
            A string showing the space's type and key attributes.
        """
        parts = [f"field_name={self.field_name!r}"]
        if self.default is not UNSET:
            parts.append(f"default={self.default!r}")
        if self.description is not None:
            parts.append(f"description={self.description!r}")
        return f"{self.__class__.__name__}({', '.join(parts)})"
