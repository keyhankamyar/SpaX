"""Utility functions for SpaX.

This module provides helper functions used throughout the SpaX library,
particularly for type checking, validation, and type annotation processing.
"""

from types import UnionType
from typing import Any, Union, get_args, get_origin


def is_comparable(value: Any) -> bool:
    """Check if a value can be safely compared using equality operators.

    This function determines whether a value implements proper equality comparison
    that can be used reliably in search spaces. It's used to validate choices in
    categorical spaces and other places where we need to compare values.

    A value is considered comparable if it:
    - Is a primitive type (int, float, str, bool, None, type), OR
    - Has a custom __eq__ method (not just the default object.__eq__)
    - The __eq__ method is callable
    - The equality operator behaves reasonably (not always True or False)

    Args:
        value: The value to check for comparability.

    Returns:
        True if the value can be safely compared, False otherwise.

    Examples:
        >>> is_comparable(5)
        True
        >>> is_comparable("hello")
        True
        >>> is_comparable([1, 2, 3])
        True
        >>> is_comparable(object())  # Plain object with no custom __eq__
        False
    """
    # Primitives and types are always comparable
    if isinstance(
        value,
        (
            int,
            float,
            str,
            bool,
            type(None),
            type,
            list,
            set,
            tuple,
            dict,
        ),
    ):
        return True

    # Check if __eq__ exists (should always be True, but defensive check)
    if not hasattr(value, "__eq__"):
        return False

    # Check if __eq__ is not None
    if value.__eq__ is None:
        return False

    # Check if __eq__ is callable
    if not callable(value.__eq__):
        return False

    # Check for custom __eq__ (not just inherited from object)
    # Objects with default __eq__ only compare by identity, which is usually not what we want
    if type(value).__eq__ is object.__eq__:  # type: ignore
        return False

    # Sanity check: value should equal itself
    # (catches broken __eq__ implementations that always return False or NaN-like cases)
    if value != value:
        return False

    # Final sanity check: value should not equal arbitrary objects
    # (catches broken __eq__ implementations that always return True)
    return bool(object() != value)


def type_from_annotation(annotation: Any, type_name: str) -> type | None:
    """Extract a type by name from a type annotation, including Unions.

    This function searches through type annotations (including Union types)
    to find a type with the given name. This is useful for deserializing
    Config objects where we need to find the correct Config subclass based
    on a type discriminator.

    Args:
        annotation: The type annotation to search. Can be a single type,
            Union, or types.UnionType.
        type_name: The name of the type to find (e.g., "ResNet").

    Returns:
        The type with matching name if found, None otherwise.

    Examples:
        >>> from typing import Union
        >>> type_from_annotation(Union[int, str], "int")
        <class 'int'>
        >>> type_from_annotation(Union[ResNet, VGG], "ResNet")
        <class 'ResNet'>
        >>> type_from_annotation(int, "int")
        <class 'int'>
        >>> type_from_annotation(int, "str")
        None
    """
    # Check if it's a union type (typing.Union or types.UnionType from |)
    if get_origin(annotation) is Union or isinstance(annotation, UnionType):
        # Search through union members
        for arg in get_args(annotation):
            if isinstance(arg, type) and arg.__name__ == type_name:
                return arg
    # Check if it's directly the type we're looking for
    elif isinstance(annotation, type) and annotation.__name__ == type_name:
        return annotation

    return None
