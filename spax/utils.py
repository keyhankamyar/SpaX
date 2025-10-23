"""Utility functions for spax."""

from typing import Any


def is_comparable(value: Any) -> bool:
    """Check if value is comparable for use in conditions.

    Args:
        value: The value to check for comparability.

    Returns:
        True if the value can be safely compared using ==, False otherwise.

    Notes:
        This function checks if a value is suitable for use in equality
        comparisons within spax conditions and categorical spaces.
        It accepts:
        - Primitive types (int, float, str, bool, None)
        - Type objects
        - Objects with custom __eq__ methods (not using object's default)
    """
    # Allow primitives and common types
    if isinstance(value, (int, float, str, bool, type(None), type)):
        return True

    # Comparable must have __eq__ which all objects do by default but good to check
    if not hasattr(value, "__eq__"):
        return False

    # Also good to check it should not be None
    if value.__eq__ is None:
        return False

    # Also good to check it should be callable
    if not callable(value.__eq__):
        return False

    # Check for custom __eq__ (not inherited from object)
    if type(value).__eq__ is object.__eq__:
        return False

    # Let's compare it to itself to see it does not always return False
    if value != value:
        return False

    # Let's compare it to object to see it does not always return True
    return object() != value
