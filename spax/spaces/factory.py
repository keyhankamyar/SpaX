"""
Automatic space inference from type annotations and Pydantic Fields.
This module provides utilities to automatically create Space objects from
standard Python type hints and Pydantic field definitions.
"""

from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin

from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from .base import UNSET, Space
from .categorical import Categorical
from .numeric import Float, Int


def _infer_numeric_space(
    annotation: type[int] | type[float],
    field_info: FieldInfo,
    default: Any,
    description: str | None,
) -> Space | None:
    """
    Infer a numeric space from Pydantic Field metadata.

    Args:
        annotation: Either int or float type
        field_info: Pydantic FieldInfo with metadata
        default: Default value extracted from field_info
        description: Description extracted from field_info

    Returns:
        FloatSpace or IntSpace if valid metadata exist, None otherwise
    """
    gt = None
    ge = None
    lt = None
    le = None

    # Check metadata
    if hasattr(field_info, "metadata"):
        for constraint in field_info.metadata:
            if hasattr(constraint, "gt"):
                gt = constraint.gt
            if hasattr(constraint, "ge"):
                ge = constraint.ge
            if hasattr(constraint, "lt"):
                lt = constraint.lt
            if hasattr(constraint, "le"):
                le = constraint.le

    # Check if we have exactly one lower and one upper bound
    lower_bounds = [b for b in [gt, ge] if b is not None]
    upper_bounds = [b for b in [lt, le] if b is not None]

    if len(lower_bounds) != 1 or len(upper_bounds) != 1:
        # Not enough constraints for automatic inference
        return None

    # Create the appropriate space
    if annotation is int:
        return Int(gt=gt, ge=ge, lt=lt, le=le, default=default, description=description)
    else:  # float
        return Float(
            gt=gt, ge=ge, lt=lt, le=le, default=default, description=description
        )


def _flatten_union_to_choices(union_args: tuple[Any, ...]) -> list[Any]:
    """
    Flatten Union types into a list of choices for Categorical.

    Handles:
    - Config subclasses → kept as-is
    - bool → expanded to [True, False]
    - Literal[...] → expanded to list of literal values
    - None → kept as-is
    """
    from spax.config import Config

    choices: list[Any] = []

    for arg in union_args:
        origin = get_origin(arg)

        # Handle Literal types
        if origin is Literal:
            literal_values = get_args(arg)
            choices.extend(literal_values)
        # Handle bool
        elif arg is bool:
            choices.extend([True, False])
        # Handle None
        elif arg is type(None):
            choices.append(None)
        # Handle Config subclasses
        elif isinstance(arg, type) and issubclass(arg, Config):
            choices.append(arg)
        # Unsupported types
        else:
            return []

    return choices


def infer_space_from_field_info(field_info: FieldInfo) -> Space | None:
    """
    Infer a Space from a Pydantic FieldInfo.

    Args:
        field_info: Pydantic FieldInfo with constraints

    Returns:
        An inferred Space object, or None if inference is not possible
    """
    annotation = field_info.annotation
    origin = get_origin(annotation)

    # Extract common field info attributes
    default = UNSET
    description = None

    # Handle default value
    if field_info.default is not PydanticUndefined:
        default = field_info.default

    # Handle description
    if field_info.description is not None:
        description = field_info.description

    # Check for both types.UnionType (|) and typing.Union
    if origin is Union or isinstance(annotation, UnionType):
        union_args = get_args(annotation)
        choices = _flatten_union_to_choices(union_args)
        if not choices:
            return None
        return Categorical(choices, default=default, description=description)

    # Handle bool type
    if annotation is bool:
        return Categorical([True, False], default=default, description=description)

    # Handle Literal types
    if origin is Literal:
        choices = list(get_args(annotation))
        if not choices:
            return None
        return Categorical(choices, default=default, description=description)

    # Handle numeric types with Pydantic Field constraints
    if annotation in (int, float):
        return _infer_numeric_space(annotation, field_info, default, description)

    return None
