"""Factory functions for automatic space inference from type annotations.

This module provides utilities to automatically infer search spaces from
Python type annotations and Pydantic Field constraints. This enables a more
concise syntax where users can use standard type hints instead of explicitly
defining spaces.
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
    """Infer a numeric space from type annotation and Pydantic constraints.

    Attempts to create a FloatSpace or IntSpace based on the type annotation
    and any gt/ge/lt/le constraints in the field metadata.

    Args:
        annotation: The numeric type (int or float).
        field_info: Pydantic FieldInfo with potential constraints.
        default: Default value for the space.
        description: Description for the space.

    Returns:
        A FloatSpace or IntSpace if constraints are sufficient, None otherwise.
    """
    gt = None
    ge = None
    lt = None
    le = None

    # Check metadata for constraints
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
    """Flatten a Union type into a list of choices for categorical space.

    Extracts all possible values from a Union that can be represented as
    categorical choices. Handles Literal types, bool, None, and Config types.

    Args:
        union_args: The arguments from get_args() on a Union type.

    Returns:
        List of choices if all union members are compatible, empty list otherwise.
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
        # Unsupported types - cannot infer
        else:
            return []

    return choices


def infer_space_from_field_info(field_info: FieldInfo) -> Space | None:
    """Infer a search space from Pydantic FieldInfo.

    Attempts to automatically create an appropriate Space based on the field's
    type annotation and constraints. Supports:
    - Union/| types -> CategoricalSpace
    - bool -> CategoricalSpace([True, False])
    - Literal types -> CategoricalSpace
    - int/float with constraints -> IntSpace/FloatSpace

    Args:
        field_info: Pydantic FieldInfo containing type annotation and metadata.

    Returns:
        An inferred Space if possible, None if inference is not possible.

    Examples:
        >>> from pydantic import Field
        >>> from pydantic.fields import FieldInfo
        >>>
        >>> # Union inference
        >>> field = FieldInfo(annotation=Literal["a", "b", "c"])
        >>> space = infer_space_from_field_info(field)
        >>> # Returns CategoricalSpace(["a", "b", "c"])
        >>>
        >>> # Numeric inference with constraints
        >>> field = FieldInfo(annotation=int, metadata=[Field(ge=1, le=10)])
        >>> space = infer_space_from_field_info(field)
        >>> # Returns IntSpace(ge=1, le=10)
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

    # Cannot infer - return None
    return None
