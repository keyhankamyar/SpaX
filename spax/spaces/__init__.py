# spax/spaces/__init__.py
"""
Search space definitions for hyperparameter optimization.

This package provides space types (Float, Int, Categorical, Conditional) that define
searchable parameter ranges with validation and sampling capabilities.
"""

from .base import UNSET, Space
from .categorical import Categorical, CategoricalSpace, Choice
from .conditional import (
    Condition,
    Conditional,
    ConditionalSpace,
)
from .conditions import (
    And,
    EqualsTo,
    FieldCondition,
    In,
    IsInstance,
    Lambda,
    LargerThan,
    Not,
    NotEqualsTo,
    NotIn,
    Or,
    SmallerThan,
)
from .numeric import Float, FloatSpace, Int, IntSpace

__all__ = [
    # Base class
    "Space",
    "UNSET",
    # Categorical spaces
    "Categorical",
    "CategoricalSpace",
    "Choice",
    # Numeric spaces
    "Float",
    "FloatSpace",
    "Int",
    "IntSpace",
    # Conditional spaces
    "Conditional",
    "ConditionalSpace",
    # Conditions
    "Condition",
    "FieldCondition",
    "EqualsTo",
    "NotEqualsTo",
    "In",
    "NotIn",
    "SmallerThan",
    "LargerThan",
    "IsInstance",
    "And",
    "Or",
    "Not",
    "Lambda",
]
