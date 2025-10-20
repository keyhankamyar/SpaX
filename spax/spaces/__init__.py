"""
Space definitions for SpaX configuration parameters.

This module provides the core Space types used to define search spaces for
hyperparameter optimization and configuration management.
"""

from .base import UNSET, Space
from .categorical import Categorical, CategoricalSpace, Choice
from .conditional import Conditional, ConditionalSpace
from .conditions import (
    And,
    AttributeCondition,
    Condition,
    EqualsTo,
    FieldCondition,
    In,
    IsInstance,
    Lambda,
    LargerThan,
    MultiFieldLambdaCondition,
    Not,
    NotEqualsTo,
    NotIn,
    ObjectCondition,
    Or,
    SmallerThan,
)
from .factory import infer_space_from_field_info
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
    # Conditions - Base classes
    "Condition",
    "AttributeCondition",
    "ObjectCondition",
    # Conditions - Attribute conditions
    "FieldCondition",
    "MultiFieldLambdaCondition",
    # Conditions - Object conditions
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
    # Auto inference
    "infer_space_from_field_info",
]
