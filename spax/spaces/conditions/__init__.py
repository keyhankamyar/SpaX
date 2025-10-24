"""Conditions for conditional search spaces.

This module provides condition classes for making parameters conditional on
other parameters. There are two main types:

- AttributeConditions: Evaluate fields of config objects (FieldCondition, MultiFieldLambdaCondition)
- ObjectConditions: Evaluate single values (EqualsTo, In, LargerThan, And, Or, Not, etc.)

AttributeConditions are used at the top level of ConditionalSpace, while
ObjectConditions are typically wrapped in FieldCondition to check specific fields.
"""

from .attribute_conditions import (
    AttributeCondition,
    FieldCondition,
    MultiFieldLambdaCondition,
)
from .base import Condition
from .object_conditions import (
    And,
    EqualsTo,
    In,
    IsInstance,
    Lambda,
    LargerThan,
    Not,
    NotEqualsTo,
    NotIn,
    ObjectCondition,
    Or,
    SmallerThan,
)

__all__ = [
    # Base classes
    "Condition",
    "AttributeCondition",
    "ObjectCondition",
    # Attribute conditions
    "FieldCondition",
    "MultiFieldLambdaCondition",
    # Object conditions - Equality
    "EqualsTo",
    "NotEqualsTo",
    # Object conditions - Membership
    "In",
    "NotIn",
    # Object conditions - Comparison
    "SmallerThan",
    "LargerThan",
    # Object conditions - Type checking
    "IsInstance",
    # Object conditions - Logical
    "And",
    "Or",
    "Not",
    # Object conditions - Custom
    "Lambda",
]
