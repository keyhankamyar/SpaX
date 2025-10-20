"""
Condition system for SpaX configuration spaces.

Conditions are predicates used to control ConditionalSpace behavior. They are
divided into two categories:

- AttributeConditions: Depend on specific config fields and can be used at the
  top level of ConditionalSpace (FieldCondition, MultiFieldLambda)

- ObjectConditions: Evaluate values/objects directly and can only be used within
  FieldCondition or as nested conditions (EqualsTo, LargerThan, And, Or, etc.)
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
    # Object conditions
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
