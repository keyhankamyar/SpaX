"""Conditions for controlling conditional spaces.

This module provides condition types for making parameters conditional on
other parameter values. Conditions are used with ConditionalSpace to create
parameters whose possible values depend on the configuration state.

Condition Types:
---------------
1. AttributeConditions: Conditions that depend on config attributes
   - FieldCondition: Condition on a single field value
   - MultiFieldLambdaCondition: Custom condition on multiple fields
   - CompositeConditions: And, Or, Not (can be used as top-level)

2. ObjectConditions: Conditions that test values directly
   - Equality: EqualsTo, NotEqualsTo
   - Membership: In, NotIn
   - Comparison: SmallerThan, LargerThan
   - Type: IsInstance
   - Custom: Lambda

3. CompositeConditions: Combine/modify other conditions
   - And: All conditions must be True
   - Or: At least one condition must be True
   - Not: Negates a condition

   Special: CompositeConditions inherit from AttributeCondition, allowing
   them to be used as top-level conditions in ConditionalSpace. However,
   when used at the top level, all children must be AttributeConditions.

Usage in ConditionalSpace:
-------------------------
Top-level conditions must be AttributeConditions (including composites
with AttributeCondition children). ObjectConditions can only be used
inside AttributeConditions.

Examples:
    >>> import spax as sp
    >>>
    >>> # Valid: FieldCondition at top level
    >>> dropout_rate = sp.Conditional(
    ...     sp.FieldCondition("use_dropout", sp.EqualsTo(True)),
    ...     true=sp.Float(gt=0.0, lt=0.5),
    ...     false=0.0
    ... )
    >>>
    >>> # Valid: Composite with AttributeConditions at top level
    >>> param = sp.Conditional(
    ...     sp.And([
    ...         sp.FieldCondition("use_l2", sp.EqualsTo(True)),
    ...         sp.FieldCondition("use_dropout", sp.EqualsTo(True))
    ...     ]),
    ...     true=sp.Float(ge=0.0, le=1.0),
    ...     false=0.0
    ... )
    >>>
    >>> # Valid: ObjectCondition inside AttributeCondition
    >>> param = sp.Conditional(
    ...     sp.FieldCondition("optimizer", sp.In(["adam", "adamw"])),
    ...     true=sp.Float(ge=0.8, lt=1.0),
    ...     false=0.0
    ... )
    >>>
    >>> # Invalid: ObjectCondition at top level
    >>> param = sp.Conditional(
    ...     sp.EqualsTo(True),  # Error! No field dependencies
    ...     true=...,
    ...     false=...
    ... )
"""

from .attribute_conditions import (
    AttributeCondition,
    FieldCondition,
    MultiFieldLambdaCondition,
    ParsedFieldPath,
)
from .base import Condition
from .composite_conditions import And, Not, Or
from .object_conditions import (
    EqualsTo,
    In,
    IsInstance,
    Lambda,
    LargerThan,
    NotEqualsTo,
    NotIn,
    ObjectCondition,
    SmallerThan,
)

__all__ = [
    # Base classes
    "Condition",
    "AttributeCondition",
    "ObjectCondition",
    # Attribute condition's path
    "ParsedFieldPath",
    # Attribute conditions
    "FieldCondition",
    "MultiFieldLambdaCondition",
    # Composite conditions (special: inherit from AttributeCondition)
    "And",
    "Or",
    "Not",
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
    # Object conditions - Custom
    "Lambda",
]
