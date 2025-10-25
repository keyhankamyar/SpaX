"""Search space definitions for parameter exploration.

This module provides the core search space types for defining hyperparameters,
architectural parameters, and other configurable values in machine learning
experiments. The main space types are:

- Numeric spaces: Float, Int for continuous and discrete numeric ranges
- Categorical spaces: Categorical for discrete choice parameters
- Conditional spaces: Conditional for parameters that depend on other parameters
- Conditions: Various condition types for conditional parameters

All spaces support:
- Validation of values
- Random sampling
- Integration with HPO libraries like Optuna
- Serialization/deserialization

Examples:
    >>> import spax as sp
    >>>
    >>> # Numeric spaces
    >>> learning_rate = sp.Float(ge=1e-5, le=1e-1, distribution='log')
    >>> num_layers = sp.Int(ge=1, le=10)
    >>>
    >>> # Categorical space
    >>> optimizer = sp.Categorical(["adam", "sgd", "rmsprop"])
    >>>
    >>> # Conditional space
    >>> dropout = sp.Conditional(
    ...     sp.FieldCondition("use_dropout", sp.EqualsTo(True)),
    ...     true=sp.Float(gt=0.0, lt=0.5),
    ...     false=0.0
    ... )
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
    ParsedFieldPath,
    SmallerThan,
)
from .factory import infer_space_from_field_info
from .numeric import Float, FloatSpace, Int, IntSpace, NumberSpace

__all__ = [
    # Base classes
    "Space",
    "UNSET",
    # Numeric spaces - User-facing functions
    "Float",
    "Int",
    # Numeric spaces - Classes (for type checking/introspection)
    "NumberSpace",
    "FloatSpace",
    "IntSpace",
    # Categorical spaces - User-facing function
    "Categorical",
    # Categorical spaces - Classes and helpers
    "CategoricalSpace",
    "Choice",
    # Conditional spaces - User-facing function
    "Conditional",
    # Conditional spaces - Class
    "ConditionalSpace",
    # Conditions - Paths
    "ParsedFieldPath",
    # Conditions - Base classes
    "Condition",
    "AttributeCondition",
    "ObjectCondition",
    # Conditions - Attribute conditions
    "FieldCondition",
    "MultiFieldLambdaCondition",
    # Conditions - Object conditions (Equality)
    "EqualsTo",
    "NotEqualsTo",
    # Conditions - Object conditions (Membership)
    "In",
    "NotIn",
    # Conditions - Object conditions (Comparison)
    "SmallerThan",
    "LargerThan",
    # Conditions - Object conditions (Type checking)
    "IsInstance",
    # Conditions - Object conditions (Logical)
    "And",
    "Or",
    "Not",
    # Conditions - Object conditions (Custom)
    "Lambda",
    # Internal utilities (exposed for advanced use)
    "infer_space_from_field_info",
]
