"""SpaX: Pythonic, type-safe search space definition and exploration.

SpaX is a library for defining, exploring, visualizing, and optimizing search
spaces for hyperparameter optimization, neural architecture search, and other
machine learning experimentation tasks.

Core Components:
---------------
- Config: Base class for defining searchable configurations
- Spaces: Float, Int, Categorical, Conditional for defining parameter ranges
- Conditions: For making parameters conditional on other parameters
- Samplers: RandomSampler, TrialSampler for generating parameter values

Key Features:
------------
- Declarative search space definition with minimal code
- Type-safe with Pydantic validation
- Conditional parameters based on other parameter values
- Random sampling with reproducible seeds
- Integration with HPO libraries (Optuna, etc.)
- Multiple serialization formats (JSON, YAML, TOML)
- Override system for iterative space narrowing

For more information, see the documentation at https://github.com/keyhankamyar/SpaX
"""

from .config import Config
from .samplers import RandomSampler, TrialSampler
from .spaces import (
    UNSET,
    And,
    Categorical,
    CategoricalSpace,
    Choice,
    Conditional,
    ConditionalSpace,
    EqualsTo,
    FieldCondition,
    Float,
    FloatSpace,
    In,
    Int,
    IntSpace,
    IsInstance,
    Lambda,
    LargerThan,
    MultiFieldLambdaCondition,
    Not,
    NotEqualsTo,
    NotIn,
    Or,
    SmallerThan,
)

__version__ = "0.2.0"

__all__ = [
    # Core configuration class
    "Config",
    # Samplers
    "RandomSampler",
    "TrialSampler",
    # Space types - User-facing functions
    "Float",
    "Int",
    "Categorical",
    "Conditional",
    "Choice",
    # Space types - Classes (for type checking and introspection)
    "FloatSpace",
    "IntSpace",
    "CategoricalSpace",
    "ConditionalSpace",
    "UNSET",
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
    # Version
    "__version__",
]
