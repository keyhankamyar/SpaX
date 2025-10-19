"""
Spax: Search Space Exploration and Optimization

A unified library for defining, exploring, visualizing, and optimizing
hyperparameter search spaces with seamless integration into ML workflows.

Example:
    >>> import spax as sp
    >>>
    >>> class ModelConfig(sp.Config):
    ...     learning_rate: float = sp.Float(1e-5, 1e-1, "log")
    ...     batch_size: int = sp.Int(8, 128, "log")
    ...     activation: str = sp.Categorical(["relu", "gelu", "silu"])
    ...     optimizer: str = sp.Categorical(["adam", "sgd"])
    ...     # Conditional: different LR range based on optimizer
    ...     lr_multiplier: float = sp.Conditional(
    ...         sp.FieldCondition("optimizer", sp.EqualsTo("sgd")),
    ...         true=sp.Float(0.1, 10.0),
    ...         false=sp.Float(0.01, 1.0)
    ...     )
    ...
    >>> # Sample a random configuration
    >>> config = ModelConfig.random()
    >>>
    >>> # Or create with specific values
    >>> config = ModelConfig(
    ...     learning_rate=0.001,
    ...     batch_size=32,
    ...     activation="relu",
    ...     optimizer="adam",
    ...     lr_multiplier=0.5
    ... )
"""

from .config import Config
from .distributions import (
    LOG,
    UNIFORM,
    CategoricalDistribution,
    Distribution,
    LogDistribution,
    NumberDistribution,
    UniformDistribution,
)
from .spaces import (
    UNSET,
    And,
    Categorical,
    CategoricalSpace,
    Choice,
    Condition,
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
    Not,
    NotEqualsTo,
    NotIn,
    Or,
    SmallerThan,
    Space,
)

__version__ = "0.1.0"

__all__ = [
    # Core config
    "Config",
    # Space types (user-facing functions)
    "Float",
    "Int",
    "Categorical",
    "Conditional",
    "Choice",
    # Space classes (for type checking and introspection)
    "FloatSpace",
    "IntSpace",
    "CategoricalSpace",
    "ConditionalSpace",
    "Space",
    "UNSET",
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
    # Distributions
    "Distribution",
    "NumberDistribution",
    "UniformDistribution",
    "LogDistribution",
    "CategoricalDistribution",
    "UNIFORM",
    "LOG",
    # Version
    "__version__",
]
