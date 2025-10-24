"""SpaX: Search space eXploration.

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

Quick Start:
-----------
    >>> import spax as sp
    >>>
    >>> # Define a searchable configuration
    >>> class MyConfig(sp.Config):
    ...     learning_rate: float = sp.Float(ge=1e-5, le=1e-1, distribution='log')
    ...     num_layers: int = sp.Int(ge=1, le=10)
    ...     optimizer: str = sp.Categorical(["adam", "sgd", "rmsprop"])
    ...     use_dropout: bool
    ...     dropout_rate: float = sp.Conditional(
    ...         sp.FieldCondition("use_dropout", sp.EqualsTo(True)),
    ...         true=sp.Float(gt=0.0, lt=0.5),
    ...         false=0.0
    ...     )
    >>>
    >>> # Sample random configurations
    >>> config = MyConfig.random(seed=42)
    >>> print(config.learning_rate, config.num_layers, config.optimizer)
    >>>
    >>> # Apply overrides to narrow search space
    >>> override = {"num_layers": {"ge": 5, "le": 7}}
    >>> config = MyConfig.random(seed=42, override=override)
    >>>
    >>> # Integration with Optuna
    >>> import optuna
    >>>
    >>> def objective(trial):
    ...     sampler = sp.TrialSampler(trial)
    ...     config = MyConfig.random(sampler=sampler)
    ...     return train_and_evaluate(config)
    >>>
    >>> study = optuna.create_study(direction="maximize")
    >>> study.optimize(objective, n_trials=100)

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
