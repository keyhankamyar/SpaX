"""Samplers for generating parameter values from search spaces.

This module provides sampler implementations that generate parameter values
from search spaces. Samplers provide a unified interface for different sampling
strategies:

- RandomSampler: Random sampling with uniform or log-uniform distributions
- TrialSampler: Wrapper for Optuna Trial objects for Bayesian optimization

All samplers implement the Sampler abstract base class and can be used with
Config.random() to generate random configurations.

Examples:
    >>> import spax as sp
    >>>
    >>> # Random sampling with seed
    >>> config = MyConfig.random(seed=42)
    >>>
    >>> # Using RandomSampler directly
    >>> sampler = sp.RandomSampler(seed=42)
    >>> config = MyConfig.random(sampler=sampler)
    >>>
    >>> # Optuna integration
    >>> import optuna
    >>> def objective(trial):
    ...     sampler = sp.TrialSampler(trial)
    ...     config = MyConfig.random(sampler=sampler)
    ...     return evaluate(config)
    >>> study = optuna.create_study()
    >>> study.optimize(objective, n_trials=100)
"""

from .base import Sampler
from .random import RandomSampler
from .trial import TrialSampler

__all__ = [
    # Base class
    "Sampler",
    # Sampler implementations
    "RandomSampler",
    "TrialSampler",
]
