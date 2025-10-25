"""Optuna trial sampler for Bayesian optimization integration.

This module provides TrialSampler, which wraps Optuna Trial objects to enable
integration with Optuna's hyperparameter optimization algorithms. It can also
wrap any object that implements the Optuna Trial interface.
"""

import inspect
from typing import Any, Literal

from .base import Sampler


class TrialSampler(Sampler):
    """Sampler that wraps an Optuna Trial object.

    TrialSampler enables integration with Optuna's hyperparameter optimization
    by wrapping an Optuna Trial or FrozenTrial object. It translates between
    SpaX's sampling interface and Optuna's suggest methods.

    The trial object must be either:
    - An instance of optuna.Trial or optuna.FrozenTrial, OR
    - Any object implementing the methods:
        - suggest_int(name: str, low: int, high: int, log: bool) -> int
        - suggest_float(name: str, low: float, high: float, log: bool) -> float
        - suggest_categorical(name: str, choices: list) -> Any

    Examples:
        >>> import spax as sp
        >>> import optuna
        >>>
        >>> def objective(trial):
        ...     # Create sampler from Optuna trial
        ...     sampler = sp.samplers.TrialSampler(trial)
        ...
        ...     # Sample a configuration
        ...     config = MyConfig.random(sampler=sampler)
        ...
        ...     # Train and evaluate
        ...     score = train_and_evaluate(config)
        ...     return score
        >>>
        >>> study = optuna.create_study(direction="maximize")
        >>> study.optimize(objective, n_trials=100)
    """

    def __init__(self, trial: Any) -> None:
        """Initialize a TrialSampler with an Optuna Trial object.

        Args:
            trial: An Optuna Trial/FrozenTrial object, or any object implementing
                the required suggest_int, suggest_float, and suggest_categorical
                methods with the correct signatures.

        Raises:
            TypeError: If trial doesn't implement the required interface.
        """
        # Check if it's an Optuna Trial/FrozenTrial
        is_optuna_trial = False
        try:
            import optuna  # pyright: ignore[reportMissingImports]  # noqa: I001

            if isinstance(trial, (optuna.Trial, optuna.trial.FrozenTrial)):
                is_optuna_trial = True
        except ImportError:
            # Optuna not installed, check interface manually
            pass

        if not is_optuna_trial:
            # Validate that trial implements the required interface
            self._validate_trial_interface(trial)

        self._trial = trial
        self._record: dict[str, Any] = {}

    def _validate_trial_interface(self, trial: Any) -> None:
        """Validate that trial implements the required Optuna interface.

        Args:
            trial: The trial object to validate.

        Raises:
            TypeError: If trial doesn't implement the required methods correctly.
        """
        # Check suggest_int
        if not hasattr(trial, "suggest_int"):
            raise TypeError(
                "Trial object must implement suggest_int method. "
                "Expected signature: suggest_int(name: str, low: int, high: int, log: bool) -> int"
            )

        try:
            sig = inspect.signature(trial.suggest_int)
            params = list(sig.parameters.keys())
            if params != ["name", "low", "high", "log"] and params != [
                "name",
                "low",
                "high",
            ]:
                raise TypeError(
                    f"suggest_int has incorrect signature. "
                    f"Expected (name, low, high, log) or (name, low, high), got {params}"
                )
        except (ValueError, TypeError) as e:
            raise TypeError(f"Could not validate suggest_int signature: {e}") from e

        # Check suggest_float
        if not hasattr(trial, "suggest_float"):
            raise TypeError(
                "Trial object must implement suggest_float method. "
                "Expected signature: suggest_float(name: str, low: float, high: float, log: bool) -> float"
            )

        try:
            sig = inspect.signature(trial.suggest_float)
            params = list(sig.parameters.keys())
            if params != ["name", "low", "high", "log"] and params != [
                "name",
                "low",
                "high",
            ]:
                raise TypeError(
                    f"suggest_float has incorrect signature. "
                    f"Expected (name, low, high, log) or (name, low, high), got {params}"
                )
        except (ValueError, TypeError) as e:
            raise TypeError(f"Could not validate suggest_float signature: {e}") from e

        # Check suggest_categorical
        if not hasattr(trial, "suggest_categorical"):
            raise TypeError(
                "Trial object must implement suggest_categorical method. "
                "Expected signature: suggest_categorical(name: str, choices: list) -> Any"
            )

        try:
            sig = inspect.signature(trial.suggest_categorical)
            params = list(sig.parameters.keys())
            if params != ["name", "choices"]:
                raise TypeError(
                    f"suggest_categorical has incorrect signature. "
                    f"Expected (name, choices), got {params}"
                )
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"Could not validate suggest_categorical signature: {e}"
            ) from e

    @property
    def trial(self) -> Any:
        """The wrapped Optuna Trial object."""
        return self._trial

    @property
    def record(self) -> dict[str, Any]:
        """Get a copy of all sampled parameter values.

        Returns:
            Dictionary mapping parameter names to their sampled values.
        """
        return self._record.copy()

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        low_inclusive: bool,
        high_inclusive: bool,
        distribution: Literal["log", "uniform"],
    ) -> int:
        """Sample an integer value using the Optuna trial.

        Args:
            name: Parameter name for tracking.
            low: Lower bound of the range.
            high: Upper bound of the range.
            low_inclusive: Whether the lower bound is inclusive.
            high_inclusive: Whether the upper bound is inclusive.
            distribution: Sampling distribution ('uniform' or 'log').

        Returns:
            Sampled integer value from Optuna's sampler.
        """
        # Adjust bounds based on inclusivity
        # Optuna's suggest_int uses inclusive bounds by default
        if not low_inclusive:
            low = low + 1
        if not high_inclusive:
            high = high - 1

        # Map distribution to Optuna's log parameter
        log = distribution == "log"

        value = int(self._trial.suggest_int(name=name, low=low, high=high, log=log))
        self._record[name] = value
        return value

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        low_inclusive: bool,
        high_inclusive: bool,
        distribution: Literal["log", "uniform"],
    ) -> float:
        """Sample a float value using the Optuna trial.

        Args:
            name: Parameter name for tracking.
            low: Lower bound of the range.
            high: Upper bound of the range.
            low_inclusive: Whether the lower bound is inclusive.
            high_inclusive: Whether the upper bound is inclusive.
            distribution: Sampling distribution ('uniform' or 'log').

        Returns:
            Sampled float value from Optuna's sampler.
        """
        # Adjust bounds based on inclusivity (small epsilon for floats)
        if not low_inclusive:
            low = low + 1e-10
        if not high_inclusive:
            high = high - 1e-10

        # Map distribution to Optuna's log parameter
        log = distribution == "log"

        value = float(self._trial.suggest_float(name=name, low=low, high=high, log=log))
        self._record[name] = value
        return value

    def suggest_categorical(
        self,
        name: str,
        choices: list[Any],
        weights: list[float],  # noqa: ARG002
    ) -> Any:
        """Sample a categorical value using the Optuna trial.

        Note: Optuna's suggest_categorical does not support weights,
        so the weights parameter is ignored.

        Args:
            name: Parameter name for tracking.
            choices: List of possible values to choose from.
            weights: Weights for each choice (ignored by Optuna).

        Returns:
            One of the choices, selected by Optuna's sampler.
        """
        value = self._trial.suggest_categorical(name=name, choices=choices)
        self._record[name] = value
        return value
