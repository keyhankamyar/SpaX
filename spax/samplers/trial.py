from typing import Any, Literal

from .base import Sampler


class TrialSampler(Sampler):
    """
    Sampler that wraps an Optuna Trial object, or any object that implements:
    suggest_int(name, low, high, log)
    suggest_float(name, low, high, log)
    suggest_categorical(name, choices)
    """

    def __init__(self, trial: Any) -> None:
        """Initialize with an Optuna trial.

        Args:
            trial: An optuna.Trial object
        """
        self.trial = trial
        self.record: dict[str, Any] = {}

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        low_inclusive: bool,
        high_inclusive: bool,
        distribution: Literal["log", "uniform"],
    ) -> int:
        """Suggest an integer using Optuna's suggest_int."""
        # Adjust bounds based on inclusivity
        # Optuna's suggest_int uses inclusive bounds by default
        if not low_inclusive:
            low = low + 1
        if not high_inclusive:
            high = high - 1

        # Map distribution
        log = distribution == "log"

        self.record[name] = self.trial.suggest_int(
            name=name, low=low, high=high, log=log
        )
        return self.record[name]

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        low_inclusive: bool,
        high_inclusive: bool,
        distribution: Literal["log", "uniform"],
    ) -> float:
        """Suggest a float using Optuna's suggest_float."""
        if not low_inclusive:
            low = low + 1e-10
        if not high_inclusive:
            high = high - 1e-10

        # Map distribution
        log = distribution == "log"

        self.record[name] = self.trial.suggest_float(
            name=name, low=low, high=high, log=log
        )
        return self.record[name]

    def suggest_categorical(
        self,
        name: str,
        choices: list[Any],
        weights: list[float],  # noqa: ARG002
    ) -> Any:
        """Suggest a categorical choice using Optuna's suggest_categorical.

        Note: Optuna's suggest_categorical doesn't support weights directly,
        so weights are ignored. For weighted sampling, use RandomSampler or
        implement custom sampling logic.
        """
        self.record[name] = self.trial.suggest_categorical(name=name, choices=choices)
        return self.record[name]
