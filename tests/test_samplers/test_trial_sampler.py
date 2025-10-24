"""Tests for TrialSampler."""

import pytest

from spax.samplers import TrialSampler


class MockTrial:
    """Mock trial object that implements the Optuna Trial interface."""

    def __init__(self):
        self.suggestions = {}

    def suggest_int(self, name: str, low: int, high: int, log: bool = False) -> int:
        """Mock suggest_int method."""
        # Simple deterministic sampling based on name hash
        value = (hash(name) % (high - low + 1)) + low
        self.suggestions[name] = value
        return value

    def suggest_float(
        self, name: str, low: float, high: float, log: bool = False
    ) -> float:
        """Mock suggest_float method."""
        # Simple deterministic sampling based on name hash
        ratio = (hash(name) % 1000) / 1000.0
        value = low + ratio * (high - low)
        self.suggestions[name] = value
        return value

    def suggest_categorical(self, name: str, choices: list) -> any:
        """Mock suggest_categorical method."""
        # Simple deterministic choice based on name hash
        idx = hash(name) % len(choices)
        value = choices[idx]
        self.suggestions[name] = value
        return value


class IncompleteTrial:
    """Mock trial that is missing some required methods."""

    def suggest_int(self, name: str, low: int, high: int, log: bool = False) -> int:
        return low


class WrongSignatureTrial:
    """Mock trial with wrong method signatures."""

    def suggest_int(self, name: str, low: int) -> int:  # Missing high and log
        return low

    def suggest_float(
        self, name: str, low: float, high: float, log: bool = False
    ) -> float:
        return low

    def suggest_categorical(self, name: str, choices: list) -> any:
        return choices[0]


class TestTrialSamplerInit:
    """Tests for TrialSampler initialization."""

    def test_init_with_valid_mock_trial(self):
        """Test initialization with valid mock trial."""
        trial = MockTrial()
        sampler = TrialSampler(trial)

        assert sampler.trial is trial
        assert sampler.record == {}

    def test_init_with_incomplete_trial_raises_error(self):
        """Test that incomplete trial raises TypeError."""
        trial = IncompleteTrial()

        with pytest.raises(TypeError, match="must implement suggest_float"):
            TrialSampler(trial)

    def test_init_with_wrong_signature_raises_error(self):
        """Test that wrong signature raises TypeError."""
        trial = WrongSignatureTrial()

        with pytest.raises(TypeError, match="suggest_int has incorrect signature"):
            TrialSampler(trial)

    def test_init_with_missing_suggest_int(self):
        """Test that missing suggest_int raises TypeError."""
        trial = type(
            "BadTrial",
            (),
            {
                "suggest_float": lambda self, name, low, high, log: low,
                "suggest_categorical": lambda self, name, choices: choices[0],
            },
        )()

        with pytest.raises(TypeError, match="must implement suggest_int"):
            TrialSampler(trial)

    def test_init_with_missing_suggest_float(self):
        """Test that missing suggest_float raises TypeError."""
        trial = type(
            "BadTrial",
            (),
            {
                "suggest_int": lambda self, name, low, high, log: low,
                "suggest_categorical": lambda self, name, choices: choices[0],
            },
        )()

        with pytest.raises(TypeError, match="must implement suggest_float"):
            TrialSampler(trial)

    def test_init_with_missing_suggest_categorical(self):
        """Test that missing suggest_categorical raises TypeError."""
        trial = type(
            "BadTrial",
            (),
            {
                "suggest_int": lambda self, name, low, high, log: low,
                "suggest_float": lambda self, name, low, high, log: low,
            },
        )()

        with pytest.raises(TypeError, match="must implement suggest_categorical"):
            TrialSampler(trial)


class TestTrialSamplerProperties:
    """Tests for TrialSampler properties."""

    def test_trial_property(self):
        """Test that trial property returns the wrapped trial."""
        trial = MockTrial()
        sampler = TrialSampler(trial)

        assert sampler.trial is trial

    def test_record_property_returns_copy(self):
        """Test that record property returns a copy."""
        trial = MockTrial()
        sampler = TrialSampler(trial)

        sampler.suggest_int("param", 1, 10, True, True, "uniform")

        record1 = sampler.record
        record2 = sampler.record

        # Should be equal but not the same object
        assert record1 == record2
        assert record1 is not record2


class TestSuggestInt:
    """Tests for suggest_int method."""

    def test_suggest_int_inclusive_bounds(self):
        """Test suggest_int with inclusive bounds."""
        trial = MockTrial()
        sampler = TrialSampler(trial)

        value = sampler.suggest_int("param", 1, 10, True, True, "uniform")

        # Check that trial's method was called
        assert "param" in trial.suggestions
        # Check that sampler recorded the value
        assert sampler.record["param"] == value
        # Check that value is in expected range
        assert 1 <= value <= 10

    def test_suggest_int_exclusive_low_bound(self):
        """Test suggest_int with exclusive lower bound."""
        trial = MockTrial()
        sampler = TrialSampler(trial)

        # When low_inclusive=False, low should be adjusted by +1
        value = sampler.suggest_int("param", 1, 10, False, True, "uniform")

        # The trial should have been called with adjusted bounds (2, 10)
        assert value >= 2  # Since we adjusted low from 1 to 2
        assert value <= 10

    def test_suggest_int_exclusive_high_bound(self):
        """Test suggest_int with exclusive upper bound."""
        trial = MockTrial()
        sampler = TrialSampler(trial)

        # When high_inclusive=False, high should be adjusted by -1
        value = sampler.suggest_int("param", 1, 10, True, False, "uniform")

        # The trial should have been called with adjusted bounds (1, 9)
        assert value >= 1
        assert value <= 9

    def test_suggest_int_log_distribution(self):
        """Test suggest_int with log distribution."""
        trial = MockTrial()
        sampler = TrialSampler(trial)

        value = sampler.suggest_int("param", 1, 1000, True, True, "log")

        # Just verify it was called and recorded
        assert "param" in sampler.record
        assert value == sampler.record["param"]


class TestSuggestFloat:
    """Tests for suggest_float method."""

    def test_suggest_float_inclusive_bounds(self):
        """Test suggest_float with inclusive bounds."""
        trial = MockTrial()
        sampler = TrialSampler(trial)

        value = sampler.suggest_float("param", 0.0, 1.0, True, True, "uniform")

        # Check that trial's method was called
        assert "param" in trial.suggestions
        # Check that sampler recorded the value
        assert sampler.record["param"] == value
        # Check that value is in expected range
        assert 0.0 <= value <= 1.0

    def test_suggest_float_exclusive_low_bound(self):
        """Test suggest_float with exclusive lower bound."""
        trial = MockTrial()
        sampler = TrialSampler(trial)

        value = sampler.suggest_float("param", 0.0, 1.0, False, True, "uniform")

        # Value should be > 0.0 (adjusted by epsilon)
        assert value > 0.0
        assert value <= 1.0

    def test_suggest_float_exclusive_high_bound(self):
        """Test suggest_float with exclusive upper bound."""
        trial = MockTrial()
        sampler = TrialSampler(trial)

        value = sampler.suggest_float("param", 0.0, 1.0, True, False, "uniform")

        # Value should be < 1.0 (adjusted by epsilon)
        assert value >= 0.0
        assert value < 1.0

    def test_suggest_float_log_distribution(self):
        """Test suggest_float with log distribution."""
        trial = MockTrial()
        sampler = TrialSampler(trial)

        value = sampler.suggest_float("param", 1e-5, 1.0, True, True, "log")

        # Just verify it was called and recorded
        assert "param" in sampler.record
        assert value == sampler.record["param"]


class TestSuggestCategorical:
    """Tests for suggest_categorical method."""

    def test_suggest_categorical_basic(self):
        """Test basic categorical sampling."""
        trial = MockTrial()
        sampler = TrialSampler(trial)

        choices = ["a", "b", "c"]
        value = sampler.suggest_categorical("param", choices, [1.0, 1.0, 1.0])

        # Check that trial's method was called
        assert "param" in trial.suggestions
        # Check that sampler recorded the value
        assert sampler.record["param"] == value
        # Check that value is one of the choices
        assert value in choices

    def test_suggest_categorical_ignores_weights(self):
        """Test that weights are ignored (Optuna doesn't support them)."""
        trial = MockTrial()
        sampler = TrialSampler(trial)

        choices = ["a", "b", "c"]
        # Weights should be ignored
        value = sampler.suggest_categorical("param", choices, [10.0, 1.0, 1.0])

        # Just verify it works and returns a valid choice
        assert value in choices

    def test_suggest_categorical_single_choice(self):
        """Test categorical with single choice."""
        trial = MockTrial()
        sampler = TrialSampler(trial)

        value = sampler.suggest_categorical("param", ["only"], [1.0])

        assert value == "only"


class TestTrialSamplerIntegration:
    """Integration tests for TrialSampler."""

    def test_multiple_parameters(self):
        """Test sampling multiple parameters."""
        trial = MockTrial()
        sampler = TrialSampler(trial)

        int_val = sampler.suggest_int("layers", 1, 10, True, True, "uniform")
        float_val = sampler.suggest_float("lr", 0.0, 1.0, True, True, "uniform")
        cat_val = sampler.suggest_categorical("optimizer", ["adam", "sgd"], [1.0, 1.0])

        # Check record
        assert len(sampler.record) == 3
        assert sampler.record["layers"] == int_val
        assert sampler.record["lr"] == float_val
        assert sampler.record["optimizer"] == cat_val

        # Check trial also recorded them
        assert len(trial.suggestions) == 3

    def test_overwriting_parameter(self):
        """Test that sampling same parameter overwrites."""
        trial = MockTrial()
        sampler = TrialSampler(trial)

        value1 = sampler.suggest_int("param", 1, 10, True, True, "uniform")
        value2 = sampler.suggest_int("param", 20, 30, True, True, "uniform")

        # Record should have latest value
        assert len(sampler.record) == 1
        assert sampler.record["param"] == value2
        assert value2 != value1  # Different ranges


class TestTrialWithOptionalLogParameter:
    """Test trial objects where log parameter has default value."""

    def test_trial_with_default_log_parameter(self):
        """Test trial where log parameter has default value."""

        class TrialWithDefaults:
            def suggest_int(
                self, name: str, low: int, high: int, log: bool = False
            ) -> int:
                return (hash(name) % (high - low + 1)) + low

            def suggest_float(
                self, name: str, low: float, high: float, log: bool = False
            ) -> float:
                ratio = (hash(name) % 1000) / 1000.0
                return low + ratio * (high - low)

            def suggest_categorical(self, name: str, choices: list) -> any:
                return choices[hash(name) % len(choices)]

        trial = TrialWithDefaults()
        sampler = TrialSampler(trial)

        # Should work fine
        value = sampler.suggest_int("param", 1, 10, True, True, "uniform")
        assert 1 <= value <= 10


class TestRealOptunaIntegration:
    """Tests for real Optuna integration (if available)."""

    def test_with_real_optuna_trial(self):
        """Test with real Optuna trial if available."""
        try:
            import optuna
        except ImportError:
            pytest.skip("Optuna not installed")

        def objective(trial):
            sampler = TrialSampler(trial)

            # Sample some parameters
            int_val = sampler.suggest_int("x", 1, 10, True, True, "uniform")
            float_val = sampler.suggest_float("y", 0.0, 1.0, True, True, "uniform")
            cat_val = sampler.suggest_categorical("z", ["a", "b"], [1.0, 1.0])

            # Check record
            assert len(sampler.record) == 3
            assert sampler.record["x"] == int_val
            assert sampler.record["y"] == float_val
            assert sampler.record["z"] == cat_val

            # Return dummy score
            return int_val + float_val

        study = optuna.create_study()
        study.optimize(objective, n_trials=5, show_progress_bar=False)

        assert len(study.trials) == 5

    def test_with_frozen_trial(self):
        """Test with Optuna FrozenTrial if available."""
        try:
            import optuna
        except ImportError:
            pytest.skip("Optuna not installed")

        # Create a study and run one trial
        study = optuna.create_study()
        study.optimize(
            lambda trial: trial.suggest_int("x", 1, 10),
            n_trials=1,
            show_progress_bar=False,
        )

        # Get frozen trial
        frozen_trial = study.trials[0]

        # Should be able to create sampler with frozen trial
        sampler = TrialSampler(frozen_trial)
        assert sampler.trial is frozen_trial
