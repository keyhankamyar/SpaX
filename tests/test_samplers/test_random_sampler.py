"""Tests for RandomSampler."""

import math

import pytest

from spax.samplers import RandomSampler


class TestRandomSamplerInit:
    """Tests for RandomSampler initialization."""

    def test_init_without_seed(self):
        """Test initialization without seed."""
        sampler = RandomSampler()
        assert sampler.record == {}

    def test_init_with_seed(self):
        """Test initialization with seed."""
        sampler = RandomSampler(seed=42)
        assert sampler.record == {}

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        sampler1 = RandomSampler(seed=42)
        value1 = sampler1.suggest_float("test", 0.0, 1.0, True, True, "uniform")

        sampler2 = RandomSampler(seed=42)
        value2 = sampler2.suggest_float("test", 0.0, 1.0, True, True, "uniform")

        assert value1 == value2

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        sampler1 = RandomSampler(seed=42)
        value1 = sampler1.suggest_float("test", 0.0, 1.0, True, True, "uniform")

        sampler2 = RandomSampler(seed=43)
        value2 = sampler2.suggest_float("test", 0.0, 1.0, True, True, "uniform")

        assert value1 != value2


class TestRandomSamplerRecord:
    """Tests for RandomSampler record tracking."""

    def test_record_is_empty_initially(self):
        """Test that record is empty on initialization."""
        sampler = RandomSampler(seed=42)
        assert sampler.record == {}

    def test_record_tracks_int_suggestions(self):
        """Test that record tracks integer suggestions."""
        sampler = RandomSampler(seed=42)
        value = sampler.suggest_int("param1", 1, 10, True, True, "uniform")

        assert "param1" in sampler.record
        assert sampler.record["param1"] == value

    def test_record_tracks_float_suggestions(self):
        """Test that record tracks float suggestions."""
        sampler = RandomSampler(seed=42)
        value = sampler.suggest_float("param1", 0.0, 1.0, True, True, "uniform")

        assert "param1" in sampler.record
        assert sampler.record["param1"] == value

    def test_record_tracks_categorical_suggestions(self):
        """Test that record tracks categorical suggestions."""
        sampler = RandomSampler(seed=42)
        value = sampler.suggest_categorical("param1", ["a", "b", "c"], [1.0, 1.0, 1.0])

        assert "param1" in sampler.record
        assert sampler.record["param1"] == value

    def test_record_tracks_multiple_parameters(self):
        """Test that record tracks multiple parameters."""
        sampler = RandomSampler(seed=42)

        int_val = sampler.suggest_int("int_param", 1, 10, True, True, "uniform")
        float_val = sampler.suggest_float(
            "float_param", 0.0, 1.0, True, True, "uniform"
        )
        cat_val = sampler.suggest_categorical("cat_param", ["a", "b"], [1.0, 1.0])

        assert len(sampler.record) == 3
        assert sampler.record["int_param"] == int_val
        assert sampler.record["float_param"] == float_val
        assert sampler.record["cat_param"] == cat_val

    def test_record_returns_copy(self):
        """Test that record property returns a copy."""
        sampler = RandomSampler(seed=42)
        sampler.suggest_int("param1", 1, 10, True, True, "uniform")

        record1 = sampler.record
        record2 = sampler.record

        # Should be equal but not the same object
        assert record1 == record2
        assert record1 is not record2


class TestSuggestInt:
    """Tests for suggest_int method."""

    def test_suggest_int_uniform_inclusive(self):
        """Test uniform integer sampling with inclusive bounds."""
        sampler = RandomSampler(seed=42)

        # Sample many values to check range
        values = [
            sampler.suggest_int(f"p{i}", 1, 10, True, True, "uniform")
            for i in range(100)
        ]

        # All values should be in range [1, 10]
        assert all(1 <= v <= 10 for v in values)
        assert min(values) >= 1
        assert max(values) <= 10

    def test_suggest_int_uniform_exclusive_low(self):
        """Test uniform integer sampling with exclusive lower bound."""
        sampler = RandomSampler(seed=42)

        values = [
            sampler.suggest_int(f"p{i}", 1, 10, False, True, "uniform")
            for i in range(100)
        ]

        # All values should be in range (1, 10] = [2, 10]
        assert all(2 <= v <= 10 for v in values)
        assert 1 not in values  # Should never sample 1

    def test_suggest_int_uniform_exclusive_high(self):
        """Test uniform integer sampling with exclusive upper bound."""
        sampler = RandomSampler(seed=42)

        values = [
            sampler.suggest_int(f"p{i}", 1, 10, True, False, "uniform")
            for i in range(100)
        ]

        # All values should be in range [1, 10) = [1, 9]
        assert all(1 <= v <= 9 for v in values)
        assert 10 not in values  # Should never sample 10

    def test_suggest_int_uniform_exclusive_both(self):
        """Test uniform integer sampling with both bounds exclusive."""
        sampler = RandomSampler(seed=42)

        values = [
            sampler.suggest_int(f"p{i}", 1, 10, False, False, "uniform")
            for i in range(100)
        ]

        # All values should be in range (1, 10) = [2, 9]
        assert all(2 <= v <= 9 for v in values)
        assert 1 not in values
        assert 10 not in values

    def test_suggest_int_log_distribution(self):
        """Test log-uniform integer sampling."""
        sampler = RandomSampler(seed=42)

        values = [
            sampler.suggest_int(f"p{i}", 1, 1000, True, True, "log") for i in range(100)
        ]

        # All values should be in range
        assert all(1 <= v <= 1000 for v in values)

        # Log distribution should favor smaller values
        # Count values in lower half vs upper half
        lower_half = sum(1 for v in values if v <= 500)
        upper_half = sum(1 for v in values if v > 500)

        # Should have more samples in lower half (not a strict test, probabilistic)
        assert lower_half > upper_half * 0.5  # At least somewhat biased

    def test_suggest_int_invalid_distribution(self):
        """Test that invalid distribution raises error."""
        sampler = RandomSampler(seed=42)

        with pytest.raises(ValueError, match="Unknown distribution"):
            sampler.suggest_int("param", 1, 10, True, True, "invalid")

    def test_suggest_int_single_value_range(self):
        """Test sampling from single-value range."""
        sampler = RandomSampler(seed=42)

        values = [
            sampler.suggest_int(f"p{i}", 5, 5, True, True, "uniform") for i in range(10)
        ]

        # All values should be 5
        assert all(v == 5 for v in values)


class TestSuggestFloat:
    """Tests for suggest_float method."""

    def test_suggest_float_uniform_inclusive(self):
        """Test uniform float sampling with inclusive bounds."""
        sampler = RandomSampler(seed=42)

        values = [
            sampler.suggest_float(f"p{i}", 0.0, 1.0, True, True, "uniform")
            for i in range(100)
        ]

        # All values should be in range [0.0, 1.0]
        assert all(0.0 <= v <= 1.0 for v in values)

    def test_suggest_float_uniform_exclusive_low(self):
        """Test uniform float sampling with exclusive lower bound."""
        sampler = RandomSampler(seed=42)

        values = [
            sampler.suggest_float(f"p{i}", 0.0, 1.0, False, True, "uniform")
            for i in range(100)
        ]

        # All values should be > 0.0
        assert all(v > 0.0 for v in values)
        assert all(v <= 1.0 for v in values)

    def test_suggest_float_uniform_exclusive_high(self):
        """Test uniform float sampling with exclusive upper bound."""
        sampler = RandomSampler(seed=42)

        values = [
            sampler.suggest_float(f"p{i}", 0.0, 1.0, True, False, "uniform")
            for i in range(100)
        ]

        # All values should be < 1.0
        assert all(v >= 0.0 for v in values)
        assert all(v < 1.0 for v in values)

    def test_suggest_float_log_distribution(self):
        """Test log-uniform float sampling."""
        sampler = RandomSampler(seed=42)

        values = [
            sampler.suggest_float(f"p{i}", 1e-5, 1.0, True, True, "log")
            for i in range(100)
        ]

        # All values should be in range
        assert all(1e-5 <= v <= 1.0 for v in values)

        # Log distribution should favor smaller values
        # Use log scale to check distribution
        log_values = [math.log10(v) for v in values]
        median_log = sum(log_values) / len(log_values)

        # Median should be closer to geometric mean than arithmetic mean
        log_low = math.log10(1e-5)
        log_high = math.log10(1.0)
        geometric_mean_log = (log_low + log_high) / 2

        # Check that median is reasonably close to geometric mean
        assert abs(median_log - geometric_mean_log) < 1.0

    def test_suggest_float_invalid_distribution(self):
        """Test that invalid distribution raises error."""
        sampler = RandomSampler(seed=42)

        with pytest.raises(ValueError, match="Unknown distribution"):
            sampler.suggest_float("param", 0.0, 1.0, True, True, "invalid")

    def test_suggest_float_negative_range(self):
        """Test sampling from negative range."""
        sampler = RandomSampler(seed=42)

        values = [
            sampler.suggest_float(f"p{i}", -1.0, 0.0, True, True, "uniform")
            for i in range(100)
        ]

        assert all(-1.0 <= v <= 0.0 for v in values)


class TestSuggestCategorical:
    """Tests for suggest_categorical method."""

    def test_suggest_categorical_uniform_weights(self):
        """Test categorical sampling with uniform weights."""
        sampler = RandomSampler(seed=42)

        choices = ["a", "b", "c"]
        weights = [1.0, 1.0, 1.0]

        values = [
            sampler.suggest_categorical(f"p{i}", choices, weights) for i in range(300)
        ]

        # All values should be in choices
        assert all(v in choices for v in values)

        # With uniform weights, each choice should appear roughly equally
        counts = {c: values.count(c) for c in choices}

        # Each should appear roughly 100 times (allow some variance)
        for count in counts.values():
            assert 50 < count < 150  # Loose bounds for randomness

    def test_suggest_categorical_weighted(self):
        """Test categorical sampling with non-uniform weights."""
        sampler = RandomSampler(seed=42)

        choices = ["a", "b", "c"]
        weights = [10.0, 1.0, 1.0]  # 'a' is much more likely

        values = [
            sampler.suggest_categorical(f"p{i}", choices, weights) for i in range(300)
        ]

        # Count occurrences
        counts = {c: values.count(c) for c in choices}

        # 'a' should appear much more frequently
        assert counts["a"] > counts["b"] * 3
        assert counts["a"] > counts["c"] * 3

    def test_suggest_categorical_single_choice(self):
        """Test categorical sampling with single choice."""
        sampler = RandomSampler(seed=42)

        values = [
            sampler.suggest_categorical(f"p{i}", ["only"], [1.0]) for i in range(10)
        ]

        # All values should be the single choice
        assert all(v == "only" for v in values)

    def test_suggest_categorical_different_types(self):
        """Test categorical sampling with different value types."""
        sampler = RandomSampler(seed=42)

        # Test with integers
        int_values = [
            sampler.suggest_categorical(f"p{i}", [1, 2, 3], [1.0, 1.0, 1.0])
            for i in range(50)
        ]
        assert all(v in [1, 2, 3] for v in int_values)

        # Test with mixed types
        mixed_values = [
            sampler.suggest_categorical(f"q{i}", [1, "a", None], [1.0, 1.0, 1.0])
            for i in range(50)
        ]
        assert all(v in [1, "a", None] for v in mixed_values)


class TestRandomSamplerIntegration:
    """Integration tests for RandomSampler."""

    def test_multiple_parameters_same_sampler(self):
        """Test sampling multiple different parameters with one sampler."""
        sampler = RandomSampler(seed=42)

        # Sample different types
        int_val = sampler.suggest_int("layers", 1, 10, True, True, "uniform")
        float_val = sampler.suggest_float("lr", 1e-5, 1e-1, True, True, "log")
        cat_val = sampler.suggest_categorical("optimizer", ["adam", "sgd"], [1.0, 1.0])

        # Check record
        assert len(sampler.record) == 3
        assert sampler.record["layers"] == int_val
        assert sampler.record["lr"] == float_val
        assert sampler.record["optimizer"] == cat_val

    def test_overwriting_parameter(self):
        """Test that sampling same parameter name overwrites previous value."""
        sampler = RandomSampler(seed=42)

        value1 = sampler.suggest_int("param", 1, 10, True, True, "uniform")
        value2 = sampler.suggest_int("param", 1, 10, True, True, "uniform")

        # Record should only have the latest value
        assert len(sampler.record) == 1
        assert sampler.record["param"] == value2
        assert value1 != value2  # Should be different due to RNG state
