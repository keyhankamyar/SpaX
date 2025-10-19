"""Tests for distribution classes."""

import pytest

from spax.distributions import (
    LOG,
    UNIFORM,
    CategoricalDistribution,
    LogDistribution,
    UniformDistribution,
)


class TestUniformDistribution:
    """Tests for UniformDistribution."""

    def test_sample_in_range(self):
        """Test that samples fall within the specified range."""
        dist = UniformDistribution()
        for _ in range(100):
            value = dist.sample(0.0, 10.0)
            assert 0.0 <= value <= 10.0

    def test_invalid_range_raises_error(self):
        """Test that invalid ranges raise ValueError."""
        dist = UniformDistribution()
        with pytest.raises(ValueError, match="must be less than"):
            dist.sample(10.0, 5.0)

        with pytest.raises(ValueError, match="must be less than"):
            dist.sample(5.0, 5.0)


class TestLogDistribution:
    """Tests for LogDistribution."""

    def test_sample_in_range(self):
        """Test that samples fall within the specified range."""
        dist = LogDistribution()
        for _ in range(100):
            value = dist.sample(1e-5, 1e-1)
            assert 1e-5 <= value <= 1e-1

    def test_negative_low_raises_error(self):
        """Test that negative low bound raises ValueError."""
        dist = LogDistribution()
        with pytest.raises(ValueError, match="requires low > 0"):
            dist.sample(-1.0, 10.0)

    def test_zero_low_raises_error(self):
        """Test that zero low bound raises ValueError."""
        dist = LogDistribution()
        with pytest.raises(ValueError, match="requires low > 0"):
            dist.sample(0.0, 10.0)

    def test_invalid_range_raises_error(self):
        """Test that invalid ranges raise ValueError."""
        dist = LogDistribution()
        with pytest.raises(ValueError, match="must be less than"):
            dist.sample(10.0, 5.0)


class TestCategoricalDistribution:
    """Tests for CategoricalDistribution."""

    def test_sample_returns_valid_choice(self):
        """Test that samples are always from the choices list."""
        dist = CategoricalDistribution()
        choices = ["a", "b", "c"]
        weights = [1.0, 1.0, 1.0]

        for _ in range(50):
            value = dist.sample(choices, weights)
            assert value in choices

    def test_weighted_sampling(self):
        """Test that weights affect sampling probability."""
        dist = CategoricalDistribution()
        choices = ["a", "b"]
        weights = [0.99, 0.01]  # Heavily favor "a"

        samples = [dist.sample(choices, weights) for _ in range(1000)]
        a_count = samples.count("a")

        # With 99% weight on "a", expect roughly 990/1000 (allow some variance)
        assert a_count > 900

    def test_empty_choices_raises_error(self):
        """Test that empty choices list raises ValueError."""
        dist = CategoricalDistribution()
        with pytest.raises(ValueError, match="empty choices"):
            dist.sample([], [])

    def test_mismatched_lengths_raises_error(self):
        """Test that mismatched lengths raise ValueError."""
        dist = CategoricalDistribution()
        with pytest.raises(ValueError, match="must match"):
            dist.sample(["a", "b"], [1.0])

    def test_negative_weights_raise_error(self):
        """Test that negative weights raise ValueError."""
        dist = CategoricalDistribution()
        with pytest.raises(ValueError, match="non-negative"):
            dist.sample(["a", "b"], [1.0, -1.0])

    def test_all_zero_weights_raise_error(self):
        """Test that all-zero weights raise ValueError."""
        dist = CategoricalDistribution()
        with pytest.raises(ValueError, match="cannot all be zero"):
            dist.sample(["a", "b"], [0.0, 0.0])


class TestSingletonInstances:
    """Tests for singleton distribution instances."""

    def test_uniform_singleton(self):
        """Test that UNIFORM is a UniformDistribution instance."""
        assert isinstance(UNIFORM, UniformDistribution)

    def test_log_singleton(self):
        """Test that LOG is a LogDistribution instance."""
        assert isinstance(LOG, LogDistribution)
