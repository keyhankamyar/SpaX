"""Tests for distribution sampling functionality."""

import pytest

from spax.distributions import (
    LOG,
    UNIFORM,
    CategoricalDistribution,
    LogDistribution,
    UniformDistribution,
)


class TestUniformDistribution:
    """Test uniform distribution sampling."""

    def test_sample_basic(self):
        """Test basic uniform sampling within range."""
        dist = UniformDistribution()
        for _ in range(100):
            value = dist.sample(0.0, 1.0)
            assert 0.0 <= value <= 1.0

    def test_sample_different_ranges(self):
        """Test sampling from different ranges."""
        dist = UniformDistribution()

        # Negative to positive
        value = dist.sample(-10.0, 10.0)
        assert -10.0 <= value <= 10.0

        # Large range
        value = dist.sample(0.0, 1000.0)
        assert 0.0 <= value <= 1000.0

        # Small range
        value = dist.sample(0.999, 1.001)
        assert 0.999 <= value <= 1.001

    def test_invalid_range_equal(self):
        """Test error when low equals high."""
        dist = UniformDistribution()
        with pytest.raises(ValueError, match="must be less than"):
            dist.sample(5.0, 5.0)

    def test_invalid_range_inverted(self):
        """Test error when low > high."""
        dist = UniformDistribution()
        with pytest.raises(ValueError, match="must be less than"):
            dist.sample(10.0, 5.0)


class TestLogDistribution:
    """Test log-uniform distribution sampling."""

    def test_sample_basic(self):
        """Test basic log sampling within range."""
        dist = LogDistribution()
        for _ in range(100):
            value = dist.sample(1.0, 100.0)
            assert 1.0 <= value <= 100.0

    def test_sample_wide_range(self):
        """Test sampling from wide exponential range."""
        dist = LogDistribution()
        value = dist.sample(1e-6, 1e6)
        assert 1e-6 <= value <= 1e6

    def test_log_distribution_bias(self):
        """Test that log distribution favors smaller values."""
        dist = LogDistribution()
        samples = [dist.sample(1.0, 1000.0) for _ in range(1000)]

        # In log space, more samples should be < 100 than > 100
        below_100 = sum(1 for s in samples if s < 100)
        above_100 = sum(1 for s in samples if s > 100)

        # Should have roughly 2/3 below 100 in log space
        assert below_100 > above_100

    def test_invalid_low_zero_or_negative(self):
        """Test error when low <= 0."""
        dist = LogDistribution()
        with pytest.raises(ValueError, match="requires low > 0"):
            dist.sample(0.0, 10.0)

        with pytest.raises(ValueError, match="requires low > 0"):
            dist.sample(-1.0, 10.0)

    def test_invalid_range_equal(self):
        """Test error when low equals high."""
        dist = LogDistribution()
        with pytest.raises(ValueError, match="must be less than"):
            dist.sample(5.0, 5.0)

    def test_invalid_range_inverted(self):
        """Test error when low > high."""
        dist = LogDistribution()
        with pytest.raises(ValueError, match="must be less than"):
            dist.sample(10.0, 5.0)


class TestCategoricalDistribution:
    """Test categorical distribution sampling."""

    def test_sample_equal_weights(self):
        """Test sampling with equal weights."""
        dist = CategoricalDistribution()
        choices = ["a", "b", "c"]
        weights = [1.0, 1.0, 1.0]

        # Sample many times and check all choices appear
        samples = [dist.sample(choices, weights) for _ in range(100)]
        assert set(samples) == set(choices)

    def test_sample_weighted(self):
        """Test sampling with unequal weights."""
        dist = CategoricalDistribution()
        choices = ["rare", "common"]
        weights = [1.0, 99.0]

        samples = [dist.sample(choices, weights) for _ in range(1000)]
        common_count = samples.count("common")

        # Should heavily favor 'common'
        assert common_count > 900

    def test_sample_single_choice(self):
        """Test sampling with single choice."""
        dist = CategoricalDistribution()
        choices = ["only"]
        weights = [1.0]

        for _ in range(10):
            assert dist.sample(choices, weights) == "only"

    def test_sample_various_types(self):
        """Test sampling with different value types."""
        dist = CategoricalDistribution()

        # Integers
        value = dist.sample([1, 2, 3], [1.0, 1.0, 1.0])
        assert value in [1, 2, 3]

        # Mixed types
        value = dist.sample([1, "two", 3.0, None], [1.0, 1.0, 1.0, 1.0])
        assert value in [1, "two", 3.0, None]

    def test_empty_choices_error(self):
        """Test error with empty choices."""
        dist = CategoricalDistribution()
        with pytest.raises(ValueError, match="empty choices"):
            dist.sample([], [])

    def test_mismatched_lengths_error(self):
        """Test error when choices and weights have different lengths."""
        dist = CategoricalDistribution()
        with pytest.raises(ValueError, match="must match"):
            dist.sample(["a", "b"], [1.0])

    def test_negative_weights_error(self):
        """Test error with negative weights."""
        dist = CategoricalDistribution()
        with pytest.raises(ValueError, match="non-negative"):
            dist.sample(["a", "b"], [1.0, -1.0])

    def test_all_zero_weights_error(self):
        """Test error when all weights are zero."""
        dist = CategoricalDistribution()
        with pytest.raises(ValueError, match="cannot all be zero"):
            dist.sample(["a", "b"], [0.0, 0.0])


class TestSingletonInstances:
    """Test that singleton instances work correctly."""

    def test_uniform_singleton(self):
        """Test UNIFORM singleton."""
        assert isinstance(UNIFORM, UniformDistribution)
        value = UNIFORM.sample(0.0, 1.0)
        assert 0.0 <= value <= 1.0

    def test_log_singleton(self):
        """Test LOG singleton."""
        assert isinstance(LOG, LogDistribution)
        value = LOG.sample(1.0, 10.0)
        assert 1.0 <= value <= 10.0
