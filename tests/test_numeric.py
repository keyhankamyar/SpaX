"""Tests for numeric search spaces."""

import pytest

from spax.spaces import Float, FloatSpace, Int, IntSpace


class TestFloatSpace:
    """Tests for FloatSpace."""

    def test_creation_with_valid_params(self):
        """Test creating a float space with valid parameters."""
        space = Float(0.0, 1.0)
        assert isinstance(space, FloatSpace)
        assert space.low == 0.0
        assert space.high == 1.0
        assert space.bounds == "both"

    def test_sample_in_range(self):
        """Test that samples fall within the specified range."""
        space = Float(0.0, 10.0)
        for _ in range(100):
            value = space.sample()
            assert 0.0 <= value <= 10.0

    def test_log_distribution_sampling(self):
        """Test sampling with log distribution."""
        space = Float(1e-5, 1e-1, distribution="log")
        for _ in range(100):
            value = space.sample()
            assert 1e-5 <= value <= 1e-1

    def test_validate_valid_float(self):
        """Test validation accepts valid float values."""
        space = Float(0.0, 10.0)
        space.field_name = "test_field"

        assert space.validate(5.0) == 5.0
        assert space.validate(0.0) == 0.0
        assert space.validate(10.0) == 10.0

    def test_validate_coerces_int_to_float(self):
        """Test that integers are coerced to floats."""
        space = Float(0.0, 10.0)
        space.field_name = "test_field"

        result = space.validate(5)
        assert result == 5.0
        assert isinstance(result, float)

    def test_validate_out_of_range_both_bounds(self):
        """Test validation rejects out-of-range values with both bounds."""
        space = Float(0.0, 10.0, bounds="both")
        space.field_name = "test_field"

        with pytest.raises(ValueError, match="must be in"):
            space.validate(-0.1)

        with pytest.raises(ValueError, match="must be in"):
            space.validate(10.1)

    def test_validate_low_bound_only(self):
        """Test validation with low bound only (exclusive high)."""
        space = Float(0.0, 10.0, bounds="low")
        space.field_name = "test_field"

        assert space.validate(0.0) == 0.0  # Low inclusive
        assert space.validate(9.9) == 9.9

        with pytest.raises(ValueError, match="must be in"):
            space.validate(10.0)  # High exclusive

    def test_validate_high_bound_only(self):
        """Test validation with high bound only (exclusive low)."""
        space = Float(0.0, 10.0, bounds="high")
        space.field_name = "test_field"

        assert space.validate(10.0) == 10.0  # High inclusive
        assert space.validate(0.1) == 0.1

        with pytest.raises(ValueError, match="must be in"):
            space.validate(0.0)  # Low exclusive

    def test_validate_no_bounds(self):
        """Test validation with no bounds (both exclusive)."""
        space = Float(0.0, 10.0, bounds="none")
        space.field_name = "test_field"

        assert space.validate(5.0) == 5.0

        with pytest.raises(ValueError, match="must be in"):
            space.validate(0.0)

        with pytest.raises(ValueError, match="must be in"):
            space.validate(10.0)

    def test_invalid_low_high_raises_error(self):
        """Test that low >= high raises an assertion error."""
        with pytest.raises(AssertionError, match="must be less than"):
            Float(10.0, 5.0)

        with pytest.raises(AssertionError, match="must be less than"):
            Float(5.0, 5.0)

    def test_invalid_distribution_string_raises_error(self):
        """Test that invalid distribution string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown distribution"):
            Float(0.0, 1.0, distribution="invalid")  # type: ignore


class TestIntSpace:
    """Tests for IntSpace."""

    def test_creation_with_valid_params(self):
        """Test creating an int space with valid parameters."""
        space = Int(0, 10)
        assert isinstance(space, IntSpace)
        assert space.low == 0
        assert space.high == 10
        assert space.bounds == "both"

    def test_sample_in_range(self):
        """Test that samples fall within the specified range."""
        space = Int(0, 10)
        for _ in range(100):
            value = space.sample()
            assert 0 <= value <= 10
            assert isinstance(value, int)

    def test_sample_returns_int(self):
        """Test that sample always returns an integer."""
        space = Int(1, 100, distribution="log")
        for _ in range(50):
            value = space.sample()
            assert isinstance(value, int)
            assert value == int(value)

    def test_validate_valid_int(self):
        """Test validation accepts valid integer values."""
        space = Int(0, 10)
        space.field_name = "test_field"

        assert space.validate(5) == 5
        assert space.validate(0) == 0
        assert space.validate(10) == 10

    def test_validate_float_with_integer_value(self):
        """Test that floats representing integers are accepted."""
        space = Int(0, 10)
        space.field_name = "test_field"

        result = space.validate(5.0)
        assert result == 5
        assert isinstance(result, int)

    def test_validate_rejects_non_integer_float(self):
        """Test that non-integer floats are rejected."""
        space = Int(0, 10)
        space.field_name = "test_field"

        with pytest.raises(ValueError, match="Expected integer value"):
            space.validate(5.5)

    def test_validate_bounds_checking(self):
        """Test that bounds are checked correctly."""
        space = Int(0, 10, bounds="both")
        space.field_name = "test_field"

        with pytest.raises(ValueError, match="must be in"):
            space.validate(-1)

        with pytest.raises(ValueError, match="must be in"):
            space.validate(11)

    def test_requires_integer_bounds(self):
        """Test that low and high must be integers."""
        with pytest.raises(AssertionError, match="must be int"):
            Int(0.5, 10)  # type: ignore

        with pytest.raises(AssertionError, match="must be int"):
            Int(0, 10.5)  # type: ignore

    def test_invalid_low_high_raises_error(self):
        """Test that low >= high raises an assertion error."""
        with pytest.raises(AssertionError, match="must be less than"):
            Int(10, 5)

        with pytest.raises(AssertionError, match="must be less than"):
            Int(5, 5)
