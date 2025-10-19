"""Tests for numeric search spaces."""

import pytest

from spax.spaces import UNSET, Float, FloatSpace, Int, IntSpace


class TestFloatSpace:
    """Tests for FloatSpace."""

    def test_creation_with_valid_params(self):
        """Test creating a float space with valid parameters."""
        space = Float(ge=0.0, le=1.0)
        assert isinstance(space, FloatSpace)
        assert space.low == 0.0
        assert space.high == 1.0
        assert space.low_inclusive is True
        assert space.high_inclusive is True

    def test_sample_in_range(self):
        """Test that samples fall within the specified range."""
        space = Float(ge=0.0, le=10.0)
        for _ in range(100):
            value = space.sample()
            assert 0.0 <= value <= 10.0

    def test_log_distribution_sampling(self):
        """Test sampling with log distribution."""
        space = Float(ge=1e-5, le=1e-1, distribution="log")
        for _ in range(100):
            value = space.sample()
            assert 1e-5 <= value <= 1e-1

    def test_validate_valid_float(self):
        """Test validation accepts valid float values."""
        space = Float(ge=0.0, le=10.0)
        space.field_name = "test_field"
        assert space.validate(5.0) == 5.0
        assert space.validate(0.0) == 0.0
        assert space.validate(10.0) == 10.0

    def test_validate_coerces_int_to_float(self):
        """Test that integers are coerced to floats."""
        space = Float(ge=0.0, le=10.0)
        space.field_name = "test_field"
        result = space.validate(5)
        assert result == 5.0
        assert isinstance(result, float)

    def test_validate_out_of_range_both_inclusive(self):
        """Test validation rejects out-of-range values with both bounds inclusive."""
        space = Float(ge=0.0, le=10.0)
        space.field_name = "test_field"
        with pytest.raises(ValueError, match="must be >="):
            space.validate(-0.1)
        with pytest.raises(ValueError, match="must be <="):
            space.validate(10.1)

    def test_validate_low_inclusive_high_exclusive(self):
        """Test validation with low inclusive, high exclusive."""
        space = Float(ge=0.0, lt=10.0)
        space.field_name = "test_field"
        assert space.validate(0.0) == 0.0  # Low inclusive
        assert space.validate(9.9) == 9.9
        with pytest.raises(ValueError, match="must be <"):
            space.validate(10.0)  # High exclusive

    def test_validate_low_exclusive_high_inclusive(self):
        """Test validation with low exclusive, high inclusive."""
        space = Float(gt=0.0, le=10.0)
        space.field_name = "test_field"
        assert space.validate(10.0) == 10.0  # High inclusive
        assert space.validate(0.1) == 0.1
        with pytest.raises(ValueError, match="must be >"):
            space.validate(0.0)  # Low exclusive

    def test_validate_both_exclusive(self):
        """Test validation with both bounds exclusive."""
        space = Float(gt=0.0, lt=10.0)
        space.field_name = "test_field"
        assert space.validate(5.0) == 5.0
        with pytest.raises(ValueError, match="must be >"):
            space.validate(0.0)
        with pytest.raises(ValueError, match="must be <"):
            space.validate(10.0)

    def test_invalid_low_high_raises_error(self):
        """Test that low >= high raises an assertion error."""
        with pytest.raises(AssertionError, match="must be less than"):
            Float(ge=10.0, le=5.0)
        with pytest.raises(AssertionError, match="must be less than"):
            Float(ge=5.0, le=5.0)

    def test_requires_exactly_one_lower_bound(self):
        """Test that exactly one of gt/ge must be specified."""
        with pytest.raises(ValueError, match="Exactly one of 'gt'.*or 'ge'"):
            Float(lt=10.0)  # Missing lower bound
        with pytest.raises(ValueError, match="Exactly one of 'gt'.*or 'ge'"):
            Float(gt=0.0, ge=0.0, lt=10.0)  # Both lower bounds

    def test_requires_exactly_one_upper_bound(self):
        """Test that exactly one of lt/le must be specified."""
        with pytest.raises(ValueError, match="Exactly one of 'lt'.*or 'le'"):
            Float(ge=0.0)  # Missing upper bound
        with pytest.raises(ValueError, match="Exactly one of 'lt'.*or 'le'"):
            Float(ge=0.0, lt=10.0, le=10.0)  # Both upper bounds

    def test_invalid_distribution_string_raises_error(self):
        """Test that invalid distribution string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown distribution"):
            Float(ge=0.0, le=1.0, distribution="invalid")  # type: ignore

    def test_default_value(self):
        """Test float space with default value."""
        space = Float(ge=0.0, le=10.0, default=5.0)
        assert space.default == 5.0

    def test_default_value_validated(self):
        """Test that default value is validated on creation."""
        with pytest.raises(ValueError, match="Invalid default value"):
            Float(ge=0.0, le=10.0, default=15.0)

    def test_description(self):
        """Test float space with description."""
        space = Float(ge=0.0, le=1.0, description="Learning rate")
        assert space.description == "Learning rate"

    def test_unset_default(self):
        """Test that default is UNSET when not provided."""
        space = Float(ge=0.0, le=1.0)
        assert space.default is UNSET


class TestIntSpace:
    """Tests for IntSpace."""

    def test_creation_with_valid_params(self):
        """Test creating an int space with valid parameters."""
        space = Int(ge=0, le=10)
        assert isinstance(space, IntSpace)
        assert space.low == 0
        assert space.high == 10
        assert space.low_inclusive is True
        assert space.high_inclusive is True

    def test_sample_in_range(self):
        """Test that samples fall within the specified range."""
        space = Int(ge=0, le=10)
        for _ in range(100):
            value = space.sample()
            assert 0 <= value <= 10
            assert isinstance(value, int)

    def test_sample_returns_int(self):
        """Test that sample always returns an integer."""
        space = Int(ge=1, le=100, distribution="log")
        for _ in range(50):
            value = space.sample()
            assert isinstance(value, int)
            assert value == int(value)

    def test_validate_valid_int(self):
        """Test validation accepts valid integer values."""
        space = Int(ge=0, le=10)
        space.field_name = "test_field"
        assert space.validate(5) == 5
        assert space.validate(0) == 0
        assert space.validate(10) == 10

    def test_validate_float_with_integer_value(self):
        """Test that floats representing integers are accepted."""
        space = Int(ge=0, le=10)
        space.field_name = "test_field"
        result = space.validate(5.0)
        assert result == 5
        assert isinstance(result, int)

    def test_validate_rejects_non_integer_float(self):
        """Test that non-integer floats are rejected."""
        space = Int(ge=0, le=10)
        space.field_name = "test_field"
        with pytest.raises(ValueError, match="Expected integer value"):
            space.validate(5.5)

    def test_validate_bounds_checking(self):
        """Test that bounds are checked correctly."""
        space = Int(ge=0, le=10)
        space.field_name = "test_field"
        with pytest.raises(ValueError, match="must be >="):
            space.validate(-1)
        with pytest.raises(ValueError, match="must be <="):
            space.validate(11)

    def test_requires_integer_bounds(self):
        """Test that low and high must be integers."""
        with pytest.raises(TypeError, match="must be an integer"):
            Int(ge=0.5, le=10)  # type: ignore
        with pytest.raises(TypeError, match="must be an integer"):
            Int(ge=0, le=10.5)  # type: ignore

    def test_invalid_low_high_raises_error(self):
        """Test that low >= high raises an assertion error."""
        with pytest.raises(AssertionError, match="must be less than"):
            Int(ge=10, le=5)
        with pytest.raises(AssertionError, match="must be less than"):
            Int(ge=5, le=5)

    def test_default_value(self):
        """Test int space with default value."""
        space = Int(ge=0, le=10, default=5)
        assert space.default == 5

    def test_default_value_validated(self):
        """Test that default value is validated on creation."""
        with pytest.raises(ValueError, match="Invalid default value"):
            Int(ge=0, le=10, default=15)

    def test_default_must_be_integer(self):
        """Test that default must be an integer."""
        with pytest.raises(TypeError, match="default must be an integer"):
            Int(ge=0, le=10, default=5.5)  # type: ignore

    def test_description(self):
        """Test int space with description."""
        space = Int(ge=1, le=10, description="Number of layers")
        assert space.description == "Number of layers"
