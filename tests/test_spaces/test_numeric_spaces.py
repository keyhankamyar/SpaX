"""Tests for numeric spaces."""

import pytest

from spax.distributions import UNIFORM
from spax.spaces import UNSET, Float, FloatSpace, Int, IntSpace


class TestFloatSpace:
    """Test FloatSpace."""

    def test_basic_float_validation(self):
        """Test basic float validation."""
        space = FloatSpace(ge=0.0, le=10.0)
        space.field_name = "test_field"  # Set field_name for validation

        assert space.validate(5.0) == 5.0
        assert space.validate(0.0) == 0.0
        assert space.validate(10.0) == 10.0

    def test_float_accepts_int(self):
        """Test that FloatSpace accepts integer values."""
        space = FloatSpace(ge=0.0, le=10.0)
        space.field_name = "test_field"

        assert space.validate(5) == 5.0
        assert isinstance(space.validate(5), float)

    def test_exclusive_lower_bound(self):
        """Test exclusive lower bound (gt)."""
        space = FloatSpace(gt=0.0, le=10.0)
        space.field_name = "test_field"

        assert space.validate(0.1) == 0.1
        assert space.validate(5.0) == 5.0

        with pytest.raises(ValueError, match="must be >"):
            space.validate(0.0)

        with pytest.raises(ValueError, match="must be >"):
            space.validate(-1.0)

    def test_inclusive_lower_bound(self):
        """Test inclusive lower bound (ge)."""
        space = FloatSpace(ge=0.0, le=10.0)
        space.field_name = "test_field"

        assert space.validate(0.0) == 0.0
        assert space.validate(5.0) == 5.0

        with pytest.raises(ValueError, match="must be >="):
            space.validate(-0.1)

    def test_exclusive_upper_bound(self):
        """Test exclusive upper bound (lt)."""
        space = FloatSpace(ge=0.0, lt=10.0)
        space.field_name = "test_field"

        assert space.validate(9.9) == 9.9
        assert space.validate(5.0) == 5.0

        with pytest.raises(ValueError, match="must be <"):
            space.validate(10.0)

        with pytest.raises(ValueError, match="must be <"):
            space.validate(11.0)

    def test_inclusive_upper_bound(self):
        """Test inclusive upper bound (le)."""
        space = FloatSpace(ge=0.0, le=10.0)
        space.field_name = "test_field"

        assert space.validate(10.0) == 10.0
        assert space.validate(5.0) == 5.0

        with pytest.raises(ValueError, match="must be <="):
            space.validate(10.1)

    def test_all_bound_combinations(self):
        """Test all combinations of inclusive/exclusive bounds."""
        # gt + lt
        space = FloatSpace(gt=0.0, lt=10.0)
        space.field_name = "test_field"
        assert space.validate(5.0) == 5.0
        with pytest.raises(ValueError):
            space.validate(0.0)
        with pytest.raises(ValueError):
            space.validate(10.0)

        # gt + le
        space = FloatSpace(gt=0.0, le=10.0)
        space.field_name = "test_field"
        assert space.validate(5.0) == 5.0
        with pytest.raises(ValueError):
            space.validate(0.0)
        assert space.validate(10.0) == 10.0

        # ge + lt
        space = FloatSpace(ge=0.0, lt=10.0)
        space.field_name = "test_field"
        assert space.validate(5.0) == 5.0
        assert space.validate(0.0) == 0.0
        with pytest.raises(ValueError):
            space.validate(10.0)

        # ge + le
        space = FloatSpace(ge=0.0, le=10.0)
        space.field_name = "test_field"
        assert space.validate(5.0) == 5.0
        assert space.validate(0.0) == 0.0
        assert space.validate(10.0) == 10.0

    def test_negative_ranges(self):
        """Test ranges with negative values."""
        space = FloatSpace(ge=-10.0, le=-1.0)
        space.field_name = "test_field"

        assert space.validate(-5.0) == -5.0
        assert space.validate(-10.0) == -10.0
        assert space.validate(-1.0) == -1.0

        with pytest.raises(ValueError):
            space.validate(0.0)

    def test_very_small_range(self):
        """Test very small range."""
        space = FloatSpace(ge=0.0, le=0.001)
        space.field_name = "test_field"

        assert space.validate(0.0) == 0.0
        assert space.validate(0.0005) == 0.0005
        assert space.validate(0.001) == 0.001

    def test_large_range(self):
        """Test large range."""
        space = FloatSpace(ge=0.0, le=1e10)
        space.field_name = "test_field"

        assert space.validate(1e5) == 1e5
        assert space.validate(1e10) == 1e10

    def test_uniform_distribution(self):
        """Test uniform distribution sampling."""
        space = FloatSpace(ge=0.0, le=10.0, distribution="uniform")

        samples = [space.sample() for _ in range(100)]

        # All samples in range
        assert all(0.0 <= s <= 10.0 for s in samples)

        # Check it's actually sampling (not always same value)
        assert len(set(samples)) > 10

    def test_log_distribution(self):
        """Test log distribution sampling."""
        space = FloatSpace(ge=1.0, le=1000.0, distribution="log")

        samples = [space.sample() for _ in range(100)]

        # All samples in range
        assert all(1.0 <= s <= 1000.0 for s in samples)

        # Log distribution should favor smaller values
        below_100 = sum(1 for s in samples if s < 100)
        above_100 = sum(1 for s in samples if s >= 100)
        assert below_100 > above_100

    def test_custom_distribution_object(self):
        """Test custom distribution object."""
        space = FloatSpace(ge=0.0, le=10.0, distribution=UNIFORM)

        samples = [space.sample() for _ in range(50)]
        assert all(0.0 <= s <= 10.0 for s in samples)

    def test_default_value(self):
        """Test default value handling."""
        space = FloatSpace(ge=0.0, le=10.0, default=5.0)

        assert space.default == 5.0

    def test_default_value_validation(self):
        """Test that default value is validated."""
        # Valid default
        space = FloatSpace(ge=0.0, le=10.0, default=5.0)
        assert space.default == 5.0

        # Invalid default
        with pytest.raises(ValueError, match="Invalid default"):
            FloatSpace(ge=0.0, le=10.0, default=15.0)

    def test_no_default(self):
        """Test space without default."""
        space = FloatSpace(ge=0.0, le=10.0)
        assert space.default is UNSET

    def test_description(self):
        """Test description field."""
        space = FloatSpace(ge=0.0, le=10.0, description="Test space")
        assert space.description == "Test space"

    def test_invalid_value_type(self):
        """Test error with non-numeric value."""
        space = FloatSpace(ge=0.0, le=10.0)
        space.field_name = "test_field"

        with pytest.raises(ValueError, match="Expected numeric value"):
            space.validate("not a number")

    def test_missing_bounds_error(self):
        """Test error when bounds are not properly specified."""
        # No lower bound
        with pytest.raises(ValueError, match="Exactly one of 'gt'"):
            FloatSpace(lt=10.0)

        # No upper bound
        with pytest.raises(ValueError, match="Exactly one of 'lt'"):
            FloatSpace(ge=0.0)

        # Both gt and ge
        with pytest.raises(ValueError, match="Exactly one of 'gt'"):
            FloatSpace(gt=0.0, ge=0.0, le=10.0)

        # Both lt and le
        with pytest.raises(ValueError, match="Exactly one of 'lt'"):
            FloatSpace(ge=0.0, lt=10.0, le=10.0)

    def test_invalid_range_error(self):
        """Test error when lower bound >= upper bound."""
        with pytest.raises(AssertionError, match="must be less than"):
            FloatSpace(ge=10.0, le=5.0)

        with pytest.raises(AssertionError, match="must be less than"):
            FloatSpace(ge=10.0, le=10.0)

    def test_invalid_distribution_string_error(self):
        """Test error with invalid distribution string."""
        with pytest.raises(ValueError, match="Unknown distribution"):
            FloatSpace(ge=0.0, le=10.0, distribution="invalid")

    def test_invalid_distribution_type_error(self):
        """Test error with invalid distribution type."""
        with pytest.raises(ValueError, match="NumberDistribution or string"):
            FloatSpace(ge=0.0, le=10.0, distribution=123)

    def test_repr(self):
        """Test string representation."""
        space = FloatSpace(ge=0.0, le=10.0, distribution="uniform")
        repr_str = repr(space)

        assert "FloatSpace" in repr_str
        assert "ge=0.0" in repr_str
        assert "le=10.0" in repr_str
        assert "distribution" in repr_str

    def test_repr_with_default(self):
        """Test repr with default value."""
        space = FloatSpace(ge=0.0, le=10.0, default=5.0)
        repr_str = repr(space)

        assert "default=5.0" in repr_str

    def test_descriptor_protocol(self):
        """Test that FloatSpace works as a descriptor."""

        class TestConfig:
            value = FloatSpace(ge=0.0, le=10.0)

        # Access from class returns descriptor
        assert isinstance(TestConfig.value, FloatSpace)

        # Access from instance returns value
        config = TestConfig()
        config.value = 5.0
        assert config.value == 5.0

    def test_field_name_propagation(self):
        """Test that field_name is set via __set_name__."""

        class TestConfig:
            my_value = FloatSpace(ge=0.0, le=10.0)

        assert TestConfig.my_value.field_name == "my_value"

    def test_validation_uses_field_name_in_error(self):
        """Test that validation errors include field name."""
        space = FloatSpace(ge=0.0, le=10.0)
        space.field_name = "test_field"

        with pytest.raises(ValueError, match="test_field"):
            space.validate(15.0)


class TestIntSpace:
    """Test IntSpace."""

    def test_basic_int_validation(self):
        """Test basic integer validation."""
        space = IntSpace(ge=0, le=10)
        space.field_name = "test_field"

        assert space.validate(5) == 5
        assert space.validate(0) == 0
        assert space.validate(10) == 10

    def test_int_accepts_float_integers(self):
        """Test that IntSpace accepts floats representing integers."""
        space = IntSpace(ge=0, le=10)
        space.field_name = "test_field"

        assert space.validate(5.0) == 5
        assert isinstance(space.validate(5.0), int)

    def test_int_rejects_non_integer_floats(self):
        """Test that IntSpace rejects floats with fractional parts."""
        space = IntSpace(ge=0, le=10)
        space.field_name = "test_field"

        with pytest.raises(ValueError, match="Expected integer value"):
            space.validate(5.5)

    def test_bounds_must_be_integers(self):
        """Test that bounds must be integers at initialization."""
        # Valid integer bounds
        space = IntSpace(ge=0, le=10)
        assert space.low == 0
        assert space.high == 10

        # Invalid float bounds
        with pytest.raises(TypeError, match="must be an integer"):
            IntSpace(ge=0.5, le=10)

        with pytest.raises(TypeError, match="must be an integer"):
            IntSpace(ge=0, le=10.5)

    def test_default_must_be_integer(self):
        """Test that default must be an integer."""
        # Valid integer default
        space = IntSpace(ge=0, le=10, default=5)
        assert space.default == 5

        # Invalid float default
        with pytest.raises(TypeError, match="must be an integer"):
            IntSpace(ge=0, le=10, default=5.5)

    def test_exclusive_bounds(self):
        """Test exclusive bounds for integers."""
        # gt excludes the lower value
        space = IntSpace(gt=0, le=10)
        space.field_name = "test_field"
        assert space.validate(1) == 1
        with pytest.raises(ValueError, match="must be >"):
            space.validate(0)

        # lt excludes the upper value
        space = IntSpace(ge=0, lt=10)
        space.field_name = "test_field"
        assert space.validate(9) == 9
        with pytest.raises(ValueError, match="must be <"):
            space.validate(10)

    def test_sampling_returns_integers(self):
        """Test that sampling returns proper integers."""
        space = IntSpace(ge=0, le=100)

        samples = [space.sample() for _ in range(50)]

        # All are integers
        assert all(isinstance(s, int) for s in samples)

        # All in range
        assert all(0 <= s <= 100 for s in samples)

    def test_sampling_with_exclusive_bounds(self):
        """Test sampling with exclusive bounds."""
        space = IntSpace(gt=0, lt=10)

        samples = [space.sample() for _ in range(100)]

        # All in valid range (1-9)
        assert all(1 <= s <= 9 for s in samples)

        # Should not include boundaries
        assert 0 not in samples
        assert 10 not in samples

    def test_small_int_range(self):
        """Test small integer range."""
        space = IntSpace(ge=5, le=7)

        samples = [space.sample() for _ in range(50)]

        # All in range
        assert all(s in [5, 6, 7] for s in samples)

        # All values should appear
        assert 5 in samples
        assert 6 in samples
        assert 7 in samples

    def test_negative_int_range(self):
        """Test negative integer range."""
        space = IntSpace(ge=-10, le=-1)
        space.field_name = "test_field"

        assert space.validate(-5) == -5
        assert space.validate(-10) == -10
        assert space.validate(-1) == -1

        with pytest.raises(ValueError):
            space.validate(0)

    def test_log_distribution_with_integers(self):
        """Test log distribution with integer space."""
        space = IntSpace(ge=1, le=1000, distribution="log")

        samples = [space.sample() for _ in range(100)]

        # All are integers in range
        assert all(isinstance(s, int) for s in samples)
        assert all(1 <= s <= 1000 for s in samples)

        # Log distribution bias
        below_100 = sum(1 for s in samples if s < 100)
        above_100 = sum(1 for s in samples if s >= 100)
        assert below_100 > above_100

    def test_repr(self):
        """Test string representation."""
        space = IntSpace(ge=0, le=10)
        repr_str = repr(space)

        assert "IntSpace" in repr_str
        assert "ge=0" in repr_str
        assert "le=10" in repr_str


class TestFloatFactory:
    """Test Float() factory function."""

    def test_float_factory_creates_space(self):
        """Test that Float() creates a FloatSpace."""
        space = Float(ge=0.0, le=10.0)
        assert isinstance(space, FloatSpace)

    def test_float_factory_all_parameters(self):
        """Test Float() with all parameters."""
        space = Float(
            ge=0.0, le=10.0, distribution="log", default=5.0, description="Test"
        )

        assert isinstance(space, FloatSpace)
        assert space.low == 0.0
        assert space.high == 10.0
        assert space.default == 5.0
        assert space.description == "Test"


class TestIntFactory:
    """Test Int() factory function."""

    def test_int_factory_creates_space(self):
        """Test that Int() creates an IntSpace."""
        space = Int(ge=0, le=10)
        assert isinstance(space, IntSpace)

    def test_int_factory_all_parameters(self):
        """Test Int() with all parameters."""
        space = Int(ge=0, le=10, distribution="uniform", default=5, description="Test")

        assert isinstance(space, IntSpace)
        assert space.low == 0
        assert space.high == 10
        assert space.default == 5
        assert space.description == "Test"


class TestNumericSpaceEdgeCases:
    """Test edge cases for numeric spaces."""

    def test_multiple_samples_different(self):
        """Test that multiple samples produce different values."""
        space = FloatSpace(ge=0.0, le=100.0)

        samples = [space.sample() for _ in range(100)]

        # Should have many unique values
        assert len(set(samples)) > 50
