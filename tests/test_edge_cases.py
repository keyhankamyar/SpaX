"""Tests for edge cases and error conditions not covered elsewhere."""

import pytest

from spax import Categorical, Config, Float, Int
from spax.spaces import FloatSpace


class TestConfigEdgeCases:
    """Edge cases for Config validation."""

    def test_string_input_to_config(self):
        """Test that string input to Config raises proper error."""

        class SimpleConfig(Config):
            lr: float = Float(ge=0.001, le=0.1)

        # String should fail validation
        with pytest.raises((ValueError, TypeError)):
            SimpleConfig.model_validate("not a dict")


class TestDescriptorProtocol:
    """Test Space descriptor protocol outside of Config."""

    def test_get_from_class_returns_space(self):
        """Test that accessing Space from class returns the Space itself."""

        class TestClass:
            value: float = Float(ge=0.0, le=1.0)

        # Accessing from class should return the FloatSpace
        space = TestClass.value
        assert isinstance(space, FloatSpace)

    def test_get_from_instance_returns_value(self):
        """Test that accessing from instance returns the stored value."""

        class TestClass:
            value = Float(ge=0.0, le=1.0)

        obj = TestClass()
        obj.__dict__["value"] = 0.5

        # Accessing from instance returns value
        assert obj.value == 0.5

    def test_set_validates_value(self):
        """Test that setting value through descriptor validates."""

        class TestClass:
            value = Float(ge=0.0, le=1.0)

        TestClass.value.field_name = "value"  # Manually set field name

        obj = TestClass()
        obj.value = 0.5
        assert obj.value == 0.5

        # Invalid value should raise
        with pytest.raises(ValueError):
            obj.value = 2.0


class TestConditionTypeErrors:
    """Test type errors in condition classes."""

    def test_smaller_than_non_numeric(self):
        """Test SmallerThan with non-numeric threshold."""
        from spax.spaces.conditions import SmallerThan

        with pytest.raises(TypeError, match="numeric"):
            SmallerThan("not a number")  # type: ignore

    def test_smaller_than_non_bool_or_equals(self):
        """Test SmallerThan with non-bool or_equals."""
        from spax.spaces.conditions import SmallerThan

        with pytest.raises(TypeError, match="must be bool"):
            SmallerThan(10, or_equals="not bool")  # type: ignore

    def test_larger_than_non_numeric(self):
        """Test LargerThan with non-numeric threshold."""
        from spax.spaces.conditions import LargerThan

        with pytest.raises(TypeError, match="numeric"):
            LargerThan("not a number")  # type: ignore

    def test_larger_than_call_with_non_numeric(self):
        """Test calling LargerThan with non-numeric value."""
        from spax.spaces.conditions import LargerThan

        cond = LargerThan(10)

        with pytest.raises(TypeError, match="numeric"):
            cond("not a number")

    def test_smaller_than_call_with_non_numeric(self):
        """Test calling SmallerThan with non-numeric value."""
        from spax.spaces.conditions import SmallerThan

        cond = SmallerThan(10)

        with pytest.raises(TypeError, match="numeric"):
            cond("not a number")

    def test_is_instance_non_type(self):
        """Test IsInstance with non-type argument."""
        from spax.spaces.conditions import IsInstance

        with pytest.raises(TypeError, match="type"):
            IsInstance("not a type")  # type: ignore

    def test_is_instance_tuple_with_non_type(self):
        """Test IsInstance with tuple containing non-type."""
        from spax.spaces.conditions import IsInstance

        with pytest.raises(TypeError, match="must be types"):
            IsInstance((int, "not a type"))  # type: ignore

    def test_and_with_non_condition(self):
        """Test And with non-Condition object in list."""
        from spax.spaces.conditions import And, EqualsTo

        with pytest.raises(TypeError, match="Condition instances"):
            And([EqualsTo("x"), "not a condition"])  # type: ignore

    def test_or_with_non_condition(self):
        """Test Or with non-Condition object in list."""
        from spax.spaces.conditions import EqualsTo, Or

        with pytest.raises(TypeError, match="Condition instances"):
            Or([EqualsTo("x"), "not a condition"])  # type: ignore


class TestNumericSpaceEdgeCases:
    """Edge cases for numeric spaces."""

    def test_float_space_field_name_none_error(self):
        """Test that _check_bounds raises RuntimeError when field_name is None."""
        space = Float(ge=0.0, le=1.0)
        # Don't set field_name, leave it as None
        with pytest.raises(RuntimeError, match="field_name is None"):
            space.validate(0.5)

    def test_int_space_non_integer_bounds(self):
        """Test that Int with float bounds raises TypeError."""
        with pytest.raises(TypeError, match="must be an integer"):
            Int(ge=0.5, le=10)  # type: ignore

    def test_int_space_bool_bounds(self):
        """Test that Int with bool bounds raises TypeError."""
        with pytest.raises(TypeError, match="must be an integer"):
            Int(ge=True, le=False)  # type: ignore

    def test_float_validate_non_numeric_string(self):
        """Test Float.validate with non-numeric string."""
        space = Float(ge=0.0, le=1.0)
        space.field_name = "test"

        with pytest.raises(ValueError, match="Expected numeric"):
            space.validate("not a number")

    def test_int_validate_string(self):
        """Test Int.validate with string."""
        space = Int(ge=0, le=10)
        space.field_name = "test"

        with pytest.raises(ValueError, match="Expected int"):
            space.validate("5")


class TestCategoricalEdgeCases:
    """Edge cases for categorical spaces."""

    def test_categorical_with_numeric_choices(self):
        """Test categorical with numeric choices works correctly."""
        space = Categorical([1, 2, 3])
        space.field_name = "test"

        assert space.validate(1) == 1
        assert space.validate(2) == 2

        with pytest.raises(ValueError, match="not in allowed choices"):
            space.validate(4)

    def test_categorical_sample_distribution_single_choice(self):
        """Test categorical with single choice."""
        space = Categorical(["only_choice"])

        # Should always return the only choice
        for _ in range(10):
            assert space.sample() == "only_choice"


class TestConditionalEdgeCases:
    """Edge cases for conditional spaces."""

    def test_conditional_validate_without_config_context(self):
        """Test that calling validate (not validate_with_config) just returns value."""
        from spax.spaces import Conditional, EqualsTo, FieldCondition, Float

        space = Conditional(
            condition=FieldCondition("x", EqualsTo(1)),
            true=Float(ge=0.0, le=1.0),
            false=Float(ge=1.0, le=2.0),
        )
        space.field_name = "test"

        # validate() without config just returns the value
        # (actual validation happens in Config's validate_spaces)
        result = space.validate(0.5)
        assert result == 0.5
