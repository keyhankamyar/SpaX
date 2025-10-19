"""Tests for condition classes."""

import pytest

from spax.spaces.conditions import (
    And,
    EqualsTo,
    FieldCondition,
    In,
    IsInstance,
    Lambda,
    LargerThan,
    Not,
    NotEqualsTo,
    NotIn,
    Or,
    SmallerThan,
)


class TestEqualsTo:
    """Tests for EqualsTo condition."""

    def test_equals_true(self):
        """Test condition returns True for equal values."""
        cond = EqualsTo("relu")
        assert cond("relu") is True

    def test_equals_false(self):
        """Test condition returns False for unequal values."""
        cond = EqualsTo("relu")
        assert cond("gelu") is False

    def test_numeric_equality(self):
        """Test with numeric values."""
        cond = EqualsTo(42)
        assert cond(42) is True
        assert cond(43) is False


class TestNotEqualsTo:
    """Tests for NotEqualsTo condition."""

    def test_not_equals_true(self):
        """Test condition returns True for unequal values."""
        cond = NotEqualsTo("none")
        assert cond("relu") is True

    def test_not_equals_false(self):
        """Test condition returns False for equal values."""
        cond = NotEqualsTo("none")
        assert cond("none") is False


class TestIn:
    """Tests for In condition."""

    def test_in_list_true(self):
        """Test condition returns True when value is in list."""
        cond = In(["relu", "gelu", "silu"])
        assert cond("relu") is True
        assert cond("gelu") is True

    def test_in_list_false(self):
        """Test condition returns False when value is not in list."""
        cond = In(["relu", "gelu", "silu"])
        assert cond("tanh") is False

    def test_in_set(self):
        """Test with set instead of list."""
        cond = In({"a", "b", "c"})
        assert cond("a") is True
        assert cond("d") is False

    def test_in_tuple(self):
        """Test with tuple instead of list."""
        cond = In(("x", "y", "z"))
        assert cond("y") is True
        assert cond("w") is False


class TestNotIn:
    """Tests for NotIn condition."""

    def test_not_in_list_true(self):
        """Test condition returns True when value is not in list."""
        cond = NotIn(["deprecated1", "deprecated2"])
        assert cond("modern") is True

    def test_not_in_list_false(self):
        """Test condition returns False when value is in list."""
        cond = NotIn(["deprecated1", "deprecated2"])
        assert cond("deprecated1") is False


class TestSmallerThan:
    """Tests for SmallerThan condition."""

    def test_smaller_than_strict(self):
        """Test strict less than comparison."""
        cond = SmallerThan(10, or_equals=False)
        assert cond(5) is True
        assert cond(10) is False
        assert cond(15) is False

    def test_smaller_than_or_equals(self):
        """Test less than or equal comparison."""
        cond = SmallerThan(10, or_equals=True)
        assert cond(5) is True
        assert cond(10) is True
        assert cond(15) is False

    def test_with_floats(self):
        """Test with floating point values."""
        cond = SmallerThan(5.5, or_equals=True)
        assert cond(5.5) is True
        assert cond(5.4) is True
        assert cond(5.6) is False


class TestLargerThan:
    """Tests for LargerThan condition."""

    def test_larger_than_strict(self):
        """Test strict greater than comparison."""
        cond = LargerThan(10, or_equals=False)
        assert cond(15) is True
        assert cond(10) is False
        assert cond(5) is False

    def test_larger_than_or_equals(self):
        """Test greater than or equal comparison."""
        cond = LargerThan(10, or_equals=True)
        assert cond(15) is True
        assert cond(10) is True
        assert cond(5) is False

    def test_with_floats(self):
        """Test with floating point values."""
        cond = LargerThan(5.5, or_equals=True)
        assert cond(5.5) is True
        assert cond(5.6) is True
        assert cond(5.4) is False


class TestIsInstance:
    """Tests for IsInstance condition."""

    def test_is_instance_single_type(self):
        """Test with single type."""
        cond = IsInstance(int)
        assert cond(5) is True
        assert cond(5.0) is False
        assert cond("5") is False

    def test_is_instance_multiple_types(self):
        """Test with tuple of types."""
        cond = IsInstance((int, float))
        assert cond(5) is True
        assert cond(5.5) is True
        assert cond("5") is False

    def test_with_custom_class(self):
        """Test with custom class."""

        class MyClass:
            pass

        cond = IsInstance(MyClass)
        obj = MyClass()
        assert cond(obj) is True
        assert cond("not MyClass") is False


class TestAnd:
    """Tests for And condition."""

    def test_all_true(self):
        """Test returns True when all conditions are True."""
        cond = And([LargerThan(0), SmallerThan(100)])
        assert cond(50) is True

    def test_one_false(self):
        """Test returns False when any condition is False."""
        cond = And([LargerThan(0), SmallerThan(100)])
        assert cond(150) is False
        assert cond(-5) is False

    def test_short_circuit(self):
        """Test that And short-circuits on first False."""
        call_count = []

        def track_call(val):
            call_count.append(val)
            return False

        cond = And(
            [
                Lambda(track_call),
                Lambda(lambda x: call_count.append("should not be called") or True),
            ]
        )

        cond("test")
        assert len(call_count) == 1  # Second condition should not be called

    def test_empty_conditions_raises_error(self):
        """Test that empty conditions list raises ValueError."""
        with pytest.raises(ValueError, match="at least one condition"):
            And([])

    def test_non_condition_raises_error(self):
        """Test that non-Condition objects raise TypeError."""
        with pytest.raises(TypeError, match="must be Condition instances"):
            And([EqualsTo("x"), "not a condition"])  # type: ignore


class TestOr:
    """Tests for Or condition."""

    def test_one_true(self):
        """Test returns True when any condition is True."""
        cond = Or([EqualsTo("relu"), EqualsTo("gelu")])
        assert cond("relu") is True
        assert cond("gelu") is True

    def test_all_false(self):
        """Test returns False when all conditions are False."""
        cond = Or([EqualsTo("relu"), EqualsTo("gelu")])
        assert cond("silu") is False

    def test_short_circuit(self):
        """Test that Or short-circuits on first True."""
        call_count = []

        def track_call(val):
            call_count.append(val)
            return True

        cond = Or(
            [
                Lambda(track_call),
                Lambda(lambda x: call_count.append("should not be called") or False),
            ]
        )

        cond("test")
        assert len(call_count) == 1  # Second condition should not be called

    def test_empty_conditions_raises_error(self):
        """Test that empty conditions list raises ValueError."""
        with pytest.raises(ValueError, match="at least one condition"):
            Or([])


class TestNot:
    """Tests for Not condition."""

    def test_negates_true(self):
        """Test that Not negates True to False."""
        cond = Not(EqualsTo("relu"))
        assert cond("relu") is False

    def test_negates_false(self):
        """Test that Not negates False to True."""
        cond = Not(EqualsTo("relu"))
        assert cond("gelu") is True

    def test_non_condition_raises_error(self):
        """Test that non-Condition objects raise TypeError."""
        with pytest.raises(TypeError, match="must be a Condition instance"):
            Not("not a condition")  # type: ignore


class TestLambda:
    """Tests for Lambda condition."""

    def test_custom_function(self):
        """Test with custom function."""
        cond = Lambda(lambda x: x % 2 == 0)
        assert cond(4) is True
        assert cond(5) is False

    def test_without_description(self):
        """Test repr without description."""
        cond = Lambda(lambda x: x > 0)
        repr_str = repr(cond)
        # Should show Lambda with function name and signature
        assert "Lambda" in repr_str
        assert "lambda" in repr_str.lower()

    def test_non_callable_raises_error(self):
        """Test that non-callable raises TypeError."""
        with pytest.raises(TypeError, match="must be callable"):
            Lambda("not callable")  # type: ignore


class TestFieldCondition:
    """Tests for FieldCondition."""

    def test_field_condition_basic(self):
        """Test basic field condition."""

        class MockConfig:
            optimizer = "adam"

        cond = FieldCondition("optimizer", EqualsTo("adam"))
        config = MockConfig()
        assert cond(config) is True

    def test_field_condition_false(self):
        """Test field condition returns False."""

        class MockConfig:
            optimizer = "sgd"

        cond = FieldCondition("optimizer", EqualsTo("adam"))
        config = MockConfig()
        assert cond(config) is False

    def test_nested_field_condition(self):
        """Test nested field conditions."""

        class InnerConfig:
            num_layers = 5

        class OuterConfig:
            inner = InnerConfig()

        cond = FieldCondition("inner", FieldCondition("num_layers", SmallerThan(10)))
        config = OuterConfig()
        assert cond(config) is True

    def test_missing_field_raises_error(self):
        """Test that missing field raises AttributeError."""

        class MockConfig:
            pass

        cond = FieldCondition("nonexistent", EqualsTo("value"))
        config = MockConfig()

        with pytest.raises(AttributeError, match="has no field"):
            cond(config)
