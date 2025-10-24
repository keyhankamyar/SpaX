"""Tests for object-level conditions."""

import pytest

from spax.spaces.conditions import (
    EqualsTo,
    In,
    IsInstance,
    Lambda,
    LargerThan,
    NotEqualsTo,
    NotIn,
    SmallerThan,
)


class TestEqualsTo:
    """Test EqualsTo condition."""

    def test_equals_basic(self):
        """Test basic equality check."""
        cond = EqualsTo(5)
        assert cond(5) is True
        assert cond(4) is False

    def test_equals_string(self):
        """Test with string values."""
        cond = EqualsTo("hello")
        assert cond("hello") is True
        assert cond("world") is False

    def test_equals_none(self):
        """Test with None value."""
        cond = EqualsTo(None)
        assert cond(None) is True
        assert cond(0) is False

    def test_equals_float(self):
        """Test with float values."""
        cond = EqualsTo(3.14)
        assert cond(3.14) is True
        assert cond(3.15) is False

    def test_repr(self):
        """Test string representation."""
        cond = EqualsTo(42)
        assert repr(cond) == "EqualsTo(42)"


class TestNotEqualsTo:
    """Test NotEqualsTo condition."""

    def test_not_equals_basic(self):
        """Test basic inequality check."""
        cond = NotEqualsTo(5)
        assert cond(4) is True
        assert cond(5) is False

    def test_not_equals_none(self):
        """Test with None value."""
        cond = NotEqualsTo(None)
        assert cond(0) is True
        assert cond(None) is False

    def test_repr(self):
        """Test string representation."""
        cond = NotEqualsTo("test")
        assert repr(cond) == "NotEqualsTo('test')"


class TestIn:
    """Test In condition."""

    def test_in_list(self):
        """Test membership in list."""
        cond = In([1, 2, 3])
        assert cond(2) is True
        assert cond(4) is False

    def test_in_set(self):
        """Test membership in set."""
        cond = In({1, 2, 3})
        assert cond(2) is True
        assert cond(4) is False

    def test_in_tuple(self):
        """Test membership in tuple."""
        cond = In((1, 2, 3))
        assert cond(2) is True
        assert cond(4) is False

    def test_in_strings(self):
        """Test with string values."""
        cond = In(["apple", "banana", "cherry"])
        assert cond("banana") is True
        assert cond("orange") is False

    def test_in_mixed_types(self):
        """Test with mixed types."""
        cond = In([1, "two", 3.0, None])
        assert cond(1) is True
        assert cond("two") is True
        assert cond(3.0) is True
        assert cond(None) is True
        assert cond(2) is False

    def test_repr(self):
        """Test string representation."""
        cond = In([1, 2, 3])
        assert "In(" in repr(cond)


class TestNotIn:
    """Test NotIn condition."""

    def test_not_in_list(self):
        """Test non-membership in list."""
        cond = NotIn([1, 2, 3])
        assert cond(4) is True
        assert cond(2) is False

    def test_not_in_empty(self):
        """Test with empty collection."""
        cond = NotIn([])
        assert cond(1) is True
        assert cond("anything") is True


class TestSmallerThan:
    """Test SmallerThan condition."""

    def test_smaller_than_basic(self):
        """Test basic less than comparison."""
        cond = SmallerThan(10)
        assert cond(5) is True
        assert cond(10) is False
        assert cond(15) is False

    def test_smaller_than_or_equals(self):
        """Test less than or equal comparison."""
        cond = SmallerThan(10, or_equals=True)
        assert cond(5) is True
        assert cond(10) is True
        assert cond(15) is False

    def test_smaller_than_floats(self):
        """Test with float values."""
        cond = SmallerThan(3.14)
        assert cond(3.0) is True
        assert cond(3.14) is False
        assert cond(3.2) is False

    def test_smaller_than_negative(self):
        """Test with negative values."""
        cond = SmallerThan(0)
        assert cond(-1) is True
        assert cond(0) is False
        assert cond(1) is False

    def test_non_numeric_init_error(self):
        """Test error with non-numeric initialization."""
        with pytest.raises(TypeError, match="numeric"):
            SmallerThan("10")

    def test_non_numeric_call_error(self):
        """Test error with non-numeric value."""
        cond = SmallerThan(10)
        with pytest.raises(TypeError, match="numeric"):
            cond("5")

    def test_or_equals_non_bool_error(self):
        """Test error with non-bool or_equals."""
        with pytest.raises(TypeError, match="bool"):
            SmallerThan(10, or_equals="yes")

    def test_repr(self):
        """Test string representation."""
        cond = SmallerThan(10, or_equals=True)
        assert repr(cond) == "SmallerThan(10, or_equals=True)"


class TestLargerThan:
    """Test LargerThan condition."""

    def test_larger_than_basic(self):
        """Test basic greater than comparison."""
        cond = LargerThan(10)
        assert cond(15) is True
        assert cond(10) is False
        assert cond(5) is False

    def test_larger_than_or_equals(self):
        """Test greater than or equal comparison."""
        cond = LargerThan(10, or_equals=True)
        assert cond(15) is True
        assert cond(10) is True
        assert cond(5) is False

    def test_larger_than_floats(self):
        """Test with float values."""
        cond = LargerThan(3.14)
        assert cond(3.2) is True
        assert cond(3.14) is False
        assert cond(3.0) is False


class TestIsInstance:
    """Test IsInstance condition."""

    def test_isinstance_single_type(self):
        """Test with single type."""
        cond = IsInstance(int)
        assert cond(5) is True
        assert cond(5.0) is False
        assert cond("5") is False

    def test_isinstance_tuple_of_types(self):
        """Test with tuple of types."""
        cond = IsInstance((int, float))
        assert cond(5) is True
        assert cond(5.0) is True
        assert cond("5") is False

    def test_isinstance_custom_class(self):
        """Test with custom class."""

        class MyClass:
            pass

        cond = IsInstance(MyClass)
        obj = MyClass()
        assert cond(obj) is True
        assert cond("not MyClass") is False

    def test_non_type_error(self):
        """Test error with non-type value."""
        with pytest.raises(TypeError):
            IsInstance("not a type")

    def test_repr(self):
        """Test string representation."""
        cond = IsInstance(int)
        assert "IsInstance" in repr(cond)


class TestLambda:
    """Test Lambda condition."""

    def test_lambda_basic(self):
        """Test basic lambda condition."""
        cond = Lambda(lambda x: x > 5)
        assert cond(10) is True
        assert cond(3) is False

    def test_lambda_complex(self):
        """Test complex lambda condition."""
        cond = Lambda(lambda x: x % 2 == 0)
        assert cond(4) is True
        assert cond(5) is False

    def test_lambda_string_operations(self):
        """Test lambda with string operations."""
        cond = Lambda(lambda x: len(x) > 3)
        assert cond("hello") is True
        assert cond("hi") is False

    def test_lambda_non_callable_error(self):
        """Test error with non-callable."""
        with pytest.raises(TypeError, match="callable"):
            Lambda("not callable")

    def test_lambda_non_bool_return_error(self):
        """Test error when lambda doesn't return bool."""
        cond = Lambda(lambda x: x + 1)
        with pytest.raises(TypeError, match="return bool"):
            cond(5)

    def test_lambda_repr(self):
        """Test string representation."""
        cond = Lambda(lambda x: x > 5)
        assert "Lambda" in repr(cond)


class TestNonComparableHandling:
    """Test handling of non-comparable objects."""

    def test_equals_to_non_comparable_init(self):
        """Test EqualsTo rejects non-comparable values at initialization."""

        class NotComparable:
            pass

        with pytest.raises(TypeError, match="comparable"):
            EqualsTo(NotComparable())

    def test_equals_to_non_comparable_call(self):
        """Test EqualsTo rejects non-comparable values when called."""
        cond = EqualsTo(5)

        class NotComparable:
            pass

        with pytest.raises(TypeError, match="comparable"):
            cond(NotComparable())

    def test_not_equals_to_non_comparable_init(self):
        """Test NotEqualsTo rejects non-comparable values at initialization."""

        class NotComparable:
            pass

        with pytest.raises(TypeError, match="comparable"):
            NotEqualsTo(NotComparable())

    def test_not_equals_to_non_comparable_call(self):
        """Test NotEqualsTo rejects non-comparable values when called."""
        cond = NotEqualsTo(5)

        class NotComparable:
            pass

        with pytest.raises(TypeError, match="comparable"):
            cond(NotComparable())

    def test_in_non_comparable_init(self):
        """Test In rejects non-comparable values at initialization."""

        class NotComparable:
            pass

        with pytest.raises(TypeError, match="comparable"):
            In([1, NotComparable(), 3])

    def test_in_non_comparable_call(self):
        """Test In rejects non-comparable values when called."""
        cond = In([1, 2, 3])

        class NotComparable:
            pass

        with pytest.raises(TypeError, match="comparable"):
            cond(NotComparable())

    def test_not_in_non_comparable_init(self):
        """Test NotIn rejects non-comparable values at initialization."""

        class NotComparable:
            pass

        with pytest.raises(TypeError, match="comparable"):
            NotIn([1, NotComparable(), 3])

    def test_not_in_non_comparable_call(self):
        """Test NotIn rejects non-comparable values when called."""
        cond = NotIn([1, 2, 3])

        class NotComparable:
            pass

        with pytest.raises(TypeError, match="comparable"):
            cond(NotComparable())

    def test_comparable_with_custom_eq(self):
        """Test that objects with custom __eq__ are accepted."""

        class Comparable:
            def __init__(self, val):
                self.val = val

            def __eq__(self, other):
                if isinstance(other, Comparable):
                    return self.val == other.val
                return False

        obj1 = Comparable(5)
        obj2 = Comparable(5)
        obj3 = Comparable(10)

        # Should work with EqualsTo
        cond = EqualsTo(obj1)
        assert cond(obj2) is True
        assert cond(obj3) is False

        # Should work with In
        cond = In([obj1, obj3])
        assert cond(obj2) is True
