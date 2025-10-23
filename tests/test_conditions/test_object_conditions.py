"""Tests for object-level conditions."""

import pytest

from spax.spaces.conditions import (
    And,
    EqualsTo,
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


class TestAnd:
    """Test And condition composition."""

    def test_and_all_true(self):
        """Test And when all conditions are true."""
        cond = And([LargerThan(5), SmallerThan(10)])
        assert cond(7) is True

    def test_and_one_false(self):
        """Test And when one condition is false."""
        cond = And([LargerThan(5), SmallerThan(10)])
        assert cond(3) is False
        assert cond(12) is False

    def test_and_multiple_conditions(self):
        """Test And with multiple conditions."""
        cond = And([LargerThan(0), SmallerThan(100), NotEqualsTo(50)])
        assert cond(25) is True
        assert cond(50) is False
        assert cond(-1) is False
        assert cond(101) is False

    def test_and_single_condition(self):
        """Test And with single condition."""
        cond = And([EqualsTo(5)])
        assert cond(5) is True
        assert cond(6) is False

    def test_and_empty_error(self):
        """Test error with empty conditions."""
        with pytest.raises(ValueError, match="at least one"):
            And([])

    def test_and_non_condition_error(self):
        """Test error with non-Condition object."""
        with pytest.raises(TypeError, match="Condition instances"):
            And([EqualsTo(5), "not a condition"])

    def test_and_non_iterable_error(self):
        """Test error with non-iterable."""
        with pytest.raises(TypeError, match="iterable"):
            And(5)


class TestOr:
    """Test Or condition composition."""

    def test_or_one_true(self):
        """Test Or when one condition is true."""
        cond = Or([EqualsTo(5), EqualsTo(10)])
        assert cond(5) is True
        assert cond(10) is True

    def test_or_all_false(self):
        """Test Or when all conditions are false."""
        cond = Or([EqualsTo(5), EqualsTo(10)])
        assert cond(7) is False

    def test_or_multiple_conditions(self):
        """Test Or with multiple conditions."""
        cond = Or([SmallerThan(0), LargerThan(100), EqualsTo(50)])
        assert cond(-5) is True
        assert cond(105) is True
        assert cond(50) is True
        assert cond(25) is False

    def test_or_empty_error(self):
        """Test error with empty conditions."""
        with pytest.raises(ValueError, match="at least one"):
            Or([])


class TestNot:
    """Test Not condition negation."""

    def test_not_basic(self):
        """Test basic negation."""
        cond = Not(EqualsTo(5))
        assert cond(5) is False
        assert cond(10) is True

    def test_not_double_negation(self):
        """Test double negation."""
        cond = Not(Not(EqualsTo(5)))
        assert cond(5) is True
        assert cond(10) is False

    def test_not_with_compound(self):
        """Test Not with compound condition."""
        cond = Not(And([LargerThan(5), SmallerThan(10)]))
        assert cond(7) is False
        assert cond(3) is True
        assert cond(12) is True

    def test_not_non_condition_error(self):
        """Test error with non-Condition object."""
        with pytest.raises(TypeError, match="Condition instance"):
            Not("not a condition")


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


class TestConditionComposition:
    """Test complex condition compositions."""

    def test_nested_and_or(self):
        """Test nested And/Or conditions."""
        # (x > 5 AND x < 10) OR (x > 20 AND x < 30)
        cond = Or(
            [
                And([LargerThan(5), SmallerThan(10)]),
                And([LargerThan(20), SmallerThan(30)]),
            ]
        )

        assert cond(7) is True
        assert cond(25) is True
        assert cond(15) is False
        assert cond(3) is False

    def test_complex_composition(self):
        """Test complex multi-level composition."""
        # NOT((x IN [1,2,3]) AND (x != 2))
        cond = Not(And([In([1, 2, 3]), NotEqualsTo(2)]))

        assert cond(1) is False  # 1 is in list and != 2
        assert cond(2) is True  # 2 is in list but == 2
        assert cond(4) is True  # 4 is not in list

    def test_type_check_first_composition(self):
        """Test combining type check first to avoid type errors."""
        # Check type first, then numeric operations
        cond = And(
            [
                IsInstance((int, float)),
                Or([SmallerThan(0), LargerThan(100)]),
                Not(In([50, 150])),
            ]
        )

        assert cond(-10) is True
        assert cond(110) is True
        assert cond(50) is False
        assert cond(150) is False
        assert cond("100") is False  # Fails at IsInstance check

    def test_short_circuit_behavior(self):
        """Test that And/Or short-circuit properly."""
        # If first condition fails, second shouldn't be evaluated
        # We can't directly test short-circuiting, but we can verify correct results
        cond = And(
            [
                EqualsTo(5),
                Lambda(lambda x: x > 0),  # This would pass for 5
            ]
        )

        assert cond(5) is True
        assert cond(10) is False  # First condition fails, returns False


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
