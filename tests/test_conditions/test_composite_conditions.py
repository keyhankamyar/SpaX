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
    Or,
    SmallerThan,
)
from spax.spaces.conditions.attribute_conditions import FieldCondition


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
        with pytest.raises(ValueError, match="at least two"):
            And([EqualsTo(5)])

    def test_and_empty_error(self):
        """Test error with empty conditions."""
        with pytest.raises(ValueError, match="at least two"):
            And([])

    def test_and_non_condition_error(self):
        """Test error with non-Condition object."""
        with pytest.raises(TypeError, match="Condition instances"):
            And([EqualsTo(5), "not a condition"])

    def test_and_non_iterable_error(self):
        """Test error with non-iterable."""
        with pytest.raises(TypeError, match="iterable"):
            And(5)

    def test_and_object(self):
        """Test Or when all conditions are false."""
        cond = And([EqualsTo(5), EqualsTo(10)])
        with pytest.raises(TypeError, match="children must be AttributeConditions"):
            assert cond.get_required_fields()

    def test_and_attribute(self):
        """Test Or when all conditions are false."""
        cond = And([FieldCondition("field_1", EqualsTo(10)), FieldCondition("field_2", EqualsTo(5))])
        assert cond.get_required_fields() == {"field_1", "field_2"}


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

    def test_or_object(self):
        """Test Or when all conditions are false."""
        cond = Or([EqualsTo(5), EqualsTo(10)])
        with pytest.raises(TypeError, match="children must be AttributeConditions"):
            assert cond.get_required_fields()

    def test_or_attribute(self):
        """Test Or when all conditions are false."""
        cond = Or([FieldCondition("field_1", EqualsTo(10)), FieldCondition("field_2", EqualsTo(5))])
        assert cond.get_required_fields() == {"field_1", "field_2"}

    def test_or_multiple_conditions(self):
        """Test Or with multiple conditions."""
        cond = Or([SmallerThan(0), LargerThan(100), EqualsTo(50)])
        assert cond(-5) is True
        assert cond(105) is True
        assert cond(50) is True
        assert cond(25) is False

    def test_or_empty_error(self):
        """Test error with empty conditions."""
        with pytest.raises(ValueError, match="at least two"):
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

    def test_not_object(self):
        """Test Or when all conditions are false."""
        cond = Not(EqualsTo(5))
        with pytest.raises(TypeError, match="child must be an AttributeCondition"):
            assert cond.get_required_fields()

    def test_not_attribute(self):
        """Test Or when all conditions are false."""
        cond = Not(FieldCondition("field_1", EqualsTo(10)))
        assert cond.get_required_fields() == {"field_1"}


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
