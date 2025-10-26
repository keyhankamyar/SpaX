"""Tests for condition render() methods.

These tests verify that all condition types properly render their logic
as human-readable strings for debugging and visualization purposes.
"""

from spax.spaces.conditions import (
    And,
    EqualsTo,
    FieldCondition,
    In,
    IsInstance,
    Lambda,
    LargerThan,
    MultiFieldLambdaCondition,
    Not,
    NotEqualsTo,
    NotIn,
    Or,
    SmallerThan,
)
from spax.spaces.conditions.base import NotGiven


class TestObjectConditionRender:
    """Test render() for ObjectCondition subclasses."""

    def test_equals_to_render(self):
        """Test EqualsTo.render()."""
        cond = EqualsTo("adam")
        assert cond.render("optimizer") == "optimizer == 'adam'"

        cond = EqualsTo(42)
        assert cond.render("num_layers") == "num_layers == 42"

        cond = EqualsTo(True)
        assert cond.render("use_dropout") == "use_dropout == True"

    def test_not_equals_to_render(self):
        """Test NotEqualsTo.render()."""
        cond = NotEqualsTo("sgd")
        assert cond.render("optimizer") == "optimizer != 'sgd'"

        cond = NotEqualsTo(0)
        assert cond.render("batch_size") == "batch_size != 0"

    def test_in_render(self):
        """Test In.render()."""
        cond = In(["adam", "sgd", "rmsprop"])
        result = cond.render("optimizer")
        assert result == "optimizer in ['adam', 'sgd', 'rmsprop']"

        cond = In([1, 2, 3])
        result = cond.render("choice")
        assert result == "choice in [1, 2, 3]"

    def test_not_in_render(self):
        """Test NotIn.render()."""
        cond = NotIn(["deprecated", "old"])
        result = cond.render("optimizer")
        assert result == "optimizer not in ['deprecated', 'old']"

    def test_smaller_than_render(self):
        """Test SmallerThan.render()."""
        cond = SmallerThan(0.5)
        assert cond.render("learning_rate") == "learning_rate < 0.5"

        cond = SmallerThan(0.5, or_equals=True)
        assert cond.render("learning_rate") == "learning_rate <= 0.5"

        cond = SmallerThan(100)
        assert cond.render("batch_size") == "batch_size < 100"

    def test_larger_than_render(self):
        """Test LargerThan.render()."""
        cond = LargerThan(0.001)
        assert cond.render("learning_rate") == "learning_rate > 0.001"

        cond = LargerThan(0.001, or_equals=True)
        assert cond.render("learning_rate") == "learning_rate >= 0.001"

        cond = LargerThan(10)
        assert cond.render("num_layers") == "num_layers > 10"

    def test_is_instance_render(self):
        """Test IsInstance.render()."""
        cond = IsInstance(str)
        assert cond.render("name") == "isinstance(name, str)"

        cond = IsInstance((int, float))
        result = cond.render("value")
        assert result == "isinstance(value, (int, float))"

    def test_lambda_render(self):
        """Test Lambda.render()."""
        cond = Lambda(lambda x: x % 2 == 0)
        assert cond.render("value") == "lambda value"

        cond = Lambda(lambda x: x > 0 and x < 100)
        assert cond.render("batch_size") == "lambda batch_size"


class TestFieldConditionRender:
    """Test render() for FieldCondition."""

    def test_simple_field_condition_render(self):
        """Test FieldCondition with simple field path."""
        cond = FieldCondition("optimizer", EqualsTo("adam"))
        # When called without field_name (top-level), uses internal path
        assert cond.render() == "optimizer == 'adam'"

        cond = FieldCondition("use_dropout", EqualsTo(True))
        assert cond.render() == "use_dropout == True"

    def test_nested_field_condition_render(self):
        """Test FieldCondition with nested field path."""
        cond = FieldCondition("model.optimizer.name", EqualsTo("adam"))
        assert cond.render() == "model.optimizer.name == 'adam'"

        cond = FieldCondition("trainer.batch_size", LargerThan(32))
        assert cond.render() == "trainer.batch_size > 32"

    def test_field_condition_render_with_context(self):
        """Test FieldCondition.render() when given external field_name context."""
        cond = FieldCondition("optimizer", EqualsTo("adam"))
        # When field_name is provided (nested context), prefix it
        assert cond.render("config") == "config.optimizer == 'adam'"

    def test_field_condition_with_complex_object_condition(self):
        """Test FieldCondition with various ObjectCondition types."""
        cond = FieldCondition("optimizer", In(["adam", "sgd"]))
        assert cond.render() == "optimizer in ['adam', 'sgd']"

        cond = FieldCondition("num_layers", SmallerThan(10, or_equals=True))
        assert cond.render() == "num_layers <= 10"


class TestMultiFieldLambdaConditionRender:
    """Test render() for MultiFieldLambdaCondition."""

    def test_multi_field_lambda_render(self):
        """Test MultiFieldLambdaCondition.render()."""
        cond = MultiFieldLambdaCondition(
            ["batch_size", "num_layers"],
            lambda data: data["batch_size"] * data["num_layers"] < 1000,
        )
        # MultiFieldLambdaCondition always renders as generic "lambda data"
        assert cond.render() == "lambda data"

    def test_multi_field_lambda_render_with_context(self):
        """Test MultiFieldLambdaCondition.render() with field_name context."""
        cond = MultiFieldLambdaCondition(
            ["optimizer.name", "learning_rate"],
            lambda data: data["optimizer.name"] == "adam"
            and data["learning_rate"] > 0.001,
        )
        # field_name is ignored for MultiFieldLambdaCondition
        assert cond.render("config") == "lambda data"
        assert cond.render() == "lambda data"


class TestCompositeConditionRender:
    """Test render() for composite conditions (And, Or, Not)."""

    def test_and_render_top_level(self):
        """Test And.render() at top level (no field_name context)."""
        cond = And(
            [
                FieldCondition("use_dropout", EqualsTo(True)),
                FieldCondition("use_l2", EqualsTo(True)),
            ]
        )
        result = cond.render()
        assert result == "use_dropout == True AND use_l2 == True"

    def test_and_render_nested(self):
        """Test And.render() with field_name context (nested)."""
        # When And is used inside another condition with object conditions
        cond = And([EqualsTo("adam"), NotEqualsTo("sgd")])
        result = cond.render("optimizer")
        assert result == "optimizer == 'adam' AND optimizer != 'sgd'"

    def test_or_render_top_level(self):
        """Test Or.render() at top level."""
        cond = Or(
            [
                FieldCondition("optimizer", EqualsTo("adam")),
                FieldCondition("optimizer", EqualsTo("sgd")),
            ]
        )
        result = cond.render()
        assert result == "optimizer == 'adam' OR optimizer == 'sgd'"

    def test_or_render_nested(self):
        """Test Or.render() with field_name context."""
        cond = Or([EqualsTo(1), EqualsTo(2), EqualsTo(3)])
        result = cond.render("choice")
        assert result == "choice == 1 OR choice == 2 OR choice == 3"

    def test_not_render_top_level(self):
        """Test Not.render() at top level."""
        cond = Not(FieldCondition("use_batch_norm", EqualsTo(True)))
        result = cond.render()
        assert result == "NOT (use_batch_norm == True)"

    def test_not_render_nested(self):
        """Test Not.render() with field_name context."""
        cond = Not(EqualsTo("adam"))
        result = cond.render("optimizer")
        assert result == "NOT (optimizer == 'adam')"

    def test_nested_composite_conditions(self):
        """Test deeply nested composite conditions."""
        cond = And(
            [
                FieldCondition("use_dropout", EqualsTo(True)),
                Or(
                    [
                        FieldCondition("optimizer", EqualsTo("adam")),
                        FieldCondition("optimizer", EqualsTo("sgd")),
                    ]
                ),
            ]
        )
        result = cond.render()
        expected = "use_dropout == True AND optimizer == 'adam' OR optimizer == 'sgd'"
        assert result == expected

    def test_not_with_and(self):
        """Test Not wrapping And condition."""
        cond = Not(
            And(
                [
                    FieldCondition("use_dropout", EqualsTo(True)),
                    FieldCondition("use_l2", EqualsTo(True)),
                ]
            )
        )
        result = cond.render()
        expected = "NOT (use_dropout == True AND use_l2 == True)"
        assert result == expected


class TestRenderEdgeCases:
    """Test edge cases and special scenarios for render()."""

    def test_render_with_special_characters(self):
        """Test rendering with special characters in values."""
        cond = EqualsTo("optimizer_v2.0")
        assert cond.render("name") == "name == 'optimizer_v2.0'"

        cond = In(["a-b", "c_d", "e.f"])
        result = cond.render("choice")
        assert result == "choice in ['a-b', 'c_d', 'e.f']"

    def test_render_with_numeric_types(self):
        """Test rendering with various numeric types."""
        cond = EqualsTo(1.5e-5)
        result = cond.render("lr")
        assert "1.5e-05" in result or "1.5e-5" in result

        cond = LargerThan(1e6)
        result = cond.render("param")
        assert "1000000" in result or "1e+06" in result or "1e+6" in result

    def test_render_with_none(self):
        """Test rendering with None values."""
        cond = EqualsTo(None)
        assert cond.render("value") == "value == None"

        cond = NotEqualsTo(None)
        assert cond.render("value") == "value != None"

    def test_field_condition_with_composite_in_object_position(self):
        """Test FieldCondition wrapping composite of object conditions."""
        # This should work: composite used as inner condition
        inner_composite = And([EqualsTo("adam"), NotEqualsTo("sgd")])
        cond = FieldCondition("optimizer", inner_composite)
        result = cond.render()
        assert result == "optimizer == 'adam' AND optimizer != 'sgd'"


class TestRenderNotGivenBehavior:
    """Test the NotGiven sentinel behavior in render()."""

    def test_not_given_default_behavior(self):
        """Test that NotGiven is the default for AttributeCondition.render()."""
        cond = FieldCondition("optimizer", EqualsTo("adam"))
        # Calling with no args should use NotGiven
        result1 = cond.render()
        # Explicitly passing NotGiven should be identical
        result2 = cond.render(NotGiven)
        assert result1 == result2
        assert result1 == "optimizer == 'adam'"

    def test_not_given_vs_string_context(self):
        """Test difference between NotGiven and string field_name."""
        cond = FieldCondition("name", EqualsTo("adam"))

        # NotGiven: use internal field path
        result1 = cond.render()
        assert result1 == "name == 'adam'"

        # String context: prefix the field path
        result2 = cond.render("optimizer")
        assert result2 == "optimizer.name == 'adam'"
