"""Tests for attribute-level conditions."""

import pytest

from spax.spaces.conditions import (
    And,
    EqualsTo,
    FieldCondition,
    In,
    LargerThan,
    MultiFieldLambdaCondition,
    Not,
    NotEqualsTo,
    Or,
    SmallerThan,
)


class TestFieldCondition:
    """Test FieldCondition."""

    def test_basic_field_check(self):
        """Test basic field condition checking."""

        class MockConfig:
            value = 10

        cond = FieldCondition("value", EqualsTo(10))
        assert cond(MockConfig()) is True

        cond = FieldCondition("value", EqualsTo(5))
        assert cond(MockConfig()) is False

    def test_field_with_various_object_conditions(self):
        """Test FieldCondition with different ObjectConditions."""

        class MockConfig:
            number = 42
            text = "hello"
            flag = True

        config = MockConfig()

        # EqualsTo
        assert FieldCondition("number", EqualsTo(42))(config) is True

        # In
        assert FieldCondition("text", In(["hello", "world"]))(config) is True

        # LargerThan
        assert FieldCondition("number", LargerThan(40))(config) is True

        # SmallerThan
        assert FieldCondition("number", SmallerThan(50))(config) is True

        # NotEqualsTo
        assert FieldCondition("flag", NotEqualsTo(False))(config) is True

    def test_field_with_compound_object_conditions(self):
        """Test FieldCondition with compound ObjectConditions."""

        class MockConfig:
            value = 25

        config = MockConfig()

        # And condition
        cond = FieldCondition("value", And([LargerThan(20), SmallerThan(30)]))
        assert cond(config) is True

        # Or condition
        cond = FieldCondition("value", Or([EqualsTo(10), EqualsTo(25)]))
        assert cond(config) is True

        # Not condition
        cond = FieldCondition("value", Not(EqualsTo(10)))
        assert cond(config) is True

    def test_nested_field_conditions_two_levels(self):
        """Test two-level nested FieldConditions."""

        class Inner:
            value = 100

        class Outer:
            inner = Inner()

        config = Outer()

        # outer.inner.value == 100
        cond = FieldCondition("inner", FieldCondition("value", EqualsTo(100)))
        assert cond(config) is True

        cond = FieldCondition("inner", FieldCondition("value", EqualsTo(50)))
        assert cond(config) is False

    def test_nested_field_conditions_three_levels(self):
        """Test three-level nested FieldConditions."""

        class Level3:
            num = 42

        class Level2:
            level3 = Level3()

        class Level1:
            level2 = Level2()

        config = Level1()

        # config.level2.level3.num > 40
        cond = FieldCondition(
            "level2", FieldCondition("level3", FieldCondition("num", LargerThan(40)))
        )
        assert cond(config) is True

    def test_nested_field_conditions_four_plus_levels(self):
        """Test deeply nested FieldConditions (4+ levels)."""

        class L5:
            value = "deep"

        class L4:
            l5 = L5()

        class L3:
            l4 = L4()

        class L2:
            l3 = L3()

        class L1:
            l2 = L2()

        config = L1()

        # config.l2.l3.l4.l5.value == "deep"
        cond = FieldCondition(
            "l2",
            FieldCondition(
                "l3",
                FieldCondition(
                    "l4",
                    FieldCondition("l5", FieldCondition("value", EqualsTo("deep"))),
                ),
            ),
        )
        assert cond(config) is True

    def test_nested_with_compound_conditions(self):
        """Test nested FieldConditions with compound ObjectConditions."""

        class Inner:
            x = 15
            y = 25

        class Outer:
            inner = Inner()

        config = Outer()

        # outer.inner.x is in range [10, 20]
        cond = FieldCondition(
            "inner", FieldCondition("x", And([LargerThan(10), SmallerThan(20)]))
        )
        assert cond(config) is True

    def test_field_not_exist_error(self):
        """Test error when field doesn't exist."""

        class MockConfig:
            value = 10

        cond = FieldCondition("nonexistent", EqualsTo(10))

        with pytest.raises(AttributeError, match="no field 'nonexistent'"):
            cond(MockConfig())

    def test_nested_field_not_exist_error(self):
        """Test error when nested field doesn't exist at any level."""

        class Inner:
            value = 10

        class Outer:
            inner = Inner()

        config = Outer()

        # First level missing
        cond = FieldCondition("missing", FieldCondition("value", EqualsTo(10)))
        with pytest.raises(AttributeError, match="no field 'missing'"):
            cond(config)

        # Second level missing
        cond = FieldCondition("inner", FieldCondition("missing", EqualsTo(10)))
        with pytest.raises(AttributeError, match="no field 'missing'"):
            cond(config)

    def test_non_string_field_name_error(self):
        """Test error with non-string field name."""
        with pytest.raises(TypeError, match="field_name must be str"):
            FieldCondition(123, EqualsTo(10))

    def test_non_condition_error(self):
        """Test error with non-Condition object."""
        with pytest.raises(TypeError, match="Condition instance"):
            FieldCondition("value", "not a condition")

    def test_get_required_fields(self):
        """Test get_required_fields returns correct field name."""
        cond = FieldCondition("my_field", EqualsTo(10))
        assert cond.get_required_fields() == {"my_field"}

    def test_get_required_fields_nested(self):
        """Test get_required_fields for nested conditions returns only immediate field."""
        cond = FieldCondition("outer", FieldCondition("inner", EqualsTo(10)))
        # Should only return the immediate field, not nested ones
        assert cond.get_required_fields() == {"outer"}

    def test_repr(self):
        """Test string representation."""
        cond = FieldCondition("my_field", EqualsTo(10))
        repr_str = repr(cond)
        assert "FieldCondition" in repr_str
        assert "my_field" in repr_str
        assert "EqualsTo(10)" in repr_str


class TestMultiFieldLambdaCondition:
    """Test MultiFieldLambdaCondition."""

    def test_basic_two_field_lambda(self):
        """Test basic lambda with two fields."""

        class MockConfig:
            x = 10
            y = 20

        cond = MultiFieldLambdaCondition(["x", "y"], lambda x, y: x + y == 30)
        assert cond(MockConfig()) is True

        cond = MultiFieldLambdaCondition(["x", "y"], lambda x, y: x > y)
        assert cond(MockConfig()) is False

    def test_three_field_lambda(self):
        """Test lambda with three fields."""

        class MockConfig:
            a = 5
            b = 10
            c = 15

        cond = MultiFieldLambdaCondition(
            ["a", "b", "c"], lambda a, b, c: a + b + c == 30
        )
        assert cond(MockConfig()) is True

    def test_four_plus_field_lambda(self):
        """Test lambda with four or more fields."""

        class MockConfig:
            w = 1
            x = 2
            y = 3
            z = 4

        cond = MultiFieldLambdaCondition(
            ["w", "x", "y", "z"], lambda w, x, y, z: w + x + y + z == 10
        )
        assert cond(MockConfig()) is True

    def test_complex_lambda_logic(self):
        """Test complex lambda expressions."""

        class MockConfig:
            min_val = 10
            max_val = 100
            current = 50

        # Check if current is within range
        cond = MultiFieldLambdaCondition(
            ["min_val", "max_val", "current"],
            lambda min_val, max_val, current: min_val <= current <= max_val,
        )
        assert cond(MockConfig()) is True

    def test_string_field_operations(self):
        """Test lambda with string operations."""

        class MockConfig:
            first_name = "John"
            last_name = "Doe"

        cond = MultiFieldLambdaCondition(
            ["first_name", "last_name"],
            lambda first_name, last_name: len(first_name) + len(last_name) == 7,
        )
        assert cond(MockConfig()) is True

    def test_field_names_not_iterable_error(self):
        """Test error when field_names is not iterable."""
        with pytest.raises(TypeError, match="iterable"):
            MultiFieldLambdaCondition(123, lambda x: True)

    def test_field_names_string_error(self):
        """Test error when field_names is a string (should be iterable of strings)."""
        with pytest.raises(TypeError, match="iterable"):
            MultiFieldLambdaCondition("field", lambda field: True)

    def test_field_names_empty_error(self):
        """Test error with empty field_names."""
        with pytest.raises(ValueError, match="cannot be empty"):
            MultiFieldLambdaCondition([], lambda: True)

    def test_field_names_duplicates_error(self):
        """Test error with duplicate field names."""
        with pytest.raises(ValueError, match="cannot contain duplicates"):
            MultiFieldLambdaCondition(["x", "y", "x"], lambda x, y: True)

    def test_field_names_non_string_error(self):
        """Test error when field_names contains non-strings."""
        with pytest.raises(TypeError, match="must be strings"):
            MultiFieldLambdaCondition([1, 2], lambda: True)

    def test_non_callable_func_error(self):
        """Test error with non-callable func."""
        with pytest.raises(TypeError, match="callable"):
            MultiFieldLambdaCondition(["x"], "not callable")

    def test_func_signature_mismatch_error(self):
        """Test error when func signature doesn't match field_names."""
        # Too many parameters
        with pytest.raises(TypeError, match="Could not validate function signature"):
            MultiFieldLambdaCondition(["x"], lambda x, y: True)

        # Too few parameters
        with pytest.raises(TypeError, match="Could not validate function signature"):
            MultiFieldLambdaCondition(["x", "y"], lambda x: True)

        # Wrong parameter names
        with pytest.raises(TypeError, match="Could not validate function signature"):
            MultiFieldLambdaCondition(["x", "y"], lambda a, b: True)

    def test_func_non_bool_return_error(self):
        """Test error when func doesn't return bool."""

        class MockConfig:
            x = 10

        cond = MultiFieldLambdaCondition(["x"], lambda x: x + 1)

        with pytest.raises(TypeError, match="return bool"):
            cond(MockConfig())

    def test_field_not_exist_error(self):
        """Test error when field doesn't exist on config."""

        class MockConfig:
            x = 10

        cond = MultiFieldLambdaCondition(
            ["x", "nonexistent"], lambda x, nonexistent: True
        )

        with pytest.raises(AttributeError, match="no field 'nonexistent'"):
            cond(MockConfig())

    def test_get_required_fields(self):
        """Test get_required_fields returns all field names."""
        cond = MultiFieldLambdaCondition(
            ["field1", "field2", "field3"], lambda field1, field2, field3: True
        )
        assert cond.get_required_fields() == {"field1", "field2", "field3"}

    def test_repr(self):
        """Test string representation."""
        cond = MultiFieldLambdaCondition(["x", "y"], lambda x, y: x > y)
        repr_str = repr(cond)
        assert "MultiFieldLambdaCondition" in repr_str
        assert "x" in repr_str or "y" in repr_str


class TestAttributeConditionIntegration:
    """Test integration scenarios with attribute conditions."""

    def test_multiple_field_conditions_with_and(self):
        """Test combining multiple FieldConditions with And."""

        class MockConfig:
            x = 15
            y = 25

        config = MockConfig()

        cond = And(
            [FieldCondition("x", LargerThan(10)), FieldCondition("y", SmallerThan(30))]
        )
        assert cond(config) is True

    def test_multiple_field_conditions_with_or(self):
        """Test combining multiple FieldConditions with Or."""

        class MockConfig:
            status = "active"
            priority = 1

        config = MockConfig()

        cond = Or(
            [
                FieldCondition("status", EqualsTo("inactive")),
                FieldCondition("priority", EqualsTo(1)),
            ]
        )
        assert cond(config) is True

    def test_field_and_multifield_combined(self):
        """Test combining FieldCondition with MultiFieldLambdaCondition."""

        class MockConfig:
            enabled = True
            min_val = 10
            max_val = 100

        config = MockConfig()

        cond = And(
            [
                FieldCondition("enabled", EqualsTo(True)),
                MultiFieldLambdaCondition(
                    ["min_val", "max_val"], lambda min_val, max_val: max_val > min_val
                ),
            ]
        )
        assert cond(config) is True

    def test_nested_field_with_multifield(self):
        """Test nested FieldCondition combined with MultiFieldLambdaCondition."""

        class Inner:
            threshold = 50

        class Outer:
            inner = Inner()
            current_value = 75

        config = Outer()

        # Complex: inner.threshold < 60 AND current_value > 70
        cond = And(
            [
                FieldCondition("inner", FieldCondition("threshold", SmallerThan(60))),
                FieldCondition("current_value", LargerThan(70)),
            ]
        )
        assert cond(config) is True

    def test_complex_nested_scenario(self):
        """Test complex nested scenario similar to real Config usage."""

        class LayerConfig:
            hidden_dim = 256
            activation = "relu"

        class EncoderConfig:
            num_layers = 128
            layer_config = LayerConfig()

        class ModelConfig:
            encoder_config = EncoderConfig()
            use_preprocessing = True

        config = ModelConfig()

        # Check: encoder_config.layer_config.hidden_dim < 512 AND use_preprocessing == True
        cond = And(
            [
                FieldCondition(
                    "encoder_config",
                    FieldCondition(
                        "layer_config", FieldCondition("hidden_dim", SmallerThan(512))
                    ),
                ),
                FieldCondition("use_preprocessing", EqualsTo(True)),
            ]
        )
        assert cond(config) is True
