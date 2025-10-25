"""Tests for conditional spaces."""

import pytest

from spax import Config
from spax.spaces import (
    UNSET,
    Conditional,
    ConditionalSpace,
    EqualsTo,
    FieldCondition,
    FloatSpace,
    IntSpace,
    MultiFieldLambdaCondition,
    SmallerThan,
)


class TestConditionalSpaceBasic:
    """Test basic ConditionalSpace functionality."""

    def test_requires_attribute_condition_error(self):
        """Test that ConditionalSpace requires AttributeCondition at top level."""
        # ObjectCondition should fail
        with pytest.raises(TypeError, match="AttributeCondition"):
            ConditionalSpace(
                condition=EqualsTo(5),  # ObjectCondition, not AttributeCondition
                true=FloatSpace(ge=0.0, le=10.0),
                false=FloatSpace(ge=10.0, le=20.0),
            )

    def test_basic_field_condition_true_branch(self):
        """Test basic conditional activating true branch."""

        class MockConfig:
            mode = "advanced"

        space = ConditionalSpace(
            condition=FieldCondition("mode", EqualsTo("advanced")),
            true=FloatSpace(ge=0.0, le=100.0),
            false=FloatSpace(ge=0.0, le=10.0),
        )
        space.field_name = "value"

        config = MockConfig()

        # Should use true branch (0-100 range)
        assert space.validate_with_config(50.0, config) == 50.0
        assert space.validate_with_config(100.0, config) == 100.0

        # Should reject values outside true branch range
        with pytest.raises(ValueError):
            space.validate_with_config(150.0, config)

    def test_basic_field_condition_false_branch(self):
        """Test basic conditional activating false branch."""

        class MockConfig:
            mode = "simple"

        space = ConditionalSpace(
            condition=FieldCondition("mode", EqualsTo("advanced")),
            true=FloatSpace(ge=0.0, le=100.0),
            false=FloatSpace(ge=0.0, le=10.0),
        )
        space.field_name = "value"

        config = MockConfig()

        # Should use false branch (0-10 range)
        assert space.validate_with_config(5.0, config) == 5.0
        assert space.validate_with_config(10.0, config) == 10.0

        # Should reject values outside false branch range
        with pytest.raises(ValueError):
            space.validate_with_config(50.0, config)

    def test_both_branches_as_spaces(self):
        """Test conditional with both branches as Space objects."""

        class MockConfig:
            use_large_range = True

        space = ConditionalSpace(
            condition=FieldCondition("use_large_range", EqualsTo(True)),
            true=IntSpace(ge=0, le=1000),
            false=IntSpace(ge=0, le=10),
        )
        space.field_name = "value"

        config = MockConfig()
        assert space.validate_with_config(500, config) == 500

        config.use_large_range = False
        assert space.validate_with_config(5, config) == 5

    def test_both_branches_as_fixed_values(self):
        """Test conditional with both branches as fixed values."""

        class MockConfig:
            enabled = True

        space = ConditionalSpace(
            condition=FieldCondition("enabled", EqualsTo(True)), true=10, false=0
        )
        space.field_name = "value"

        config = MockConfig()
        assert space.validate_with_config(10, config) == 10

        with pytest.raises(ValueError, match="Expected fixed value"):
            space.validate_with_config(5, config)

        config.enabled = False
        assert space.validate_with_config(0, config) == 0

    def test_true_space_false_fixed(self):
        """Test conditional with true as Space, false as fixed value."""

        class MockConfig:
            mode = "variable"

        space = ConditionalSpace(
            condition=FieldCondition("mode", EqualsTo("variable")),
            true=FloatSpace(ge=0.0, le=10.0),
            false=5.0,
        )
        space.field_name = "value"

        config = MockConfig()
        # True branch - can be any value in range
        assert space.validate_with_config(7.5, config) == 7.5

        config.mode = "fixed"
        # False branch - must be exactly 5.0
        assert space.validate_with_config(5.0, config) == 5.0
        with pytest.raises(ValueError, match="Expected fixed value"):
            space.validate_with_config(7.5, config)

    def test_true_fixed_false_space(self):
        """Test conditional with true as fixed value, false as Space."""

        class MockConfig:
            use_default = True

        space = ConditionalSpace(
            condition=FieldCondition("use_default", EqualsTo(True)),
            true=42,
            false=IntSpace(ge=0, le=100),
        )
        space.field_name = "value"

        config = MockConfig()
        # True branch - must be exactly 42
        assert space.validate_with_config(42, config) == 42

        config.use_default = False
        # False branch - can be any value in range
        assert space.validate_with_config(75, config) == 75

    def test_branches_as_config_types(self):
        """Test conditional with Config types as branches."""

        class ConfigA(Config):
            x: int = 1

        class ConfigB(Config):
            y: int = 2

        class MockConfig:
            use_a = True

        space = ConditionalSpace(
            condition=FieldCondition("use_a", EqualsTo(True)),
            true=ConfigA,
            false=ConfigB,
        )
        space.field_name = "config"

        mock_config = MockConfig()

        # True branch - accepts ConfigA instance
        instance_a = ConfigA()
        assert space.validate_with_config(instance_a, mock_config) == instance_a

        # Should reject ConfigB when condition is true
        instance_b = ConfigB()
        with pytest.raises(ValueError):
            space.validate_with_config(instance_b, mock_config)

        mock_config.use_a = False
        # False branch - accepts ConfigB instance
        assert space.validate_with_config(instance_b, mock_config) == instance_b


class TestConditionalValidation:
    """Test ConditionalSpace validation behavior."""

    def test_condition_evaluation_failure(self):
        """Test error when condition evaluation fails."""

        class MockConfig:
            pass  # Missing 'mode' field

        space = ConditionalSpace(
            condition=FieldCondition("mode", EqualsTo("test")), true=10, false=20
        )
        space.field_name = "value"

        config = MockConfig()

        with pytest.raises(RuntimeError, match="Failed to evaluate condition"):
            space.validate_with_config(10, config)

    def test_value_doesnt_match_active_branch(self):
        """Test error when value doesn't match the active branch."""

        class MockConfig:
            mode = "fixed"

        space = ConditionalSpace(
            condition=FieldCondition("mode", EqualsTo("fixed")),
            true=42,
            false=IntSpace(ge=0, le=100),
        )
        space.field_name = "value"

        config = MockConfig()

        # Active branch is true (fixed value 42), trying to validate 50
        with pytest.raises(ValueError, match="Expected fixed value"):
            space.validate_with_config(50, config)


class TestNestedConditionals:
    """Test nested ConditionalSpace structures."""

    def test_two_level_nested_conditional(self):
        """Test conditional inside conditional (2 levels)."""

        class MockConfig:
            mode = "advanced"
            size = "large"

        # Inner conditional: if size=="large" then 100-1000, else 10-100
        inner_conditional = ConditionalSpace(
            condition=FieldCondition("size", EqualsTo("large")),
            true=IntSpace(ge=100, le=1000),
            false=IntSpace(ge=10, le=100),
        )

        # Outer conditional: if mode=="advanced" then inner_conditional, else 0-10
        outer_conditional = ConditionalSpace(
            condition=FieldCondition("mode", EqualsTo("advanced")),
            true=inner_conditional,
            false=IntSpace(ge=0, le=10),
        )
        outer_conditional.field_name = "value"

        config = MockConfig()

        # mode="advanced" and size="large" -> 100-1000 range
        assert outer_conditional.validate_with_config(500, config) == 500

        config.size = "small"
        # mode="advanced" and size="small" -> 10-100 range
        assert outer_conditional.validate_with_config(50, config) == 50

        config.mode = "simple"
        # mode="simple" -> 0-10 range (ignores size)
        assert outer_conditional.validate_with_config(5, config) == 5

    def test_three_level_nested_conditional(self):
        """Test deeply nested conditionals (3 levels)."""

        class MockConfig:
            level1 = True
            level2 = True
            level3 = True

        # Level 3
        level3_cond = ConditionalSpace(
            condition=FieldCondition("level3", EqualsTo(True)), true=100, false=200
        )

        # Level 2
        level2_cond = ConditionalSpace(
            condition=FieldCondition("level2", EqualsTo(True)),
            true=level3_cond,
            false=300,
        )

        # Level 1
        level1_cond = ConditionalSpace(
            condition=FieldCondition("level1", EqualsTo(True)),
            true=level2_cond,
            false=400,
        )
        level1_cond.field_name = "value"

        config = MockConfig()

        # All true: 100
        assert level1_cond.validate_with_config(100, config) == 100

        # level1=T, level2=T, level3=F: 200
        config.level3 = False
        assert level1_cond.validate_with_config(200, config) == 200

        # level1=T, level2=F: 300 (ignores level3)
        config.level2 = False
        assert level1_cond.validate_with_config(300, config) == 300

        # level1=F: 400 (ignores level2 and level3)
        config.level1 = False
        assert level1_cond.validate_with_config(400, config) == 400

    def test_nested_field_conditions_in_conditional(self):
        """Test ConditionalSpace with nested FieldConditions."""

        class Inner:
            threshold = 50

        class Outer:
            inner = Inner()

        # Condition checks: outer.inner.threshold < 100
        space = ConditionalSpace(
            condition=FieldCondition(
                "inner", FieldCondition("threshold", SmallerThan(100))
            ),
            true="low_threshold",
            false="high_threshold",
        )
        space.field_name = "mode"

        config = Outer()

        # threshold is 50 < 100, so true branch
        assert space.validate_with_config("low_threshold", config) == "low_threshold"

        config.inner.threshold = 150
        # threshold is 150 >= 100, so false branch
        assert space.validate_with_config("high_threshold", config) == "high_threshold"


class TestMultiFieldLambdaCondition:
    """Test ConditionalSpace with MultiFieldLambdaCondition."""

    def test_multifield_lambda_condition(self):
        """Test ConditionalSpace with MultiFieldLambdaCondition (2 fields)."""

        class MockConfig:
            min_val = 10
            max_val = 100

        # Condition: max_val > min_val
        space = ConditionalSpace(
            condition=MultiFieldLambdaCondition(
                ["min_val", "max_val"],
                lambda d: d["max_val"] > d["min_val"],
            ),
            true="valid_range",
            false="invalid_range",
        )
        space.field_name = "status"

        config = MockConfig()

        # max_val (100) > min_val (10), true branch
        assert space.validate_with_config("valid_range", config) == "valid_range"

        config.max_val = 5
        # max_val (5) < min_val (10), false branch
        assert space.validate_with_config("invalid_range", config) == "invalid_range"

    def test_complex_multifield_condition(self):
        """Test ConditionalSpace with MultiFieldLambdaCondition (3 fields)."""

        class MockConfig:
            width = 10
            height = 20
            depth = 5

        # Condition: width * height * depth < 1000
        space = ConditionalSpace(
            condition=MultiFieldLambdaCondition(
                ["width", "height", "depth"],
                lambda d: d["width"] * d["height"] * d["depth"] < 1000,
            ),
            true=IntSpace(ge=0, le=10),
            false=IntSpace(ge=10, le=100),
        )
        space.field_name = "value"

        config = MockConfig()

        # 10 * 20 * 5 = 1000, not < 1000, so false branch
        assert space.validate_with_config(50, config) == 50

        config.depth = 4
        # 10 * 20 * 4 = 800 < 1000, so true branch
        assert space.validate_with_config(5, config) == 5


class TestConditionalSpaceProperties:
    """Test ConditionalSpace properties and metadata."""

    def test_no_default(self):
        """Test space without default."""
        space = ConditionalSpace(
            condition=FieldCondition("x", EqualsTo(5)), true=10, false=20
        )

        assert space.default is UNSET

    def test_description(self):
        """Test description field."""
        space = ConditionalSpace(
            condition=FieldCondition("x", EqualsTo(5)),
            true=10,
            false=20,
            description="Test conditional",
        )

        assert space.description == "Test conditional"

    def test_field_name_propagation_to_branches(self):
        """Test that field_name propagates to nested spaces."""

        class TestClass:
            value = ConditionalSpace(
                condition=FieldCondition("mode", EqualsTo("test")),
                true=FloatSpace(ge=0.0, le=10.0),
                false=IntSpace(ge=0, le=100),
            )

        # Field name should be set on the conditional
        assert TestClass.value.field_name == "value"

        # And on the branches
        assert TestClass.value.true_branch.field_name == "value"
        assert TestClass.value.false_branch.field_name == "value"

    def test_repr(self):
        """Test string representation."""
        space = ConditionalSpace(
            condition=FieldCondition("x", EqualsTo(5)),
            true=FloatSpace(ge=0.0, le=10.0),
            false=IntSpace(ge=0, le=100),
        )
        repr_str = repr(space)

        assert "ConditionalSpace" in repr_str
        assert "condition" in repr_str
        assert "true" in repr_str
        assert "false" in repr_str


class TestConditionalFactory:
    """Test Conditional() factory function."""

    def test_conditional_factory_creates_space(self):
        """Test that Conditional() creates a ConditionalSpace."""
        space = Conditional(
            condition=FieldCondition("x", EqualsTo(5)), true=10, false=20
        )

        assert isinstance(space, ConditionalSpace)

    def test_conditional_factory_all_parameters(self):
        """Test Conditional() with all parameters."""
        space = Conditional(
            condition=FieldCondition("mode", EqualsTo("test")),
            true=FloatSpace(ge=0.0, le=10.0),
            false=IntSpace(ge=0, le=100),
            description="Test conditional",
        )

        assert isinstance(space, ConditionalSpace)
        assert space.description == "Test conditional"


# ============================================================================
# NEW TESTS FOR DEPENDENCY TRACKING / DOTTED PATHS
# ============================================================================


class TestDependencyIntrospection:
    """Tests for get_required_fields() / get_required_paths() behavior."""

    def test_fieldcondition_required_paths_simple(self):
        cond = FieldCondition("foo", EqualsTo(1))

        # Top-level dependency
        assert cond.get_required_fields() == {"foo"}

        # Required paths should include just "foo"
        paths = cond.get_required_paths()
        assert [p.raw for p in paths] == ["foo"]

    def test_fieldcondition_required_paths_nested(self):
        # Outer.field 'outer' wraps inner FieldCondition("inner", ...)
        nested = FieldCondition("outer", FieldCondition("inner", EqualsTo(10)))

        # Only the root "outer" should be required as a top-level field
        assert nested.get_required_fields() == {"outer"}

        # Required paths should include:
        # - "outer" (the immediate root)
        # - "outer.inner" (the nested access)
        raw_paths = [p.raw for p in nested.get_required_paths()]
        assert "outer" in raw_paths
        assert "outer.inner" in raw_paths
        # determinism check: root first, then extended path
        assert raw_paths[0] == "outer"

    def test_multifield_required_fields_and_paths(self):
        cond = MultiFieldLambdaCondition(
            ["model.hidden_dim", "trainer.batch_size"],
            lambda d: d["model.hidden_dim"] > 0 and d["trainer.batch_size"] > 0,
        )

        # Required fields should be the ROOTS of the dotted paths
        # -> {"model", "trainer"}
        assert cond.get_required_fields() == {"model", "trainer"}

        # Required paths should include both full dotted paths, sorted
        paths = cond.get_required_paths()
        raw_paths = [p.raw for p in paths]
        assert set(raw_paths) == {"model.hidden_dim", "trainer.batch_size"}

        # Should be deterministic ordering (sorted by raw string)
        assert raw_paths == sorted(raw_paths)

    def test_conditional_space_with_dotted_field_condition(self):
        """ConditionalSpace should work with nested FieldCondition chains."""

        class OptimizerConfig:
            name = "adam"

        class TrainerConfig:
            optimizer = OptimizerConfig()
            lr = 1e-3

        class RootConfig:
            trainer = TrainerConfig()

        # If trainer.optimizer.name == "adam", then lr space is (1e-5, 1e-2)
        lr_space = ConditionalSpace(
            condition=FieldCondition(
                "trainer",
                FieldCondition(
                    "optimizer",
                    FieldCondition("name", EqualsTo("adam")),
                ),
            ),
            true=FloatSpace(ge=1e-5, le=1e-2),
            false=FloatSpace(ge=1e-3, le=1e-1),
        )
        lr_space.field_name = "lr"

        cfg = RootConfig()

        # matches true branch because optimizer.name == "adam"
        assert lr_space.validate_with_config(5e-4, cfg) == pytest.approx(5e-4)

        # flip optimizer name -> goes to false branch
        cfg.trainer.optimizer.name = "sgd"
        assert lr_space.validate_with_config(5e-2, cfg) == pytest.approx(5e-2)

        # now false branch is active, so low values should fail validation
        with pytest.raises(ValueError):
            lr_space.validate_with_config(5e-4, cfg)
