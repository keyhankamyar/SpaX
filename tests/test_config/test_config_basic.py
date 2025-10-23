"""Tests for basic Config functionality."""

from typing import Literal

from pydantic import Field
import pytest

from spax import Categorical, Conditional, Config, Float, Int
from spax.spaces import EqualsTo, FieldCondition


class TestConfigCreation:
    """Test basic Config creation and instantiation."""

    def test_empty_config(self):
        """Test creating an empty config."""

        class EmptyConfig(Config):
            pass

        config = EmptyConfig()
        assert isinstance(config, EmptyConfig)
        assert isinstance(config, Config)

    def test_config_with_int_space(self):
        """Test config with IntSpace field."""

        class MyConfig(Config):
            value: int = Int(ge=0, le=10)

        config = MyConfig(value=5)
        assert config.value == 5

    def test_config_with_float_space(self):
        """Test config with FloatSpace field."""

        class MyConfig(Config):
            rate: float = Float(ge=0.0, le=1.0)

        config = MyConfig(rate=0.5)
        assert config.rate == 0.5

    def test_config_with_categorical_space(self):
        """Test config with CategoricalSpace field."""

        class MyConfig(Config):
            mode: str = Categorical(["train", "eval", "test"])

        config = MyConfig(mode="train")
        assert config.mode == "train"

    def test_config_with_conditional_space(self):
        """Test config with ConditionalSpace field."""

        class MyConfig(Config):
            enabled: bool = Categorical([True, False])
            value: int = Conditional(
                condition=FieldCondition("enabled", EqualsTo(True)),
                true=Int(ge=0, le=100),
                false=0,
            )

        config = MyConfig(enabled=True, value=50)
        assert config.enabled is True
        assert config.value == 50

    def test_config_with_multiple_space_types(self):
        """Test config with multiple space types."""

        class MyConfig(Config):
            count: int = Int(ge=1, le=100)
            rate: float = Float(ge=0.0, le=1.0)
            mode: str = Categorical(["a", "b", "c"])
            flag: bool = Categorical([True, False])

        config = MyConfig(count=10, rate=0.5, mode="b", flag=True)

        assert config.count == 10
        assert config.rate == 0.5
        assert config.mode == "b"
        assert config.flag is True

    def test_config_with_fixed_values(self):
        """Test config with fixed default values."""

        class MyConfig(Config):
            name: str = "default_name"
            version: int = 1
            enabled: bool = True

        config = MyConfig()
        assert config.name == "default_name"
        assert config.version == 1
        assert config.enabled is True

    def test_config_with_default_factory(self):
        """Test config with default_factory."""

        class MyConfig(Config):
            items: list = Field(default_factory=list)
            mapping: dict = Field(default_factory=dict)

        config1 = MyConfig()
        config2 = MyConfig()

        # Should be different instances
        assert config1.items is not config2.items
        assert config1.mapping is not config2.mapping

        # Modifying one shouldn't affect the other
        config1.items.append(1)
        assert len(config1.items) == 1
        assert len(config2.items) == 0

    def test_config_with_inferred_bool(self):
        """Test config with bool type (inferred as Categorical)."""

        class MyConfig(Config):
            flag: bool

        config = MyConfig(flag=True)
        assert config.flag is True

        config = MyConfig(flag=False)
        assert config.flag is False

    def test_config_with_inferred_literal(self):
        """Test config with Literal type (inferred as Categorical)."""

        class MyConfig(Config):
            mode: Literal["a", "b", "c"]

        config = MyConfig(mode="b")
        assert config.mode == "b"

    def test_config_with_inferred_union(self):
        """Test config with Union type (inferred as Categorical)."""

        class MyConfig(Config):
            value: Literal["a", "b"] | None

        config = MyConfig(value="a")
        assert config.value == "a"

        config = MyConfig(value=None)
        assert config.value is None


class TestConfigFieldAccess:
    """Test accessing and setting config fields."""

    def test_dot_notation_access(self):
        """Test accessing fields via dot notation."""

        class MyConfig(Config):
            x: int = 10
            y: str = "hello"

        config = MyConfig()

        assert config.x == 10
        assert config.y == "hello"

    def test_field_assignment(self):
        """Test setting fields after creation."""

        class MyConfig(Config):
            value: int = Int(ge=0, le=10)

        config = MyConfig(value=5)
        assert config.value == 5

        config.value = 8
        assert config.value == 8

    def test_field_assignment_validates(self):
        """Test that field assignment triggers validation."""

        class MyConfig(Config):
            value: int = Int(ge=0, le=10)

        config = MyConfig(value=5)

        # Valid assignment
        config.value = 7
        assert config.value == 7

        # Invalid assignment should fail
        with pytest.raises(ValueError):
            config.value = 15

    def test_multiple_instances_independent(self):
        """Test that multiple config instances are independent."""

        class MyConfig(Config):
            value: int = Int(ge=0, le=10, default=5)

        config1 = MyConfig()
        config2 = MyConfig()

        config1.value = 3
        config2.value = 7

        assert config1.value == 3
        assert config2.value == 7


class TestConfigValidation:
    """Test config validation behavior."""

    def test_valid_values_accepted(self):
        """Test that valid values are accepted."""

        class MyConfig(Config):
            x: int = Int(ge=0, le=10)
            y: float = Float(ge=0.0, le=1.0)
            z: str = Categorical(["a", "b", "c"])

        # All valid
        config = MyConfig(x=5, y=0.5, z="b")
        assert config.x == 5
        assert config.y == 0.5
        assert config.z == "b"

        # Boundary values
        config = MyConfig(x=0, y=0.0, z="a")
        assert config.x == 0

        config = MyConfig(x=10, y=1.0, z="c")
        assert config.x == 10

    def test_invalid_values_rejected(self):
        """Test that invalid values are rejected."""

        class MyConfig(Config):
            x: int = Int(ge=0, le=10)
            y: str = Categorical(["a", "b"])

        # x out of range
        with pytest.raises(ValueError):
            MyConfig(x=15, y="a")

        with pytest.raises(ValueError):
            MyConfig(x=-5, y="a")

        # y not in choices
        with pytest.raises(ValueError):
            MyConfig(x=5, y="invalid")

    def test_type_coercion(self):
        """Test type coercion (float to int for IntSpace)."""

        class MyConfig(Config):
            value: int = Int(ge=0, le=10)

        # Float that represents an integer should be accepted
        config = MyConfig(value=5.0)
        assert config.value == 5
        assert isinstance(config.value, int)

        # Float with fractional part should be rejected
        with pytest.raises(ValueError, match="Expected integer"):
            MyConfig(value=5.5)

    def test_wrong_type_rejected(self):
        """Test that wrong types are rejected."""

        class MyConfig(Config):
            value: int = Int(ge=0, le=10)

        with pytest.raises(ValueError):
            MyConfig(value="not a number")

    def test_field_name_in_error_message(self):
        """Test that validation errors include field name."""

        class MyConfig(Config):
            my_value: int = Int(ge=0, le=10)

        try:
            MyConfig(my_value=15)
            pytest.fail("Should have raised ValueError")
        except ValueError as e:
            assert "my_value" in str(e)

    def test_validation_on_instantiation(self):
        """Test that validation happens during __init__."""

        class MyConfig(Config):
            value: int = Int(ge=0, le=10)

        # Should validate immediately
        with pytest.raises(ValueError):
            MyConfig(value=100)

    def test_validation_on_assignment(self):
        """Test that validation happens on field assignment."""

        class MyConfig(Config):
            value: int = Int(ge=0, le=10)

        config = MyConfig(value=5)

        # Should validate on assignment
        with pytest.raises(ValueError):
            config.value = 100


class TestConfigDefaults:
    """Test default value behavior."""

    def test_space_default_used(self):
        """Test that space default is used when field not provided."""

        class MyConfig(Config):
            value: int = Int(ge=0, le=10, default=5)

        config = MyConfig()
        assert config.value == 5

    def test_fixed_default_used(self):
        """Test that fixed default is used."""

        class MyConfig(Config):
            name: str = "default"
            count: int = 0

        config = MyConfig()
        assert config.name == "default"
        assert config.count == 0

    def test_default_factory_called(self):
        """Test that default_factory is called."""
        call_count = 0

        def make_list():
            nonlocal call_count
            call_count += 1
            return []

        class MyConfig(Config):
            items: list = Field(default_factory=make_list)

        config1 = MyConfig()
        config2 = MyConfig()

        # Factory should be called twice
        assert call_count == 2

        # Should be different instances
        assert config1.items is not config2.items

    def test_can_override_defaults(self):
        """Test that defaults can be overridden."""

        class MyConfig(Config):
            x: int = Int(ge=0, le=10, default=5)
            y: str = "default"

        config = MyConfig(x=7, y="custom")
        assert config.x == 7
        assert config.y == "custom"

    def test_missing_required_field_error(self):
        """Test error when required field is missing."""

        class MyConfig(Config):
            required: int = Int(ge=0, le=10)

        with pytest.raises(RuntimeError):
            MyConfig()

    def test_partial_defaults(self):
        """Test config with some defaults and some required."""

        class MyConfig(Config):
            required: int = Int(ge=0, le=10)
            optional: int = Int(ge=0, le=10, default=5)

        config = MyConfig(required=3)
        assert config.required == 3
        assert config.optional == 5


class TestConfigRepr:
    """Test config string representation."""

    def test_repr_simple(self):
        """Test repr with simple config."""

        class MyConfig(Config):
            x: int = 5
            y: str = "hello"

        config = MyConfig()
        repr_str = repr(config)

        assert "MyConfig" in repr_str
        assert "x=5" in repr_str
        assert "y='hello'" in repr_str

    def test_repr_with_spaces(self):
        """Test repr with space fields."""

        class MyConfig(Config):
            value: int = Int(ge=0, le=10, default=5)

        config = MyConfig()
        repr_str = repr(config)

        assert "MyConfig" in repr_str
        assert "value=5" in repr_str

    def test_str_same_as_repr(self):
        """Test that str() produces same output as repr()."""

        class MyConfig(Config):
            x: int = 5

        config = MyConfig()

        assert str(config) == repr(config)


class TestConfigEquality:
    """Test config equality comparison."""

    def test_configs_with_same_values_equal(self):
        """Test that configs with same values are equal."""

        class MyConfig(Config):
            x: int = 5
            y: str = "hello"

        config1 = MyConfig()
        config2 = MyConfig()

        # Pydantic BaseModel provides equality
        assert config1 == config2

    def test_configs_with_different_values_not_equal(self):
        """Test that configs with different values are not equal."""

        class MyConfig(Config):
            value: int = Int(ge=0, le=10)

        config1 = MyConfig(value=5)
        config2 = MyConfig(value=7)

        assert config1 != config2


class TestConfigWithConditionals:
    """Test config with conditional fields in basic scenarios."""

    def test_simple_conditional_true_branch(self):
        """Test conditional field using true branch."""

        class MyConfig(Config):
            mode: str = Categorical(["simple", "advanced"])
            value: int = Conditional(
                condition=FieldCondition("mode", EqualsTo("advanced")),
                true=Int(ge=0, le=100),
                false=Int(ge=0, le=10),
            )

        config = MyConfig(mode="advanced", value=50)
        assert config.mode == "advanced"
        assert config.value == 50

    def test_simple_conditional_false_branch(self):
        """Test conditional field using false branch."""

        class MyConfig(Config):
            mode: str = Categorical(["simple", "advanced"])
            value: int = Conditional(
                condition=FieldCondition("mode", EqualsTo("advanced")),
                true=Int(ge=0, le=100),
                false=Int(ge=0, le=10),
            )

        config = MyConfig(mode="simple", value=5)
        assert config.mode == "simple"
        assert config.value == 5

    def test_conditional_validation_enforces_active_branch(self):
        """Test that conditional validation uses the active branch."""

        class MyConfig(Config):
            mode: str = Categorical(["simple", "advanced"])
            value: int = Conditional(
                condition=FieldCondition("mode", EqualsTo("advanced")),
                true=Int(ge=0, le=100),
                false=Int(ge=0, le=10),
            )

        # Mode is "simple", so value must be 0-10
        config = MyConfig(mode="simple", value=5)
        assert config.value == 5

        # Mode is "simple", value=50 is out of range for false branch
        with pytest.raises(ValueError):
            MyConfig(mode="simple", value=50)
