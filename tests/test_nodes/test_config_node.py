"""Tests for ConfigNode."""

from typing import Literal

from pydantic import Field
import pytest

from spax import Categorical, Conditional, Config, Float, Int
from spax.nodes import (
    CategoricalNode,
    ConditionalNode,
    ConfigNode,
    FixedNode,
    NumberNode,
)
from spax.spaces import (
    CategoricalSpace,
    EqualsTo,
    FieldCondition,
    FloatSpace,
    IntSpace,
)


class TestConfigNodeBasic:
    """Test basic ConfigNode functionality."""

    def test_error_with_non_type(self):
        """Test error when creating with non-type."""
        with pytest.raises(TypeError, match="must be a type"):
            ConfigNode("not a type")

    def test_error_with_non_config_class(self):
        """Test error when creating with non-Config class."""

        class NotConfig:
            pass

        with pytest.raises(TypeError, match="must be a Config class"):
            ConfigNode(NotConfig)


class TestConfigNodeChildren:
    """Test ConfigNode children population."""

    def test_children_with_space_fields(self):
        """Test children created for fields with Space declarations."""

        class MyConfig(Config):
            x: int = Int(ge=0, le=10)
            y: float = Float(ge=0.0, le=1.0)
            z: str = Categorical(["a", "b", "c"])

        node = MyConfig._node

        assert isinstance(node._children["x"], NumberNode)
        assert isinstance(node._children["x"].space, IntSpace)

        assert isinstance(node._children["y"], NumberNode)
        assert isinstance(node._children["y"].space, FloatSpace)

        assert isinstance(node._children["z"], CategoricalNode)
        assert isinstance(node._children["z"].space, CategoricalSpace)

    def test_children_with_fixed_defaults(self):
        """Test children created for fields with fixed default values."""

        class MyConfig(Config):
            x: int = 42
            y: str = "hello"

        node = MyConfig._node

        assert isinstance(node._children["x"], FixedNode)
        assert node._children["x"].default == 42

        assert isinstance(node._children["y"], FixedNode)
        assert node._children["y"].default == "hello"

    def test_children_with_default_factory(self):
        """Test children created for fields with default_factory."""

        class MyConfig(Config):
            items: list = Field(default_factory=list)

        node = MyConfig._node

        assert isinstance(node._children["items"], FixedNode)
        assert node._children["items"].is_factory is True

    def test_children_with_nested_config(self):
        """Test children created for nested Config fields."""

        class InnerConfig(Config):
            value: int = 10

        class OuterConfig(Config):
            inner: InnerConfig

        node = OuterConfig._node

        # Child should be the InnerConfig's node
        assert node._children["inner"] is InnerConfig._node

    def test_children_with_inferable_types(self):
        """Test children with type annotations that can be inferred."""

        class MyConfig(Config):
            flag: bool  # Should infer Categorical([True, False])
            mode: Literal["a", "b", "c"]  # Should infer Categorical

        node = MyConfig._node

        # bool should create CategoricalNode
        assert isinstance(node._children["flag"], CategoricalNode)
        assert set(node._children["flag"].space.choices) == {True, False}

        # Literal should create CategoricalNode
        assert isinstance(node._children["mode"], CategoricalNode)
        assert node._children["mode"].space.choices == ["a", "b", "c"]

    def test_conditional_field(self):
        """Test children with ConditionalSpace."""

        class MyConfig(Config):
            mode: str = Categorical(["simple", "advanced"])
            value: int = Conditional(
                condition=FieldCondition("mode", EqualsTo("advanced")),
                true=Int(ge=0, le=100),
                false=Int(ge=0, le=10),
            )

        node = MyConfig._node

        assert isinstance(node._children["value"], ConditionalNode)


class TestConfigNodeFieldOrdering:
    """Test ConfigNode field ordering and dependency resolution."""

    def test_simple_field_order(self):
        """Test field order without dependencies."""

        class MyConfig(Config):
            c: int = 3
            a: int = 1
            b: int = 2

        node = MyConfig._node

        # Without dependencies, should maintain definition order
        field_names = list(node._field_order)
        assert field_names == ["c", "a", "b"]

    def test_ordered_children_generator(self):
        """Test ordered_children() generator."""

        class MyConfig(Config):
            x: int = 1
            y: int = 2
            z: int = 3

        node = MyConfig._node

        children = list(node.ordered_children())
        assert len(children) == 3

        # Each item should be (field_name, node)
        assert children[0][0] == "x"
        assert isinstance(children[0][1], FixedNode)

        assert children[1][0] == "y"
        assert children[2][0] == "z"

    def test_dependency_ordering(self):
        """Test that fields are ordered by dependencies."""

        class MyConfig(Config):
            mode: str = Categorical(["a", "b"])
            # value depends on mode, should come after
            value: int = Conditional(
                condition=FieldCondition("mode", EqualsTo("a")), true=10, false=20
            )

        node = MyConfig._node

        field_order = list(node._field_order)

        # mode should come before value
        assert field_order.index("mode") < field_order.index("value")

    def test_multi_level_dependencies(self):
        """Test ordering with multiple dependency levels."""

        class MyConfig(Config):
            a: int = 1
            # c depends on b
            c: int = Conditional(
                condition=FieldCondition("b", EqualsTo(2)), true=30, false=31
            )
            # b depends on a
            b: int = Conditional(
                condition=FieldCondition("a", EqualsTo(1)), true=20, false=21
            )

        node = MyConfig._node

        field_order = list(node._field_order)

        # Order should be: a, b, c
        assert field_order.index("a") < field_order.index("b")
        assert field_order.index("b") < field_order.index("c")

    def test_circular_dependency_error(self):
        """Test error on circular dependencies."""
        with pytest.raises(TypeError, match="Circular dependency"):

            class MyConfig(Config):
                # a depends on b
                a: int = Conditional(
                    condition=FieldCondition("b", EqualsTo(2)), true=10, false=11
                )
                # b depends on a - circular!
                b: int = Conditional(
                    condition=FieldCondition("a", EqualsTo(1)), true=20, false=21
                )

    def test_missing_dependency_error(self):
        """Test error when dependency field doesn't exist."""
        with pytest.raises(TypeError, match="unknown field"):

            class MyConfig(Config):
                a: int = Conditional(
                    condition=FieldCondition("nonexistent", EqualsTo(5)),
                    true=10,
                    false=20,
                )


class TestConfigNodeSimplification:
    """Test ConfigNode simplification of single-choice categoricals."""

    def test_single_choice_categorical_simplified_to_fixed(self):
        """Test that single-choice categorical with non-Config is simplified to FixedNode."""

        class MyConfig(Config):
            value: str = Categorical(["only_option"])

        node = MyConfig._node

        # Should be simplified to FixedNode
        assert isinstance(node._children["value"], FixedNode)
        assert node._children["value"].default == "only_option"

    def test_single_choice_categorical_with_config_simplified(self):
        """Test that single-choice categorical with Config type is simplified to ConfigNode."""

        class InnerConfig(Config):
            x: int = 1

        class OuterConfig(Config):
            inner: InnerConfig | None = Categorical([InnerConfig])

        node = OuterConfig._node

        # Should be simplified to the ConfigNode directly
        assert node._children["inner"] is InnerConfig._node

    def test_multi_choice_categorical_not_simplified(self):
        """Test that multi-choice categorical is not simplified."""

        class MyConfig(Config):
            value: str = Categorical(["a", "b"])

        node = MyConfig._node

        # Should remain as CategoricalNode
        assert isinstance(node._children["value"], CategoricalNode)


class TestConfigNodeValidation:
    """Test ConfigNode validate_spaces method."""

    def test_validate_simple_config(self):
        """Test validation with simple config."""

        class MyConfig(Config):
            x: int = Int(ge=0, le=10)
            y: str = Categorical(["a", "b"])

        node = MyConfig._node

        data = {"x": 5, "y": "a"}
        validated = node.validate_spaces(data)

        assert validated["x"] == 5
        assert validated["y"] == "a"

    def test_validate_with_defaults(self):
        """Test validation uses defaults when values not provided."""

        class MyConfig(Config):
            x: int = Int(ge=0, le=10, default=5)
            y: str = "hello"

        node = MyConfig._node

        data = {}
        validated = node.validate_spaces(data)

        assert validated["x"] == 5
        assert validated["y"] == "hello"

    def test_validate_with_default_factory(self):
        """Test validation uses default_factory when value not provided."""

        class MyConfig(Config):
            items: list = Field(default_factory=list)

        node = MyConfig._node

        data = {}
        validated = node.validate_spaces(data)

        assert validated["items"] == []

    def test_validate_missing_field_without_default_error(self):
        """Test error when field missing and no default."""

        class MyConfig(Config):
            x: int = Int(ge=0, le=10)

        node = MyConfig._node

        with pytest.raises(RuntimeError, match="not provided"):
            node.validate_spaces({})

    def test_validate_out_of_range_error(self):
        """Test error when value out of range."""

        class MyConfig(Config):
            x: int = Int(ge=0, le=10)

        node = MyConfig._node

        with pytest.raises(ValueError, match="Validation failed"):
            node.validate_spaces({"x": 15})

    def test_validate_conditional_field(self):
        """Test validation with conditional field."""

        class MyConfig(Config):
            mode: str = Categorical(["simple", "advanced"])
            value: int = Conditional(
                condition=FieldCondition("mode", EqualsTo("advanced")),
                true=Int(ge=0, le=100),
                false=Int(ge=0, le=10),
            )

        node = MyConfig._node

        # Mode is advanced, value can be 0-100
        data = {"mode": "advanced", "value": 50}
        validated = node.validate_spaces(data)
        assert validated["value"] == 50

        # Mode is simple, value must be 0-10
        data = {"mode": "simple", "value": 5}
        validated = node.validate_spaces(data)
        assert validated["value"] == 5

        # Mode is simple, value out of range
        with pytest.raises(ValueError):
            node.validate_spaces({"mode": "simple", "value": 50})

    def test_validate_preserves_extra_fields(self):
        """Test that extra fields not in Config are preserved."""

        class MyConfig(Config):
            x: int = 1

        node = MyConfig._node

        data = {"x": 1, "extra": "value"}
        validated = node.validate_spaces(data)

        assert validated["x"] == 1
        assert validated["extra"] == "value"

    def test_validate_non_dict_error(self):
        """Test error when data is not a dict."""

        class MyConfig(Config):
            x: int = 1

        node = MyConfig._node

        with pytest.raises(ValueError, match="Got .* which is"):
            node.validate_spaces("not a dict")


class TestConfigNodeGetChild:
    """Test ConfigNode get_child method."""

    def test_get_existing_child(self):
        """Test getting existing child."""

        class MyConfig(Config):
            x: int = 1
            y: int = 2

        node = MyConfig._node

        child_x = node.get_child("x")
        assert child_x is not None
        assert isinstance(child_x, FixedNode)

    def test_get_nonexistent_child(self):
        """Test getting nonexistent child returns None."""

        class MyConfig(Config):
            x: int = 1

        node = MyConfig._node

        child = node.get_child("nonexistent")
        assert child is None
