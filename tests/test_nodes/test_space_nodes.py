"""Tests for SpaceNode classes."""

import pytest

from spax import Config
from spax.nodes import (
    CategoricalNode,
    ConditionalNode,
    FixedNode,
    NumberNode,
    SpaceNode,
)
from spax.spaces import (
    CategoricalSpace,
    ConditionalSpace,
    EqualsTo,
    FieldCondition,
    FloatSpace,
    IntSpace,
)


class TestSpaceNode:
    """Test base SpaceNode class."""

    def test_creation_with_space(self):
        """Test SpaceNode creation with a Space."""
        space = FloatSpace(ge=0.0, le=10.0)
        node = SpaceNode(space)

        assert node.space is space

    def test_error_with_non_space(self):
        """Test error when creating with non-Space object."""
        with pytest.raises(TypeError, match="must be a Space"):
            SpaceNode("not a space")


class TestNumberNode:
    """Test NumberNode."""

    def test_creation_with_float_space(self):
        """Test NumberNode creation with FloatSpace."""
        space = FloatSpace(ge=0.0, le=10.0)
        node = NumberNode(space)

        assert node.space is space
        assert isinstance(node.space, FloatSpace)

    def test_creation_with_int_space(self):
        """Test NumberNode creation with IntSpace."""
        space = IntSpace(ge=0, le=10)
        node = NumberNode(space)

        assert node.space is space
        assert isinstance(node.space, IntSpace)

    def test_error_with_non_number_space(self):
        """Test error when creating with non-NumberSpace."""
        space = CategoricalSpace(["a", "b", "c"])

        with pytest.raises(TypeError, match="must be a NumberSpace"):
            NumberNode(space)


class TestCategoricalNode:
    """Test CategoricalNode."""

    def test_creation_with_categorical_space(self):
        """Test CategoricalNode creation with CategoricalSpace."""
        space = CategoricalSpace(["a", "b", "c"])
        node = CategoricalNode(space)

        assert node.space is space
        assert isinstance(node.space, CategoricalSpace)

    def test_error_with_non_categorical_space(self):
        """Test error when creating with non-CategoricalSpace."""
        space = FloatSpace(ge=0.0, le=10.0)

        with pytest.raises(TypeError, match="must be a CategoricalSpace"):
            CategoricalNode(space)

    def test_children_with_simple_values(self):
        """Test children created for simple values."""
        space = CategoricalSpace([1, 2, 3])
        node = CategoricalNode(space)

        # Should have 3 children
        assert len(node._children) == 3

        # All should be FixedNodes
        for i in range(3):
            assert isinstance(node._children[i], FixedNode)
            assert node._children[i].default == space.choices[i]

    def test_children_with_config_types(self):
        """Test children created for Config types."""

        class ConfigA(Config):
            x: int = 1

        class ConfigB(Config):
            y: int = 2

        space = CategoricalSpace([ConfigA, ConfigB])
        node = CategoricalNode(space)

        # Should have 2 children
        assert len(node._children) == 2

        # Children should be ConfigNodes (accessed via Config._node)
        assert node._children[0] is ConfigA._node
        assert node._children[1] is ConfigB._node

    def test_children_mixed_config_and_values(self):
        """Test children with mix of Config types and values."""

        class MyConfig(Config):
            x: int = 1

        space = CategoricalSpace([MyConfig, "string", 42])
        node = CategoricalNode(space)

        assert len(node._children) == 3

        # First child is ConfigNode
        assert node._children[0] is MyConfig._node

        # Other children are FixedNodes
        assert isinstance(node._children[1], FixedNode)
        assert node._children[1].default == "string"

        assert isinstance(node._children[2], FixedNode)
        assert node._children[2].default == 42


class TestConditionalNode:
    """Test ConditionalNode."""

    def test_creation_with_conditional_space(self):
        """Test ConditionalNode creation with ConditionalSpace."""
        space = ConditionalSpace(
            condition=FieldCondition("x", EqualsTo(5)),
            true=FloatSpace(ge=0.0, le=10.0),
            false=IntSpace(ge=0, le=100),
        )
        node = ConditionalNode(space)

        assert node.space is space
        assert isinstance(node.space, ConditionalSpace)

    def test_error_with_non_conditional_space(self):
        """Test error when creating with non-ConditionalSpace."""
        space = FloatSpace(ge=0.0, le=10.0)

        with pytest.raises(TypeError, match="must be a ConditionalSpace"):
            ConditionalNode(space)

    def test_dependencies_extraction(self):
        """Test that dependencies are extracted from condition."""
        space = ConditionalSpace(
            condition=FieldCondition("mode", EqualsTo("advanced")), true=10, false=20
        )
        node = ConditionalNode(space)

        assert node.dependencies == {"mode"}

    def test_true_branch_as_number_space(self):
        """Test true branch node creation with NumberSpace."""
        space = ConditionalSpace(
            condition=FieldCondition("x", EqualsTo(5)),
            true=FloatSpace(ge=0.0, le=10.0),
            false=20,
        )
        node = ConditionalNode(space)

        assert isinstance(node._true_node, NumberNode)
        assert isinstance(node._true_node.space, FloatSpace)

    def test_false_branch_as_categorical_space(self):
        """Test false branch node creation with CategoricalSpace."""
        space = ConditionalSpace(
            condition=FieldCondition("x", EqualsTo(5)),
            true=10,
            false=CategoricalSpace(["a", "b", "c"]),
        )
        node = ConditionalNode(space)

        assert isinstance(node._false_node, CategoricalNode)
        assert isinstance(node._false_node.space, CategoricalSpace)

    def test_branch_as_fixed_value(self):
        """Test branch node creation with fixed value."""
        space = ConditionalSpace(
            condition=FieldCondition("x", EqualsTo(5)), true=42, false=100
        )
        node = ConditionalNode(space)

        assert isinstance(node._true_node, FixedNode)
        assert node._true_node.default == 42

        assert isinstance(node._false_node, FixedNode)
        assert node._false_node.default == 100

    def test_branch_as_config_type(self):
        """Test branch node creation with Config type."""

        class MyConfig(Config):
            x: int = 1

        space = ConditionalSpace(
            condition=FieldCondition("enabled", EqualsTo(True)),
            true=MyConfig,
            false=None,
        )
        node = ConditionalNode(space)

        # True branch should be ConfigNode
        assert node._true_node is MyConfig._node

        # False branch should be FixedNode with None
        assert isinstance(node._false_node, FixedNode)
        assert node._false_node.default is None

    def test_nested_conditional(self):
        """Test ConditionalNode with nested ConditionalSpace."""
        inner_space = ConditionalSpace(
            condition=FieldCondition("y", EqualsTo(10)),
            true=FloatSpace(ge=0.0, le=1.0),
            false=FloatSpace(ge=1.0, le=10.0),
        )

        outer_space = ConditionalSpace(
            condition=FieldCondition("x", EqualsTo(5)), true=inner_space, false=100
        )

        node = ConditionalNode(outer_space)

        # True branch should be another ConditionalNode
        assert isinstance(node._true_node, ConditionalNode)
        assert isinstance(node._true_node.space, ConditionalSpace)

        # Dependencies only from top-level condition
        assert node.dependencies == {"x"}
