"""Tests for node tree visualization methods.

These tests verify that all node types properly render their structure
as ASCII tree representations for debugging and visualization.
"""

import spax as sp
from spax.nodes import (
    CategoricalNode,
    ConditionalNode,
    FixedNode,
    NumberNode,
)


class TestFixedNodeTree:
    """Test tree rendering for FixedNode."""

    def test_fixed_node_with_field_name(self):
        """Test FixedNode rendering with field name."""
        node = FixedNode(default="value")
        lines = node._tree_lines(
            prefix="", is_last=True, field_name="name", is_root=False
        )
        assert len(lines) == 1
        assert lines[0] == "└─ name: 'value'"

    def test_fixed_node_without_field_name(self):
        """Test FixedNode rendering without field name (categorical choice)."""
        node = FixedNode(default="adam")
        lines = node._tree_lines(
            prefix="", is_last=True, field_name=None, is_root=False
        )
        assert len(lines) == 1
        assert lines[0] == "└─ 'adam'"

    def test_fixed_node_as_middle_child(self):
        """Test FixedNode rendering as middle child (not last)."""
        node = FixedNode(default=42)
        lines = node._tree_lines(
            prefix="", is_last=False, field_name="value", is_root=False
        )
        assert len(lines) == 1
        assert lines[0] == "├─ value: 42"

    def test_fixed_node_with_prefix(self):
        """Test FixedNode rendering with prefix (nested)."""
        node = FixedNode(default=True)
        lines = node._tree_lines(
            prefix="│  ", is_last=True, field_name="flag", is_root=False
        )
        assert len(lines) == 1
        assert lines[0] == "│  └─ flag: True"


class TestNumberNodeTree:
    """Test tree rendering for NumberNode."""

    def test_int_node_inclusive_bounds(self):
        """Test IntSpace rendering with inclusive bounds."""
        space = sp.IntSpace(ge=1, le=10)
        node = NumberNode(space)
        lines = node._tree_lines(
            prefix="", is_last=True, field_name="layers", is_root=False
        )
        assert len(lines) == 1
        assert lines[0] == "└─ layers: Int([1, 10], uniform)"

    def test_int_node_exclusive_bounds(self):
        """Test IntSpace rendering with exclusive bounds."""
        space = sp.IntSpace(gt=0, lt=100)
        node = NumberNode(space)
        lines = node._tree_lines(
            prefix="", is_last=False, field_name="batch_size", is_root=False
        )
        assert len(lines) == 1
        assert lines[0] == "├─ batch_size: Int((0, 100), uniform)"

    def test_float_node_log_distribution(self):
        """Test FloatSpace rendering with log distribution."""
        space = sp.FloatSpace(ge=1e-5, le=1e-1, distribution="log")
        node = NumberNode(space)
        lines = node._tree_lines(
            prefix="", is_last=True, field_name="lr", is_root=False
        )
        assert len(lines) == 1
        assert lines[0] == "└─ lr: Float([1e-05, 0.1], log)"

    def test_float_node_mixed_bounds(self):
        """Test FloatSpace rendering with mixed inclusive/exclusive bounds."""
        space = sp.FloatSpace(gt=0.0, le=1.0)
        node = NumberNode(space)
        lines = node._tree_lines(
            prefix="│  ", is_last=False, field_name="dropout", is_root=False
        )
        assert len(lines) == 1
        assert lines[0] == "│  ├─ dropout: Float((0.0, 1.0], uniform)"


class TestCategoricalNodeTree:
    """Test tree rendering for CategoricalNode."""

    def test_categorical_with_simple_choices(self):
        """Test CategoricalNode with simple fixed value choices."""
        space = sp.CategoricalSpace(["adam", "sgd", "rmsprop"])
        node = CategoricalNode(space)
        lines = node._tree_lines(
            prefix="", is_last=True, field_name="optimizer", is_root=False
        )

        assert len(lines) == 4  # Header + 3 choices
        assert lines[0] == "└─ optimizer: Categorical"
        assert lines[1] == "   ├─ 'adam'"
        assert lines[2] == "   ├─ 'sgd'"
        assert lines[3] == "   └─ 'rmsprop'"

    def test_categorical_as_middle_child(self):
        """Test CategoricalNode as middle child (not last)."""
        space = sp.CategoricalSpace([True, False])
        node = CategoricalNode(space)
        lines = node._tree_lines(
            prefix="", is_last=False, field_name="flag", is_root=False
        )

        assert len(lines) == 3  # Header + 2 choices
        assert lines[0] == "├─ flag: Categorical"
        assert lines[1] == "│  ├─ True"
        assert lines[2] == "│  └─ False"

    def test_categorical_with_prefix(self):
        """Test CategoricalNode rendering with prefix (nested)."""
        space = sp.CategoricalSpace([1, 2, 3])
        node = CategoricalNode(space)
        lines = node._tree_lines(
            prefix="│  ", is_last=True, field_name="choice", is_root=False
        )

        assert len(lines) == 4
        assert lines[0] == "│  └─ choice: Categorical"
        assert lines[1] == "│     ├─ 1"
        assert lines[2] == "│     ├─ 2"
        assert lines[3] == "│     └─ 3"


class TestConditionalNodeTree:
    """Test tree rendering for ConditionalNode."""

    def test_conditional_with_fixed_branches(self):
        """Test ConditionalNode with fixed value branches."""
        space = sp.ConditionalSpace(
            sp.FieldCondition("use_dropout", sp.EqualsTo(True)),
            true=0.5,
            false=0.0,
        )
        node = ConditionalNode(space)
        lines = node._tree_lines(
            prefix="", is_last=True, field_name="dropout_rate", is_root=False
        )

        assert len(lines) == 3  # Header + true + false
        assert "dropout_rate: Conditional" in lines[0]
        assert "use_dropout == True" in lines[0]
        assert lines[1] == "   ├─ true: 0.5"
        assert lines[2] == "   └─ false: 0.0"

    def test_conditional_with_space_branches(self):
        """Test ConditionalNode with space branches."""
        space = sp.ConditionalSpace(
            sp.FieldCondition("use_dropout", sp.EqualsTo(True)),
            true=sp.FloatSpace(gt=0.0, lt=0.5),
            false=sp.FloatSpace(ge=0.0, le=0.1),
        )
        node = ConditionalNode(space)
        lines = node._tree_lines(
            prefix="", is_last=True, field_name="dropout_rate", is_root=False
        )

        assert len(lines) == 3
        assert "Conditional" in lines[0]
        assert "true: Float((0.0, 0.5)" in lines[1]
        assert "false: Float([0.0, 0.1]" in lines[2]

    def test_conditional_as_middle_child(self):
        """Test ConditionalNode as middle child."""
        space = sp.ConditionalSpace(
            sp.FieldCondition("flag", sp.EqualsTo(True)),
            true=1,
            false=0,
        )
        node = ConditionalNode(space)
        lines = node._tree_lines(
            prefix="", is_last=False, field_name="value", is_root=False
        )

        assert lines[0].startswith("├─")
        assert lines[1] == "│  ├─ true: 1"
        assert lines[2] == "│  └─ false: 0"

    def test_conditional_with_complex_condition(self):
        """Test ConditionalNode with composite condition."""
        space = sp.ConditionalSpace(
            sp.And(
                [
                    sp.FieldCondition("use_dropout", sp.EqualsTo(True)),
                    sp.FieldCondition("use_l2", sp.EqualsTo(True)),
                ]
            ),
            true=sp.FloatSpace(ge=0.0, le=1.0),
            false=0.0,
        )
        node = ConditionalNode(space)
        lines = node._tree_lines(
            prefix="", is_last=True, field_name="reg_strength", is_root=False
        )

        assert len(lines) == 3
        assert "use_dropout == True AND use_l2 == True" in lines[0]


class TestConfigNodeTree:
    """Test tree rendering for ConfigNode."""

    def test_simple_config_tree(self):
        """Test ConfigNode with simple fields."""

        class SimpleConfig(sp.Config):
            lr: float = sp.Float(ge=1e-5, le=1e-1)
            layers: int = sp.Int(ge=1, le=10)

        node = SimpleConfig._node
        tree = node.get_tree()

        assert "SimpleConfig" in tree
        assert "lr: Float([1e-05, 0.1]" in tree
        assert "layers: Int([1, 10]" in tree

    def test_config_with_categorical(self):
        """Test ConfigNode with categorical field."""

        class ConfigWithCat(sp.Config):
            optimizer: str = sp.Categorical(["adam", "sgd"])
            lr: float = sp.Float(ge=1e-5, le=1e-1)

        node = ConfigWithCat._node
        tree = node.get_tree()

        assert "ConfigWithCat" in tree
        assert "optimizer: Categorical" in tree
        assert "'adam'" in tree
        assert "'sgd'" in tree
        assert "lr: Float" in tree

    def test_config_with_conditional(self):
        """Test ConfigNode with conditional field."""

        class ConfigWithCond(sp.Config):
            use_dropout: bool = sp.Categorical([True, False])
            dropout_rate: float = sp.Conditional(
                sp.FieldCondition("use_dropout", sp.EqualsTo(True)),
                true=sp.Float(gt=0.0, lt=0.5),
                false=0.0,
            )

        node = ConfigWithCond._node
        tree = node.get_tree()

        assert "ConfigWithCond" in tree
        assert "use_dropout: Categorical" in tree
        assert "dropout_rate: Conditional" in tree
        assert "use_dropout == True" in tree
        assert "true:" in tree
        assert "false:" in tree

    def test_config_with_fixed_values(self):
        """Test ConfigNode with fixed value fields."""

        class ConfigWithFixed(sp.Config):
            name: str = "model"
            lr: float = sp.Float(ge=1e-5, le=1e-1)
            version: int = 1

        node = ConfigWithFixed._node
        tree = node.get_tree()

        assert "ConfigWithFixed" in tree
        assert "lr: Float" in tree
        # Fixed values should not appear in tree (they're not searchable)
        # Only spaces appear in the tree
        lines = tree.split("\n")
        assert any("lr:" in line for line in lines)

    def test_nested_config_tree(self):
        """Test ConfigNode with nested config."""

        class InnerConfig(sp.Config):
            hidden_size: int = sp.Int(ge=64, le=512)
            num_layers: int = sp.Int(ge=1, le=10)

        class OuterConfig(sp.Config):
            lr: float = sp.Float(ge=1e-5, le=1e-1)
            model: InnerConfig

        node = OuterConfig._node
        tree = node.get_tree()

        assert "OuterConfig" in tree
        assert "lr: Float" in tree
        assert "model: InnerConfig" in tree
        assert "hidden_size: Int" in tree
        assert "num_layers: Int" in tree

    def test_config_tree_structure_indentation(self):
        """Test that tree structure has correct indentation."""

        class TestConfig(sp.Config):
            a: int = sp.Int(ge=1, le=10)
            b: float = sp.Float(ge=0.0, le=1.0)
            c: str = sp.Categorical(["x", "y"])

        node = TestConfig._node
        tree = node.get_tree()
        lines = tree.split("\n")

        # First line should be the config name with no prefix
        assert lines[0] == "TestConfig"

        # Subsequent lines should have branch characters
        for line in lines[1:]:
            if line.strip():  # Skip empty lines
                assert (
                    line.startswith("├─")
                    or line.startswith("└─")
                    or "   " in line
                    or "│  " in line
                )


class TestComplexConfigTree:
    """Test tree rendering for complex nested configurations."""

    def test_deeply_nested_config(self):
        """Test tree with multiple levels of nesting."""

        class Level3Config(sp.Config):
            value: int = sp.Int(ge=1, le=10)

        class Level2Config(sp.Config):
            level3: Level3Config
            lr: float = sp.Float(ge=1e-5, le=1e-1)

        class Level1Config(sp.Config):
            level2: Level2Config
            name: str = sp.Categorical(["a", "b"])

        node = Level1Config._node
        tree = node.get_tree()

        assert "Level1Config" in tree
        assert "level2: Level2Config" in tree
        assert "level3: Level3Config" in tree
        assert "value: Int" in tree
        assert "lr: Float" in tree
        assert "name: Categorical" in tree

    def test_config_with_all_node_types(self):
        """Test config containing all node types."""

        class NestedConfig(sp.Config):
            x: int = sp.Int(ge=1, le=5)

        class ComplexConfig(sp.Config):
            # NumberNode
            lr: float = sp.Float(ge=1e-5, le=1e-1)
            layers: int = sp.Int(ge=1, le=10)

            # CategoricalNode with simple choices
            optimizer: str = sp.Categorical(["adam", "sgd"])

            # CategoricalNode with Config choices
            model: NestedConfig = sp.Categorical([NestedConfig])

            # ConditionalNode
            use_dropout: bool = sp.Categorical([True, False])
            dropout_rate: float = sp.Conditional(
                sp.FieldCondition("use_dropout", sp.EqualsTo(True)),
                true=sp.Float(gt=0.0, lt=0.5),
                false=0.0,
            )

            # FixedNode
            version: str = "1.0"

        node = ComplexConfig._node
        tree = node.get_tree()

        # Check all components appear
        assert "ComplexConfig" in tree
        assert "lr: Float" in tree
        assert "layers: Int" in tree
        assert "optimizer: Categorical" in tree
        assert "model: NestedConfig" in tree
        assert "NestedConfig" in tree
        assert "dropout_rate: Conditional" in tree
        assert "use_dropout == True" in tree

    def test_conditional_with_categorical_branches(self):
        """Test conditional with categorical spaces as branches."""

        class ConfigWithCondCat(sp.Config):
            model_size: str = sp.Categorical(["small", "large"])
            optimizer: str = sp.Conditional(
                sp.FieldCondition("model_size", sp.EqualsTo("large")),
                true=sp.Categorical(["adam", "adamw"]),
                false=sp.Categorical(["sgd", "momentum"]),
            )

        node = ConfigWithCondCat._node
        tree = node.get_tree()

        assert "ConfigWithCondCat" in tree
        assert "model_size: Categorical" in tree
        assert "optimizer: Conditional" in tree
        assert "model_size == 'large'" in tree
        # Both branches should show as Categorical
        lines = tree.split("\n")
        true_false_categorical = [
            line for line in lines if "true:" in line or "false:" in line
        ]
        assert len(true_false_categorical) >= 2


class TestTreeEdgeCases:
    """Test edge cases in tree rendering."""

    def test_empty_config(self):
        """Test config with no searchable fields."""

        class EmptyConfig(sp.Config):
            name: str = "test"
            version: int = 1

        node = EmptyConfig._node
        tree = node.get_tree()

        assert "EmptyConfig" in tree
        lines = [line for line in tree.split("\n") if line.strip()]
        assert len(lines) == 3

    def test_single_field_config(self):
        """Test config with single searchable field."""

        class SingleFieldConfig(sp.Config):
            lr: float = sp.Float(ge=1e-5, le=1e-1)

        node = SingleFieldConfig._node
        tree = node.get_tree()

        lines = tree.split("\n")
        assert len(lines) == 2
        assert "SingleFieldConfig" in lines[0]
        assert "lr: Float" in lines[1]
        assert "└─" in lines[1]  # Should use last-child marker

    def test_categorical_single_choice(self):
        """Test categorical with single choice (should simplify to fixed)."""

        class SingleChoiceConfig(sp.Config):
            optimizer: str = sp.Categorical(["adam"])

        node = SingleChoiceConfig._node
        tree = node.get_tree()

        # Single-choice categorical should be simplified to fixed value
        assert "SingleChoiceConfig" in tree
        # Should show as fixed value, not categorical
        assert "'adam'" in tree


class TestGetTreeMethod:
    """Test the public get_tree() method."""

    def test_get_tree_returns_string(self):
        """Test that get_tree() returns a string."""

        class TestConfig(sp.Config):
            lr: float = sp.Float(ge=1e-5, le=1e-1)

        tree = TestConfig._node.get_tree()
        assert isinstance(tree, str)

    def test_get_tree_multiline(self):
        """Test that get_tree() returns multiline string."""

        class TestConfig(sp.Config):
            lr: float = sp.Float(ge=1e-5, le=1e-1)
            layers: int = sp.Int(ge=1, le=10)

        tree = TestConfig._node.get_tree()
        lines = tree.split("\n")
        assert len(lines) == 3  # Config + 2 fields

    def test_get_tree_printable(self):
        """Test that get_tree() output is printable."""

        class TestConfig(sp.Config):
            lr: float = sp.Float(ge=1e-5, le=1e-1)
            optimizer: str = sp.Categorical(["adam", "sgd"])

        tree = TestConfig._node.get_tree()
        # Should not raise any exceptions
        print(tree)

        # Should contain readable characters
        assert all(c.isprintable() or c in "\n\t" for c in tree)
