import pytest

import spax as sp

# ============================================================================
# Basic Override Tests
# ============================================================================


def test_fixed_node_cannot_override():
    """Fixed nodes should raise an error when attempting to override."""

    class SimpleConfig(sp.Config):
        fixed_value: int = 42

    with pytest.raises(ValueError, match="Cannot override fixed value"):
        SimpleConfig._node.apply_override({"fixed_value": 100})


def test_number_override_to_fixed():
    """Test overriding a numeric space to a fixed value."""

    class SimpleConfig(sp.Config):
        value: int = sp.Int(ge=0, le=100)

    override = {"value": 42}
    new_node = SimpleConfig._node.apply_override(override)

    from spax.nodes import FixedNode

    assert isinstance(new_node._children["value"], FixedNode)
    assert new_node._children["value"].default == 42


def test_number_override_narrow_range():
    """Test overriding a numeric space to narrow its range."""

    class SimpleConfig(sp.Config):
        value: int = sp.Int(ge=0, le=100)

    override = {"value": {"ge": 10, "le": 50}}
    new_node = SimpleConfig._node.apply_override(override)

    from spax.nodes import NumberNode

    assert isinstance(new_node._children["value"], NumberNode)
    assert new_node._children["value"]._space.low == 10
    assert new_node._children["value"]._space.high == 50
    assert new_node._children["value"]._space.low_inclusive is True
    assert new_node._children["value"]._space.high_inclusive is True


def test_number_override_out_of_bounds():
    """Test that overriding outside original bounds raises an error."""

    class SimpleConfig(sp.Config):
        value: int = sp.Int(ge=0, le=100)

    # Try to expand range beyond original
    with pytest.raises(ValueError, match="does not fit within original space"):
        SimpleConfig._node.apply_override({"value": {"ge": -10, "le": 50}})

    with pytest.raises(ValueError, match="does not fit within original space"):
        SimpleConfig._node.apply_override({"value": {"ge": 10, "le": 150}})


def test_number_override_invalid_value():
    """Test that invalid fixed value raises an error."""

    class SimpleConfig(sp.Config):
        value: int = sp.Int(ge=0, le=100)

    with pytest.raises(ValueError, match="not valid for this space"):
        SimpleConfig._node.apply_override({"value": 200})


def test_categorical_override_to_fixed():
    """Test overriding a categorical space to a fixed value."""

    class SimpleConfig(sp.Config):
        choice: str = sp.Categorical(["a", "b", "c"])

    override = {"choice": "b"}
    new_node = SimpleConfig._node.apply_override(override)

    from spax.nodes import FixedNode

    assert isinstance(new_node._children["choice"], FixedNode)
    assert new_node._children["choice"].default == "b"


def test_categorical_override_subset():
    """Test overriding a categorical space to a subset."""

    class SimpleConfig(sp.Config):
        choice: str = sp.Categorical(["a", "b", "c"])

    override = {"choice": ["a", "c"]}
    new_node = SimpleConfig._node.apply_override(override)

    from spax.nodes import CategoricalNode

    assert isinstance(new_node._children["choice"], CategoricalNode)
    assert set(new_node._children["choice"]._space.choices) == {"a", "c"}


def test_categorical_override_single_element_list():
    """Test that single-element list becomes FixedNode."""

    class SimpleConfig(sp.Config):
        choice: str = sp.Categorical(["a", "b", "c"])

    override = {"choice": ["b"]}
    new_node = SimpleConfig._node.apply_override(override)

    from spax.nodes import FixedNode

    assert isinstance(new_node._children["choice"], FixedNode)
    assert new_node._children["choice"].default == "b"


def test_categorical_override_invalid_choice():
    """Test that invalid choice raises an error."""

    class SimpleConfig(sp.Config):
        choice: str = sp.Categorical(["a", "b", "c"])

    with pytest.raises(ValueError, match="not in original choices"):
        SimpleConfig._node.apply_override({"choice": ["a", "d"]})


# ============================================================================
# Nested Config Override Tests
# ============================================================================


def test_nested_config_override():
    """Test overriding fields in nested configs."""

    class InnerConfig(sp.Config):
        value: int = sp.Int(ge=0, le=100)
        choice: str = sp.Categorical(["a", "b"])

    class OuterConfig(sp.Config):
        inner: InnerConfig
        outer_value: int = sp.Int(ge=0, le=10)

    override = {"inner": {"value": 50, "choice": "a"}, "outer_value": 5}

    new_node = OuterConfig._node.apply_override(override)

    from spax.nodes import ConfigNode, FixedNode

    assert isinstance(new_node._children["inner"], ConfigNode)
    assert isinstance(new_node._children["inner"]._children["value"], FixedNode)
    assert new_node._children["inner"]._children["value"].default == 50
    assert isinstance(new_node._children["outer_value"], FixedNode)


def test_deeply_nested_config_override():
    """Test overriding in deeply nested configs (3+ levels)."""

    class Level3Config(sp.Config):
        deep_value: int = sp.Int(ge=0, le=100)

    class Level2Config(sp.Config):
        level3: Level3Config
        mid_value: float = sp.Float(ge=0.0, le=1.0)

    class Level1Config(sp.Config):
        level2: Level2Config
        top_value: str = sp.Categorical(["x", "y", "z"])

    override = {
        "level2": {"level3": {"deep_value": {"ge": 20, "le": 80}}, "mid_value": 0.5},
        "top_value": "y",
    }

    new_node = Level1Config._node.apply_override(override)

    from spax.nodes import NumberNode

    deep_node = new_node._children["level2"]._children["level3"]._children["deep_value"]
    assert isinstance(deep_node, NumberNode)
    assert deep_node._space.low == 20
    assert deep_node._space.high == 80


def test_partial_nested_override():
    """Test that unspecified nested fields remain unchanged."""

    class InnerConfig(sp.Config):
        value1: int = sp.Int(ge=0, le=100)
        value2: int = sp.Int(ge=0, le=100)

    class OuterConfig(sp.Config):
        inner: InnerConfig

    # Only override one field in inner
    override = {"inner": {"value1": 50}}

    new_node = OuterConfig._node.apply_override(override)

    from spax.nodes import FixedNode, NumberNode

    assert isinstance(new_node._children["inner"]._children["value1"], FixedNode)
    assert isinstance(new_node._children["inner"]._children["value2"], NumberNode)


# ============================================================================
# Config Inheritance Override Tests
# ============================================================================


def test_config_inheritance_override():
    """Test overriding configs with inheritance."""

    class BaseConfig(sp.Config):
        base_value: int = sp.Int(ge=0, le=100)
        shared_value: int = sp.Int(ge=0, le=50)

    class DerivedConfig(BaseConfig):
        derived_value: float = sp.Float(ge=0.0, le=1.0)

    override = {
        "base_value": 42,
        "shared_value": {"ge": 10, "le": 30},
        "derived_value": 0.5,
    }

    new_node = DerivedConfig._node.apply_override(override)

    from spax.nodes import FixedNode, NumberNode

    assert isinstance(new_node._children["base_value"], FixedNode)
    assert isinstance(new_node._children["shared_value"], NumberNode)
    assert isinstance(new_node._children["derived_value"], FixedNode)


def test_multi_level_inheritance_override():
    """Test overriding with multiple levels of inheritance."""

    class Level1Config(sp.Config):
        l1_value: int = sp.Int(ge=0, le=100)

    class Level2Config(Level1Config):
        l2_value: int = sp.Int(ge=0, le=100)

    class Level3Config(Level2Config):
        l3_value: int = sp.Int(ge=0, le=100)

    override = {"l1_value": 10, "l2_value": 20, "l3_value": 30}

    new_node = Level3Config._node.apply_override(override)

    assert new_node._children["l1_value"].default == 10
    assert new_node._children["l2_value"].default == 20
    assert new_node._children["l3_value"].default == 30


# ============================================================================
# Categorical with Config Choices Override Tests
# ============================================================================


def test_categorical_config_choice_override():
    """Test overriding categorical space with Config choices."""

    class ConfigA(sp.Config):
        a_value: int = sp.Int(ge=0, le=100)

    class ConfigB(sp.Config):
        b_value: float = sp.Float(ge=0.0, le=1.0)

    class MainConfig(sp.Config):
        choice: ConfigA | ConfigB

    # Fix to ConfigA
    override = {"choice": "ConfigA"}
    new_node = MainConfig._node.apply_override(override)

    from spax.nodes import ConfigNode

    assert isinstance(new_node._children["choice"], ConfigNode)
    assert new_node._children["choice"]._config_class == ConfigA


def test_categorical_config_choice_with_nested_override():
    """Test overriding both the choice and nested fields."""

    class ConfigA(sp.Config):
        a_value: int = sp.Int(ge=0, le=100)

    class ConfigB(sp.Config):
        b_value: float = sp.Float(ge=0.0, le=1.0)

    class MainConfig(sp.Config):
        choice: ConfigA | ConfigB

    # Fix to ConfigA and override its field
    override = {"choice": {"ConfigA": {"a_value": 50}}}
    new_node = MainConfig._node.apply_override(override)

    from spax.nodes import ConfigNode, FixedNode

    assert isinstance(new_node._children["choice"], ConfigNode)
    assert isinstance(new_node._children["choice"]._children["a_value"], FixedNode)


def test_categorical_config_multiple_choices_with_overrides():
    """Test narrowing Config choices while overriding their internals."""

    class ConfigA(sp.Config):
        a_value: int = sp.Int(ge=0, le=100)

    class ConfigB(sp.Config):
        b_value: float = sp.Float(ge=0.0, le=1.0)

    class ConfigC(sp.Config):
        c_value: str = sp.Categorical(["x", "y"])

    class MainConfig(sp.Config):
        choice: ConfigA | ConfigB | ConfigC

    # Keep only A and B, override A's internals
    override = {"choice": {"ConfigA": {"a_value": {"ge": 20, "le": 80}}, "ConfigB": {}}}
    new_node = MainConfig._node.apply_override(override)

    from spax.nodes import CategoricalNode, NumberNode

    assert isinstance(new_node._children["choice"], CategoricalNode)
    assert len(new_node._children["choice"]._space.choices) == 2

    # Check ConfigA's override was applied
    config_a_node = new_node._children["choice"]._children["ConfigA"]
    assert isinstance(config_a_node._children["a_value"], NumberNode)
    assert config_a_node._children["a_value"]._space.low == 20


# ============================================================================
# Conditional Space Override Tests
# ============================================================================


def test_conditional_override_true_branch():
    """Test overriding the true branch of a conditional."""

    class MainConfig(sp.Config):
        use_feature: bool
        value: int = sp.Conditional(
            condition=sp.FieldCondition("use_feature", sp.EqualsTo(True)),
            true=sp.Int(ge=0, le=100),
            false=0,
        )

    override = {"value": {"true": {"ge": 20, "le": 80}}}
    new_node = MainConfig._node.apply_override(override)

    from spax.nodes import ConditionalNode, NumberNode

    assert isinstance(new_node._children["value"], ConditionalNode)
    assert isinstance(new_node._children["value"]._true_node, NumberNode)
    assert new_node._children["value"]._true_node._space.low == 20


def test_conditional_override_false_branch():
    """Test overriding the false branch of a conditional."""

    class ConfigA(sp.Config):
        a_value: int = sp.Int(ge=0, le=100)

    class MainConfig(sp.Config):
        mode: str = sp.Categorical(["simple", "complex"])
        config: ConfigA | None = sp.Conditional(
            condition=sp.FieldCondition("mode", sp.EqualsTo("complex")),
            true=ConfigA,
            false=None,
        )

    # Can't override None, but let's test it doesn't crash
    override = {"config": {"true": {"a_value": 50}}}
    new_node = MainConfig._node.apply_override(override)

    from spax.nodes import ConditionalNode, ConfigNode, FixedNode

    assert isinstance(new_node._children["config"], ConditionalNode)
    assert isinstance(new_node._children["config"]._true_node, ConfigNode)
    assert isinstance(
        new_node._children["config"]._true_node._children["a_value"], FixedNode
    )


def test_conditional_override_both_branches():
    """Test overriding both branches of a conditional."""

    class MainConfig(sp.Config):
        use_large: bool
        value: int = sp.Conditional(
            condition=sp.FieldCondition("use_large", sp.EqualsTo(True)),
            true=sp.Int(ge=0, le=1000),
            false=sp.Int(ge=0, le=100),
        )

    override = {
        "value": {"true": {"ge": 500, "le": 1000}, "false": {"ge": 0, "le": 50}}
    }
    new_node = MainConfig._node.apply_override(override)

    from spax.nodes import ConditionalNode, NumberNode

    cond_node = new_node._children["value"]
    assert isinstance(cond_node, ConditionalNode)
    assert isinstance(cond_node._true_node, NumberNode)
    assert cond_node._true_node._space.low == 500
    assert isinstance(cond_node._false_node, NumberNode)
    assert cond_node._false_node._space.high == 50


# ============================================================================
# Real-World Scenario Tests
# ============================================================================


def test_ml_model_config_override():
    """Test realistic ML model configuration override."""

    class MLPConfig(sp.Config):
        hidden_dim: int = sp.Int(gt=16, lt=4096)
        activation: str = sp.Categorical(["relu", "gelu", "silu"])
        dropout: float = sp.Float(ge=0.0, le=0.5)

    class CNNConfig(sp.Config):
        channels: int = sp.Int(gt=8, lt=512)
        kernel_size: int = sp.Int(ge=1, le=7)
        activation: str = sp.Categorical(["relu", "gelu"])

    class ModelConfig(sp.Config):
        num_layers: int = sp.Int(ge=1, le=12)
        layer_type: MLPConfig | CNNConfig
        use_batch_norm: bool
        learning_rate: float = sp.Float(gt=1e-5, lt=1e-1, distribution="log")

    # Realistic override: narrow search space after initial exploration
    override = {
        "num_layers": {"ge": 4, "le": 8},  # Focus on mid-range
        "layer_type": {  # Only use MLP with specific constraints
            "MLPConfig": {
                "hidden_dim": {"gt": 128, "lt": 512},
                "activation": ["relu", "gelu"],  # Remove silu
            }
        },
        "use_batch_norm": True,  # Fix based on prior results
        "learning_rate": {"gt": 1e-4, "lt": 1e-2},  # Narrow LR range
    }

    new_node = ModelConfig._node.apply_override(override)

    from spax.nodes import CategoricalNode, ConfigNode, FixedNode, NumberNode

    # Check all overrides applied correctly
    assert isinstance(new_node._children["num_layers"], NumberNode)
    assert new_node._children["num_layers"]._space.low == 4

    assert isinstance(new_node._children["layer_type"], ConfigNode)
    mlp_node = new_node._children["layer_type"]
    assert isinstance(mlp_node._children["hidden_dim"], NumberNode)
    assert mlp_node._children["hidden_dim"]._space.low == 128

    assert isinstance(mlp_node._children["activation"], CategoricalNode)
    assert set(mlp_node._children["activation"]._space.choices) == {"relu", "gelu"}

    assert isinstance(new_node._children["use_batch_norm"], FixedNode)
    assert new_node._children["use_batch_norm"].default is True

    assert isinstance(new_node._children["learning_rate"], NumberNode)


def test_rl_environment_config_override():
    """Test RL environment configuration with conditional spaces."""

    class DiscreteActionConfig(sp.Config):
        num_actions: int = sp.Int(ge=2, le=20)

    class ContinuousActionConfig(sp.Config):
        action_dim: int = sp.Int(ge=1, le=10)
        action_scale: float = sp.Float(ge=0.1, le=10.0)

    class EnvConfig(sp.Config):
        action_type: str = sp.Categorical(["discrete", "continuous"])
        action_config: DiscreteActionConfig | ContinuousActionConfig = sp.Conditional(
            condition=sp.FieldCondition("action_type", sp.EqualsTo("discrete")),
            true=DiscreteActionConfig,
            false=ContinuousActionConfig,
        )
        episode_length: int = sp.Int(ge=100, le=10000)

    # Override to focus on discrete actions
    override = {
        "action_type": "discrete",
        "action_config": {"true": {"num_actions": {"ge": 4, "le": 10}}},
        "episode_length": 1000,
    }

    new_node = EnvConfig._node.apply_override(override)

    from spax.nodes import ConditionalNode, ConfigNode, FixedNode, NumberNode

    assert isinstance(new_node._children["action_type"], FixedNode)
    assert isinstance(new_node._children["action_config"], ConditionalNode)

    discrete_node = new_node._children["action_config"]._true_node
    assert isinstance(discrete_node, ConfigNode)
    assert isinstance(discrete_node._children["num_actions"], NumberNode)
    assert discrete_node._children["num_actions"]._space.low == 4


def test_complex_nested_conditional_override():
    """Test complex nested conditional with multiple dependencies."""

    class SimplePreprocessor(sp.Config):
        normalize: bool

    class ComplexPreprocessor(sp.Config):
        method: str = sp.Categorical(["standard", "minmax", "robust"])
        clip_range: float = sp.Float(ge=1.0, le=10.0)

    class ModelConfig(sp.Config):
        use_preprocessing: bool
        preprocessing_type: str = sp.Conditional(
            condition=sp.FieldCondition("use_preprocessing", sp.EqualsTo(True)),
            true=sp.Categorical(["simple", "complex"]),
            false="none",
        )
        preprocessor: SimplePreprocessor | ComplexPreprocessor | None = sp.Conditional(
            condition=sp.FieldCondition("use_preprocessing", sp.EqualsTo(True)),
            true=sp.Conditional(
                condition=sp.FieldCondition(
                    "preprocessing_type", sp.EqualsTo("simple")
                ),
                true=SimplePreprocessor,
                false=ComplexPreprocessor,
            ),
            false=None,
        )

    # Override nested conditional
    override = {
        "use_preprocessing": True,
        "preprocessing_type": {"true": ["complex"]},
        "preprocessor": {
            "true": {
                "false": {
                    "method": ["standard", "minmax"],
                    "clip_range": {"ge": 2.0, "le": 5.0},
                }
            }
        },
    }

    new_node = ModelConfig._node.apply_override(override)

    # Verify the complex nested structure
    from spax.nodes import ConditionalNode, FixedNode

    assert isinstance(new_node._children["use_preprocessing"], FixedNode)
    assert isinstance(new_node._children["preprocessing_type"], ConditionalNode)
    assert isinstance(new_node._children["preprocessor"], ConditionalNode)


# ============================================================================
# Error Handling Tests
# ============================================================================


def test_override_unknown_field():
    """Test that overriding unknown field raises error."""

    class SimpleConfig(sp.Config):
        value: int = sp.Int(ge=0, le=100)

    with pytest.raises(ValueError, match="unknown fields"):
        SimpleConfig._node.apply_override({"unknown_field": 42})


def test_override_wrong_type():
    """Test that wrong override type raises error."""

    class SimpleConfig(sp.Config):
        value: int = sp.Int(ge=0, le=100)

    with pytest.raises(TypeError, match="must be a dict"):
        SimpleConfig._node.apply_override("not a dict")


def test_number_override_invalid_keys():
    """Test that invalid dict keys in number override raise error."""

    class SimpleConfig(sp.Config):
        value: int = sp.Int(ge=0, le=100)

    with pytest.raises(ValueError, match="Invalid override keys"):
        SimpleConfig._node.apply_override({"value": {"min": 0, "max": 50}})


def test_conditional_override_invalid_keys():
    """Test that invalid keys in conditional override raise error."""

    class MainConfig(sp.Config):
        flag: bool
        value: int = sp.Conditional(
            condition=sp.FieldCondition("flag", sp.EqualsTo(True)),
            true=sp.Int(ge=0, le=100),
            false=0,
        )

    with pytest.raises(ValueError, match="Invalid override keys"):
        MainConfig._node.apply_override({"value": {"maybe": 50}})
