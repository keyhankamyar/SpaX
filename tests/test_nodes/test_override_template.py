import spax as sp


def test_fixed_node_template():
    """Test that FixedNode returns None for template."""

    class SimpleConfig(sp.Config):
        x: int = 5

    template = SimpleConfig.get_override_template()

    # Fixed values should not appear in template (None is filtered out)
    assert "x" not in template


def test_number_node_template():
    """Test override template for numeric spaces."""

    class SimpleConfig(sp.Config):
        x: int = sp.Int(ge=0, le=100)
        y: float = sp.Float(gt=0.0, lt=1.0)

    template = SimpleConfig.get_override_template()

    # Check integer template
    assert "x" in template
    assert "ge" in template["x"]
    assert "le" in template["x"]
    assert template["x"]["ge"] == 0
    assert template["x"]["le"] == 100

    # Check float template
    assert "y" in template
    assert "gt" in template["y"]
    assert "lt" in template["y"]
    assert template["y"]["gt"] == 0.0
    assert template["y"]["lt"] == 1.0


def test_categorical_simple_template():
    """Test override template for simple categorical."""

    class SimpleConfig(sp.Config):
        choice: str = sp.Categorical(["a", "b", "c"])

    template = SimpleConfig.get_override_template()

    # Should be a list of choices
    assert "choice" in template
    assert template["choice"] == ["a", "b", "c"]


def test_categorical_config_template():
    """Test override template for categorical with Config choices."""

    class ConfigA(sp.Config):
        a: int = sp.Int(ge=0, le=10)

    class ConfigB(sp.Config):
        b: float = sp.Float(ge=0.0, le=1.0)

    class MainConfig(sp.Config):
        choice: ConfigA | ConfigB

    template = MainConfig.get_override_template()

    # Should be a dict with config names as keys
    assert "choice" in template
    assert isinstance(template["choice"], dict)
    assert "ConfigA" in template["choice"]
    assert "ConfigB" in template["choice"]

    # Check nested templates
    assert "a" in template["choice"]["ConfigA"]
    assert "b" in template["choice"]["ConfigB"]


def test_conditional_template():
    """Test override template for conditional spaces."""

    class SimpleConfig(sp.Config):
        use_feature: bool
        value: int = sp.Conditional(
            condition=sp.FieldCondition("use_feature", sp.EqualsTo(True)),
            true=sp.Int(ge=0, le=100),
            false=0,
        )

    template = SimpleConfig.get_override_template()

    # Conditional should have true/false branches
    assert "value" in template
    assert isinstance(template["value"], dict)
    assert "true" in template["value"]
    # false branch is Fixed, so should not appear
    assert "false" not in template["value"]

    # True branch should have numeric template
    assert "ge" in template["value"]["true"]
    assert "le" in template["value"]["true"]


def test_nested_config_template():
    """Test override template for nested configs."""

    class InnerConfig(sp.Config):
        x: int = sp.Int(ge=0, le=10)
        y: float = sp.Float(ge=0.0, le=1.0)

    class OuterConfig(sp.Config):
        inner: InnerConfig
        z: int = sp.Int(ge=0, le=100)

    template = OuterConfig.get_override_template()

    # Should have nested structure
    assert "inner" in template
    assert "z" in template

    # Inner should have its own template
    assert "x" in template["inner"]
    assert "y" in template["inner"]


def test_complex_template():
    """Test override template for complex nested structure."""

    class MLPConfig(sp.Config):
        hidden_dim: int = sp.Int(gt=16, lt=4096)
        activation: str = sp.Categorical(["relu", "gelu", "silu"])

    class CNNConfig(sp.Config):
        hidden_channels: int = sp.Int(gt=16, lt=4096)
        kernel_size: int = sp.Int(ge=1, le=64)

    class EncoderConfig(sp.Config):
        num_layers: int = sp.Int(ge=1, le=64)
        layer_config: MLPConfig | CNNConfig

    class ModelConfig(sp.Config):
        encoder: EncoderConfig
        use_head: bool
        head: MLPConfig | None = sp.Conditional(
            condition=sp.FieldCondition("use_head", sp.EqualsTo(True)),
            true=MLPConfig,
            false=None,
        )

    template = ModelConfig.get_override_template()

    # Check top-level structure
    assert "encoder" in template
    assert "use_head" in template
    assert "head" in template

    # Check encoder structure
    assert "num_layers" in template["encoder"]
    assert "layer_config" in template["encoder"]
    assert isinstance(template["encoder"]["layer_config"], dict)
    assert "MLPConfig" in template["encoder"]["layer_config"]
    assert "CNNConfig" in template["encoder"]["layer_config"]

    # Check conditional head
    assert "true" in template["head"]
    assert "hidden_dim" in template["head"]["true"]


def test_template_can_create_valid_override():
    """Test that template structure can be used to create valid overrides."""

    class SimpleConfig(sp.Config):
        x: int = sp.Int(ge=0, le=100)
        y: str = sp.Categorical(["a", "b", "c"])

    # Get template
    template = SimpleConfig.get_override_template()

    # Create override based on template structure
    override = template.copy()
    override["x"] = {"ge": 10, "le": 50}
    override["y"] = ["a", "b"]

    # Should work without errors
    config = SimpleConfig.random(seed=42, override=override)

    # Verify constraints
    assert 10 <= config.x <= 50
    assert config.y in ["a", "b"]


def test_template_with_mixed_spaces():
    """Test template with mix of different space types."""

    class MixedConfig(sp.Config):
        fixed_value: int = 42
        int_space: int = sp.Int(ge=0, le=100)
        float_space: float = sp.Float(ge=0.0, le=1.0)
        categorical: str = sp.Categorical(["a", "b", "c"])
        boolean: bool

    template = MixedConfig.get_override_template()

    # Fixed values should not appear
    assert "fixed_value" not in template

    # Spaces should appear
    assert "int_space" in template
    assert "float_space" in template
    assert "categorical" in template
    assert "boolean" in template

    # Check types
    assert isinstance(template["int_space"], dict)
    assert isinstance(template["float_space"], dict)
    assert isinstance(template["categorical"], list)
    assert isinstance(template["boolean"], list)


def test_empty_template_for_all_fixed():
    """Test that config with only fixed values has empty template."""

    class AllFixedConfig(sp.Config):
        x: int = 5
        y: str = "hello"
        z: float = 3.14

    template = AllFixedConfig.get_override_template()

    # Should be empty or have no tunable fields
    assert template == {}


def test_template_preserves_original_bounds():
    """Test that template shows original bounds correctly."""

    class SimpleConfig(sp.Config):
        x: int = sp.Int(gt=10, lt=100)

    template = SimpleConfig.get_override_template()

    # Should use gt/lt (not ge/le) since original uses exclusive bounds
    assert "gt" in template["x"]
    assert "lt" in template["x"]
    assert template["x"]["gt"] == 10
    assert template["x"]["lt"] == 100


def test_template_roundtrip_no_changes():
    """Test that unmodified template doesn't change the space."""

    class SimpleConfig(sp.Config):
        x: int = sp.Int(ge=0, le=100)
        y: str = sp.Categorical(["a", "b", "c"])

    # Get template and use as-is (no modifications)
    template = SimpleConfig.get_override_template()

    # Get nodes
    original_node = SimpleConfig.get_node()
    template_node = SimpleConfig.get_node(override=template)

    # Hashes should be identical (no change)
    assert original_node.get_space_hash() == template_node.get_space_hash()


def test_template_roundtrip_narrow_numeric():
    """Test narrowing numeric bounds via template."""

    class SimpleConfig(sp.Config):
        x: int = sp.Int(ge=0, le=100)
        y: float = sp.Float(gt=0.0, lt=10.0)

    # Get template
    template = SimpleConfig.get_override_template()

    # Modify the template
    template["x"]["ge"] = 20
    template["x"]["le"] = 80
    template["y"]["gt"] = 1.0
    template["y"]["lt"] = 5.0

    # Apply modified template
    node = SimpleConfig.get_node(override=template)

    # Check narrowed ranges
    from spax.nodes import NumberNode

    assert isinstance(node._children["x"], NumberNode)
    assert node._children["x"]._space.low == 20
    assert node._children["x"]._space.high == 80

    assert isinstance(node._children["y"], NumberNode)
    assert node._children["y"]._space.low == 1.0
    assert node._children["y"]._space.high == 5.0


def test_template_roundtrip_fix_numeric():
    """Test fixing numeric value by replacing template with value."""

    class SimpleConfig(sp.Config):
        x: int = sp.Int(ge=0, le=100)

    # Get template
    template = SimpleConfig.get_override_template()

    # Replace dict with fixed value
    template["x"] = 42

    node = SimpleConfig.get_node(override=template)

    # Should be FixedNode now
    from spax.nodes import FixedNode

    assert isinstance(node._children["x"], FixedNode)
    assert node._children["x"].default == 42


def test_template_roundtrip_subset_categorical():
    """Test subsetting categorical via template."""

    class SimpleConfig(sp.Config):
        choice: str = sp.Categorical(["a", "b", "c", "d"])

    # Get template (should be list of choices)
    template = SimpleConfig.get_override_template()
    assert template["choice"] == ["a", "b", "c", "d"]

    # Modify to subset
    template["choice"] = ["a", "c"]

    node = SimpleConfig.get_node(override=template)

    # Should still be CategoricalNode with subset
    from spax.nodes import CategoricalNode

    assert isinstance(node._children["choice"], CategoricalNode)
    assert node._children["choice"]._space.choices == ["a", "c"]


def test_template_roundtrip_fix_categorical():
    """Test fixing categorical by replacing list with single value."""

    class SimpleConfig(sp.Config):
        choice: str = sp.Categorical(["a", "b", "c"])

    # Get template
    template = SimpleConfig.get_override_template()

    # Replace list with single value
    template["choice"] = "b"

    node = SimpleConfig.get_node(override=template)

    # Should be FixedNode now
    from spax.nodes import FixedNode

    assert isinstance(node._children["choice"], FixedNode)
    assert node._children["choice"].default == "b"


def test_template_roundtrip_nested_config():
    """Test modifying nested config via template."""

    class InnerConfig(sp.Config):
        x: int = sp.Int(ge=0, le=100)
        y: float = sp.Float(ge=0.0, le=1.0)

    class OuterConfig(sp.Config):
        inner: InnerConfig
        z: int = sp.Int(ge=0, le=50)

    # Get template
    template = OuterConfig.get_override_template()

    # Modify using the template structure
    template["inner"]["x"]["ge"] = 10
    template["inner"]["x"]["le"] = 90
    template["z"] = 25  # Fix to value

    node = OuterConfig.get_node(override=template)

    # Check nested modification
    inner_node = node._children["inner"]
    from spax.nodes import FixedNode, NumberNode

    assert isinstance(inner_node._children["x"], NumberNode)
    assert inner_node._children["x"]._space.low == 10
    assert inner_node._children["x"]._space.high == 90

    # Check top-level fixed
    assert isinstance(node._children["z"], FixedNode)
    assert node._children["z"].default == 25


def test_template_roundtrip_categorical_configs():
    """Test modifying categorical with config choices via template."""

    class ConfigA(sp.Config):
        a: int = sp.Int(ge=0, le=100)

    class ConfigB(sp.Config):
        b: float = sp.Float(ge=0.0, le=10.0)

    class MainConfig(sp.Config):
        choice: ConfigA | ConfigB

    # Get template (should be dict with config names)
    template = MainConfig.get_override_template()
    assert "ConfigA" in template["choice"]
    assert "ConfigB" in template["choice"]

    # Modify nested configs using template
    template["choice"]["ConfigA"]["a"]["ge"] = 10
    template["choice"]["ConfigA"]["a"]["le"] = 50
    template["choice"]["ConfigB"]["b"]["ge"] = 1.0
    template["choice"]["ConfigB"]["b"]["le"] = 5.0

    node = MainConfig.get_node(override=template)

    # Should still be CategoricalNode
    from spax.nodes import CategoricalNode

    assert isinstance(node._children["choice"], CategoricalNode)

    # Check nested modifications
    config_a_node = node._children["choice"]._children["ConfigA"]
    assert config_a_node._children["a"]._space.low == 10
    assert config_a_node._children["a"]._space.high == 50

    config_b_node = node._children["choice"]._children["ConfigB"]
    assert config_b_node._children["b"]._space.low == 1.0
    assert config_b_node._children["b"]._space.high == 5.0


def test_template_roundtrip_fix_categorical_config():
    """Test fixing categorical config choice by keeping only one in template."""

    class ConfigA(sp.Config):
        a: int = sp.Int(ge=0, le=100)

    class ConfigB(sp.Config):
        b: float = sp.Float(ge=0.0, le=10.0)

    class MainConfig(sp.Config):
        choice: ConfigA | ConfigB

    # Get template
    template = MainConfig.get_override_template()

    # Keep only ConfigA and modify it
    template["choice"] = {"ConfigA": {"a": {"ge": 20, "le": 80}}}

    node = MainConfig.get_node(override=template)

    # Should be ConfigNode now (fixed to ConfigA)
    from spax.nodes import ConfigNode

    assert isinstance(node._children["choice"], ConfigNode)
    assert node._children["choice"]._config_class == ConfigA

    # Check nested modification
    assert node._children["choice"]._children["a"]._space.low == 20


def test_template_roundtrip_conditional():
    """Test modifying conditional branches via template."""

    class InnerConfig(sp.Config):
        value: int = sp.Int(ge=0, le=100)

    class MainConfig(sp.Config):
        use_feature: bool
        feature: InnerConfig | None = sp.Conditional(
            condition=sp.FieldCondition("use_feature", sp.EqualsTo(True)),
            true=InnerConfig,
            false=None,
        )

    # Get template
    template = MainConfig.get_override_template()

    # Template should have true branch (false is None/Fixed)
    assert "true" in template["feature"]

    # Modify true branch using template
    template["feature"]["true"]["value"]["ge"] = 10
    template["feature"]["true"]["value"]["le"] = 50

    node = MainConfig.get_node(override=template)

    # Should still be ConditionalNode
    from spax.nodes import ConditionalNode

    assert isinstance(node._children["feature"], ConditionalNode)

    # Check true branch modification
    true_node = node._children["feature"]._true_node
    assert true_node._children["value"]._space.low == 10
    assert true_node._children["value"]._space.high == 50


def test_template_roundtrip_complex_nested():
    """Test complex nested structure with multiple modifications using template."""

    class MLPConfig(sp.Config):
        hidden_dim: int = sp.Int(gt=16, lt=4096)
        activation: str = sp.Categorical(["relu", "gelu", "silu", "tanh"])
        dropout: float = sp.Float(ge=0.0, le=0.9)

    class CNNConfig(sp.Config):
        hidden_channels: int = sp.Int(gt=16, lt=4096)
        kernel_size: int = sp.Int(ge=1, le=64)
        activation: str = sp.Categorical(["relu", "gelu"])

    class EncoderConfig(sp.Config):
        num_layers: int = sp.Int(ge=1, le=64)
        layer_config: MLPConfig | CNNConfig

    class ModelConfig(sp.Config):
        encoder: EncoderConfig
        use_head: bool
        head: MLPConfig | None = sp.Conditional(
            condition=sp.FieldCondition("use_head", sp.EqualsTo(True)),
            true=MLPConfig,
            false=None,
        )

    # Get template
    template = ModelConfig.get_override_template()

    # Modify template extensively
    template["encoder"]["num_layers"]["ge"] = 4
    template["encoder"]["num_layers"]["le"] = 32

    template["encoder"]["layer_config"]["MLPConfig"]["hidden_dim"]["gt"] = 64
    template["encoder"]["layer_config"]["MLPConfig"]["hidden_dim"]["lt"] = 2048
    template["encoder"]["layer_config"]["MLPConfig"]["activation"] = ["relu", "gelu"]

    template["encoder"]["layer_config"]["CNNConfig"]["kernel_size"]["ge"] = 3
    template["encoder"]["layer_config"]["CNNConfig"]["kernel_size"]["le"] = 7

    template["use_head"] = [True, False]

    template["head"]["true"]["dropout"]["ge"] = 0.1
    template["head"]["true"]["dropout"]["le"] = 0.5

    node = ModelConfig.get_node(override=template)

    # Verify encoder modifications
    encoder_node = node._children["encoder"]
    assert encoder_node._children["num_layers"]._space.low == 4
    assert encoder_node._children["num_layers"]._space.high == 32

    # Verify MLP modifications
    layer_config_node = encoder_node._children["layer_config"]
    mlp_node = layer_config_node._children["MLPConfig"]
    assert mlp_node._children["hidden_dim"]._space.low == 64
    assert mlp_node._children["hidden_dim"]._space.high == 2048
    assert mlp_node._children["activation"]._space.choices == ["relu", "gelu"]

    # Verify CNN modifications
    cnn_node = layer_config_node._children["CNNConfig"]
    assert cnn_node._children["kernel_size"]._space.low == 3
    assert cnn_node._children["kernel_size"]._space.high == 7

    # Verify head conditional modifications
    from spax.nodes import ConditionalNode

    assert isinstance(node._children["head"], ConditionalNode)
    head_true_node = node._children["head"]._true_node
    assert head_true_node._children["dropout"]._space.low == 0.1
    assert head_true_node._children["dropout"]._space.high == 0.5


def test_template_partial_modification():
    """Test that modifying only some fields in template works."""

    class ComplexConfig(sp.Config):
        a: int = sp.Int(ge=0, le=100)
        b: float = sp.Float(ge=0.0, le=1.0)
        c: str = sp.Categorical(["x", "y", "z"])

    # Get template
    template = ComplexConfig.get_override_template()

    # Only modify one field, leave others as-is
    template["a"]["ge"] = 20
    template["a"]["le"] = 80
    # Don't modify b and c

    node = ComplexConfig.get_node(override=template)

    # a should be modified
    from spax.nodes import NumberNode

    assert isinstance(node._children["a"], NumberNode)
    assert node._children["a"]._space.low == 20

    # b should be unchanged (template keeps original bounds)
    assert node._children["b"]._space.low == 0.0
    assert node._children["b"]._space.high == 1.0

    # c should be unchanged (template keeps all choices)
    assert node._children["c"]._space.choices == ["x", "y", "z"]


def test_template_remove_some_choices():
    """Test removing some categorical choices from template list."""

    class SimpleConfig(sp.Config):
        choice: str = sp.Categorical(["a", "b", "c", "d", "e"])

    # Get template
    template = SimpleConfig.get_override_template()
    assert len(template["choice"]) == 5

    # Remove some choices from the list
    template["choice"].remove("b")
    template["choice"].remove("d")

    node = SimpleConfig.get_node(override=template)

    # Should have only 3 choices now
    from spax.nodes import CategoricalNode

    assert isinstance(node._children["choice"], CategoricalNode)
    assert node._children["choice"]._space.choices == ["a", "c", "e"]


def test_template_sampling_with_modifications():
    """Test that modified template produces valid samples."""

    class TestConfig(sp.Config):
        x: int = sp.Int(ge=0, le=100)
        y: float = sp.Float(ge=0.0, le=10.0)
        choice: str = sp.Categorical(["a", "b", "c", "d"])

    # Get and modify template
    template = TestConfig.get_override_template()
    template["x"]["ge"] = 20
    template["x"]["le"] = 80
    template["y"]["ge"] = 1.0
    template["y"]["le"] = 5.0
    template["choice"] = ["a", "b"]

    # Sample multiple times
    for seed in range(10):
        config = TestConfig.random(seed=seed, override=template)

        # Verify all constraints from modified template
        assert 20 <= config.x <= 80
        assert 1.0 <= config.y <= 5.0
        assert config.choice in ["a", "b"]
