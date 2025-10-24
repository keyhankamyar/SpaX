import spax as sp


def test_fixed_node_signature():
    """Test signature generation for FixedNode."""

    class SimpleConfig(sp.Config):
        x: int = 5
        y: str = "hello"

    node = SimpleConfig._node

    # Check that fixed values appear in signature
    sig = node.get_signature()
    assert "Fixed(5)" in sig
    assert 'Fixed("hello")' in sig


def test_number_node_signature():
    """Test signature generation for NumberNode."""

    class SimpleConfig(sp.Config):
        x: int = sp.Int(ge=0, le=100)
        y: float = sp.Float(gt=0.0, lt=1.0, distribution="log")

    node = SimpleConfig._node
    sig = node.get_signature()

    # Check integer space
    assert "IntSpace" in sig
    assert "low=0" in sig
    assert "high=100" in sig
    assert "low_inclusive=True" in sig
    assert "high_inclusive=True" in sig

    # Check float space
    assert "FloatSpace" in sig
    assert "low=0.0" in sig
    assert "high=1.0" in sig
    assert "low_inclusive=False" in sig
    assert "high_inclusive=False" in sig
    assert "distribution=log" in sig


def test_categorical_node_signature():
    """Test signature generation for CategoricalNode."""

    class SimpleConfig(sp.Config):
        choice: str = sp.Categorical(["a", "b", "c"])

    node = SimpleConfig._node
    sig = node.get_signature()

    # Check that all choices appear
    assert "Categorical" in sig
    assert "a:Fixed" in sig
    assert "b:Fixed" in sig
    assert "c:Fixed" in sig


def test_categorical_config_signature():
    """Test signature for categorical with Config choices."""

    class ConfigA(sp.Config):
        a: int = sp.Int(ge=0, le=10)

    class ConfigB(sp.Config):
        b: float = sp.Float(ge=0.0, le=1.0)

    class MainConfig(sp.Config):
        choice: ConfigA | ConfigB

    node = MainConfig._node
    sig = node.get_signature()

    # Check that both config types appear with their internals
    assert "Categorical" in sig
    assert "ConfigA" in sig
    assert "ConfigB" in sig
    assert "IntSpace" in sig
    assert "FloatSpace" in sig


def test_conditional_node_signature():
    """Test signature generation for ConditionalNode."""

    class SimpleConfig(sp.Config):
        use_feature: bool
        value: int = sp.Conditional(
            condition=sp.FieldCondition("use_feature", sp.EqualsTo(True)),
            true=sp.Int(ge=0, le=100),
            false=0,
        )

    node = SimpleConfig._node
    sig = node.get_signature()

    # Check conditional structure
    assert "Conditional" in sig
    assert "condition=" in sig
    assert "true=" in sig
    assert "false=" in sig
    assert "IntSpace" in sig
    assert "Fixed(0)" in sig


def test_nested_config_signature():
    """Test signature for nested configs."""

    class InnerConfig(sp.Config):
        x: int = sp.Int(ge=0, le=10)

    class OuterConfig(sp.Config):
        inner: InnerConfig
        y: float = sp.Float(ge=0.0, le=1.0)

    node = OuterConfig._node
    sig = node.get_signature()

    # Check nested structure
    assert "OuterConfig" in sig
    assert "InnerConfig" in sig
    assert "inner=" in sig
    assert "y=" in sig


def test_signature_determinism():
    """Test that signatures are deterministic."""

    class SimpleConfig(sp.Config):
        x: int = sp.Int(ge=0, le=100)
        y: float = sp.Float(ge=0.0, le=1.0)
        choice: str = sp.Categorical(["a", "b", "c"])

    node = SimpleConfig._node

    # Get signature multiple times
    sig1 = node.get_signature()
    sig2 = node.get_signature()
    sig3 = node.get_signature()

    # All should be identical
    assert sig1 == sig2 == sig3


def test_signature_changes_with_override():
    """Test that signature changes when overrides are applied."""

    class SimpleConfig(sp.Config):
        x: int = sp.Int(ge=0, le=100)

    node1 = SimpleConfig._node
    node2 = SimpleConfig._node.apply_override({"x": {"ge": 10, "le": 50}})

    sig1 = node1.get_signature()
    sig2 = node2.get_signature()

    # Signatures should be different
    assert sig1 != sig2

    # Original should have wider bounds
    assert "low=0" in sig1
    assert "high=100" in sig1

    # Overridden should have narrower bounds
    assert "low=10" in sig2
    assert "high=50" in sig2


def test_hash_determinism():
    """Test that hashes are deterministic."""

    class SimpleConfig(sp.Config):
        x: int = sp.Int(ge=0, le=100)
        y: float = sp.Float(ge=0.0, le=1.0)

    node = SimpleConfig._node

    # Get hash multiple times
    hash1 = node.get_space_hash()
    hash2 = node.get_space_hash()
    hash3 = node.get_space_hash()

    # All should be identical
    assert hash1 == hash2 == hash3

    # Should be a valid SHA256 hash (64 hex chars)
    assert len(hash1) == 64
    assert all(c in "0123456789abcdef" for c in hash1)


def test_hash_changes_with_space():
    """Test that hash changes when space changes."""

    class Config1(sp.Config):
        x: int = sp.Int(ge=0, le=100)

    class Config2(sp.Config):
        x: int = sp.Int(ge=0, le=50)

    hash1 = Config1._node.get_space_hash()
    hash2 = Config2._node.get_space_hash()

    # Different spaces should have different hashes
    assert hash1 != hash2


def test_hash_with_override():
    """Test hash changes with overrides."""

    class SimpleConfig(sp.Config):
        x: int = sp.Int(ge=0, le=100)

    node1 = SimpleConfig._node
    node2 = node1.apply_override({"x": {"ge": 10, "le": 50}})

    hash1 = node1.get_space_hash()
    hash2 = node2.get_space_hash()

    # Different overrides should produce different hashes
    assert hash1 != hash2


def test_complex_nested_signature():
    """Test signature for complex nested structure."""

    class MLPConfig(sp.Config):
        hidden_dim: int = sp.Int(gt=16, lt=4096)
        activation: str = sp.Categorical(["relu", "gelu", "silu"])

    class CNNConfig(sp.Config):
        hidden_channels: int = sp.Int(gt=16, lt=4096)
        kernel_size: int = sp.Int(ge=1, le=64)

    class EncoderConfig(sp.Config):
        num_layers: int = sp.Int(ge=1, le=64, default=8)
        layer_config: MLPConfig | CNNConfig

    class ModelConfig(sp.Config):
        encoder: EncoderConfig
        use_head: bool
        head: MLPConfig | None = sp.Conditional(
            condition=sp.FieldCondition("use_head", sp.EqualsTo(True)),
            true=MLPConfig,
            false=None,
        )

    node = ModelConfig._node
    sig = node.get_signature()

    # Check all components appear
    assert "ModelConfig" in sig
    assert "EncoderConfig" in sig
    assert "MLPConfig" in sig
    assert "CNNConfig" in sig
    assert "Conditional" in sig

    # Should be deterministic
    sig2 = node.get_signature()
    assert sig == sig2


def test_weighted_categorical_signature():
    """Test signature includes weights for categorical."""

    class SimpleConfig(sp.Config):
        choice: str = sp.Categorical(
            [
                sp.Choice("a", weight=2.0),
                sp.Choice("b", weight=1.0),
                sp.Choice("c", weight=3.0),
            ]
        )

    node = SimpleConfig._node
    sig = node.get_signature()

    # Weights should be normalized to probabilities and included
    assert "weights=" in sig
    # Probabilities should sum to 1: [2/6, 1/6, 3/6] = [0.333..., 0.166..., 0.5]
    assert "0.3333" in sig or "0.16" in sig or "0.5" in sig


def test_signature_override_number_to_fixed():
    """Test signature changes when overriding number to fixed value."""

    class SimpleConfig(sp.Config):
        x: int = sp.Int(ge=0, le=100)

    node1 = SimpleConfig._node
    node2 = node1.apply_override({"x": 50})

    sig1 = node1.get_signature()
    sig2 = node2.get_signature()

    assert sig1 != sig2
    assert "IntSpace" in sig1
    assert "Fixed(50)" in sig2


def test_signature_override_number_narrow_range():
    """Test signature changes when narrowing numeric range."""

    class SimpleConfig(sp.Config):
        x: int = sp.Int(ge=0, le=100)

    node1 = SimpleConfig._node
    node2 = node1.apply_override({"x": {"ge": 20, "le": 80}})
    node3 = node1.apply_override({"x": {"ge": 10, "le": 90}})

    sig1 = node1.get_signature()
    sig2 = node2.get_signature()
    sig3 = node3.get_signature()

    # All should be different
    assert sig1 != sig2
    assert sig1 != sig3
    assert sig2 != sig3

    # Check bounds in signatures
    assert "low=0" in sig1 and "high=100" in sig1
    assert "low=20" in sig2 and "high=80" in sig2
    assert "low=10" in sig3 and "high=90" in sig3


def test_signature_override_categorical_to_fixed():
    """Test signature changes when fixing categorical choice."""

    class SimpleConfig(sp.Config):
        choice: str = sp.Categorical(["a", "b", "c"])

    node1 = SimpleConfig._node
    node2 = node1.apply_override({"choice": "b"})

    sig1 = node1.get_signature()
    sig2 = node2.get_signature()

    assert sig1 != sig2
    assert "Categorical" in sig1
    assert 'Fixed("b")' in sig2


def test_signature_override_categorical_subset():
    """Test signature changes when subsetting categorical choices."""

    class SimpleConfig(sp.Config):
        choice: str = sp.Categorical(["a", "b", "c", "d"])

    node1 = SimpleConfig._node
    node2 = node1.apply_override({"choice": ["a", "c"]})
    node3 = node1.apply_override({"choice": ["b", "d"]})

    sig1 = node1.get_signature()
    sig2 = node2.get_signature()
    sig3 = node3.get_signature()

    # All should be different
    assert sig1 != sig2
    assert sig1 != sig3
    assert sig2 != sig3

    # Check choices
    assert (
        "a:Fixed" in sig1
        and "b:Fixed" in sig1
        and "c:Fixed" in sig1
        and "d:Fixed" in sig1
    )
    assert "a:Fixed" in sig2 and "c:Fixed" in sig2
    assert "b:Fixed" in sig3 and "d:Fixed" in sig3


def test_signature_override_nested_config():
    """Test signature changes when overriding nested config fields."""

    class InnerConfig(sp.Config):
        x: int = sp.Int(ge=0, le=100)
        y: int = sp.Int(ge=0, le=100)

    class OuterConfig(sp.Config):
        inner: InnerConfig

    node1 = OuterConfig._node
    node2 = node1.apply_override({"inner": {"x": 50}})
    node3 = node1.apply_override({"inner": {"y": 75}})
    node4 = node1.apply_override({"inner": {"x": 50, "y": 75}})

    sig1 = node1.get_signature()
    sig2 = node2.get_signature()
    sig3 = node3.get_signature()
    sig4 = node4.get_signature()

    # All should be different
    assert sig1 != sig2
    assert sig1 != sig3
    assert sig1 != sig4
    assert sig2 != sig3
    assert sig2 != sig4
    assert sig3 != sig4


def test_signature_override_categorical_config_fix_choice():
    """Test signature when fixing categorical config choice."""

    class ConfigA(sp.Config):
        a: int = sp.Int(ge=0, le=10)

    class ConfigB(sp.Config):
        b: float = sp.Float(ge=0.0, le=1.0)

    class MainConfig(sp.Config):
        choice: ConfigA | ConfigB

    node1 = MainConfig._node
    node2 = node1.apply_override({"choice": "ConfigA"})
    node3 = node1.apply_override({"choice": "ConfigB"})

    sig1 = node1.get_signature()
    sig2 = node2.get_signature()
    sig3 = node3.get_signature()

    # All different
    assert sig1 != sig2
    assert sig1 != sig3
    assert sig2 != sig3

    # Original has both
    assert "ConfigA" in sig1 and "ConfigB" in sig1
    # Overridden have only one choice each
    assert "ConfigA" in sig2 and "ConfigB" not in sig2
    assert "ConfigB" in sig3 and "ConfigA" not in sig3


def test_signature_override_categorical_config_with_nested_override():
    """Test signature when overriding categorical config with nested overrides."""

    class ConfigA(sp.Config):
        a: int = sp.Int(ge=0, le=100)

    class ConfigB(sp.Config):
        b: float = sp.Float(ge=0.0, le=1.0)

    class MainConfig(sp.Config):
        choice: ConfigA | ConfigB

    node1 = MainConfig._node
    node2 = node1.apply_override({"choice": {"ConfigA": {"a": {"ge": 10, "le": 50}}}})
    node3 = node1.apply_override({"choice": {"ConfigB": {"b": {"ge": 0.2, "le": 0.8}}}})
    node4 = node1.apply_override(
        {
            "choice": {
                "ConfigA": {"a": {"ge": 10, "le": 50}},
                "ConfigB": {"b": {"ge": 0.2, "le": 0.8}},
            }
        }
    )

    sig1 = node1.get_signature()
    sig2 = node2.get_signature()
    sig3 = node3.get_signature()
    sig4 = node4.get_signature()

    # All should be different
    assert len({sig1, sig2, sig3, sig4}) == 4

    # Check bounds
    assert "low=0" in sig1 and "high=100" in sig1
    assert "low=10" in sig2 and "high=50" in sig2
    assert "low=0.2" in sig3 and "high=0.8" in sig3


def test_signature_override_conditional_branches():
    """Test signature changes when overriding conditional branches."""

    class SimpleConfig(sp.Config):
        use_feature: bool
        value: int = sp.Conditional(
            condition=sp.FieldCondition("use_feature", sp.EqualsTo(True)),
            true=sp.Int(ge=0, le=100),
            false=0,
        )

    node1 = SimpleConfig._node
    node2 = node1.apply_override({"value": {"true": {"ge": 10, "le": 50}}})
    node3 = node1.apply_override({"value": {"true": 25}})

    sig1 = node1.get_signature()
    sig2 = node2.get_signature()
    sig3 = node3.get_signature()

    # All different
    assert sig1 != sig2
    assert sig1 != sig3
    assert sig2 != sig3

    # Check changes
    assert "low=0" in sig1 and "high=100" in sig1
    assert "low=10" in sig2 and "high=50" in sig2
    assert "Fixed(25)" in sig3


def test_signature_override_multiple_fields():
    """Test signature when overriding multiple fields."""

    class SimpleConfig(sp.Config):
        x: int = sp.Int(ge=0, le=100)
        y: float = sp.Float(ge=0.0, le=1.0)
        choice: str = sp.Categorical(["a", "b", "c"])

    node1 = SimpleConfig._node
    node2 = node1.apply_override({"x": 50})
    node3 = node1.apply_override({"y": 0.5})
    node4 = node1.apply_override({"choice": "b"})
    node5 = node1.apply_override({"x": 50, "y": 0.5})
    node6 = node1.apply_override({"x": 50, "choice": "b"})
    node7 = node1.apply_override({"y": 0.5, "choice": "b"})
    node8 = node1.apply_override({"x": 50, "y": 0.5, "choice": "b"})

    sigs = [
        node1.get_signature(),
        node2.get_signature(),
        node3.get_signature(),
        node4.get_signature(),
        node5.get_signature(),
        node6.get_signature(),
        node7.get_signature(),
        node8.get_signature(),
    ]

    # All should be unique
    assert len(set(sigs)) == 8


def test_hash_override_consistency():
    """Test that same override produces same hash."""

    class SimpleConfig(sp.Config):
        x: int = sp.Int(ge=0, le=100)
        y: float = sp.Float(ge=0.0, le=1.0)

    override = {"x": {"ge": 10, "le": 50}, "y": {"ge": 0.2, "le": 0.8}}

    node1 = SimpleConfig._node.apply_override(override)
    node2 = SimpleConfig._node.apply_override(override)

    hash1 = node1.get_space_hash()
    hash2 = node2.get_space_hash()

    # Same override should produce same hash
    assert hash1 == hash2


def test_hash_override_order_independence():
    """Test that override dict key order doesn't affect hash."""

    class SimpleConfig(sp.Config):
        x: int = sp.Int(ge=0, le=100)
        y: float = sp.Float(ge=0.0, le=1.0)

    # Same overrides in different order
    override1 = {"x": 50, "y": 0.5}
    override2 = {"y": 0.5, "x": 50}

    node1 = SimpleConfig._node.apply_override(override1)
    node2 = SimpleConfig._node.apply_override(override2)

    hash1 = node1.get_space_hash()
    hash2 = node2.get_space_hash()

    # Should produce same hash
    assert hash1 == hash2


def test_signature_deep_nested_override():
    """Test signature with deeply nested overrides."""

    class Level3Config(sp.Config):
        z: int = sp.Int(ge=0, le=10)

    class Level2Config(sp.Config):
        y: int = sp.Int(ge=0, le=20)
        level3: Level3Config

    class Level1Config(sp.Config):
        x: int = sp.Int(ge=0, le=30)
        level2: Level2Config

    node1 = Level1Config._node
    node2 = node1.apply_override({"level2": {"level3": {"z": 5}}})
    node3 = node1.apply_override({"x": 15, "level2": {"y": 10, "level3": {"z": 5}}})

    sig1 = node1.get_signature()
    sig2 = node2.get_signature()
    sig3 = node3.get_signature()

    # All different
    assert sig1 != sig2
    assert sig1 != sig3
    assert sig2 != sig3


def test_signature_override_preserves_unmodified_fields():
    """Test that overriding one field doesn't change signature of others."""

    class SimpleConfig(sp.Config):
        x: int = sp.Int(ge=0, le=100)
        y: int = sp.Int(ge=0, le=100)

    node1 = SimpleConfig._node
    node2 = node1.apply_override({"x": 50})

    sig1 = node1.get_signature()
    sig2 = node2.get_signature()

    # y's signature should be identical in both
    assert (
        "y=IntSpace(low=0, high=100, low_inclusive=True, high_inclusive=True, distribution=uniform)"
        in sig1
    )
    assert (
        "y=IntSpace(low=0, high=100, low_inclusive=True, high_inclusive=True, distribution=uniform)"
        in sig2
    )
