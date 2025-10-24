import spax as sp


def test_parameter_names_simple():
    """Test parameter name generation for simple config."""

    class SimpleConfig(sp.Config):
        x: int = sp.Int(ge=0, le=100)
        y: float = sp.Float(ge=0.0, le=1.0)
        choice: str = sp.Categorical(["a", "b", "c"])

    names = set(SimpleConfig.get_parameter_names())
    assert names == {
        "SimpleConfig.x",
        "SimpleConfig.y",
        "SimpleConfig.choice",
    }


def test_parameter_names_nested():
    """Test parameter name generation with nested configs."""

    class InnerConfig(sp.Config):
        value: int = sp.Int(ge=0, le=10)

    class OuterConfig(sp.Config):
        x: int = sp.Int(ge=0, le=100)
        inner: InnerConfig

    names = set(OuterConfig.get_parameter_names())
    assert names == {
        "OuterConfig.x",
        "OuterConfig.inner::InnerConfig.value",
    }


def test_parameter_names_categorical_config():
    """Test parameter name generation with categorical config choices."""

    class ConfigA(sp.Config):
        a: int = sp.Int(ge=0, le=10)

    class ConfigB(sp.Config):
        b: float = sp.Float(ge=0.0, le=1.0)

    class MainConfig(sp.Config):
        choice: ConfigA | ConfigB

    names = set(MainConfig.get_parameter_names())
    expected = {
        "MainConfig.choice",
        "MainConfig.choice::ConfigA.a",
        "MainConfig.choice::ConfigB.b",
    }
    assert names == expected


def test_parameter_names_deeply_nested():
    """Test parameter name generation with deeply nested configs."""

    class Level3Config(sp.Config):
        z: int = sp.Int(ge=0, le=10)

    class Level2Config(sp.Config):
        y: float = sp.Float(ge=0.0, le=1.0)
        level3: Level3Config

    class Level1Config(sp.Config):
        x: int = sp.Int(ge=0, le=100)
        level2: Level2Config

    names = set(Level1Config.get_parameter_names())
    expected = {
        "Level1Config.x",
        "Level1Config.level2::Level2Config.y",
        "Level1Config.level2::Level2Config.level3::Level3Config.z",
    }
    assert names == expected


def test_parameter_names_conditional():
    """Test parameter name generation with conditional spaces."""

    class ConfigA(sp.Config):
        a: int = sp.Int(ge=0, le=10)

    class MainConfig(sp.Config):
        use_feature: bool = sp.Categorical([True, False])
        feature_config: ConfigA | None = sp.Conditional(
            condition=sp.FieldCondition("use_feature", sp.EqualsTo(True)),
            true=ConfigA,
            false=None,
        )

    names = set(MainConfig.get_parameter_names())
    expected = {
        "MainConfig.use_feature",
        "MainConfig.feature_config::true_branch::ConfigA.a",
    }
    assert names == expected


def test_parameter_names_complex_nested():
    """Test parameter name generation with complex nesting including categoricals."""

    class MLPConfig(sp.Config):
        hidden_dim: int = sp.Int(ge=16, le=512)
        activation: str = sp.Categorical(["relu", "gelu"])

    class CNNConfig(sp.Config):
        channels: int = sp.Int(ge=8, le=256)
        kernel_size: int = sp.Int(ge=1, le=7)

    class EncoderConfig(sp.Config):
        num_layers: int = sp.Int(ge=1, le=10)
        layer_config: MLPConfig | CNNConfig

    class ModelConfig(sp.Config):
        learning_rate: float = sp.Float(ge=1e-5, le=1e-1)
        encoder: EncoderConfig

    names = set(ModelConfig.get_parameter_names())
    expected = {
        "ModelConfig.learning_rate",
        "ModelConfig.encoder::EncoderConfig.num_layers",
        "ModelConfig.encoder::EncoderConfig.layer_config",
        "ModelConfig.encoder::EncoderConfig.layer_config::MLPConfig.hidden_dim",
        "ModelConfig.encoder::EncoderConfig.layer_config::MLPConfig.activation",
        "ModelConfig.encoder::EncoderConfig.layer_config::CNNConfig.channels",
        "ModelConfig.encoder::EncoderConfig.layer_config::CNNConfig.kernel_size",
    }
    assert names == expected


def test_parameter_names_with_inheritance():
    """Test parameter name generation with config inheritance."""

    class BaseConfig(sp.Config):
        base_param: int = sp.Int(ge=0, le=10)

    class DerivedConfig(BaseConfig):
        derived_param: float = sp.Float(ge=0.0, le=1.0)

    names = set(DerivedConfig.get_parameter_names())
    expected = {
        "DerivedConfig.base_param",
        "DerivedConfig.derived_param",
    }
    assert names == expected


def test_parameter_names_inheritance_with_override():
    """Test parameter name generation when derived class overrides parent field."""

    class BaseConfig(sp.Config):
        x: int = sp.Int(ge=0, le=100)
        y: int = sp.Int(ge=0, le=10)

    class DerivedConfig(BaseConfig):
        y: int = sp.Int(ge=5, le=20)  # Override with different bounds
        z: float = sp.Float(ge=0.0, le=1.0)

    names = set(DerivedConfig.get_parameter_names())
    expected = {
        "DerivedConfig.x",
        "DerivedConfig.y",
        "DerivedConfig.z",
    }
    assert names == expected


def test_parameter_names_multilevel_inheritance():
    """Test parameter name generation with multiple inheritance levels."""

    class Level1Config(sp.Config):
        a: int = sp.Int(ge=0, le=10)

    class Level2Config(Level1Config):
        b: int = sp.Int(ge=0, le=20)

    class Level3Config(Level2Config):
        c: int = sp.Int(ge=0, le=30)

    names = set(Level3Config.get_parameter_names())
    expected = {
        "Level3Config.a",
        "Level3Config.b",
        "Level3Config.c",
    }
    assert names == expected


def test_parameter_names_inheritance_with_nested():
    """Test parameter name generation with inheritance and nested configs."""

    class InnerConfig(sp.Config):
        value: int = sp.Int(ge=0, le=10)

    class BaseConfig(sp.Config):
        base_x: int = sp.Int(ge=0, le=100)

    class DerivedConfig(BaseConfig):
        derived_y: float = sp.Float(ge=0.0, le=1.0)
        inner: InnerConfig

    names = set(DerivedConfig.get_parameter_names())
    expected = {
        "DerivedConfig.base_x",
        "DerivedConfig.derived_y",
        "DerivedConfig.inner::InnerConfig.value",
    }
    assert names == expected


def test_parameter_names_multiple_conditionals():
    """Test parameter name generation with multiple conditional branches."""

    class FeatureA(sp.Config):
        param_a: int = sp.Int(ge=0, le=10)

    class FeatureB(sp.Config):
        param_b: float = sp.Float(ge=0.0, le=1.0)

    class MainConfig(sp.Config):
        use_a: bool = sp.Categorical([True, False])
        use_b: bool = sp.Categorical([True, False])

        feature_a: FeatureA | None = sp.Conditional(
            condition=sp.FieldCondition("use_a", sp.EqualsTo(True)),
            true=FeatureA,
            false=None,
        )

        feature_b: FeatureB | None = sp.Conditional(
            condition=sp.FieldCondition("use_b", sp.EqualsTo(True)),
            true=FeatureB,
            false=None,
        )

    names = set(MainConfig.get_parameter_names())
    expected = {
        "MainConfig.use_a",
        "MainConfig.use_b",
        "MainConfig.feature_a::true_branch::FeatureA.param_a",
        "MainConfig.feature_b::true_branch::FeatureB.param_b",
    }
    assert names == expected
