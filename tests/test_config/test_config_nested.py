"""Tests for nested Config structures."""

import pytest

from spax import Categorical, Conditional, Config, Float, Int
from spax.spaces import EqualsTo, FieldCondition, SmallerThan


class TestBasicNesting:
    """Test basic nested config structures."""

    def test_one_level_nesting(self):
        """Test simple one-level nesting."""

        class InnerConfig(Config):
            value: int = 10

        class OuterConfig(Config):
            inner: InnerConfig

        inner = InnerConfig()
        outer = OuterConfig(inner=inner)

        assert outer.inner.value == 10

    def test_two_level_nesting(self):
        """Test two-level nesting."""

        class Level2(Config):
            value: int = 20

        class Level1(Config):
            level2: Level2

        class Level0(Config):
            level1: Level1

        l2 = Level2()
        l1 = Level1(level2=l2)
        l0 = Level0(level1=l1)

        assert l0.level1.level2.value == 20

    def test_three_plus_level_nesting(self):
        """Test deeply nested configs (3+ levels)."""

        class Level4(Config):
            x: int = 4

        class Level3(Config):
            level4: Level4

        class Level2(Config):
            level3: Level3

        class Level1(Config):
            level2: Level2

        class Level0(Config):
            level1: Level1

        l4 = Level4()
        l3 = Level3(level4=l4)
        l2 = Level2(level3=l3)
        l1 = Level1(level2=l2)
        l0 = Level0(level1=l1)

        assert l0.level1.level2.level3.level4.x == 4

    def test_multiple_nested_configs_same_level(self):
        """Test multiple nested configs at same level."""

        class ConfigA(Config):
            a: int = 1

        class ConfigB(Config):
            b: int = 2

        class ConfigC(Config):
            c: int = 3

        class ParentConfig(Config):
            config_a: ConfigA
            config_b: ConfigB
            config_c: ConfigC

        parent = ParentConfig(
            config_a=ConfigA(), config_b=ConfigB(), config_c=ConfigC()
        )

        assert parent.config_a.a == 1
        assert parent.config_b.b == 2
        assert parent.config_c.c == 3

    def test_nested_config_with_defaults(self):
        """Test nested config with default values."""

        class InnerConfig(Config):
            x: int = Int(ge=0, le=10, default=5)
            y: str = "default"

        class OuterConfig(Config):
            inner: InnerConfig

        inner = InnerConfig()  # Uses defaults
        outer = OuterConfig(inner=inner)

        assert outer.inner.x == 5
        assert outer.inner.y == "default"


class TestNestedConfigsWithSpaces:
    """Test nested configs with their own space definitions."""

    def test_nested_config_with_int_space(self):
        """Test nested config with IntSpace fields."""

        class InnerConfig(Config):
            value: int = Int(ge=0, le=100)

        class OuterConfig(Config):
            inner: InnerConfig
            outer_value: int = Int(ge=0, le=10)

        inner = InnerConfig(value=50)
        outer = OuterConfig(inner=inner, outer_value=5)

        assert outer.inner.value == 50
        assert outer.outer_value == 5

    def test_nested_config_validation_independent(self):
        """Test that nested config validation is independent."""

        class InnerConfig(Config):
            value: int = Int(ge=0, le=10)

        class OuterConfig(Config):
            inner: InnerConfig
            value: int = Int(ge=0, le=100)  # Different range

        # Inner's range is 0-10, outer's is 0-100
        inner = InnerConfig(value=5)
        outer = OuterConfig(inner=inner, value=50)

        assert outer.inner.value == 5
        assert outer.value == 50

        # Inner should reject value > 10
        with pytest.raises(ValueError):
            InnerConfig(value=15)

        # But outer accepts it
        inner_valid = InnerConfig(value=5)
        outer = OuterConfig(inner=inner_valid, value=15)
        assert outer.value == 15

    def test_nested_config_with_categorical(self):
        """Test nested config with Categorical fields."""

        class InnerConfig(Config):
            mode: str = Categorical(["a", "b", "c"])

        class OuterConfig(Config):
            inner: InnerConfig
            outer_mode: str = Categorical(["x", "y"])

        inner = InnerConfig(mode="b")
        outer = OuterConfig(inner=inner, outer_mode="x")

        assert outer.inner.mode == "b"
        assert outer.outer_mode == "x"

    def test_deeply_nested_configs_with_spaces(self):
        """Test deeply nested configs each with their own spaces."""

        class Level3(Config):
            value: int = Int(ge=0, le=10)

        class Level2(Config):
            level3: Level3
            value: int = Int(ge=0, le=100)

        class Level1(Config):
            level2: Level2
            value: int = Int(ge=0, le=1000)

        l3 = Level3(value=5)
        l2 = Level2(level3=l3, value=50)
        l1 = Level1(level2=l2, value=500)

        assert l1.value == 500
        assert l1.level2.value == 50
        assert l1.level2.level3.value == 5


class TestNestedFieldConditions:
    """Test accessing nested fields in conditions."""

    def test_condition_on_nested_field(self):
        """Test conditional depending on nested config field."""

        class InnerConfig(Config):
            threshold: int = Int(ge=0, le=100, default=50)

        class OuterConfig(Config):
            inner: InnerConfig
            mode: str = Conditional(
                condition=FieldCondition(
                    "inner", FieldCondition("threshold", SmallerThan(50))
                ),
                true="low",
                false="high",
            )

        # threshold = 30 < 50, so mode = "low"
        inner = InnerConfig(threshold=30)
        outer = OuterConfig(inner=inner)
        assert outer.mode == "low"

        # threshold = 70 >= 50, so mode = "high"
        inner = InnerConfig(threshold=70)
        outer = OuterConfig(inner=inner)
        assert outer.mode == "high"

    def test_two_level_nested_field_condition(self):
        """Test condition on two-level nested field."""

        class Level2(Config):
            value: int = Int(ge=0, le=100, default=50)

        class Level1(Config):
            level2: Level2

        class Level0(Config):
            level1: Level1
            flag: bool = Conditional(
                condition=FieldCondition(
                    "level1",
                    FieldCondition("level2", FieldCondition("value", SmallerThan(50))),
                ),
                true=True,
                false=False,
            )

        l2 = Level2(value=30)
        l1 = Level1(level2=l2)
        l0 = Level0(level1=l1)

        assert l0.flag is True

        l2 = Level2(value=70)
        l1 = Level1(level2=l2)
        l0 = Level0(level1=l1)

        assert l0.flag is False

    def test_multiple_nested_conditions(self):
        """Test multiple conditions on different nested fields."""

        class ConfigA(Config):
            x: int = 10

        class ConfigB(Config):
            y: int = 20

        class ParentConfig(Config):
            config_a: ConfigA
            config_b: ConfigB
            mode: str = Conditional(
                condition=FieldCondition("config_a", FieldCondition("x", EqualsTo(10))),
                true="a_is_10",
                false="a_is_not_10",
            )
            flag: bool = Conditional(
                condition=FieldCondition("config_b", FieldCondition("y", EqualsTo(20))),
                true=True,
                false=False,
            )

        parent = ParentConfig(config_a=ConfigA(), config_b=ConfigB())

        assert parent.mode == "a_is_10"
        assert parent.flag is True


class TestNestedInCategorical:
    """Test nested configs in categorical spaces."""

    def test_categorical_with_config_types(self):
        """Test categorical space with Config type choices."""

        class ConfigA(Config):
            a: int = 1

        class ConfigB(Config):
            b: int = 2

        class ParentConfig(Config):
            choice: ConfigA | ConfigB = Categorical([ConfigA, ConfigB])

        # Can use ConfigA
        parent = ParentConfig(choice=ConfigA())
        assert isinstance(parent.choice, ConfigA)
        assert parent.choice.a == 1

        # Can use ConfigB
        parent = ParentConfig(choice=ConfigB())
        assert isinstance(parent.choice, ConfigB)
        assert parent.choice.b == 2

    def test_categorical_mixed_config_and_none(self):
        """Test categorical with Config type and None."""

        class MyConfig(Config):
            x: int = 10

        class ParentConfig(Config):
            optional: MyConfig | None = Categorical([MyConfig, None])

        # Can be Config
        parent = ParentConfig(optional=MyConfig())
        assert parent.optional.x == 10

        # Can be None
        parent = ParentConfig(optional=None)
        assert parent.optional is None

    def test_categorical_multiple_config_types(self):
        """Test categorical with multiple different Config types."""

        class ConfigA(Config):
            a: int = 1

        class ConfigB(Config):
            b: str = "b"

        class ConfigC(Config):
            c: float = 3.14

        class ParentConfig(Config):
            choice: ConfigA | ConfigB | ConfigC = Categorical(
                [ConfigA, ConfigB, ConfigC]
            )

        parent = ParentConfig(choice=ConfigA())
        assert isinstance(parent.choice, ConfigA)

        parent = ParentConfig(choice=ConfigB())
        assert isinstance(parent.choice, ConfigB)

        parent = ParentConfig(choice=ConfigC())
        assert isinstance(parent.choice, ConfigC)


class TestNestedInConditional:
    """Test nested configs in conditional spaces."""

    def test_conditional_with_config_branches(self):
        """Test conditional with Config types as branches."""

        class ConfigA(Config):
            a: int = 1

        class ConfigB(Config):
            b: int = 2

        class ParentConfig(Config):
            use_a: bool = True
            config: ConfigA | ConfigB = Conditional(
                condition=FieldCondition("use_a", EqualsTo(True)),
                true=ConfigA,
                false=ConfigB,
            )

        # use_a = True, should be ConfigA
        parent = ParentConfig(use_a=True, config=ConfigA())
        assert isinstance(parent.config, ConfigA)

        # use_a = False, should be ConfigB
        parent = ParentConfig(use_a=False, config=ConfigB())
        assert isinstance(parent.config, ConfigB)

    def test_conditional_config_or_none(self):
        """Test conditional that is Config or None."""

        class MyConfig(Config):
            value: int = 10

        class ParentConfig(Config):
            enabled: bool = True
            config: MyConfig | None = Conditional(
                condition=FieldCondition("enabled", EqualsTo(True)),
                true=MyConfig,
                false=None,
            )

        # enabled = True
        parent = ParentConfig(enabled=True, config=MyConfig())
        assert parent.config.value == 10

        # enabled = False
        parent = ParentConfig(enabled=False, config=None)
        assert parent.config is None

    def test_nested_conditional_with_nested_configs(self):
        """Test nested conditional with nested config branches."""

        class ConfigA(Config):
            a: int = 1

        class ConfigB(Config):
            b: int = 2

        class ConfigC(Config):
            c: int = 3

        class ParentConfig(Config):
            level: int = Int(ge=0, le=2, default=0)
            config: ConfigA | ConfigB | ConfigC = Conditional(
                condition=FieldCondition("level", EqualsTo(0)),
                true=ConfigA,
                false=Conditional(
                    condition=FieldCondition("level", EqualsTo(1)),
                    true=ConfigB,
                    false=ConfigC,
                ),
            )

        # level = 0 -> ConfigA
        parent = ParentConfig(level=0, config=ConfigA())
        assert isinstance(parent.config, ConfigA)

        # level = 1 -> ConfigB
        parent = ParentConfig(level=1, config=ConfigB())
        assert isinstance(parent.config, ConfigB)

        # level = 2 -> ConfigC
        parent = ParentConfig(level=2, config=ConfigC())
        assert isinstance(parent.config, ConfigC)


class TestComplexNestedScenarios:
    """Test complex real-world-like nested scenarios."""

    def test_ml_model_like_structure(self):
        """Test ML model-like nested config structure."""

        class LayerConfig(Config):
            hidden_dim: int = Int(ge=16, le=1024, default=128)
            activation: str = Categorical(["relu", "gelu", "silu"], default="relu")

        class EncoderConfig(Config):
            num_layers: int = Int(ge=1, le=32, default=6)
            layer_config: LayerConfig

        class ModelConfig(Config):
            encoder: EncoderConfig
            use_decoder: bool = False
            decoder: EncoderConfig | None = Conditional(
                condition=FieldCondition("use_decoder", EqualsTo(True)),
                true=EncoderConfig,
                false=None,
            )

        # Model with only encoder
        layer = LayerConfig()
        encoder = EncoderConfig(layer_config=layer)
        model = ModelConfig(encoder=encoder, use_decoder=False, decoder=None)

        assert model.encoder.num_layers == 6
        assert model.encoder.layer_config.hidden_dim == 128
        assert model.decoder is None

        # Model with encoder and decoder
        decoder = EncoderConfig(num_layers=3, layer_config=LayerConfig(hidden_dim=256))
        model = ModelConfig(encoder=encoder, use_decoder=True, decoder=decoder)

        assert model.decoder.num_layers == 3
        assert model.decoder.layer_config.hidden_dim == 256

    def test_hierarchical_config_system(self):
        """Test hierarchical configuration system."""

        class DatabaseConfig(Config):
            host: str = "localhost"
            port: int = Int(ge=1, le=65535, default=5432)

        class CacheConfig(Config):
            enabled: bool = True
            ttl: int = Int(ge=0, le=3600, default=300)

        class BackendConfig(Config):
            database: DatabaseConfig
            cache: CacheConfig

        class FrontendConfig(Config):
            theme: str = Categorical(["light", "dark"], default="light")

        class AppConfig(Config):
            backend: BackendConfig
            frontend: FrontendConfig
            debug: bool = False

        db = DatabaseConfig()
        cache = CacheConfig()
        backend = BackendConfig(database=db, cache=cache)
        frontend = FrontendConfig()
        app = AppConfig(backend=backend, frontend=frontend)

        assert app.backend.database.host == "localhost"
        assert app.backend.database.port == 5432
        assert app.backend.cache.ttl == 300
        assert app.frontend.theme == "light"

    def test_conditional_nested_with_inheritance(self):
        """Test conditional nested configs with inheritance."""

        class BaseOptimizer(Config):
            lr: float = Float(ge=1e-6, le=1.0, default=0.001)

        class Adam(BaseOptimizer):
            beta1: float = Float(ge=0.0, le=1.0, default=0.9)
            beta2: float = Float(ge=0.0, le=1.0, default=0.999)

        class SGD(BaseOptimizer):
            momentum: float = Float(ge=0.0, le=1.0, default=0.9)

        class TrainingConfig(Config):
            optimizer_type: str = Categorical(["adam", "sgd"], default="adam")
            optimizer: Adam | SGD = Conditional(
                condition=FieldCondition("optimizer_type", EqualsTo("adam")),
                true=Adam,
                false=SGD,
            )

        # Using Adam
        config = TrainingConfig(optimizer_type="adam", optimizer=Adam())
        assert isinstance(config.optimizer, Adam)
        assert config.optimizer.lr == 0.001
        assert config.optimizer.beta1 == 0.9

        # Using SGD
        config = TrainingConfig(optimizer_type="sgd", optimizer=SGD())
        assert isinstance(config.optimizer, SGD)
        assert config.optimizer.lr == 0.001
        assert config.optimizer.momentum == 0.9
