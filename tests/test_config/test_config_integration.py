"""Integration tests for Config with complex real-world scenarios."""

from spax import Categorical, Conditional, Config, Float, Int
from spax.spaces import EqualsTo, FieldCondition, LargerThan


class TestMLModelConfigs:
    """Test realistic ML model configuration structures."""

    def test_resnet_like_architecture(self):
        """Test ResNet-like architecture with conditional blocks."""

        class ConvBlock(Config):
            in_channels: int = Int(ge=1, le=2048)
            out_channels: int = Int(ge=1, le=2048)
            kernel_size: int = Categorical([1, 3, 5, 7], default=3)
            stride: int = Categorical([1, 2], default=1)

        class ResidualBlock(Config):
            num_layers: int = Int(ge=2, le=10, default=2)
            hidden_channels: int = Int(ge=16, le=2048, default=64)
            use_bottleneck: bool = False

        class ResNetConfig(Config):
            num_blocks: int = Int(ge=1, le=10, default=4)
            block_config: ResidualBlock
            initial_conv: ConvBlock
            use_projection: bool = False
            projection: ConvBlock | None = Conditional(
                condition=FieldCondition("use_projection", EqualsTo(True)),
                true=ConvBlock,
                false=None,
            )

        # Create a ResNet config
        initial = ConvBlock(in_channels=3, out_channels=64)
        block = ResidualBlock(num_layers=3, hidden_channels=128)

        resnet = ResNetConfig(
            num_blocks=4,
            block_config=block,
            initial_conv=initial,
            use_projection=False,
            projection=None,
        )

        assert resnet.num_blocks == 4
        assert resnet.block_config.hidden_channels == 128
        assert resnet.initial_conv.in_channels == 3
        assert resnet.projection is None

        # Test serialization round-trip
        data = resnet.model_dump()
        restored = ResNetConfig.model_validate(data)

        assert restored.num_blocks == 4
        assert restored.block_config.hidden_channels == 128

    def test_transformer_like_architecture(self):
        """Test Transformer-like architecture."""

        class AttentionConfig(Config):
            num_heads: int = Int(ge=1, le=32, default=8)
            head_dim: int = Int(ge=8, le=256, default=64)
            dropout: float = Float(ge=0.0, le=0.5, default=0.1)

        class FFNConfig(Config):
            hidden_dim: int = Int(ge=64, le=8192, default=2048)
            activation: str = Categorical(["relu", "gelu", "swish"], default="gelu")
            dropout: float = Float(ge=0.0, le=0.5, default=0.1)

        class TransformerLayer(Config):
            attention: AttentionConfig
            ffn: FFNConfig
            norm_first: bool = True

        class TransformerConfig(Config):
            num_layers: int = Int(ge=1, le=48, default=12)
            layer_config: TransformerLayer
            vocab_size: int = Int(ge=1000, le=100000, default=50000)
            max_seq_len: int = Int(ge=128, le=8192, default=512)

        # Create configuration
        attention = AttentionConfig(num_heads=12, head_dim=64)
        ffn = FFNConfig(hidden_dim=3072)
        layer = TransformerLayer(attention=attention, ffn=ffn, norm_first=True)

        transformer = TransformerConfig(
            num_layers=12, layer_config=layer, vocab_size=50000, max_seq_len=1024
        )

        assert transformer.num_layers == 12
        assert transformer.layer_config.attention.num_heads == 12
        assert transformer.layer_config.ffn.hidden_dim == 3072
        assert transformer.max_seq_len == 1024

        # JSON round-trip
        json_str = transformer.model_dump_json()
        restored = TransformerConfig.model_validate_json(json_str)

        assert restored.layer_config.attention.num_heads == 12

    def test_optimizer_config_union(self):
        """Test optimizer configuration with union types."""

        class BaseOptimizer(Config):
            lr: float = Float(ge=1e-6, le=1.0, default=0.001)
            weight_decay: float = Float(ge=0.0, le=0.1, default=0.0)

        class Adam(BaseOptimizer):
            beta1: float = Float(ge=0.0, le=1.0, default=0.9)
            beta2: float = Float(ge=0.0, le=1.0, default=0.999)
            eps: float = Float(ge=1e-10, le=1e-6, default=1e-8)

        class SGD(BaseOptimizer):
            momentum: float = Float(ge=0.0, le=1.0, default=0.9)
            nesterov: bool = False

        class AdamW(Adam):
            # AdamW inherits from Adam and just changes weight decay behavior
            pass

        class TrainingConfig(Config):
            optimizer_type: str = Categorical(["adam", "sgd", "adamw"], default="adam")
            optimizer: Adam | SGD | AdamW = Conditional(
                condition=FieldCondition("optimizer_type", EqualsTo("adam")),
                true=Adam,
                false=Conditional(
                    condition=FieldCondition("optimizer_type", EqualsTo("sgd")),
                    true=SGD,
                    false=AdamW,
                ),
            )
            batch_size: int = Int(ge=1, le=1024, default=32)
            epochs: int = Int(ge=1, le=1000, default=100)

        # Adam configuration
        adam_config = TrainingConfig(
            optimizer_type="adam", optimizer=Adam(lr=0.001, beta1=0.9), batch_size=64
        )

        assert isinstance(adam_config.optimizer, Adam)
        assert adam_config.optimizer.lr == 0.001

        # SGD configuration
        sgd_config = TrainingConfig(
            optimizer_type="sgd", optimizer=SGD(lr=0.01, momentum=0.9), batch_size=128
        )

        assert isinstance(sgd_config.optimizer, SGD)
        assert sgd_config.optimizer.momentum == 0.9

        # Serialization with type discrimination
        data = adam_config.model_dump()
        assert data["optimizer"]["__type__"] == "Adam"

        restored = TrainingConfig.model_validate(data)
        assert isinstance(restored.optimizer, Adam)


class TestComplexNestedStructures:
    """Test complex nested configuration structures."""

    def test_five_level_deep_nesting(self):
        """Test 5+ levels of nesting."""

        class Level5(Config):
            value: int = Int(ge=0, le=100, default=50)

        class Level4(Config):
            level5: Level5
            name: str = "level4"

        class Level3(Config):
            level4: Level4
            flag: bool = True

        class Level2(Config):
            level3: Level3
            count: int = Int(ge=0, le=10, default=5)

        class Level1(Config):
            level2: Level2
            mode: str = Categorical(["a", "b"], default="a")

        class Level0(Config):
            level1: Level1
            root_value: int = 0

        # Build from bottom up
        l5 = Level5(value=75)
        l4 = Level4(level5=l5)
        l3 = Level3(level4=l4)
        l2 = Level2(level3=l3, count=7)
        l1 = Level1(level2=l2, mode="b")
        l0 = Level0(level1=l1, root_value=42)

        # Access deeply nested value
        assert l0.level1.level2.level3.level4.level5.value == 75
        assert l0.level1.level2.count == 7
        assert l0.level1.mode == "b"

        # Serialization and deserialization
        yaml_str = l0.model_dump_yaml()
        restored = Level0.model_validate_yaml(yaml_str)

        assert restored.level1.level2.level3.level4.level5.value == 75

    def test_multiple_conditional_dependencies(self):
        """Test config with multiple conditional dependencies."""

        class ComplexConfig(Config):
            mode: str = Categorical(["simple", "medium", "complex"], default="simple")

            param1: int = Conditional(
                condition=FieldCondition("mode", EqualsTo("simple")),
                true=Int(ge=0, le=10, default=5),
                false=Int(ge=0, le=100, default=50),
            )

            param2: float = Conditional(
                condition=FieldCondition("mode", EqualsTo("complex")),
                true=Float(ge=0.0, le=1.0, default=0.5),
                false=Float(ge=0.0, le=0.1, default=0.01),
            )

            param3: str = Conditional(
                condition=FieldCondition("param1", LargerThan(50)),
                true="high",
                false="low",
            )

            final_value: int = Conditional(
                condition=FieldCondition("param3", EqualsTo("high")),
                true=1000,
                false=10,
            )

        # Simple mode
        config = ComplexConfig(mode="simple")
        assert config.param1 == 5
        assert config.param3 == "low"
        assert config.final_value == 10

        # Complex mode with high param1
        config = ComplexConfig(mode="complex", param1=75)
        assert config.param1 == 75
        assert config.param2 == 0.5
        assert config.param3 == "high"
        assert config.final_value == 1000

    def test_mix_all_space_types(self):
        """Test config with all space types mixed together."""

        class InnerA(Config):
            x: int = 1

        class InnerB(Config):
            y: int = 2

        class ComplexConfig(Config):
            # IntSpace
            count: int = Int(ge=1, le=100, default=10)

            # FloatSpace
            rate: float = Float(ge=0.0, le=1.0, default=0.5)

            # CategoricalSpace with primitives
            mode: str = Categorical(["a", "b", "c"], default="a")

            # CategoricalSpace with bool
            enabled: bool = True

            # CategoricalSpace with Config types
            inner: InnerA | InnerB = Categorical([InnerA, InnerB])

            # ConditionalSpace
            conditional_value: int = Conditional(
                condition=FieldCondition("enabled", EqualsTo(True)),
                true=Int(ge=0, le=100, default=50),
                false=0,
            )

            # Nested Config
            class NestedConfig(Config):
                nested_value: int = 5

            nested: NestedConfig

            # Fixed value
            fixed: str = "constant"

        config = ComplexConfig(
            count=20,
            rate=0.75,
            mode="b",
            enabled=True,
            inner=InnerA(),
            nested=ComplexConfig.NestedConfig(),
        )

        assert config.count == 20
        assert config.rate == 0.75
        assert config.mode == "b"
        assert config.enabled is True
        assert isinstance(config.inner, InnerA)
        assert config.conditional_value == 50
        assert config.nested.nested_value == 5
        assert config.fixed == "constant"

        # Full round-trip
        json_str = config.model_dump_json()
        restored = ComplexConfig.model_validate_json(json_str)

        assert restored.count == 20
        assert isinstance(restored.inner, InnerA)


class TestLargeConfigs:
    """Test large configuration structures."""

    def test_config_with_50_plus_fields(self):
        """Test config with many fields."""

        class LargeConfig(Config):
            f1: int = Int(ge=0, le=10, default=1)
            f2: int = Int(ge=0, le=10, default=2)
            f3: int = Int(ge=0, le=10, default=3)
            f4: int = Int(ge=0, le=10, default=4)
            f5: int = Int(ge=0, le=10, default=5)
            f6: int = Int(ge=0, le=10, default=6)
            f7: int = Int(ge=0, le=10, default=7)
            f8: int = Int(ge=0, le=10, default=8)
            f9: int = Int(ge=0, le=10, default=9)
            f10: int = Int(ge=0, le=10, default=10)
            f11: float = Float(ge=0.0, le=1.0, default=0.1)
            f12: float = Float(ge=0.0, le=1.0, default=0.2)
            f13: float = Float(ge=0.0, le=1.0, default=0.3)
            f14: float = Float(ge=0.0, le=1.0, default=0.4)
            f15: float = Float(ge=0.0, le=1.0, default=0.5)
            f16: str = Categorical(["a", "b"], default="a")
            f17: str = Categorical(["a", "b"], default="a")
            f18: str = Categorical(["a", "b"], default="a")
            f19: str = Categorical(["a", "b"], default="a")
            f20: str = Categorical(["a", "b"], default="a")
            f21: bool = True
            f22: bool = False
            f23: bool = True
            f24: bool = False
            f25: bool = True
            f26: int = 26
            f27: int = 27
            f28: int = 28
            f29: int = 29
            f30: int = 30
            f31: str = "s31"
            f32: str = "s32"
            f33: str = "s33"
            f34: str = "s34"
            f35: str = "s35"
            f36: float = 36.0
            f37: float = 37.0
            f38: float = 38.0
            f39: float = 39.0
            f40: float = 40.0
            f41: int = Int(ge=0, le=100, default=41)
            f42: int = Int(ge=0, le=100, default=42)
            f43: int = Int(ge=0, le=100, default=43)
            f44: int = Int(ge=0, le=100, default=44)
            f45: int = Int(ge=0, le=100, default=45)
            f46: int = Int(ge=0, le=100, default=46)
            f47: int = Int(ge=0, le=100, default=47)
            f48: int = Int(ge=0, le=100, default=48)
            f49: int = Int(ge=0, le=100, default=49)
            f50: int = Int(ge=0, le=100, default=50)

        config = LargeConfig()

        # Check a few fields
        assert config.f1 == 1
        assert config.f25 is True
        assert config.f50 == 50

        # Serialization should work
        data = config.model_dump()
        assert len(data) == 50

        restored = LargeConfig.model_validate(data)
        assert restored.f1 == 1
        assert restored.f50 == 50

    def test_config_with_10_nested_levels(self):
        """Test config with 10+ nested configs."""

        class L1(Config):
            v: int = 1

        class L2(Config):
            l1: L1
            v: int = 2

        class L3(Config):
            l2: L2
            v: int = 3

        class L4(Config):
            l3: L3
            v: int = 4

        class L5(Config):
            l4: L4
            v: int = 5

        class L6(Config):
            l5: L5
            v: int = 6

        class L7(Config):
            l6: L6
            v: int = 7

        class L8(Config):
            l7: L7
            v: int = 8

        class L9(Config):
            l8: L8
            v: int = 9

        class L10(Config):
            l9: L9
            v: int = 10

        # Build up
        l1 = L1()
        l2 = L2(l1=l1)
        l3 = L3(l2=l2)
        l4 = L4(l3=l3)
        l5 = L5(l4=l4)
        l6 = L6(l5=l5)
        l7 = L7(l6=l6)
        l8 = L8(l7=l7)
        l9 = L9(l8=l8)
        l10 = L10(l9=l9)

        # Access deeply nested
        assert l10.l9.l8.l7.l6.l5.l4.l3.l2.l1.v == 1
        assert l10.v == 10

        # Serialization
        data = l10.model_dump()
        restored = L10.model_validate(data)

        assert restored.l9.l8.l7.l6.l5.l4.l3.l2.l1.v == 1


class TestProductionLikeScenarios:
    """Test production-like configuration scenarios."""

    def test_microservice_config(self):
        """Test microservice-like configuration."""

        class DatabaseConfig(Config):
            host: str = "localhost"
            port: int = Int(ge=1, le=65535, default=5432)
            name: str = "mydb"
            pool_size: int = Int(ge=1, le=100, default=10)
            ssl: bool = False

        class CacheConfig(Config):
            enabled: bool = True
            host: str = Conditional(
                condition=FieldCondition("enabled", EqualsTo(True)),
                true="localhost",
                false="",
            )
            port: int = Conditional(
                condition=FieldCondition("enabled", EqualsTo(True)),
                true=Int(ge=1, le=65535, default=6379),
                false=0,
            )
            ttl: int = Int(ge=0, le=86400, default=3600)

        class LoggingConfig(Config):
            level: str = Categorical(
                ["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO"
            )
            format: str = "json"
            output: str = Categorical(["stdout", "file"], default="stdout")

        class AppConfig(Config):
            name: str = "my-service"
            version: str = "1.0.0"
            database: DatabaseConfig
            cache: CacheConfig
            logging: LoggingConfig
            debug: bool = False

        db = DatabaseConfig(host="prod-db.example.com", port=5432)
        cache = CacheConfig(enabled=True)
        logging = LoggingConfig(level="INFO")

        app = AppConfig(name="my-service", database=db, cache=cache, logging=logging)

        assert app.database.host == "prod-db.example.com"
        assert app.cache.enabled is True
        assert app.logging.level == "INFO"

        # YAML serialization (common for configs)
        yaml_str = app.model_dump_yaml()
        restored = AppConfig.model_validate_yaml(yaml_str)

        assert restored.database.host == "prod-db.example.com"

    def test_all_fields_optional_with_defaults(self):
        """Test config where all fields are optional with defaults."""

        class OptionalConfig(Config):
            field1: int = Int(ge=0, le=100, default=50)
            field2: str = Categorical(["a", "b", "c"], default="b")
            field3: float = Float(ge=0.0, le=1.0, default=0.5)
            field4: bool = True
            field5: str = "default"

        # Can create with no arguments
        config = OptionalConfig()

        assert config.field1 == 50
        assert config.field2 == "b"
        assert config.field3 == 0.5
        assert config.field4 is True
        assert config.field5 == "default"

    def test_all_fields_conditional(self):
        """Test config where all fields are conditional."""

        class AllConditionalConfig(Config):
            mode: str = Categorical(["a", "b"], default="a")

            field1: int = Conditional(
                condition=FieldCondition("mode", EqualsTo("a")), true=10, false=20
            )

            field2: str = Conditional(
                condition=FieldCondition("mode", EqualsTo("a")),
                true="mode_a",
                false="mode_b",
            )

            field3: float = Conditional(
                condition=FieldCondition("field1", EqualsTo(10)), true=1.0, false=2.0
            )

        # Mode A
        config = AllConditionalConfig(mode="a")
        assert config.field1 == 10
        assert config.field2 == "mode_a"
        assert config.field3 == 1.0

        # Mode B
        config = AllConditionalConfig(mode="b")
        assert config.field1 == 20
        assert config.field2 == "mode_b"
        assert config.field3 == 2.0
