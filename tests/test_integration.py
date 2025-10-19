"""Integration tests for spax - real-world usage scenarios."""

from spax import (
    And,
    Categorical,
    Choice,
    Conditional,
    Config,
    EqualsTo,
    FieldCondition,
    Float,
    In,
    Int,
    LargerThan,
)


class TestMLWorkflows:
    """Test realistic ML experiment configurations."""

    def test_simple_training_config(self):
        """Test a basic training configuration."""

        class TrainingConfig(Config):
            learning_rate: float = Float(ge=1e-5, le=1e-1, distribution="log")
            batch_size: int = Int(ge=8, le=128, distribution="log")
            epochs: int = Int(ge=1, le=100)
            optimizer: str = Categorical(["adam", "sgd", "adamw"])

        # Create specific config
        config = TrainingConfig(
            learning_rate=0.001, batch_size=32, epochs=10, optimizer="adam"
        )
        assert config.learning_rate == 0.001

        # Random sampling
        for _ in range(10):
            random_config = TrainingConfig.random()
            assert 1e-5 <= random_config.learning_rate <= 1e-1
            assert 8 <= random_config.batch_size <= 128
            assert random_config.optimizer in ["adam", "sgd", "adamw"]

    def test_conditional_optimizer_config(self):
        """Test config with optimizer-specific parameters."""

        class OptimizerConfig(Config):
            optimizer: str = Categorical(["adam", "sgd"])
            learning_rate: float = Float(ge=1e-5, le=1e-1, distribution="log")
            momentum: float = Conditional(
                condition=FieldCondition("optimizer", EqualsTo("sgd")),
                true=Float(ge=0.0, le=0.99),
                false=0.0,  # Adam doesn't use momentum
            )
            beta1: float = Conditional(
                condition=FieldCondition("optimizer", EqualsTo("adam")),
                true=Float(ge=0.8, le=0.999),
                false=0.9,  # Default for non-Adam
            )

        # Test SGD config
        sgd_config = OptimizerConfig(
            optimizer="sgd", learning_rate=0.01, momentum=0.9, beta1=0.9
        )
        assert sgd_config.momentum == 0.9

        # Test Adam config
        adam_config = OptimizerConfig(
            optimizer="adam", learning_rate=0.001, momentum=0.0, beta1=0.95
        )
        assert adam_config.beta1 == 0.95
        assert adam_config.momentum == 0.0

        # Random sampling respects conditionals
        for _ in range(20):
            config = OptimizerConfig.random()
            if config.optimizer == "sgd":
                assert 0.0 <= config.momentum <= 0.99
            else:  # adam
                assert config.momentum == 0.0
                assert 0.8 <= config.beta1 <= 0.999

    def test_neural_architecture_search(self):
        """Test NAS-style configuration with nested conditionals."""

        class NASConfig(Config):
            model_type: str = Categorical(["transformer", "cnn", "rnn"])
            depth: int = Conditional(
                condition=FieldCondition("model_type", EqualsTo("transformer")),
                true=Int(ge=6, le=24),
                false=Conditional(
                    condition=FieldCondition("model_type", EqualsTo("cnn")),
                    true=Int(ge=10, le=50),
                    false=Int(ge=1, le=10),  # RNN
                ),
            )
            hidden_size: int = Int(ge=64, le=1024, distribution="log")
            dropout: float = Float(ge=0.0, le=0.5)
            use_attention: bool = Conditional(
                condition=FieldCondition("model_type", In(["transformer", "rnn"])),
                true=Categorical([True, False]),
                false=False,  # CNN doesn't use attention
            )

        # Test transformer
        tf_config = NASConfig(
            model_type="transformer",
            depth=12,
            hidden_size=512,
            dropout=0.1,
            use_attention=True,
        )
        assert 6 <= tf_config.depth <= 24

        # Test CNN
        cnn_config = NASConfig(
            model_type="cnn",
            depth=30,
            hidden_size=256,
            dropout=0.2,
            use_attention=False,
        )
        assert 10 <= cnn_config.depth <= 50
        assert cnn_config.use_attention is False

        # Random sampling
        for _ in range(20):
            config = NASConfig.random()
            if config.model_type == "transformer":
                assert 6 <= config.depth <= 24
            elif config.model_type == "cnn":
                assert 10 <= config.depth <= 50
                assert config.use_attention is False
            else:  # rnn
                assert 1 <= config.depth <= 10

    def test_weighted_categorical_choices(self):
        """Test using weighted choices for biased sampling."""

        class WeightedConfig(Config):
            activation: str = Categorical(
                [
                    Choice("relu", weight=5.0),  # Prefer relu
                    Choice("gelu", weight=3.0),
                    Choice("silu", weight=1.0),  # Rarely sample
                ]
            )

        # Sample many times
        samples = [WeightedConfig.random().activation for _ in range(1000)]
        relu_count = samples.count("relu")
        gelu_count = samples.count("gelu")
        silu_count = samples.count("silu")

        # relu should be most common (5/9 ≈ 55%)
        assert relu_count > 450
        # gelu should be middle (3/9 ≈ 33%)
        assert 200 < gelu_count < 450
        # silu should be least common (1/9 ≈ 11%)
        assert silu_count < 200


class TestComplexDependencies:
    """Test complex conditional dependency scenarios."""

    def test_chain_of_dependencies(self):
        """Test a chain: A -> B -> C."""

        class ChainConfig(Config):
            use_feature_a: bool = Categorical([True, False])
            use_feature_b: bool = Conditional(
                condition=FieldCondition("use_feature_a", EqualsTo(True)),
                true=Categorical([True, False]),
                false=False,
            )
            feature_b_param: float = Conditional(
                condition=FieldCondition("use_feature_b", EqualsTo(True)),
                true=Float(ge=0.0, le=1.0),
                false=0.0,
            )

        # When feature A is off, everything else is off
        config1 = ChainConfig(
            use_feature_a=False, use_feature_b=False, feature_b_param=0.0
        )
        assert not config1.use_feature_a
        assert not config1.use_feature_b
        assert config1.feature_b_param == 0.0

        # Random sampling respects chain
        for _ in range(30):
            config = ChainConfig.random()
            if not config.use_feature_a:
                assert not config.use_feature_b
                assert config.feature_b_param == 0.0
            if not config.use_feature_b:
                assert config.feature_b_param == 0.0

    def test_multiple_conditions_and_logic(self):
        """Test using And condition for multiple requirements."""

        class MultiConditionConfig(Config):
            model_size: str = Categorical(["small", "medium", "large"])
            use_advanced: bool = Categorical([True, False])
            advanced_param: float = Conditional(
                condition=And(
                    [
                        FieldCondition("model_size", In(["medium", "large"])),
                        FieldCondition("use_advanced", EqualsTo(True)),
                    ]
                ),
                true=Float(ge=0.0, le=1.0),
                false=0.0,
            )

        # Both conditions must be true
        config1 = MultiConditionConfig(
            model_size="large", use_advanced=True, advanced_param=0.5
        )
        assert config1.advanced_param == 0.5

        # If either is false, param is 0
        config2 = MultiConditionConfig(
            model_size="small", use_advanced=True, advanced_param=0.0
        )
        assert config2.advanced_param == 0.0


class TestSpaceIntrospection:
    """Test space introspection and metadata."""

    def test_get_space_info_comprehensive(self):
        """Test getting comprehensive space information."""

        class ComplexConfig(Config):
            lr: float = Float(ge=1e-5, le=1e-1, distribution="log")
            batch_size: int = Int(ge=8, le=128)
            optimizer: str = Categorical(["adam", "sgd"])
            momentum: float = Conditional(
                condition=FieldCondition("optimizer", EqualsTo("sgd")),
                true=Float(ge=0.0, le=0.99),
                false=0.0,
            )

        info = ComplexConfig.get_space_info()

        # Check all fields are present
        assert set(info.keys()) == {"lr", "batch_size", "optimizer", "momentum"}

        # Check lr info
        assert info["lr"]["type"] == "FloatSpace"
        assert info["lr"]["distribution"] == "LogDistribution"

        # Check batch_size info
        assert info["batch_size"]["type"] == "IntSpace"

        # Check optimizer info
        assert info["optimizer"]["type"] == "CategoricalSpace"
        assert len(info["optimizer"]["choices"]) == 2

        # Check momentum info
        assert info["momentum"]["type"] == "ConditionalSpace"
        assert "depends_on" in info["momentum"]
        assert "optimizer" in info["momentum"]["depends_on"]

    def test_get_dependency_info_comprehensive(self):
        """Test getting dependency graph information."""

        class DependentConfig(Config):
            a: float = Float(ge=0.0, le=1.0)
            b: float = Conditional(
                condition=FieldCondition("a", LargerThan(0.5)),
                true=Float(ge=1.0, le=2.0),
                false=Float(ge=0.0, le=1.0),
            )
            c: float = Conditional(
                condition=FieldCondition("b", LargerThan(1.0)),
                true=Float(ge=2.0, le=3.0),
                false=Float(ge=0.0, le=2.0),
            )

        dep_info = DependentConfig.get_dependency_info()

        # Check structure
        assert "nodes" in dep_info
        assert "edges" in dep_info
        assert "order" in dep_info
        assert "dependencies" in dep_info

        # Check dependency relationships
        deps = dep_info["dependencies"]
        assert "a" in deps["b"]  # b depends on a
        assert "b" in deps["c"]  # c depends on b

        # Check ordering (a must come before b, b before c)
        order = dep_info["order"]
        assert order.index("a") < order.index("b")
        assert order.index("b") < order.index("c")


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_value_range(self):
        """Test integer ranges at boundaries."""

        class EdgeConfig(Config):
            single_int: int = Int(ge=5, le=6)  # Only 5 or 6 possible

        values = set()
        for _ in range(20):
            config = EdgeConfig.random()
            values.add(config.single_int)
        assert values.issubset({5, 6})

    def test_very_small_float_range(self):
        """Test very small float ranges."""

        class EdgeConfig(Config):
            tiny_float: float = Float(ge=0.0001, le=0.0002)

        for _ in range(20):
            config = EdgeConfig.random()
            assert 0.0001 <= config.tiny_float <= 0.0002

    def test_log_space_extreme_range(self):
        """Test log distribution over many orders of magnitude."""

        class EdgeConfig(Config):
            wide_range: float = Float(ge=1e-10, le=1e10, distribution="log")

        for _ in range(20):
            config = EdgeConfig.random()
            assert 1e-10 <= config.wide_range <= 1e10


class TestPydanticIntegration:
    """Test integration with Pydantic features."""

    def test_model_dump(self):
        """Test Pydantic's model_dump method."""

        class SimpleConfig(Config):
            lr: float = Float(ge=0.001, le=0.1)
            batch_size: int = Int(ge=8, le=128)

        config = SimpleConfig(lr=0.01, batch_size=32)
        data = config.model_dump()

        assert data == {"lr": 0.01, "batch_size": 32}

    def test_model_dump_json(self):
        """Test Pydantic's model_dump_json method."""

        class SimpleConfig(Config):
            lr: float = Float(ge=0.001, le=0.1)
            optimizer: str = Categorical(["adam", "sgd"])

        config = SimpleConfig(lr=0.01, optimizer="adam")
        json_str = config.model_dump_json()

        assert '"lr":0.01' in json_str or '"lr": 0.01' in json_str
        assert '"optimizer":"adam"' in json_str or '"optimizer": "adam"' in json_str

    def test_model_validate(self):
        """Test Pydantic's model_validate method."""

        class SimpleConfig(Config):
            lr: float = Float(ge=0.001, le=0.1)
            batch_size: int = Int(ge=8, le=128)

        data = {"lr": 0.01, "batch_size": 32}
        config = SimpleConfig.model_validate(data)

        assert config.lr == 0.01
        assert config.batch_size == 32
