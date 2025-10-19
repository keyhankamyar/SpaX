"""Tests for Config class."""

import pytest

from spax import (
    Categorical,
    Conditional,
    Config,
    EqualsTo,
    FieldCondition,
    Float,
    Int,
)


class TestBasicConfig:
    """Tests for basic Config functionality."""

    def test_simple_config_creation(self):
        """Test creating a simple config."""

        class SimpleConfig(Config):
            learning_rate: float = Float(ge=0.001, le=0.1)
            batch_size: int = Int(ge=8, le=128)

        config = SimpleConfig(learning_rate=0.01, batch_size=32)
        assert config.learning_rate == 0.01
        assert config.batch_size == 32

    def test_validation_on_creation(self):
        """Test that validation happens during creation."""

        class SimpleConfig(Config):
            learning_rate: float = Float(ge=0.001, le=0.1)

        # Valid
        SimpleConfig(learning_rate=0.01)

        # Invalid - out of range
        with pytest.raises(ValueError):
            SimpleConfig(learning_rate=1.0)

    def test_validation_on_assignment(self):
        """Test that validation happens on attribute assignment."""

        class SimpleConfig(Config):
            learning_rate: float = Float(ge=0.001, le=0.1)

        config = SimpleConfig(learning_rate=0.01)

        # Valid assignment
        config.learning_rate = 0.05
        assert config.learning_rate == 0.05

        # Invalid assignment
        with pytest.raises(ValueError):
            config.learning_rate = 1.0

    def test_categorical_field(self):
        """Test config with categorical field."""

        class SimpleConfig(Config):
            activation: str = Categorical(["relu", "gelu", "silu"])

        config = SimpleConfig(activation="relu")
        assert config.activation == "relu"

        with pytest.raises(ValueError):
            SimpleConfig(activation="tanh")

    def test_mixed_space_types(self):
        """Test config with multiple space types."""

        class MixedConfig(Config):
            learning_rate: float = Float(ge=0.001, le=0.1, distribution="log")
            batch_size: int = Int(ge=8, le=128, distribution="log")
            optimizer: str = Categorical(["adam", "sgd"])

        config = MixedConfig(learning_rate=0.01, batch_size=32, optimizer="adam")
        assert config.learning_rate == 0.01
        assert config.batch_size == 32
        assert config.optimizer == "adam"


class TestDefaultValues:
    """Tests for default value functionality."""

    def test_config_with_defaults(self):
        """Test creating config with default values."""

        class ConfigWithDefaults(Config):
            a: int = Int(ge=0, le=100, default=50)
            b: float = Float(ge=-5.0, lt=0.0, default=-0.5)
            c: int = Int(gt=-5, lt=5)

        # Only need to provide c
        config = ConfigWithDefaults(c=0)
        assert config.a == 50
        assert config.b == -0.5
        assert config.c == 0

    def test_override_defaults(self):
        """Test that provided values override defaults."""

        class ConfigWithDefaults(Config):
            a: int = Int(ge=0, le=100, default=50)
            b: float = Float(ge=-5.0, lt=0.0, default=-0.5)

        config = ConfigWithDefaults(a=75, b=-2.0)
        assert config.a == 75
        assert config.b == -2.0

    def test_random_uses_defaults(self):
        """Test that random() uses defaults when use_defaults=True."""

        class ConfigWithDefaults(Config):
            a: int = Int(ge=0, le=100, default=50)
            b: float = Float(ge=0.0, le=1.0)

        config = ConfigWithDefaults.random(use_defaults=True)
        assert config.a == 50  # Should use default
        assert 0.0 <= config.b <= 1.0  # Should sample

    def test_random_ignores_defaults(self):
        """Test that random() ignores defaults when use_defaults=False."""

        class ConfigWithDefaults(Config):
            a: int = Int(ge=0, le=100, default=50)
            b: float = Float(ge=0.0, le=1.0, default=0.5)

        values_a = []
        values_b = []
        for _ in range(20):
            config = ConfigWithDefaults.random(use_defaults=False)
            values_a.append(config.a)
            values_b.append(config.b)

        # Should have sampled different values, not always defaults
        assert len(set(values_a)) > 1
        assert len(set(values_b)) > 1

    def test_none_as_default(self):
        """Test that None can be used as a default value."""

        class ConfigWithNone(Config):
            value: None | str = Categorical([None, "a", "b"], default=None)

        config = ConfigWithNone()
        assert config.value is None


class TestRandomSampling:
    """Tests for Config.random() method."""

    def test_random_generates_valid_config(self):
        """Test that random() generates valid configs."""

        class SimpleConfig(Config):
            learning_rate: float = Float(ge=0.001, le=0.1)
            batch_size: int = Int(ge=8, le=128)

        for _ in range(10):
            config = SimpleConfig.random()
            assert 0.001 <= config.learning_rate <= 0.1
            assert 8 <= config.batch_size <= 128

    def test_random_with_categorical(self):
        """Test random sampling with categorical spaces."""

        class SimpleConfig(Config):
            activation: str = Categorical(["relu", "gelu", "silu"])

        activations = set()
        for _ in range(30):
            config = SimpleConfig.random()
            activations.add(config.activation)

        # Should have sampled multiple different activations
        assert len(activations) >= 2

    def test_random_samples_different_values(self):
        """Test that random() produces varying configs."""

        class SimpleConfig(Config):
            value: float = Float(ge=0.0, le=100.0)

        values = [SimpleConfig.random().value for _ in range(20)]
        # Should have at least some variation
        assert len(set(values)) > 1


class TestSpaceInfo:
    """Tests for Config.get_space_info() method."""

    def test_space_info_float(self):
        """Test space info for float spaces."""

        class SimpleConfig(Config):
            learning_rate: float = Float(ge=0.001, le=0.1, distribution="log")

        info = SimpleConfig.get_space_info()
        assert "learning_rate" in info
        lr_info = info["learning_rate"]
        assert lr_info["type"] == "FloatSpace"
        assert lr_info["low"] == 0.001
        assert lr_info["high"] == 0.1
        assert lr_info["distribution"] == "LogDistribution"

    def test_space_info_int(self):
        """Test space info for int spaces."""

        class SimpleConfig(Config):
            batch_size: int = Int(ge=8, le=128)

        info = SimpleConfig.get_space_info()
        assert "batch_size" in info
        bs_info = info["batch_size"]
        assert bs_info["type"] == "IntSpace"
        assert bs_info["low"] == 8
        assert bs_info["high"] == 128

    def test_space_info_categorical(self):
        """Test space info for categorical spaces."""

        class SimpleConfig(Config):
            activation: str = Categorical(["relu", "gelu", "silu"])

        info = SimpleConfig.get_space_info()
        assert "activation" in info
        act_info = info["activation"]
        assert act_info["type"] == "CategoricalSpace"
        assert act_info["choices"] == ["relu", "gelu", "silu"]
        assert len(act_info["weights"]) == 3

    def test_space_info_conditional(self):
        """Test space info for conditional spaces."""

        class SimpleConfig(Config):
            optimizer: str = Categorical(["adam", "sgd"])
            learning_rate: float = Conditional(
                condition=FieldCondition("optimizer", EqualsTo("sgd")),
                true=Float(ge=0.1, le=1.0),
                false=Float(ge=0.001, le=0.1),
            )

        info = SimpleConfig.get_space_info()
        assert "learning_rate" in info
        lr_info = info["learning_rate"]
        assert lr_info["type"] == "ConditionalSpace"
        assert "condition" in lr_info
        assert "true_branch" in lr_info
        assert "false_branch" in lr_info


class TestNestedConfig:
    """Tests for nested Config types."""

    def test_nested_config_in_categorical(self):
        """Test using Config types as categorical choices."""

        class OptimizerConfig(Config):
            lr: float = Float(ge=0.001, le=0.1)

        class MainConfig(Config):
            optimizer: OptimizerConfig = Categorical([OptimizerConfig])

        # Random should instantiate the nested config
        config = MainConfig.random()
        assert isinstance(config.optimizer, OptimizerConfig)
        assert hasattr(config.optimizer, "lr")

    def test_multiple_nested_config_types(self):
        """Test multiple config types as choices."""

        class AdamConfig(Config):
            lr: float = Float(ge=0.0001, le=0.01)

        class SGDConfig(Config):
            lr: float = Float(ge=0.01, le=0.1)
            momentum: float = Float(ge=0.0, le=0.99)

        class MainConfig(Config):
            optimizer: AdamConfig | SGDConfig = Categorical([AdamConfig, SGDConfig])

        # Sample multiple times
        configs = [MainConfig.random() for _ in range(20)]

        # Should have both types
        adam_count = sum(1 for c in configs if isinstance(c.optimizer, AdamConfig))
        sgd_count = sum(1 for c in configs if isinstance(c.optimizer, SGDConfig))

        assert adam_count > 0
        assert sgd_count > 0


class TestConditionalConfig:
    """Tests for configs with conditional spaces."""

    def test_simple_conditional(self):
        """Test simple conditional space in config."""

        class SimpleConfig(Config):
            optimizer: str = Categorical(["adam", "sgd"])
            learning_rate: float = Conditional(
                condition=FieldCondition("optimizer", EqualsTo("sgd")),
                true=Float(ge=0.1, le=1.0),
                false=Float(ge=0.001, le=0.1),
            )

        # SGD uses higher learning rate range
        config1 = SimpleConfig(optimizer="sgd", learning_rate=0.5)
        assert config1.learning_rate == 0.5

        # Adam uses lower learning rate range
        config2 = SimpleConfig(optimizer="adam", learning_rate=0.01)
        assert config2.learning_rate == 0.01

    def test_conditional_random_sampling(self):
        """Test random sampling respects conditional constraints."""

        class SimpleConfig(Config):
            optimizer: str = Categorical(["adam", "sgd"])
            learning_rate: float = Conditional(
                condition=FieldCondition("optimizer", EqualsTo("sgd")),
                true=Float(ge=0.1, le=1.0),
                false=Float(ge=0.001, le=0.1),
            )

        for _ in range(20):
            config = SimpleConfig.random()
            if config.optimizer == "sgd":
                assert 0.1 <= config.learning_rate <= 1.0
            else:
                assert 0.001 <= config.learning_rate <= 0.1

    def test_multiple_dependencies(self):
        """Test config with multiple conditional dependencies."""

        class ComplexConfig(Config):
            optimizer: str = Categorical(["adam", "sgd"])
            use_scheduler: bool = Categorical([True, False])
            learning_rate: float = Conditional(
                condition=FieldCondition("optimizer", EqualsTo("sgd")),
                true=Float(ge=0.1, le=1.0),
                false=Float(ge=0.001, le=0.1),
            )
            scheduler_gamma: float = Conditional(
                condition=FieldCondition("use_scheduler", EqualsTo(True)),
                true=Float(ge=0.1, le=0.99),
                false=0.0,
            )

        config = ComplexConfig.random()

        # Check learning rate range based on optimizer
        if config.optimizer == "sgd":
            assert 0.1 <= config.learning_rate <= 1.0
        else:
            assert 0.001 <= config.learning_rate <= 0.1

        # Check scheduler gamma based on use_scheduler
        if config.use_scheduler:
            assert 0.1 <= config.scheduler_gamma <= 0.99
        else:
            assert config.scheduler_gamma == 0.0


class TestDependencyInfo:
    """Tests for dependency graph information."""

    def test_dependency_info_no_conditionals(self):
        """Test dependency info for config without conditionals."""

        class SimpleConfig(Config):
            learning_rate: float = Float(ge=0.001, le=0.1)
            batch_size: int = Int(ge=8, le=128)

        info = SimpleConfig.get_dependency_info()
        assert "nodes" in info
        assert "edges" in info
        assert "order" in info
        assert len(info["edges"]) == 0  # No dependencies

    def test_dependency_info_with_conditional(self):
        """Test dependency info for config with conditional."""

        class SimpleConfig(Config):
            optimizer: str = Categorical(["adam", "sgd"])
            learning_rate: float = Conditional(
                condition=FieldCondition("optimizer", EqualsTo("sgd")),
                true=Float(ge=0.1, le=1.0),
                false=Float(ge=0.001, le=0.1),
            )

        info = SimpleConfig.get_dependency_info()
        assert "learning_rate" in info["nodes"]
        assert "optimizer" in info["nodes"]

        # learning_rate depends on optimizer
        assert len(info["edges"]) > 0

        # optimizer should come before learning_rate in order
        order = info["order"]
        opt_idx = order.index("optimizer")
        lr_idx = order.index("learning_rate")
        assert opt_idx < lr_idx


class TestConfigErrors:
    """Tests for error handling in Config."""

    def test_circular_dependency_raises_error(self):
        """Test that circular dependencies are caught at class creation."""
        with pytest.raises(TypeError, match="Circular dependency"):

            class CircularConfig(Config):
                a: float = Conditional(
                    condition=FieldCondition("b", EqualsTo(1.0)),
                    true=Float(ge=0.0, le=1.0),
                    false=Float(ge=1.0, le=2.0),
                )
                b: float = Conditional(
                    condition=FieldCondition("a", EqualsTo(1.0)),
                    true=Float(ge=0.0, le=1.0),
                    false=Float(ge=1.0, le=2.0),
                )

    def test_missing_dependency_field_raises_error(self):
        """Test that referencing non-existent fields raises error."""
        with pytest.raises(TypeError, match="unknown field"):

            class BadConfig(Config):
                learning_rate: float = Conditional(
                    condition=FieldCondition("nonexistent", EqualsTo("value")),
                    true=Float(ge=0.0, le=1.0),
                    false=Float(ge=1.0, le=2.0),
                )

    def test_missing_field_without_default_raises_error(self):
        """Test that missing required fields without defaults raise error."""

        class SimpleConfig(Config):
            learning_rate: float = Float(ge=0.001, le=0.1)

        with pytest.raises(RuntimeError, match="not provided.*no default"):
            SimpleConfig()

    def test_can_omit_field_with_default(self):
        """Test that fields with defaults can be omitted."""

        class SimpleConfig(Config):
            learning_rate: float = Float(ge=0.001, le=0.1, default=0.01)

        config = SimpleConfig()
        assert config.learning_rate == 0.01
