"""Tests for conditional search spaces."""

import pytest

from spax.config import Config
from spax.spaces import (
    Categorical,
    Conditional,
    ConditionalSpace,
    EqualsTo,
    FieldCondition,
    Float,
    Int,
)


class TestConditionalSpace:
    """Tests for ConditionalSpace."""

    def test_creation(self):
        """Test creating a conditional space."""
        space = Conditional(
            condition=FieldCondition("optimizer", EqualsTo("sgd")),
            true=Float(ge=0.1, le=1.0),
            false=Float(ge=0.001, le=0.1),
        )
        assert isinstance(space, ConditionalSpace)

    def test_sample_raises_error_without_config(self):
        """Test that sampling without config raises NotImplementedError."""
        space = Conditional(
            condition=FieldCondition("optimizer", EqualsTo("sgd")),
            true=Float(ge=0.1, le=1.0),
            false=Float(ge=0.001, le=0.1),
        )
        with pytest.raises(
            NotImplementedError, match="cannot be sampled independently"
        ):
            space.sample()

    def test_sample_with_config_true_branch(self):
        """Test sampling when condition is True."""

        class MockConfig:
            optimizer = "sgd"

        space = Conditional(
            condition=FieldCondition("optimizer", EqualsTo("sgd")),
            true=Float(ge=0.1, le=1.0),
            false=Float(ge=0.001, le=0.1),
        )
        config = MockConfig()

        for _ in range(20):
            value = space.sample_with_config(config)
            assert 0.1 <= value <= 1.0

    def test_sample_with_config_false_branch(self):
        """Test sampling when condition is False."""

        class MockConfig:
            optimizer = "adam"

        space = Conditional(
            condition=FieldCondition("optimizer", EqualsTo("sgd")),
            true=Float(ge=0.1, le=1.0),
            false=Float(ge=0.001, le=0.1),
        )
        config = MockConfig()

        for _ in range(20):
            value = space.sample_with_config(config)
            assert 0.001 <= value <= 0.1

    def test_validate_with_config_true_branch(self):
        """Test validation when condition is True."""

        class MockConfig:
            optimizer = "sgd"

        space = Conditional(
            condition=FieldCondition("optimizer", EqualsTo("sgd")),
            true=Float(ge=0.1, le=1.0),
            false=Float(ge=0.001, le=0.1),
        )
        space.field_name = "lr"
        config = MockConfig()

        assert space.validate_with_config(0.5, config) == 0.5

    def test_validate_with_config_false_branch(self):
        """Test validation when condition is False."""

        class MockConfig:
            optimizer = "adam"

        space = Conditional(
            condition=FieldCondition("optimizer", EqualsTo("sgd")),
            true=Float(ge=0.1, le=1.0),
            false=Float(ge=0.001, le=0.1),
        )
        space.field_name = "lr"
        config = MockConfig()

        assert space.validate_with_config(0.01, config) == 0.01

    def test_validate_with_config_rejects_invalid_value(self):
        """Test validation rejects values outside active branch range."""

        class MockConfig:
            optimizer = "sgd"

        space = Conditional(
            condition=FieldCondition("optimizer", EqualsTo("sgd")),
            true=Float(ge=0.1, le=1.0),
            false=Float(ge=0.001, le=0.1),
        )
        space.field_name = "lr"
        config = MockConfig()

        # Condition is True, so should use range [0.1, 1.0]
        with pytest.raises(ValueError):
            space.validate_with_config(0.01, config)  # Too small for true branch

    def test_fixed_value_in_branch(self):
        """Test conditional with fixed values instead of spaces."""

        class MockConfig:
            use_feature = True

        space = Conditional(
            condition=FieldCondition("use_feature", EqualsTo(True)),
            true=10,  # Fixed value
            false=0,  # Fixed value
        )
        space.field_name = "param"
        config = MockConfig()

        assert space.sample_with_config(config) == 10
        assert space.validate_with_config(10, config) == 10

        with pytest.raises(ValueError, match="Expected fixed value"):
            space.validate_with_config(5, config)

    def test_nested_conditionals(self):
        """Test nested conditional spaces."""

        class MockConfig:
            model_type = "transformer"
            size = "large"

        space = Conditional(
            condition=FieldCondition("model_type", EqualsTo("transformer")),
            true=Conditional(
                condition=FieldCondition("size", EqualsTo("large")),
                true=Int(ge=12, le=24),
                false=Int(ge=6, le=12),
            ),
            false=Int(ge=1, le=6),
        )
        space.field_name = "depth"
        config = MockConfig()

        for _ in range(20):
            value = space.sample_with_config(config)
            assert 12 <= value <= 24


class TestConditionalInConfig:
    """Tests for conditional spaces integrated with Config."""

    def test_simple_conditional_config(self):
        """Test Config with simple conditional space."""

        class MyConfig(Config):
            optimizer: str = Categorical(["adam", "sgd"])
            learning_rate: float = Conditional(
                condition=FieldCondition("optimizer", EqualsTo("sgd")),
                true=Float(ge=0.1, le=1.0),
                false=Float(ge=0.001, le=0.1),
            )

        # Create with SGD
        config1 = MyConfig(optimizer="sgd", learning_rate=0.5)
        assert config1.learning_rate == 0.5

        # Create with Adam
        config2 = MyConfig(optimizer="adam", learning_rate=0.01)
        assert config2.learning_rate == 0.01

    def test_conditional_validation_in_config(self):
        """Test that conditional validation works in Config."""

        class MyConfig(Config):
            optimizer: str = Categorical(["adam", "sgd"])
            learning_rate: float = Conditional(
                condition=FieldCondition("optimizer", EqualsTo("sgd")),
                true=Float(ge=0.1, le=1.0),
                false=Float(ge=0.001, le=0.1),
            )

        # Valid for SGD
        MyConfig(optimizer="sgd", learning_rate=0.5)

        # Invalid for SGD (too small)
        with pytest.raises(ValueError):
            MyConfig(optimizer="sgd", learning_rate=0.01)

        # Valid for Adam
        MyConfig(optimizer="adam", learning_rate=0.01)

        # Invalid for Adam (too large)
        with pytest.raises(ValueError):
            MyConfig(optimizer="adam", learning_rate=0.5)

    def test_random_sampling_with_conditional(self):
        """Test random sampling respects conditional constraints."""

        class MyConfig(Config):
            optimizer: str = Categorical(["adam", "sgd"])
            learning_rate: float = Conditional(
                condition=FieldCondition("optimizer", EqualsTo("sgd")),
                true=Float(ge=0.1, le=1.0),
                false=Float(ge=0.001, le=0.1),
            )

        for _ in range(20):
            config = MyConfig.random()
            if config.optimizer == "sgd":
                assert 0.1 <= config.learning_rate <= 1.0
            else:  # adam
                assert 0.001 <= config.learning_rate <= 0.1

    def test_nested_conditional_config(self):
        """Test Config with nested conditionals."""

        class MyConfig(Config):
            model_type: str = Categorical(["transformer", "cnn"])
            size: str = Categorical(["small", "large"])
            depth: int = Conditional(
                condition=FieldCondition("model_type", EqualsTo("transformer")),
                true=Conditional(
                    condition=FieldCondition("size", EqualsTo("large")),
                    true=Int(ge=12, le=24),
                    false=Int(ge=6, le=12),
                ),
                false=Int(ge=1, le=6),
            )

        # Transformer + large
        config1 = MyConfig(model_type="transformer", size="large", depth=18)
        assert 12 <= config1.depth <= 24

        # Transformer + small
        config2 = MyConfig(model_type="transformer", size="small", depth=8)
        assert 6 <= config2.depth <= 12

        # CNN (size doesn't matter)
        config3 = MyConfig(model_type="cnn", size="large", depth=3)
        assert 1 <= config3.depth <= 6

    def test_multiple_conditionals(self):
        """Test Config with multiple independent conditional spaces."""

        class MyConfig(Config):
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
                false=0.0,  # Fixed value when no scheduler
            )

        config = MyConfig(
            optimizer="sgd", use_scheduler=True, learning_rate=0.5, scheduler_gamma=0.9
        )
        assert config.learning_rate == 0.5
        assert config.scheduler_gamma == 0.9
