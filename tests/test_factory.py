"""Tests for automatic space inference from type annotations."""

from typing import Literal

from pydantic import Field
import pytest

import spax as sp


def test_bool_inference():
    """Test that bool fields are automatically converted to Categorical."""

    class MyConfig(sp.Config):
        use_feature: bool

    # Check that a space was created
    assert "use_feature" in MyConfig._spaces
    space = MyConfig._spaces["use_feature"]
    assert isinstance(space, sp.CategoricalSpace)
    assert space.choices == [True, False]

    # Test instantiation
    config = MyConfig(use_feature=True)
    assert config.use_feature is True

    # Test random sampling
    random_config = MyConfig.random()
    assert random_config.use_feature in [True, False]


def test_literal_inference():
    """Test that Literal fields are automatically converted to Categorical."""

    class MyConfig(sp.Config):
        activation: Literal["relu", "gelu", "silu"]

    # Check that a space was created
    assert "activation" in MyConfig._spaces
    space = MyConfig._spaces["activation"]
    assert isinstance(space, sp.CategoricalSpace)
    assert set(space.choices) == {"relu", "gelu", "silu"}

    # Test instantiation
    config = MyConfig(activation="relu")
    assert config.activation == "relu"

    # Test random sampling
    random_config = MyConfig.random()
    assert random_config.activation in ["relu", "gelu", "silu"]


def test_numeric_field_inference():
    """Test that numeric fields with Field constraints are auto-converted."""

    class MyConfig(sp.Config):
        learning_rate: float = Field(ge=0.0001, le=0.1)
        num_layers: int = Field(gt=0, lt=10)

    # Check that spaces were created
    assert "learning_rate" in MyConfig._spaces
    lr_space = MyConfig._spaces["learning_rate"]
    assert isinstance(lr_space, sp.FloatSpace)
    assert lr_space.low == 0.0001
    assert lr_space.high == 0.1

    assert "num_layers" in MyConfig._spaces
    layers_space = MyConfig._spaces["num_layers"]
    assert isinstance(layers_space, sp.IntSpace)
    assert layers_space.low == 0
    assert layers_space.high == 10

    # Test random sampling
    random_config = MyConfig.random()
    assert 0.0001 <= random_config.learning_rate <= 0.1
    assert 0 < random_config.num_layers < 10


def test_missing_space_with_default():
    """Test that fields with defaults don't require spaces."""

    class MyConfig(sp.Config):
        name: str = "default_name"
        value: int = 42

    # These should not have spaces but should work fine
    assert "name" not in MyConfig._spaces
    assert "value" not in MyConfig._spaces

    config = MyConfig()
    assert config.name == "default_name"
    assert config.value == 42


def test_missing_space_without_default_error():
    """Test that fields without spaces or defaults raise an error."""
    with pytest.raises(TypeError, match="cannot be automatically converted"):

        class MyConfig(sp.Config):
            unsupported_field: str  # No space, no default, can't infer


def test_explicit_space_overrides_inference():
    """Test that explicit spaces take precedence over inference."""

    class MyConfig(sp.Config):
        # Even though this is a bool, we define a custom space
        flag: bool = sp.Categorical([True, False, None])

    space = MyConfig._spaces["flag"]
    assert isinstance(space, sp.CategoricalSpace)
    assert space.choices == [True, False, None]  # Custom choices


def test_mixed_explicit_and_inferred():
    """Test a config with both explicit and inferred spaces."""

    class MyConfig(sp.Config):
        # Explicit space
        lr: float = sp.Float(ge=1e-5, le=1e-1, distribution="log")
        # Inferred from Literal
        activation: Literal["relu", "gelu"]
        # Inferred from bool
        use_dropout: bool
        # Has default, no space needed
        seed: int = 42

    assert isinstance(MyConfig._spaces["lr"], sp.FloatSpace)
    assert isinstance(MyConfig._spaces["activation"], sp.CategoricalSpace)
    assert isinstance(MyConfig._spaces["use_dropout"], sp.CategoricalSpace)
    assert "seed" not in MyConfig._spaces

    config = MyConfig.random()
    assert 1e-5 <= config.lr <= 1e-1
    assert config.activation in ["relu", "gelu"]
    assert config.use_dropout in [True, False]
    assert config.seed == 42
