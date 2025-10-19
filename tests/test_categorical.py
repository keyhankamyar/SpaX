"""Tests for categorical search spaces."""

import pytest

from spax.config import Config
from spax.spaces import Categorical, CategoricalSpace, Choice


class TestChoice:
    """Tests for Choice class."""

    def test_creation_with_default_weight(self):
        """Test creating a choice with default weight."""
        choice = Choice("relu")
        assert choice.value == "relu"
        assert choice.weight == 1.0

    def test_creation_with_custom_weight(self):
        """Test creating a choice with custom weight."""
        choice = Choice("gelu", weight=2.5)
        assert choice.value == "gelu"
        assert choice.weight == 2.5

    def test_non_numeric_weight_raises_error(self):
        """Test that non-numeric weights raise TypeError."""
        with pytest.raises(TypeError, match="weight must be numeric"):
            Choice("value", weight="invalid")  # type: ignore

    def test_negative_weight_raises_error(self):
        """Test that negative weights raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            Choice("value", weight=-1.0)

    def test_zero_weight_raises_error(self):
        """Test that zero weight raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            Choice("value", weight=0.0)

    def test_equality(self):
        """Test choice equality comparison."""
        choice1 = Choice("relu", weight=1.0)
        choice2 = Choice("relu", weight=1.0)
        choice3 = Choice("relu", weight=2.0)
        choice4 = Choice("gelu", weight=1.0)

        assert choice1 == choice2
        assert choice1 != choice3  # Different weight
        assert choice1 != choice4  # Different value


class TestCategoricalSpace:
    """Tests for CategoricalSpace."""

    def test_creation_with_simple_choices(self):
        """Test creating a categorical space with simple values."""
        space = Categorical(["a", "b", "c"])
        assert isinstance(space, CategoricalSpace)
        assert space.choices == ["a", "b", "c"]
        assert space.weights == [1.0, 1.0, 1.0]

    def test_creation_with_weighted_choices(self):
        """Test creating a categorical space with weighted choices."""
        space = Categorical(
            [
                Choice("a", weight=2.0),
                Choice("b", weight=1.0),
                "c",  # Default weight 1.0
            ]
        )

        assert space.choices == ["a", "b", "c"]
        assert space.weights == [2.0, 1.0, 1.0]
        # Probabilities should sum to 1
        assert abs(sum(space.probs) - 1.0) < 1e-10
        assert space.probs[0] == 0.5  # 2.0 / 4.0
        assert space.probs[1] == 0.25  # 1.0 / 4.0
        assert space.probs[2] == 0.25  # 1.0 / 4.0

    def test_sample_returns_valid_choice(self):
        """Test that sampling returns one of the choices."""
        space = Categorical(["relu", "gelu", "silu"])

        for _ in range(50):
            value = space.sample()
            assert value in ["relu", "gelu", "silu"]

    def test_weighted_sampling_distribution(self):
        """Test that weights affect sampling probability."""
        space = Categorical([Choice("a", weight=99.0), Choice("b", weight=1.0)])

        samples = [space.sample() for _ in range(1000)]
        a_count = samples.count("a")

        # With 99% weight on "a", expect roughly 990/1000
        assert a_count > 900

    def test_validate_valid_choice(self):
        """Test validation accepts valid choices."""
        space = Categorical(["relu", "gelu", "silu"])
        space.field_name = "activation"

        assert space.validate("relu") == "relu"
        assert space.validate("gelu") == "gelu"

    def test_validate_invalid_choice(self):
        """Test validation rejects invalid choices."""
        space = Categorical(["relu", "gelu", "silu"])
        space.field_name = "activation"

        with pytest.raises(ValueError, match="not in allowed choices"):
            space.validate("tanh")

    def test_validate_with_config_types(self):
        """Test validation with nested Config types."""

        class SubConfig1(Config):
            value: int = 1

        class SubConfig2(Config):
            value: int = 2

        space = Categorical([SubConfig1, SubConfig2])
        space.field_name = "nested"

        # Should accept instances of the config types
        config1 = SubConfig1()
        assert space.validate(config1) == config1

        config2 = SubConfig2()
        assert space.validate(config2) == config2

    def test_sample_with_config_types(self):
        """Test sampling with nested Config types returns a type."""

        class SubConfig1(Config):
            value: int = 1

        class SubConfig2(Config):
            value: int = 2

        space = Categorical([SubConfig1, SubConfig2])

        for _ in range(10):
            sampled = space.sample()
            assert sampled in [SubConfig1, SubConfig2]
            assert isinstance(sampled, type)

    def test_empty_choices_raises_error(self):
        """Test that empty choices list raises ValueError."""
        with pytest.raises(ValueError, match="at least one choice"):
            Categorical([])

    def test_numeric_choices_work(self):
        """Test that numeric choices work correctly."""
        space = Categorical([1, 2, 3])
        space.field_name = "test"

        assert space.validate(1) == 1
        assert space.validate(2) == 2

        for _ in range(20):
            value = space.sample()
            assert value in [1, 2, 3]
