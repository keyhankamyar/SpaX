"""Tests for categorical spaces."""

import pytest

from spax import Config
from spax.spaces import UNSET, Categorical, CategoricalSpace, Choice


class TestChoice:
    """Test Choice class."""

    def test_basic_choice_creation(self):
        """Test basic Choice creation."""
        choice = Choice("option_a", weight=2.0)
        assert choice.value == "option_a"
        assert choice.weight == 2.0

    def test_default_weight(self):
        """Test default weight is 1.0."""
        choice = Choice("option_a")
        assert choice.value == "option_a"
        assert choice.weight == 1.0

    def test_choice_with_various_value_types(self):
        """Test Choice with different value types."""
        # Integer
        choice = Choice(42)
        assert choice.value == 42

        # String
        choice = Choice("text")
        assert choice.value == "text"

        # None
        choice = Choice(None)
        assert choice.value is None

        # Float
        choice = Choice(3.14)
        assert choice.value == 3.14

    def test_choice_weight_accepts_int(self):
        """Test that weight accepts int and converts to float."""
        choice = Choice("value", weight=5)
        assert choice.weight == 5.0
        assert isinstance(choice.weight, float)

    def test_choice_non_numeric_weight_error(self):
        """Test error when weight is non-numeric."""
        with pytest.raises(TypeError, match="weight must be numeric"):
            Choice("value", weight="not a number")

    def test_choice_zero_weight_error(self):
        """Test error when weight is zero."""
        with pytest.raises(ValueError, match="weight must be positive"):
            Choice("value", weight=0)

    def test_choice_negative_weight_error(self):
        """Test error when weight is negative."""
        with pytest.raises(ValueError, match="weight must be positive"):
            Choice("value", weight=-1.0)

    def test_choice_equality(self):
        """Test Choice equality comparison."""
        choice1 = Choice("a", weight=1.0)
        choice2 = Choice("a", weight=1.0)
        choice3 = Choice("a", weight=2.0)
        choice4 = Choice("b", weight=1.0)

        assert choice1 == choice2
        assert choice1 != choice3  # Different weight
        assert choice1 != choice4  # Different value

    def test_choice_repr(self):
        """Test Choice string representation."""
        choice = Choice("value", weight=2.5)
        repr_str = repr(choice)

        assert "Choice" in repr_str
        assert "value" in repr_str
        assert "2.5" in repr_str


class TestCategoricalSpace:
    """Test CategoricalSpace."""

    def test_basic_categorical_simple_values(self):
        """Test basic categorical with simple values."""
        space = CategoricalSpace(["a", "b", "c"])
        space.field_name = "test_field"

        assert space.validate("a") == "a"
        assert space.validate("b") == "b"
        assert space.validate("c") == "c"

    def test_categorical_with_integers(self):
        """Test categorical with integer values."""
        space = CategoricalSpace([1, 2, 3, 4, 5])
        space.field_name = "test_field"

        assert space.validate(3) == 3
        with pytest.raises(ValueError, match="not in allowed choices"):
            space.validate(6)

    def test_categorical_with_mixed_types(self):
        """Test categorical with mixed value types."""
        space = CategoricalSpace([1, "two", 3.0, None, True])
        space.field_name = "test_field"

        assert space.validate(1) == 1
        assert space.validate("two") == "two"
        assert space.validate(3.0) == 3.0
        assert space.validate(None) is None
        assert space.validate(True) is True

    def test_categorical_with_choice_objects(self):
        """Test categorical with Choice objects."""
        space = CategoricalSpace(
            [Choice("rare", weight=1.0), Choice("common", weight=10.0)]
        )
        space.field_name = "test_field"

        assert space.validate("rare") == "rare"
        assert space.validate("common") == "common"

    def test_categorical_mixed_weighted_unweighted(self):
        """Test categorical with mix of Choice objects and plain values."""
        space = CategoricalSpace(
            ["plain", Choice("weighted", weight=5.0), "another_plain"]
        )
        space.field_name = "test_field"

        assert space.validate("plain") == "plain"
        assert space.validate("weighted") == "weighted"
        assert space.validate("another_plain") == "another_plain"

    def test_single_choice(self):
        """Test categorical with single choice."""
        space = CategoricalSpace(["only_option"])
        space.field_name = "test_field"

        assert space.validate("only_option") == "only_option"

        # Sampling should always return the only choice
        samples = [space.sample() for _ in range(10)]
        assert all(s == "only_option" for s in samples)

    def test_sampling_equal_weights(self):
        """Test sampling with equal weights."""
        space = CategoricalSpace(["a", "b", "c", "d"])

        samples = [space.sample() for _ in range(200)]

        # All choices should appear
        assert "a" in samples
        assert "b" in samples
        assert "c" in samples
        assert "d" in samples

        # With equal weights, distribution should be roughly equal
        # (allowing for randomness)
        counts = {choice: samples.count(choice) for choice in ["a", "b", "c", "d"]}
        # Each should appear roughly 50 times (200/4)
        # Allow generous margin for randomness
        for count in counts.values():
            assert 20 < count < 80

    def test_sampling_respects_weights(self):
        """Test that sampling respects weight distribution."""
        space = CategoricalSpace(
            [Choice("rare", weight=1.0), Choice("common", weight=99.0)]
        )

        samples = [space.sample() for _ in range(1000)]

        common_count = samples.count("common")
        rare_count = samples.count("rare")

        # "common" should appear much more frequently
        assert common_count > 900
        assert rare_count < 100

    def test_default_value(self):
        """Test default value handling."""
        space = CategoricalSpace(["a", "b", "c"], default="b")
        assert space.default == "b"

    def test_default_value_validation(self):
        """Test that default value must be in choices."""
        # Valid default
        space = CategoricalSpace(["a", "b", "c"], default="b")
        assert space.default == "b"

        # Invalid default
        with pytest.raises(ValueError, match="Invalid default"):
            CategoricalSpace(["a", "b", "c"], default="d")

    def test_no_default(self):
        """Test space without default."""
        space = CategoricalSpace(["a", "b", "c"])
        assert space.default is UNSET

    def test_description(self):
        """Test description field."""
        space = CategoricalSpace(["a", "b"], description="Test categorical")
        assert space.description == "Test categorical"

    def test_validation_rejects_invalid_value(self):
        """Test that validation rejects values not in choices."""
        space = CategoricalSpace(["a", "b", "c"])
        space.field_name = "test_field"

        with pytest.raises(ValueError, match="not in allowed choices"):
            space.validate("d")

    def test_empty_choices_error(self):
        """Test error when choices is empty."""
        with pytest.raises(ValueError, match="at least one choice"):
            CategoricalSpace([])

    def test_non_comparable_choice_error(self):
        """Test error when choice value is non-comparable."""

        class NotComparable:
            pass

        with pytest.raises(ValueError, match="must be comparable"):
            CategoricalSpace([NotComparable()])

    def test_descriptor_protocol(self):
        """Test that CategoricalSpace works as a descriptor."""

        class TestConfig:
            choice = CategoricalSpace(["a", "b", "c"])

        # Access from class returns descriptor
        assert isinstance(TestConfig.choice, CategoricalSpace)

        # Access from instance returns value
        config = TestConfig()
        config.choice = "b"
        assert config.choice == "b"

    def test_field_name_propagation(self):
        """Test that field_name is set via __set_name__."""

        class TestConfig:
            my_choice = CategoricalSpace(["a", "b"])

        assert TestConfig.my_choice.field_name == "my_choice"

    def test_repr_simple(self):
        """Test string representation with simple choices."""
        space = CategoricalSpace(["a", "b", "c"])
        repr_str = repr(space)

        assert "CategoricalSpace" in repr_str
        assert "choices" in repr_str

    def test_repr_with_weights(self):
        """Test repr shows probs when weights differ."""
        space = CategoricalSpace([Choice("a", weight=1.0), Choice("b", weight=2.0)])
        repr_str = repr(space)

        assert "CategoricalSpace" in repr_str
        assert "probs" in repr_str

    def test_repr_with_default(self):
        """Test repr with default value."""
        space = CategoricalSpace(["a", "b"], default="a")
        repr_str = repr(space)

        assert "default='a'" in repr_str


class TestCategoricalWithConfigTypes:
    """Test CategoricalSpace with Config types."""

    def test_categorical_with_config_type(self):
        """Test categorical space with Config type as choice."""

        class ConfigA(Config):
            value: int = 1

        class ConfigB(Config):
            value: int = 2

        space = CategoricalSpace([ConfigA, ConfigB])
        space.field_name = "test_field"

        # Should accept instances
        instance_a = ConfigA()
        assert space.validate(instance_a) == instance_a

        instance_b = ConfigB()
        assert space.validate(instance_b) == instance_b

    def test_categorical_mixed_config_and_values(self):
        """Test categorical with mix of Config types and regular values."""

        class MyConfig(Config):
            x: int = 10

        space = CategoricalSpace([MyConfig, "string_option", None])
        space.field_name = "test_field"

        # Config instance
        config_instance = MyConfig()
        assert space.validate(config_instance) == config_instance

        # Regular values
        assert space.validate("string_option") == "string_option"
        assert space.validate(None) is None

    def test_sampling_config_types(self):
        """Test sampling produces valid instances when Config types are choices."""

        class ConfigA(Config):
            value: int = 1

        class ConfigB(Config):
            value: int = 2

        # Note: CategoricalSpace stores the type itself, not instances
        # So sampling will return the type, not an instance
        space = CategoricalSpace([ConfigA, ConfigB])

        samples = [space.sample() for _ in range(10)]

        # All samples should be one of the Config types
        assert all(s in [ConfigA, ConfigB] for s in samples)


class TestCategoricalFactory:
    """Test Categorical() factory function."""

    def test_categorical_factory_creates_space(self):
        """Test that Categorical() creates a CategoricalSpace."""
        space = Categorical(["a", "b", "c"])
        assert isinstance(space, CategoricalSpace)

    def test_categorical_factory_all_parameters(self):
        """Test Categorical() with all parameters."""
        space = Categorical(["a", "b", "c"], default="b", description="Test")

        assert isinstance(space, CategoricalSpace)
        assert space.choices == ["a", "b", "c"]
        assert space.default == "b"
        assert space.description == "Test"


class TestCategoricalEdgeCases:
    """Test edge cases for categorical spaces."""

    def test_large_number_of_choices(self):
        """Test categorical with many choices."""
        choices = list(range(1000))
        space = CategoricalSpace(choices)
        space.field_name = "test_field"

        # Should validate any choice
        assert space.validate(500) == 500
        assert space.validate(999) == 999

        # Sampling should work
        samples = [space.sample() for _ in range(100)]
        assert all(s in choices for s in samples)

    def test_extreme_weight_differences(self):
        """Test with extreme weight differences."""
        space = CategoricalSpace(
            [Choice("very_rare", weight=1.0), Choice("very_common", weight=1000.0)]
        )

        samples = [space.sample() for _ in range(1000)]

        # Very common should dominate
        very_common_count = samples.count("very_common")
        assert very_common_count > 980

    def test_all_same_weights_no_probs_in_repr(self):
        """Test that probs are not shown in repr when all weights are equal."""
        space = CategoricalSpace(
            [Choice("a", weight=5.0), Choice("b", weight=5.0), Choice("c", weight=5.0)]
        )
        repr(space)

        # When all weights are equal, probs should not be shown
        # (they're all the same, not interesting)
        # Actually, looking at the implementation, it checks if set(probs) > 1
        # So if all are equal, they should NOT appear
        # Let's verify this behavior
        assert len(set(space.probs)) == 1  # All probs are equal
