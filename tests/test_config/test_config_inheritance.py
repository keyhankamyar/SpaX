"""Tests for Config inheritance and subclassing."""

import pytest

from spax import Categorical, Conditional, Config, Float, Int
from spax.spaces import EqualsTo, FieldCondition, LargerThan


class TestBasicInheritance:
    """Test basic inheritance patterns."""

    def test_subclass_inherits_parent_fields(self):
        """Test that subclass inherits parent fields."""

        class ParentConfig(Config):
            x: int = 10
            y: str = "hello"

        class ChildConfig(ParentConfig):
            z: float = 3.14

        config = ChildConfig()

        # Should have all fields
        assert config.x == 10
        assert config.y == "hello"
        assert config.z == 3.14

    def test_subclass_adds_new_fields(self):
        """Test subclass adding new fields."""

        class ParentConfig(Config):
            name: str = "parent"

        class ChildConfig(ParentConfig):
            version: int = 2
            enabled: bool = True

        config = ChildConfig()

        assert config.name == "parent"
        assert config.version == 2
        assert config.enabled is True

    def test_field_order_preserved(self):
        """Test that field order is preserved in inheritance."""

        class ParentConfig(Config):
            a: int = 1
            b: int = 2

        class ChildConfig(ParentConfig):
            c: int = 3
            d: int = 4

        # Fields should be in order: a, b, c, d
        fields = list(ChildConfig.model_fields.keys())
        assert fields == ["a", "b", "c", "d"]

    def test_parent_not_modified_by_child(self):
        """Test that defining child doesn't modify parent."""

        class ParentConfig(Config):
            x: int = 10

        class ChildConfig(ParentConfig):
            y: int = 20

        # Parent should only have x
        assert list(ParentConfig.model_fields.keys()) == ["x"]

        # Child should have both
        assert set(ChildConfig.model_fields.keys()) == {"x", "y"}

        # Instances are independent
        parent = ParentConfig()
        child = ChildConfig()

        assert parent.x == 10
        assert not hasattr(parent, "y")

        assert child.x == 10
        assert child.y == 20


class TestOverridingFields:
    """Test overriding parent fields in subclasses."""

    def test_override_with_different_default(self):
        """Test overriding field with different default value."""

        class ParentConfig(Config):
            value: int = Int(ge=0, le=100, default=50)

        class ChildConfig(ParentConfig):
            value: int = Int(ge=0, le=100, default=75)

        parent = ParentConfig()
        child = ChildConfig()

        assert parent.value == 50
        assert child.value == 75

    def test_override_with_narrower_range(self):
        """Test overriding with narrower range."""

        class ParentConfig(Config):
            value: int = Int(ge=0, le=100)

        class ChildConfig(ParentConfig):
            value: int = Int(ge=10, le=50)  # Narrower range

        # Parent accepts wider range
        parent = ParentConfig(value=5)
        assert parent.value == 5

        parent = ParentConfig(value=90)
        assert parent.value == 90

        # Child only accepts narrower range
        child = ChildConfig(value=30)
        assert child.value == 30

        with pytest.raises(ValueError):
            ChildConfig(value=5)  # Below child's minimum

    def test_override_with_wider_range(self):
        """Test overriding with wider range."""

        class ParentConfig(Config):
            value: int = Int(ge=10, le=50)

        class ChildConfig(ParentConfig):
            value: int = Int(ge=0, le=100)  # Wider range

        # Child accepts wider range
        child = ChildConfig(value=5)
        assert child.value == 5

        child = ChildConfig(value=90)
        assert child.value == 90

    def test_override_int_with_float(self):
        """Test overriding int field with float field."""

        class ParentConfig(Config):
            value: int = Int(ge=0, le=10)

        class ChildConfig(ParentConfig):
            value: float = Float(ge=0.0, le=10.0)

        parent = ParentConfig(value=5)
        assert isinstance(parent.value, int)

        child = ChildConfig(value=5.5)
        assert isinstance(child.value, float)
        assert child.value == 5.5

    def test_override_space_with_fixed_value(self):
        """Test overriding space with fixed value."""

        class ParentConfig(Config):
            value: int = Int(ge=0, le=100)

        class ChildConfig(ParentConfig):
            value: int = 42  # Fixed value

        # Parent accepts any value in range
        parent = ParentConfig(value=50)
        assert parent.value == 50

        # Child has fixed value
        child = ChildConfig()
        assert child.value == 42

    def test_override_fixed_value_with_space(self):
        """Test overriding fixed value with space."""

        class ParentConfig(Config):
            value: int = 42  # Fixed

        class ChildConfig(ParentConfig):
            value: int = Int(ge=0, le=100)  # Space

        # Parent has fixed value
        parent = ParentConfig()
        assert parent.value == 42

        # Child can vary
        child = ChildConfig(value=75)
        assert child.value == 75

    def test_override_categorical_choices(self):
        """Test overriding categorical with different choices."""

        class ParentConfig(Config):
            mode: str = Categorical(["a", "b", "c"])

        class ChildConfig(ParentConfig):
            mode: str = Categorical(["a", "b"])  # Fewer choices

        # Parent accepts all three
        parent = ParentConfig(mode="c")
        assert parent.mode == "c"

        # Child only accepts two
        child = ChildConfig(mode="a")
        assert child.mode == "a"

        with pytest.raises(ValueError):
            ChildConfig(mode="c")

    def test_override_multiple_fields(self):
        """Test overriding multiple fields at once."""

        class ParentConfig(Config):
            x: int = Int(ge=0, le=10, default=5)
            y: str = "parent"
            z: float = 1.0

        class ChildConfig(ParentConfig):
            x: int = Int(ge=0, le=20, default=10)  # Different range and default
            y: str = "child"  # Different default
            # z not overridden

        parent = ParentConfig()
        assert parent.x == 5
        assert parent.y == "parent"
        assert parent.z == 1.0

        child = ChildConfig()
        assert child.x == 10
        assert child.y == "child"
        assert child.z == 1.0  # Inherited unchanged


class TestMultiLevelInheritance:
    """Test multi-level inheritance chains."""

    def test_three_level_inheritance(self):
        """Test three-level inheritance A -> B -> C."""

        class ConfigA(Config):
            a: int = 1

        class ConfigB(ConfigA):
            b: int = 2

        class ConfigC(ConfigB):
            c: int = 3

        config = ConfigC()

        assert config.a == 1
        assert config.b == 2
        assert config.c == 3

    def test_override_in_middle_level(self):
        """Test overriding in middle level of inheritance."""

        class ConfigA(Config):
            value: int = Int(ge=0, le=100, default=50)

        class ConfigB(ConfigA):
            value: int = Int(ge=0, le=100, default=30)  # Override

        class ConfigC(ConfigB):
            extra: int = 10

        a = ConfigA()
        b = ConfigB()
        c = ConfigC()

        assert a.value == 50
        assert b.value == 30
        assert c.value == 30  # Inherits from B

    def test_override_in_final_level(self):
        """Test overriding in final level of inheritance."""

        class ConfigA(Config):
            value: int = Int(ge=0, le=100, default=50)

        class ConfigB(ConfigA):
            pass  # No override

        class ConfigC(ConfigB):
            value: int = Int(ge=0, le=100, default=70)  # Override here

        a = ConfigA()
        b = ConfigB()
        c = ConfigC()

        assert a.value == 50
        assert b.value == 50
        assert c.value == 70

    def test_progressive_narrowing(self):
        """Test progressively narrowing ranges through inheritance."""

        class ConfigA(Config):
            value: int = Int(ge=0, le=100)

        class ConfigB(ConfigA):
            value: int = Int(ge=10, le=90)  # Narrower

        class ConfigC(ConfigB):
            value: int = Int(ge=20, le=80)  # Even narrower

        # Each level has progressively narrower constraints
        ConfigA(value=5)  # OK
        ConfigB(value=15)  # OK

        with pytest.raises(ValueError):
            ConfigB(value=5)  # Too low for B

        ConfigC(value=50)  # OK

        with pytest.raises(ValueError):
            ConfigC(value=15)  # Too low for C


class TestInheritanceWithConditionals:
    """Test inheritance with conditional fields."""

    def test_inherit_conditional_field(self):
        """Test inheriting a conditional field."""

        class ParentConfig(Config):
            mode: str = Categorical(["simple", "advanced"])
            value: int = Conditional(
                condition=FieldCondition("mode", EqualsTo("advanced")),
                true=Int(ge=0, le=100),
                false=Int(ge=0, le=10),
            )

        class ChildConfig(ParentConfig):
            extra: int = 5

        # Child should have the conditional behavior
        child = ChildConfig(mode="advanced", value=50)
        assert child.value == 50

        child = ChildConfig(mode="simple", value=5)
        assert child.value == 5

        with pytest.raises(ValueError):
            ChildConfig(mode="simple", value=50)

    def test_override_conditional_field(self):
        """Test overriding a conditional field."""

        class ParentConfig(Config):
            enabled: bool = True
            value: int = Conditional(
                condition=FieldCondition("enabled", EqualsTo(True)),
                true=Int(ge=0, le=100),
                false=0,
            )

        class ChildConfig(ParentConfig):
            value: int = Conditional(
                condition=FieldCondition("enabled", EqualsTo(True)),
                true=Int(ge=0, le=1000),  # Wider range
                false=0,
            )

        # Parent limited to 0-100
        parent = ParentConfig(enabled=True, value=50)
        assert parent.value == 50

        with pytest.raises(ValueError):
            ParentConfig(enabled=True, value=500)

        # Child can go 0-1000
        child = ChildConfig(enabled=True, value=500)
        assert child.value == 500

    def test_add_conditional_depending_on_parent_field(self):
        """Test adding conditional that depends on parent's field."""

        class ParentConfig(Config):
            mode: str = Categorical(["a", "b", "c"])

        class ChildConfig(ParentConfig):
            value: int = Conditional(
                condition=FieldCondition("mode", EqualsTo("a")),
                true=Int(ge=0, le=10),
                false=Int(ge=10, le=100),
            )

        child = ChildConfig(mode="a", value=5)
        assert child.value == 5

        child = ChildConfig(mode="b", value=50)
        assert child.value == 50

    def test_override_condition_logic(self):
        """Test overriding with different condition logic."""

        class ParentConfig(Config):
            x: int = Int(ge=0, le=10, default=5)
            y: int = Conditional(
                condition=FieldCondition("x", LargerThan(5)), true=100, false=10
            )

        class ChildConfig(ParentConfig):
            y: int = Conditional(
                condition=FieldCondition("x", LargerThan(3)),  # Different threshold
                true=100,
                false=10,
            )

        parent = ParentConfig(x=4)
        assert parent.y == 10  # x=4 is not > 5

        child = ChildConfig(x=4)
        assert child.y == 100  # x=4 is > 3


class TestInheritanceWithNestedConfigs:
    """Test inheritance with nested config fields."""

    def test_inherit_nested_config(self):
        """Test inheriting a nested config field."""

        class InnerConfig(Config):
            value: int = 10

        class ParentConfig(Config):
            inner: InnerConfig

        class ChildConfig(ParentConfig):
            extra: int = 5

        inner = InnerConfig()
        child = ChildConfig(inner=inner, extra=5)

        assert child.inner.value == 10
        assert child.extra == 5

    def test_override_nested_config_type(self):
        """Test overriding nested config with different type."""

        class InnerA(Config):
            a: int = 1

        class InnerB(Config):
            b: int = 2

        class ParentConfig(Config):
            inner: InnerA

        class ChildConfig(ParentConfig):
            inner: InnerB  # Different type

        parent = ParentConfig(inner=InnerA())
        assert parent.inner.a == 1

        child = ChildConfig(inner=InnerB())
        assert child.inner.b == 2

    def test_extend_nested_config(self):
        """Test child's nested config extending parent's nested config."""

        class BaseInner(Config):
            x: int = 1

        class ExtendedInner(BaseInner):
            y: int = 2

        class ParentConfig(Config):
            inner: BaseInner

        class ChildConfig(ParentConfig):
            inner: ExtendedInner  # Extended version

        # Child can use extended inner config
        extended = ExtendedInner()
        child = ChildConfig(inner=extended)

        assert child.inner.x == 1
        assert child.inner.y == 2


class TestInheritanceEdgeCases:
    """Test edge cases in inheritance."""

    def test_multiple_overrides_same_field(self):
        """Test overriding the same field multiple times."""

        class ConfigA(Config):
            value: int = Int(ge=0, le=100, default=50)

        class ConfigB(ConfigA):
            value: int = Int(ge=0, le=100, default=30)

        class ConfigC(ConfigB):
            value: int = Int(ge=0, le=100, default=70)

        a = ConfigA()
        b = ConfigB()
        c = ConfigC()

        assert a.value == 50
        assert b.value == 30
        assert c.value == 70

    def test_child_adds_many_fields(self):
        """Test child adding many fields."""

        class ParentConfig(Config):
            a: int = 1

        class ChildConfig(ParentConfig):
            b: int = 2
            c: int = 3
            d: int = 4
            e: int = 5
            f: int = 6

        config = ChildConfig()

        assert config.a == 1
        assert config.b == 2
        assert config.c == 3
        assert config.d == 4
        assert config.e == 5
        assert config.f == 6

    def test_override_changes_type_dramatically(self):
        """Test overriding with completely different type."""

        class ParentConfig(Config):
            value: int = Int(ge=0, le=10)

        class ChildConfig(ParentConfig):
            value: str = Categorical(["a", "b", "c"])

        parent = ParentConfig(value=5)
        assert isinstance(parent.value, int)

        child = ChildConfig(value="a")
        assert isinstance(child.value, str)

    def test_dependency_ordering_with_new_fields(self):
        """Test dependency ordering when child adds conditional fields."""

        class ParentConfig(Config):
            mode: str = Categorical(["a", "b"])

        class ChildConfig(ParentConfig):
            # This depends on parent's mode field
            value: int = Conditional(
                condition=FieldCondition("mode", EqualsTo("a")), true=10, false=20
            )
            extra: int = 5

        # Should work - dependency ordering should be maintained
        child = ChildConfig(mode="a", extra=5)
        assert child.value == 10

        child = ChildConfig(mode="b", extra=5)
        assert child.value == 20

    def test_child_adds_dependency_on_inherited_conditional(self):
        """Test child adding field that depends on inherited conditional."""

        class ParentConfig(Config):
            x: int = Int(ge=0, le=10, default=5)
            y: int = Conditional(
                condition=FieldCondition("x", LargerThan(5)), true=100, false=10
            )

        class ChildConfig(ParentConfig):
            # Depends on y, which is conditional in parent
            z: int = Conditional(
                condition=FieldCondition("y", EqualsTo(100)), true=1000, false=1
            )

        # x=7 > 5, so y=100, so z=1000
        child = ChildConfig(x=7)
        assert child.y == 100
        assert child.z == 1000

        # x=3 <= 5, so y=10, so z=1
        child = ChildConfig(x=3)
        assert child.y == 10
        assert child.z == 1
