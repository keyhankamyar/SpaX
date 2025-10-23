"""Tests for Config serialization and deserialization."""

import json

from spax import Categorical, Conditional, Config, Int
from spax.spaces import EqualsTo, FieldCondition


class TestDictSerialization:
    """Test model_dump and model_validate with dict."""

    def test_simple_config_to_dict(self):
        """Test dumping simple config to dict."""

        class MyConfig(Config):
            x: int = 10
            y: str = "hello"

        config = MyConfig()
        data = config.model_dump()

        assert isinstance(data, dict)
        assert data["x"] == 10
        assert data["y"] == "hello"

    def test_simple_config_from_dict(self):
        """Test loading simple config from dict."""

        class MyConfig(Config):
            x: int = 10
            y: str = "hello"

        data = {"x": 20, "y": "world"}
        config = MyConfig.model_validate(data)

        assert config.x == 20
        assert config.y == "world"

    def test_config_with_spaces_to_dict(self):
        """Test dumping config with spaces to dict."""

        class MyConfig(Config):
            value: int = Int(ge=0, le=100, default=50)
            mode: str = Categorical(["a", "b", "c"], default="b")

        config = MyConfig()
        data = config.model_dump()

        assert data["value"] == 50
        assert data["mode"] == "b"

    def test_config_with_spaces_from_dict(self):
        """Test loading config with spaces from dict."""

        class MyConfig(Config):
            value: int = Int(ge=0, le=100)
            mode: str = Categorical(["a", "b", "c"])

        data = {"value": 75, "mode": "c"}
        config = MyConfig.model_validate(data)

        assert config.value == 75
        assert config.mode == "c"

    def test_round_trip_simple(self):
        """Test round-trip (dump then validate) with simple config."""

        class MyConfig(Config):
            x: int = Int(ge=0, le=10)
            y: str = "test"

        original = MyConfig(x=7)

        # Dump
        data = original.model_dump()

        # Load
        restored = MyConfig.model_validate(data)

        assert restored.x == original.x
        assert restored.y == original.y


class TestNestedConfigSerialization:
    """Test serialization with nested configs."""

    def test_nested_config_to_dict_with_type_discriminator(self):
        """Test that nested configs include __type__ discriminator."""

        class InnerConfig(Config):
            value: int = 10

        class OuterConfig(Config):
            inner: InnerConfig

        inner = InnerConfig()
        outer = OuterConfig(inner=inner)

        data = outer.model_dump()

        assert "inner" in data
        assert "__type__" in data["inner"]
        assert data["inner"]["__type__"] == "InnerConfig"
        assert data["inner"]["value"] == 10

    def test_nested_config_from_dict_with_type_discriminator(self):
        """Test loading nested config using __type__ discriminator."""

        class InnerConfig(Config):
            value: int = 10

        class OuterConfig(Config):
            inner: InnerConfig

        data = {"inner": {"__type__": "InnerConfig", "value": 20}}

        outer = OuterConfig.model_validate(data)

        assert isinstance(outer.inner, InnerConfig)
        assert outer.inner.value == 20

    def test_deeply_nested_with_discriminators(self):
        """Test deeply nested configs with type discriminators."""

        class Level3(Config):
            x: int = 3

        class Level2(Config):
            level3: Level3

        class Level1(Config):
            level2: Level2

        l3 = Level3()
        l2 = Level2(level3=l3)
        l1 = Level1(level2=l2)

        data = l1.model_dump()

        # Check discriminators at each level
        assert data["level2"]["__type__"] == "Level2"
        assert data["level2"]["level3"]["__type__"] == "Level3"
        assert data["level2"]["level3"]["x"] == 3

    def test_round_trip_nested(self):
        """Test round-trip with nested configs."""

        class InnerConfig(Config):
            value: int = Int(ge=0, le=100, default=50)

        class OuterConfig(Config):
            inner: InnerConfig
            name: str = "test"

        original = OuterConfig(inner=InnerConfig(value=75), name="original")

        # Dump
        data = original.model_dump()

        # Load
        restored = OuterConfig.model_validate(data)

        assert restored.inner.value == 75
        assert restored.name == "original"

    def test_multiple_nested_configs(self):
        """Test serialization with multiple nested configs."""

        class ConfigA(Config):
            a: int = 1

        class ConfigB(Config):
            b: int = 2

        class ParentConfig(Config):
            config_a: ConfigA
            config_b: ConfigB

        parent = ParentConfig(config_a=ConfigA(), config_b=ConfigB())

        data = parent.model_dump()

        assert data["config_a"]["__type__"] == "ConfigA"
        assert data["config_b"]["__type__"] == "ConfigB"

        restored = ParentConfig.model_validate(data)
        assert restored.config_a.a == 1
        assert restored.config_b.b == 2


class TestUnionConfigSerialization:
    """Test serialization with union of config types."""

    def test_union_config_serialization(self):
        """Test serialization of union config types."""

        class ConfigA(Config):
            a: int = 1

        class ConfigB(Config):
            b: int = 2

        class ParentConfig(Config):
            choice: ConfigA | ConfigB = Categorical([ConfigA, ConfigB])

        # Using ConfigA
        parent = ParentConfig(choice=ConfigA())
        data = parent.model_dump()

        assert data["choice"]["__type__"] == "ConfigA"
        assert data["choice"]["a"] == 1

        restored = ParentConfig.model_validate(data)
        assert isinstance(restored.choice, ConfigA)

    def test_union_with_none_serialization(self):
        """Test serialization with Config | None union."""

        class MyConfig(Config):
            x: int = 10

        class ParentConfig(Config):
            optional: MyConfig | None = Categorical([MyConfig, None])

        # With config
        parent = ParentConfig(optional=MyConfig())
        data = parent.model_dump()

        assert data["optional"]["__type__"] == "MyConfig"

        # With None
        parent = ParentConfig(optional=None)
        data = parent.model_dump()

        assert data["optional"] is None


class TestJSONSerialization:
    """Test model_dump_json and model_validate_json."""

    def test_simple_config_to_json(self):
        """Test dumping simple config to JSON."""

        class MyConfig(Config):
            x: int = 10
            y: str = "hello"

        config = MyConfig()
        json_str = config.model_dump_json()

        assert isinstance(json_str, str)

        # Parse to verify it's valid JSON
        data = json.loads(json_str)
        assert data["x"] == 10
        assert data["y"] == "hello"

    def test_simple_config_from_json(self):
        """Test loading simple config from JSON."""

        class MyConfig(Config):
            x: int = 10
            y: str = "hello"

        json_str = '{"x": 20, "y": "world"}'
        config = MyConfig.model_validate_json(json_str)

        assert config.x == 20
        assert config.y == "world"

    def test_json_with_custom_indent(self):
        """Test JSON dumping with custom indent."""

        class MyConfig(Config):
            x: int = 10
            y: str = "hello"

        config = MyConfig()

        # Default indent (2)
        json_default = config.model_dump_json()
        assert "\n" in json_default

        # No indent
        json_compact = config.model_dump_json(indent=None)
        assert "\n" not in json_compact

        # Custom indent
        json_custom = config.model_dump_json(indent=4)
        assert "    " in json_custom

    def test_nested_config_json_round_trip(self):
        """Test JSON round-trip with nested configs."""

        class InnerConfig(Config):
            value: int = Int(ge=0, le=100)

        class OuterConfig(Config):
            inner: InnerConfig
            name: str = "test"

        original = OuterConfig(inner=InnerConfig(value=75), name="original")

        # Dump to JSON
        json_str = original.model_dump_json()

        # Verify type discriminator in JSON
        data = json.loads(json_str)
        assert data["inner"]["__type__"] == "InnerConfig"

        # Load from JSON
        restored = OuterConfig.model_validate_json(json_str)

        assert restored.inner.value == 75
        assert restored.name == "original"


class TestYAMLSerialization:
    """Test model_dump_yaml and model_validate_yaml."""

    def test_simple_config_to_yaml(self):
        """Test dumping simple config to YAML."""

        class MyConfig(Config):
            x: int = 10
            y: str = "hello"

        config = MyConfig()
        yaml_str = config.model_dump_yaml()

        assert isinstance(yaml_str, str)
        assert "x: 10" in yaml_str
        assert "y: hello" in yaml_str

    def test_simple_config_from_yaml(self):
        """Test loading simple config from YAML."""

        class MyConfig(Config):
            x: int = 10
            y: str = "hello"

        yaml_str = """
x: 20
y: world
"""
        config = MyConfig.model_validate_yaml(yaml_str)

        assert config.x == 20
        assert config.y == "world"

    def test_nested_config_yaml_round_trip(self):
        """Test YAML round-trip with nested configs."""

        class InnerConfig(Config):
            value: int = 50

        class OuterConfig(Config):
            inner: InnerConfig
            name: str = "test"

        original = OuterConfig(inner=InnerConfig(), name="original")

        # Dump to YAML
        yaml_str = original.model_dump_yaml()

        # Verify structure
        assert "inner:" in yaml_str
        assert "__type__: InnerConfig" in yaml_str
        assert "value: 50" in yaml_str

        # Load from YAML
        restored = OuterConfig.model_validate_yaml(yaml_str)

        assert restored.inner.value == 50
        assert restored.name == "original"


class TestTOMLSerialization:
    """Test model_dump_toml and model_validate_toml."""

    def test_simple_config_to_toml(self):
        """Test dumping simple config to TOML."""

        class MyConfig(Config):
            x: int = 10
            y: str = "hello"

        config = MyConfig()
        toml_str = config.model_dump_toml()

        assert isinstance(toml_str, str)
        assert "x = 10" in toml_str
        assert 'y = "hello"' in toml_str or "y = 'hello'" in toml_str

    def test_simple_config_from_toml(self):
        """Test loading simple config from TOML."""

        class MyConfig(Config):
            x: int = 10
            y: str = "hello"

        toml_str = """
x = 20
y = "world"
"""
        config = MyConfig.model_validate_toml(toml_str)

        assert config.x == 20
        assert config.y == "world"

    def test_nested_config_toml_round_trip(self):
        """Test TOML round-trip with nested configs."""

        class InnerConfig(Config):
            value: int = 50

        class OuterConfig(Config):
            inner: InnerConfig
            name: str = "test"

        original = OuterConfig(inner=InnerConfig(), name="original")

        # Dump to TOML
        toml_str = original.model_dump_toml()

        # Verify structure (TOML uses [inner] for nested tables)
        assert "[inner]" in toml_str or "inner." in toml_str

        # Load from TOML
        restored = OuterConfig.model_validate_toml(toml_str)

        assert restored.inner.value == 50
        assert restored.name == "original"


class TestSerializationWithNoneValues:
    """Test serialization with None values."""

    def test_none_value_in_dict(self):
        """Test serialization with None values."""

        class MyConfig(Config):
            optional: int | None = None

        config = MyConfig()
        data = config.model_dump()

        assert data["optional"] is None

        # Round-trip
        restored = MyConfig.model_validate(data)
        assert restored.optional is None

    def test_none_nested_config(self):
        """Test serialization with None nested config."""

        class InnerConfig(Config):
            x: int = 10

        class OuterConfig(Config):
            inner: InnerConfig | None = None

        config = OuterConfig()
        data = config.model_dump()

        assert data["inner"] is None

        # Round-trip
        restored = OuterConfig.model_validate(data)
        assert restored.inner is None


class TestSerializationWithConditionals:
    """Test serialization with conditional fields."""

    def test_conditional_field_serialization(self):
        """Test serialization of conditional fields."""

        class MyConfig(Config):
            mode: str = Categorical(["simple", "advanced"], default="simple")
            value: int = Conditional(
                condition=FieldCondition("mode", EqualsTo("advanced")),
                true=Int(ge=0, le=100, default=50),
                false=Int(ge=0, le=10, default=5),
            )

        # Simple mode
        config = MyConfig(mode="simple")
        data = config.model_dump()

        assert data["mode"] == "simple"
        assert data["value"] == 5

        # Round-trip
        restored = MyConfig.model_validate(data)
        assert restored.mode == "simple"
        assert restored.value == 5

    def test_conditional_nested_config_serialization(self):
        """Test serialization of conditional nested configs."""

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

        # Using ConfigA
        parent = ParentConfig(use_a=True, config=ConfigA())
        data = parent.model_dump()

        assert data["config"]["__type__"] == "ConfigA"

        restored = ParentConfig.model_validate(data)
        assert isinstance(restored.config, ConfigA)


class TestSerializationEdgeCases:
    """Test edge cases in serialization."""

    def test_empty_config(self):
        """Test serialization of empty config."""

        class EmptyConfig(Config):
            pass

        config = EmptyConfig()
        data = config.model_dump()

        assert isinstance(data, dict)
        assert len(data) == 0

        restored = EmptyConfig.model_validate(data)
        assert isinstance(restored, EmptyConfig)

    def test_config_with_many_fields(self):
        """Test serialization with many fields."""

        class LargeConfig(Config):
            f1: int = 1
            f2: int = 2
            f3: int = 3
            f4: int = 4
            f5: int = 5
            f6: int = 6
            f7: int = 7
            f8: int = 8
            f9: int = 9
            f10: int = 10

        config = LargeConfig()
        data = config.model_dump()

        assert len(data) == 10

        restored = LargeConfig.model_validate(data)
        for i in range(1, 11):
            assert getattr(restored, f"f{i}") == i

    def test_bytes_json_input(self):
        """Test model_validate_json with bytes input."""

        class MyConfig(Config):
            x: int = 10

        json_bytes = b'{"x": 20}'
        config = MyConfig.model_validate_json(json_bytes)

        assert config.x == 20

    def test_bytes_yaml_input(self):
        """Test model_validate_yaml with bytes input."""

        class MyConfig(Config):
            x: int = 10

        yaml_bytes = b"x: 20"
        config = MyConfig.model_validate_yaml(yaml_bytes)

        assert config.x == 20

    def test_bytes_toml_input(self):
        """Test model_validate_toml with bytes input."""

        class MyConfig(Config):
            x: int = 10

        toml_bytes = b"x = 20"
        config = MyConfig.model_validate_toml(toml_bytes)

        assert config.x == 20
