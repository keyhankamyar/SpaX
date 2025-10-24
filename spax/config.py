"""
Configuration base class with integrated search space support.

This module provides the Config class that combines Pydantic's
validation with searchable parameter spaces for HPO.
"""

from types import UnionType
from typing import Any, ClassVar, Self, Union, get_args, get_origin

from pydantic import BaseModel, model_validator

from .nodes import ConfigNode
from .samplers import RandomSampler


def _type_from_annotation(annotation: Any, type_name: str) -> type | None:
    """Find a type by name within a type annotation (union, etc)."""
    # Check if it's a union type
    if get_origin(annotation) is Union or isinstance(annotation, UnionType):
        # Search through union members
        for arg in get_args(annotation):
            if isinstance(arg, type) and arg.__name__ == type_name:
                return arg

    # Check if it's directly a Config type
    elif isinstance(annotation, type) and annotation.__name__ == type_name:
        return annotation

    return None


class Config(BaseModel, validate_assignment=True):
    """
    Base class for searchable configuration objects.
    """

    _node: ClassVar[ConfigNode | None] = None

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)

        try:
            cls._node = ConfigNode(cls)
        except Exception as e:
            raise TypeError(f"Error in Config class '{cls.__name__}': {e}") from e

    @model_validator(mode="before")
    @classmethod
    def validate_spaces(cls, data: dict[str, Any]) -> dict[str, Any]:
        assert cls._node is not None
        return cls._node.validate_spaces(data)

    @classmethod
    def get_parameter_names(cls) -> list[str]:
        """Get all tunable parameter names in this configuration.

        Returns:
            List of fully qualified parameter names that can be sampled/tuned.

        Example:
            >>> class MyConfig(Config):
            ...     x: int = Int(ge=0, le=10)
            ...     y: float = Float(ge=0.0, le=1.0)
            >>> MyConfig.get_parameter_names()
            ['MyConfig.x', 'MyConfig.y']
        """
        assert cls._node is not None
        return cls._node.get_parameter_names()

    @classmethod
    def get_node(cls, override: dict[str, Any] | None = None) -> Any:
        """Get the configuration node, optionally with overrides applied.

        This provides access to the internal node structure for advanced operations
        like getting signatures, hashes, parameter names, etc.

        Args:
            override: Optional overrides to apply to the search space

        Returns:
            The ConfigNode, potentially with overrides applied

        Example:
            >>> class MyConfig(Config):
            ...     x: int = Int(ge=0, le=10)
            >>> node = MyConfig.get_node()
            >>> hash1 = node.get_space_hash()
            >>>
            >>> # With overrides
            >>> node2 = MyConfig.get_node(override={"x": {"ge": 5, "le": 8}})
            >>> hash2 = node2.get_space_hash()
            >>> assert hash1 != hash2
        """
        assert cls._node is not None

        if override is None:
            return cls._node

        return cls._node.apply_override(override)

    @classmethod
    def random(
        cls, seed: int | None = None, override: dict[str, Any] | None = None
    ) -> Self:
        """Generate a random configuration by sampling from the search space.

        Args:
            seed: Random seed for reproducibility
            override: Optional overrides to apply before sampling

        Returns:
            A randomly sampled configuration instance

        Example:
            >>> class MyConfig(Config):
            ...     x: int = Int(ge=0, le=10)
            ...     y: float = Float(ge=0.0, le=1.0)
            >>> config = MyConfig.random(seed=42)
            >>> config2 = MyConfig.random(seed=42, override={"x": 5})
        """

        node = cls.get_node(override)

        # Sample using RandomSampler
        sampler = RandomSampler(seed=seed)
        sampled_values = node.sample(sampler)

        # Create and return the config instance
        return cls.model_validate(sampled_values)

    @classmethod
    def get_override_template(cls) -> dict[str, Any]:
        """Get a template dict structure for overrides.

        This returns a dict showing all available fields and their override options.
        Users can fill this in to create custom overrides.

        Returns:
            Dict template for overrides

        Example:
            >>> class MyConfig(Config):
            ...     x: int = Int(ge=0, le=10)
            ...     y: str = Categorical(["a", "b", "c"])
            >>> template = MyConfig.get_override_template()
            >>> # Fill in the template
            >>> override = {"x": {"ge": 5, "le": 8}, "y": ["a", "b"]}
            >>> config = MyConfig.random(override=override)
        """
        assert cls._node is not None
        return cls._node.get_override_template()

    def model_dump(self) -> dict[str, Any]:
        """
        Serialize the configuration to a dictionary.

        This method extends Pydantic's model_dump to add type discrimination
        for nested Config instances. When a field contains a Config object
        (e.g., from a union type like `InnerConfig1 | InnerConfig2`), the
        serialized dict includes a `__type__` field with the class name to
        enable correct deserialization.

        Returns:
            A dictionary representation of the config with type information
            for nested Config objects.

        Example:
            >>> class Inner1(Config):
            ...     x: int = Int(ge=0, lt=10)
            ...
            >>> class Inner2(Config):
            ...     y: float = Float(ge=0, lt=1)
            ...
            >>> class Outer(Config):
            ...     name: str = "test"
            ...     inner: Inner1 | Inner2
            ...
            >>> config = Outer(name="example", inner=Inner1(x=5))
            >>> config.model_dump()
            {
                'name': 'example',
                'inner': {'__type__': 'Inner1', 'x': 5}
            }
        """
        result: dict[str, Any] = {}

        for field_name in self.__class__.model_fields:
            value = getattr(self, field_name)
            if isinstance(value, Config):
                # Serialize nested Config with type discriminator
                serialized = value.model_dump()
                serialized["__type__"] = value.__class__.__name__
                result[field_name] = serialized
            else:
                result[field_name] = value

        return result

    def model_dump_json(self, *, indent: int | None = 2, **json_kwargs: Any) -> str:
        """
        Serialize the configuration to a JSON string.

        Args:
            indent: Number of spaces for indentation (default: 2). Set to None for compact output.
            **json_kwargs: Additional arguments passed to json.dumps().

        Returns:
            A JSON string representation of the config.

        Example:
            >>> config = MyConfig(var_1=50, var_2=InnerConfig1(x=5))
            >>> json_str = config.model_dump_json()
            >>> print(json_str)
        """
        import json

        data = self.model_dump()
        return json.dumps(data, indent=indent, **json_kwargs)

    def model_dump_yaml(self, **yaml_kwargs: Any) -> str:
        """
        Serialize the configuration to a YAML string.

        Args:
            **yaml_kwargs: Additional arguments passed to yaml.dump().

        Returns:
            A YAML string representation of the config.

        Raises:
            RuntimeError: If PyYAML is not installed.

        Example:
            >>> config = MyConfig(var_1=50, var_2=InnerConfig1(x=5))
            >>> yaml_str = config.model_dump_yaml()
            >>> print(yaml_str)
        """
        try:
            import yaml
        except ImportError:
            raise RuntimeError(
                "PyYAML is required for YAML serialization. "
                "Install it with: pip install PyYAML"
            ) from None

        data = self.model_dump()
        return yaml.dump(data, sort_keys=False, **yaml_kwargs)

    def model_dump_toml(self) -> str:
        """
        Serialize the configuration to a TOML string.

        Returns:
            A TOML string representation of the config.

        Raises:
            RuntimeError: If tomli-w is not installed.

        Example:
            >>> config = MyConfig(var_1=50, var_2=InnerConfig1(x=5))
            >>> toml_str = config.model_dump_toml()
            >>> print(toml_str)
        """
        try:
            import tomli_w
        except ImportError:
            raise RuntimeError(
                "tomli-w is required for TOML serialization. "
                "Install it with: pip install tomli-w"
            ) from None

        data = self.model_dump()
        return tomli_w.dumps(data)

    @classmethod
    def model_validate(cls, data: Any) -> Self:
        """
        Deserialize and validate data into a Config instance.

        This method extends Pydantic's model_validate to handle type
        discrimination for nested Config instances. When encountering
        a dict with a `__type__` field, it uses that to instantiate
        the correct Config subclass from union types.

        Args:
            data: Input data to validate (typically a dict).

        Returns:
            A validated Config instance.

        Raises:
            ValueError: If the data is invalid or type discrimination fails.

        Example:
            >>> data = {
            ...     'name': 'example',
            ...     'inner': {'__type__': 'Inner1', 'x': 5}
            ... }
            >>> config = Outer.model_validate(data)
            >>> isinstance(config.inner, Inner1)
            True
        """
        if not isinstance(data, dict):
            # Let pydantic handle non-dict data
            return super().model_validate(data)

        # Process the data to handle __type__ discriminators
        processed_data: dict[str, Any] = {}

        for field_name, value in data.items():
            if isinstance(value, dict) and "__type__" in value:
                # This is a serialized Config - need to find the right type
                type_name = value["__type__"]

                # Get the field info to find possible Config types
                if field_name not in cls.model_fields:
                    raise ValueError(f"Unknown field '{field_name}' in {cls.__name__}")

                field_info = cls.model_fields[field_name]
                config_type = _type_from_annotation(field_info.annotation, type_name)

                if config_type is None or not issubclass(config_type, Config):
                    raise ValueError(
                        f"Could not find Config type '{type_name}' "
                        f"for field '{field_name}' in {cls.__name__}"
                    )

                # Remove __type__ and recursively deserialize
                data_without_type = {k: v for k, v in value.items() if k != "__type__"}
                processed_data[field_name] = config_type.model_validate(
                    data_without_type
                )
            else:
                processed_data[field_name] = value

        # Use pydantic's validator with processed data
        return super().model_validate(processed_data)

    @classmethod
    def model_validate_json(cls, json_data: str | bytes, **json_kwargs: Any) -> Self:
        """
        Deserialize and validate a JSON string into a Config instance.

        Args:
            json_data: JSON string or bytes to deserialize.
            **json_kwargs: Additional arguments passed to json.loads().

        Returns:
            A validated Config instance.

        Example:
            >>> json_str = '{"var_1": 50, "var_2": {"__type__": "InnerConfig1", "x": 5}}'
            >>> config = MyConfig.model_validate_json(json_str)
        """
        import json

        data = json.loads(json_data, **json_kwargs)
        return cls.model_validate(data)

    @classmethod
    def model_validate_yaml(cls, yaml_data: str | bytes, **yaml_kwargs: Any) -> Self:
        """
        Deserialize and validate a YAML string into a Config instance.

        Args:
            yaml_data: YAML string or bytes to deserialize.
            **yaml_kwargs: Additional arguments passed to yaml.safe_load().

        Returns:
            A validated Config instance.

        Raises:
            RuntimeError: If PyYAML is not installed.

        Example:
            >>> yaml_str = '''
            ... var_1: 50
            ... var_2:
            ...   __type__: InnerConfig1
            ...   x: 5
            ... '''
            >>> config = MyConfig.model_validate_yaml(yaml_str)
        """
        try:
            import yaml
        except ImportError:
            raise RuntimeError(
                "PyYAML is required for YAML deserialization. "
                "Install it with: pip install PyYAML"
            ) from None

        data = yaml.safe_load(yaml_data, **yaml_kwargs)
        return cls.model_validate(data)

    @classmethod
    def model_validate_toml(cls, toml_data: str | bytes) -> Self:
        """
        Deserialize and validate a TOML string into a Config instance.

        Args:
            toml_data: TOML string or bytes to deserialize.

        Returns:
            A validated Config instance.

        Example:
            >>> toml_str = '''
            ... var_1 = 50
            ... [var_2]
            ... __type__ = "InnerConfig1"
            ... x = 5
            ... '''
            >>> config = MyConfig.model_validate_toml(toml_str)
        """
        import tomllib

        if isinstance(toml_data, bytes):
            toml_data = toml_data.decode()

        data = tomllib.loads(toml_data)
        return cls.model_validate(data)

    def __repr__(self) -> str:
        """Return a detailed string representation."""
        field_strs = []
        for field_name in self.__class__.model_fields:
            value = getattr(self, field_name, None)
            field_strs.append(f"{field_name}={value!r}")

        return f"{self.__class__.__name__}({', '.join(field_strs)})"

    def __str__(self) -> str:
        return self.__repr__()
