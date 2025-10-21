"""
Configuration base class with integrated search space support.

This module provides the Config class that combines Pydantic's
validation with searchable parameter spaces for HPO.
"""

from types import UnionType
from typing import Any, ClassVar, Self, Union, get_args, get_origin

from pydantic import BaseModel, model_validator
from pydantic_core import PydanticUndefined

from .graph_node import GraphNode
from .spaces import (
    UNSET,
    CategoricalSpace,
    ConditionalSpace,
    Space,
    infer_space_from_field_info,
)


class Config(BaseModel):
    """
    Base class for searchable configuration objects.

    Config combines Pydantic's validation with Space definitions to create
    configuration classes that can be:
    - Validated automatically using Space constraints
    - Sampled randomly for hyperparameter search
    - Introspected to understand the search space
    - Serialized/deserialized with Pydantic's methods

    Example:
        >>> class TrainingConfig(Config):
        ...     learning_rate: float = Float(1e-5, 1e-1, "log")
        ...     batch_size: int = Int(8, 128, "log")
        ...     optimizer: str = Categorical(["adam", "sgd"])
        ...
        >>> # Create with specific values
        >>> config = TrainingConfig(learning_rate=0.001, batch_size=32, optimizer="adam")
        >>>
        >>> # Or sample randomly
        >>> random_config = TrainingConfig.random()
        >>>
        >>> # Inspect the search space
        >>> space_info = TrainingConfig.get_space_info()
    """

    _spaces: ClassVar[dict[str, Space]] = {}
    _root_node: ClassVar[GraphNode | None] = None

    model_config = {
        "validate_assignment": True,  # Validate on attribute assignment
        "frozen": False,  # Allow mutation
        "arbitrary_types_allowed": True,  # Allow Space descriptors
    }

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Called when a subclass is created. Collects Space descriptors.
        """
        super().__init_subclass__(**kwargs)

        # Inherit spaces from parent classes
        spaces: dict[str, Space] = {}
        for base in cls.__mro__[1:]:  # Skip cls itself
            if issubclass(base, Config):
                for key, value in base._spaces.items():
                    if key not in spaces:
                        spaces[key] = value

        cls._spaces = spaces

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """
        Called by Pydantic after model fields are set up.
        This is where we do space inference since model_fields is now populated.
        """
        super().__pydantic_init_subclass__(**kwargs)

        # Collect all Space fields defined in this class
        spaces: dict[str, Space] = {}

        # Add spaces from this class (can override parent and field_info)
        # Infer spaces from type annotations for fields without explicit spaces
        for field_name, field_info in cls.model_fields.items():
            if isinstance(field_info.default, Space):
                space = field_info.default
                spaces[field_name] = space
                if space.default is not UNSET:
                    field_info.default = space.default
                else:
                    field_info.default = PydanticUndefined
            else:
                # Try to infer a space from the annotation
                inferred_space = infer_space_from_field_info(field_info)
                annotation = field_info.annotation

                if inferred_space is not None:
                    # Successfully inferred a space
                    spaces[field_name] = inferred_space
                    # Set the field name for the space
                    inferred_space.field_name = field_name
                elif field_name in cls._spaces:
                    # Defined in a parent class. Because user can not provide both Field and sp.Space
                    pass
                elif (
                    field_info.default is not PydanticUndefined
                    and not isinstance(field_info.default, Space)
                ) or field_info.default_factory is not None:
                    # Has a default value, so it's okay to not have a space
                    pass
                elif isinstance(annotation, type) and issubclass(annotation, Config):
                    # Nested Config type - allowed without explicit space
                    pass
                else:
                    # No space, no inferrable type, and no default - this is an error
                    raise TypeError(
                        f"Field '{field_name}' in Config class '{cls.__name__}' has type "
                        f"'{annotation}' which cannot be automatically converted to a Space. "
                        f"Please either: (1) define an explicit Space for this field, "
                        f"(2) provide a default value, or (3) use a supported type "
                        f"(bool, Literal, int/float with Field constraints)."
                    )

        cls._spaces.update(spaces)

        # Simplify single-choice categorical spaces
        spaces_to_remove = []
        for field_name, space in cls._spaces.items():
            if isinstance(space, CategoricalSpace) and len(space.choices) == 1:
                single_choice = space.choices[0]
                # If it's a Config type, remove the space entirely
                if isinstance(single_choice, type) and issubclass(
                    single_choice, Config
                ):
                    spaces_to_remove.append(field_name)
                else:
                    # Set as default value for the field
                    field_info = cls.model_fields[field_name]
                    field_info.default = single_choice
                    spaces_to_remove.append(field_name)

        # Remove simplified spaces
        for field_name in spaces_to_remove:
            del cls._spaces[field_name]

        # Build root node for graph representation and dependency tracking
        try:
            cls._root_node = GraphNode(cls)
        except (ValueError, TypeError) as e:
            raise TypeError(f"Error in Config class '{cls.__name__}': {e}") from e

    @model_validator(mode="before")
    @classmethod
    def validate_spaces(cls, data: Any) -> Any:
        """
        Validate input data against space constraints before Pydantic validation.

        This runs before Pydantic's standard validation and ensures that
        all Space-defined fields satisfy their constraints, including
        conditional space validation with proper dependency handling.

        Args:
            data: Input data (typically a dict).

        Returns:
            Validated data dictionary.

        Raises:
            ValueError: If any value violates its space constraints.
            RuntimeError: If a field name is not present in the given data.
        """
        if not isinstance(data, dict):
            raise ValueError(f"Got {data} which is {type(data).__name__}")
            # return data

        validated: dict[str, Any] = {}

        # Create a temporary object to hold values for condition evaluation
        temp_obj = type("TempConfig", (), {})()

        assert cls._root_node
        # Get ordered nodes from root node and validate in dependency order
        for field_name, field_node in cls._root_node.ordered_children():
            space = field_node.space

            # Find the value
            if field_name in data:
                value = data[field_name]
            else:
                # If field not in data, try to use default
                if space is not None and space.default is not UNSET:
                    value = space.default
                else:
                    raise RuntimeError(
                        f"Field '{field_name}' not provided in the data and has no default value"
                    )

            # Validate it
            if space is not None:
                if isinstance(space, ConditionalSpace):
                    try:
                        value = space.validate_with_config(value, temp_obj)
                    except ValueError as e:
                        raise ValueError(
                            f"Validation failed for conditional field '{field_name}': {e}"
                        ) from e
                else:
                    # Validate non-conditional spaces
                    try:
                        value = space.validate(value)
                    except ValueError as e:
                        raise ValueError(
                            f"Validation failed for field '{field_name}': {e}"
                        ) from e

            validated[field_name] = value
            setattr(temp_obj, field_name, value)

        return validated

    def _serialize_value(self, value: Any) -> Any:
        """Recursively serialize a value, adding __type__ for Config instances."""
        if isinstance(value, Config):
            # Serialize nested Config with type discriminator
            serialized = value.model_dump()
            serialized["__type__"] = value.__class__.__name__
            return serialized
        elif isinstance(value, tuple):
            return tuple(self._serialize_value(item) for item in value)
        elif isinstance(value, list):
            return [self._serialize_value(item) for item in value]
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        else:
            return value

    def __repr__(self) -> str:
        """Return a detailed string representation."""
        field_strs = []
        for field_name in self.__class__.model_fields:
            value = getattr(self, field_name, None)
            field_strs.append(f"{field_name}={value!r}")

        return f"{self.__class__.__name__}({', '.join(field_strs)})"

    def __str__(self) -> str:
        return self.__repr__()

    @classmethod
    def _deserialize_dict(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Process a dict to instantiate nested Configs based on __type__."""
        result: dict[str, Any] = {}

        for field_name, value in data.items():
            result[field_name] = cls._deserialize_value(field_name, value)

        return result

    @classmethod
    def _deserialize_value(cls, field_name: str, value: Any) -> Any:
        """Recursively deserialize a value, handling __type__ discriminators."""
        if isinstance(value, dict) and "__type__" in value:
            # This is a serialized Config - need to find the right type
            type_name = value["__type__"]

            # Get the field info to find possible Config types
            if field_name not in cls.model_fields:
                raise ValueError(f"Unknown field '{field_name}' in {cls.__name__}")

            field_info = cls.model_fields[field_name]
            config_type = cls._find_config_type(field_info.annotation, type_name)

            if config_type is None:
                raise ValueError(
                    f"Could not find Config type '{type_name}' for field '{field_name}' "
                    f"in {cls.__name__}"
                )

            # Remove __type__ and recursively deserialize
            data_without_type = {k: v for k, v in value.items() if k != "__type__"}
            return config_type.model_validate(data_without_type)

        elif isinstance(value, dict):
            return {k: cls._deserialize_value(field_name, v) for k, v in value.items()}
        elif isinstance(value, list):
            return [cls._deserialize_value(field_name, item) for item in value]
        elif isinstance(value, tuple):
            return tuple(cls._deserialize_value(field_name, item) for item in value)
        else:
            return value

    @classmethod
    def _find_config_type(cls, annotation: Any, type_name: str) -> type[Self] | None:
        """Find a Config type by name within a type annotation (union, etc)."""
        # Check if it's a union type
        if get_origin(annotation) is Union or isinstance(annotation, UnionType):
            # Search through union members
            for arg in get_args(annotation):
                if (
                    isinstance(arg, type)
                    and issubclass(arg, Config)
                    and arg.__name__ == type_name
                ):
                    return arg

        # Check if it's directly a Config type
        elif (
            isinstance(annotation, type)
            and issubclass(annotation, Config)
            and annotation.__name__ == type_name
        ):
            return annotation

        return None

    @classmethod
    def random(cls, use_defaults: bool = True) -> Self:
        """
        Generate a random configuration by sampling all search spaces.

        This method samples each Space field randomly according to its
        distribution, and uses default values for non-space fields.
        For conditional spaces, respects dependency ordering to ensure
        conditions can be properly evaluated.
        For nested Config types in Categorical spaces, recursively
        generates random instances.

        Args:
            use_defaults: If True, use default values where specified instead of sampling.
                         If False, always sample randomly even when defaults exist.

        Returns:
            A randomly generated Config instance.

        Example:
            >>> config = TrainingConfig.random()  # Uses defaults where specified
            >>> config = TrainingConfig.random(use_defaults=False)  # Always samples
        """
        from .spaces import UNSET

        kwargs: dict[str, Any] = {}
        # Create a temporary object to hold values for condition evaluation
        temp_obj = type("TempConfig", (), {})()

        assert cls._root_node
        # Get ordered nodes from root node and sample in dependency order
        for field_name, field_node in cls._root_node.ordered_children():
            # Sample each space field
            if (space := field_node.space) is not None:
                # Use default if available and use_defaults is True
                if use_defaults and space.default is not UNSET:
                    value = space.default
                else:
                    # Sample from the space
                    if isinstance(space, ConditionalSpace):
                        # Sample with config context
                        value = space.sample_with_config(temp_obj)
                    else:
                        # Regular sampling
                        value = space.sample()

                    # Categorical or Conditional with inner Config case
                    if isinstance(value, type) and issubclass(value, Config):
                        value = value.random(use_defaults=use_defaults)

            # Check if field is a nested Config
            elif isinstance(field_node.annotation, type) and issubclass(
                field_node.annotation, Config
            ):
                value = field_node.annotation.random(use_defaults=use_defaults)
            # Field must be fixed
            else:
                # Add default values for non-space fields
                default = field_node.fixed_value
                assert default is not None
                assert field_node.is_default_factory is not None
                value = default() if field_node.is_default_factory else default

            kwargs[field_name] = value
            setattr(temp_obj, field_name, value)

        return cls(**kwargs)

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
            result[field_name] = self._serialize_value(getattr(self, field_name))

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
        processed_data = cls._deserialize_dict(data)

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

        if isinstance(toml_data, str):
            toml_data = toml_data.encode()
        data = tomllib.loads(toml_data.decode())
        return cls.model_validate(data)
