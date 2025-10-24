"""Core Config class for defining searchable configurations.

This module provides the Config base class, which is the main user-facing API
for defining configurations with search spaces. Config classes use Pydantic
for validation and add search space functionality on top.

Config classes support:
- Declarative search space definition using Space objects
- Automatic space inference from type annotations
- Random sampling with reproducible seeds
- Integration with HPO libraries (Optuna, etc.)
- Serialization/deserialization (JSON, YAML, TOML)
- Override application to narrow search spaces
- Nested configurations

Examples:
    >>> import spax as sp
    >>>
    >>> # Define a searchable configuration
    >>> class MyConfig(sp.Config):
    ...     learning_rate: float = sp.Float(ge=1e-5, le=1e-1, distribution='log')
    ...     num_layers: int = sp.Int(ge=1, le=10)
    ...     optimizer: str = sp.Categorical(["adam", "sgd", "rmsprop"])
    ...     use_dropout: bool
    ...     dropout_rate: float = sp.Conditional(
    ...         sp.FieldCondition("use_dropout", sp.EqualsTo(True)),
    ...         true=sp.Float(gt=0.0, lt=0.5),
    ...         false=0.0
    ...     )
    >>>
    >>> # Sample random configurations
    >>> config = MyConfig.random(seed=42)
    >>>
    >>> # Apply overrides to narrow search space
    >>> override = {"num_layers": {"ge": 5, "le": 7}}
    >>> config = MyConfig.random(seed=42, override=override)
    >>>
    >>> # Serialize and deserialize
    >>> json_str = config.model_dump_json()
    >>> loaded_config = MyConfig.model_validate_json(json_str)
"""

from typing import Any, ClassVar, Self

from pydantic import BaseModel, model_validator

from .nodes import ConfigNode
from .samplers import RandomSampler
from .utils import type_from_annotation


class Config(BaseModel, validate_assignment=True):
    """Base class for searchable configurations.

    Config extends Pydantic's BaseModel with search space functionality.
    Subclasses can define fields with Space objects (Float, Int, Categorical,
    Conditional) to create searchable parameters, or use regular Python types
    for fixed values or automatic inference.

    Class Attributes:
        _node: Internal ConfigNode representing the search space structure
            (automatically created when class is defined).

    Key Features:
        - Declarative search space definition
        - Automatic type inference for simple cases
        - Random sampling with seeds
        - Override application
        - Nested configurations
        - Multiple serialization formats

    Examples:
        >>> import spax as sp
        >>>
        >>> # Simple configuration
        >>> class TrainingConfig(sp.Config):
        ...     learning_rate: float = sp.Float(ge=1e-5, le=1e-1)
        ...     batch_size: int = sp.Int(ge=16, le=128)
        ...     optimizer: str = sp.Categorical(["adam", "sgd"])
        >>>
        >>> # Sample configuration
        >>> config = TrainingConfig.random(seed=42)
        >>> print(config.learning_rate, config.batch_size, config.optimizer)
        >>>
        >>> # Nested configuration
        >>> class ModelConfig(sp.Config):
        ...     num_layers: int = sp.Int(ge=1, le=10)
        ...     hidden_size: int = sp.Int(ge=64, le=512)
        >>>
        >>> class ExperimentConfig(sp.Config):
        ...     training: TrainingConfig
        ...     model: ModelConfig
        >>>
        >>> # Sample nested configuration
        >>> exp_config = ExperimentConfig.random(seed=42)
    """

    # Class variable to hold the ConfigNode (set during class creation)
    _node: ClassVar[ConfigNode | None] = None

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Hook called when a Config subclass is created.

        This creates the ConfigNode that represents the search space structure.

        Args:
            **kwargs: Additional keyword arguments from class definition.

        Raises:
            TypeError: If there's an error creating the ConfigNode.
        """
        super().__pydantic_init_subclass__(**kwargs)
        try:
            cls._node = ConfigNode(cls)
        except Exception as e:
            raise TypeError(f"Error in Config class '{cls.__name__}': {e}") from e

    @model_validator(mode="before")
    @classmethod
    def validate_spaces(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Validate all fields using their space definitions.

        This validator runs before Pydantic's validation and handles
        Space-specific validation (bounds checking, conditional evaluation, etc.).

        Args:
            data: Dictionary of field values to validate.

        Returns:
            Dictionary of validated values.

        Raises:
            ValueError: If validation fails for any field.
        """
        assert cls._node is not None
        return cls._node.validate_spaces(data)

    @classmethod
    def get_parameter_names(cls) -> list[str]:
        """Get all parameter names in the search space.

        Returns hierarchical parameter names using "::" as separator.
        Fixed values are not included in the list.

        Returns:
            List of fully-qualified parameter names.

        Examples:
            >>> class MyConfig(sp.Config):
            ...     lr: float = sp.Float(ge=1e-5, le=1e-1)
            ...     layers: int = sp.Int(ge=1, le=10)
            ...     name: str = "my_model"  # Fixed value
            >>>
            >>> MyConfig.get_parameter_names()
            ['MyConfig.lr', 'MyConfig.layers']
        """
        assert cls._node is not None
        return cls._node.get_parameter_names()

    @classmethod
    def get_node(cls, override: dict[str, Any] | None = None) -> ConfigNode:
        """Get the ConfigNode, optionally with overrides applied.

        Args:
            override: Optional dictionary of overrides to apply to the node.

        Returns:
            The ConfigNode (with overrides applied if provided).

        Examples:
            >>> node = MyConfig.get_node()
            >>> override_node = MyConfig.get_node(override={"lr": {"ge": 1e-4}})
        """
        assert cls._node is not None
        if override is None:
            return cls._node
        return cls._node.apply_override(override)

    @classmethod
    def random(
        cls, seed: int | None = None, override: dict[str, Any] | None = None
    ) -> Self:
        """Sample a random configuration.

        Generates a random configuration by sampling from all search spaces.
        Uses a RandomSampler by default, but can be extended to use other
        samplers (e.g., Optuna).

        Args:
            seed: Random seed for reproducibility. If None, uses system time.
            override: Optional dictionary of overrides to narrow the search space
                before sampling. See get_override_template() for structure.

        Returns:
            A Config instance with randomly sampled values.

        Examples:
            >>> # Basic random sampling
            >>> config = MyConfig.random(seed=42)
            >>>
            >>> # With overrides to narrow search space
            >>> config = MyConfig.random(
            ...     seed=42,
            ...     override={
            ...         "learning_rate": {"ge": 1e-4, "le": 1e-2},
            ...         "optimizer": ["adam", "sgd"]  # Exclude "rmsprop"
            ...     }
            ... )
        """
        node = cls.get_node(override)

        # Create sampler and sample from the node
        sampler = RandomSampler(seed=seed)
        sampled_values = node.sample(sampler)

        # Validate and return the config instance
        return cls.model_validate(sampled_values)

    @classmethod
    def get_override_template(cls) -> dict[str, Any]:
        """Get a template showing the structure for overrides.

        The template shows the nested dictionary structure expected for
        overriding the search space. Use this as a guide for creating
        override dictionaries.

        Returns:
            Dictionary template showing override structure.

        Examples:
            >>> template = MyConfig.get_override_template()
            >>> print(template)
            {
                'learning_rate': {'ge': 1e-05, 'le': 0.1},
                'num_layers': {'ge': 1, 'le': 10},
                'optimizer': ['adam', 'sgd', 'rmsprop']
            }
        """
        assert cls._node is not None
        return cls._node.get_override_template()

    def model_dump(self) -> dict[str, Any]:
        """Serialize the configuration to a dictionary.

        Extends Pydantic's model_dump to add type discriminators for
        nested Config objects, enabling proper deserialization.

        Returns:
            Dictionary representation with __type__ discriminators for
            nested Config objects.

        Examples:
            >>> config = MyConfig.random()
            >>> data = config.model_dump()
            >>> # data includes __type__ for nested configs
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
        """Serialize the configuration to JSON.

        Args:
            indent: Number of spaces for indentation (default=2).
            **json_kwargs: Additional arguments passed to json.dumps.

        Returns:
            JSON string representation.

        Examples:
            >>> config = MyConfig.random()
            >>> json_str = config.model_dump_json()
            >>> print(json_str)
        """
        import json

        data = self.model_dump()
        return json.dumps(data, indent=indent, **json_kwargs)

    def model_dump_yaml(self, **yaml_kwargs: Any) -> str:
        """Serialize the configuration to YAML.

        Args:
            **yaml_kwargs: Additional arguments passed to yaml.dump.

        Returns:
            YAML string representation.

        Raises:
            RuntimeError: If PyYAML is not installed.

        Examples:
            >>> config = MyConfig.random()
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
        """Serialize the configuration to TOML.

        Returns:
            TOML string representation.

        Raises:
            RuntimeError: If tomli-w is not installed.

        Examples:
            >>> config = MyConfig.random()
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
        """Validate and create a Config instance from data.

        Handles __type__ discriminators for nested Config objects,
        enabling proper deserialization of configurations.

        Args:
            data: Data to validate. Can be a dict, or any format Pydantic accepts.

        Returns:
            Validated Config instance.

        Raises:
            ValueError: If data is invalid or __type__ references unknown Config.

        Examples:
            >>> data = {"learning_rate": 0.001, "num_layers": 5}
            >>> config = MyConfig.model_validate(data)
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
                config_type = type_from_annotation(field_info.annotation, type_name)

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
        """Validate and create a Config instance from JSON.

        Args:
            json_data: JSON string or bytes.
            **json_kwargs: Additional arguments passed to json.loads.

        Returns:
            Validated Config instance.

        Examples:
            >>> json_str = '{"learning_rate": 0.001, "num_layers": 5}'
            >>> config = MyConfig.model_validate_json(json_str)
        """
        import json

        data = json.loads(json_data, **json_kwargs)
        return cls.model_validate(data)

    @classmethod
    def model_validate_yaml(cls, yaml_data: str | bytes, **yaml_kwargs: Any) -> Self:
        """Validate and create a Config instance from YAML.

        Args:
            yaml_data: YAML string or bytes.
            **yaml_kwargs: Additional arguments passed to yaml.safe_load.

        Returns:
            Validated Config instance.

        Raises:
            RuntimeError: If PyYAML is not installed.

        Examples:
            >>> yaml_str = '''
            ... learning_rate: 0.001
            ... num_layers: 5
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
        """Validate and create a Config instance from TOML.

        Args:
            toml_data: TOML string or bytes.

        Returns:
            Validated Config instance.

        Examples:
            >>> toml_str = '''
            ... learning_rate = 0.001
            ... num_layers = 5
            ... '''
            >>> config = MyConfig.model_validate_toml(toml_str)
        """
        import tomllib

        if isinstance(toml_data, bytes):
            toml_data = toml_data.decode()

        data = tomllib.loads(toml_data)
        return cls.model_validate(data)

    def __repr__(self) -> str:
        """Return a string representation of this config.

        Returns:
            String showing class name and all field values.
        """
        field_strs = []
        for field_name in self.__class__.model_fields:
            value = getattr(self, field_name, None)
            field_strs.append(f"{field_name}={value!r}")

        return f"{self.__class__.__name__}({', '.join(field_strs)})"

    def __str__(self) -> str:
        """Return a string representation of this config.

        Returns:
            Same as __repr__.
        """
        return self.__repr__()
