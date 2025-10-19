"""
Configuration base class with integrated search space support.

This module provides the Config class that combines Pydantic's
validation with searchable parameter spaces for HPO.
"""

from typing import Any, ClassVar, Self

from pydantic import BaseModel, model_validator

from .dependency_graph import DependencyGraph
from .spaces import (
    UNSET,
    CategoricalSpace,
    ConditionalSpace,
    FloatSpace,
    IntSpace,
    Space,
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
    _dependency_graph: ClassVar[DependencyGraph | None] = None

    model_config = {
        "validate_assignment": True,  # Validate on attribute assignment
        "frozen": False,  # Allow mutation
        "arbitrary_types_allowed": True,  # Allow Space descriptors
    }

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Called when a subclass is created. Collects Space descriptors
        and builds dependency graph.
        """
        super().__init_subclass__(**kwargs)

        # Collect all Space fields defined in this class and parents
        spaces: dict[str, Space] = {}

        # Inherit spaces from parent classes
        for base in cls.__mro__[1:]:  # Skip cls itself
            if hasattr(base, "_spaces"):
                for key, value in base._spaces.items():
                    if key not in spaces:
                        spaces[key] = value

        # Add spaces from this class (can override parent)
        for key, value in cls.__dict__.items():
            if isinstance(value, Space):
                spaces[key] = value

        cls._spaces = spaces

        # Build dependency graph for conditional spaces
        if spaces:
            try:
                cls._dependency_graph = DependencyGraph(spaces)
            except ValueError as e:
                raise TypeError(f"Error in Config class '{cls.__name__}': {e}") from e
        else:
            cls._dependency_graph = None

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

        # Get ordered fields if we have a dependency graph
        if cls._dependency_graph:
            ordered_fields = cls._dependency_graph.get_ordered_fields()
        else:
            ordered_fields = list(cls._spaces.keys())

        # validate non-conditional spaces and set temp values
        for field_name in ordered_fields:
            space = cls._spaces.get(field_name)

            # If field not in data, try to use default
            if field_name not in data:
                if space is not None and space.default is not UNSET:
                    value = space.default
                else:
                    raise RuntimeError(
                        f"Field '{field_name}' not provided in the data and has no default value"
                    )
            else:
                value = data[field_name]

            if space is None:
                validated[field_name] = value
                setattr(temp_obj, field_name, value)
                continue

            if isinstance(space, ConditionalSpace):
                value = data[field_name]
                try:
                    validated_value = space.validate_with_config(value, temp_obj)
                except ValueError as e:
                    raise ValueError(
                        f"Validation failed for conditional field '{field_name}': {e}"
                    ) from e
            else:
                # Validate non-conditional spaces
                try:
                    validated_value = space.validate(value)
                except ValueError as e:
                    raise ValueError(
                        f"Validation failed for field '{field_name}': {e}"
                    ) from e

            validated[field_name] = validated_value
            setattr(temp_obj, field_name, validated_value)

        # Add any non-space fields
        for field_name, value in data.items():
            if field_name not in validated:
                validated[field_name] = value

        return validated

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

        # Get ordered fields if we have a dependency graph
        if cls._dependency_graph:
            ordered_fields = cls._dependency_graph.get_ordered_fields()
        else:
            ordered_fields = list(cls._spaces.keys())

        # Sample each space field in dependency order
        for field_name in ordered_fields:
            space = cls._spaces[field_name]

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

            # Handle nested Spaces (shouldn't happen but keep for safety)
            while isinstance(value, Space):
                if isinstance(value, ConditionalSpace):
                    value = value.sample_with_config(temp_obj)
                else:
                    value = value.sample()

            # If the sampled value is a Config class (not instance), instantiate it
            if isinstance(value, type) and issubclass(value, Config):
                value = value.random(use_defaults=use_defaults)

            kwargs[field_name] = value
            setattr(temp_obj, field_name, value)

        # Add default values for non-space fields
        for field_name, field_info in cls.model_fields.items():
            if field_name not in kwargs:
                # Use default if available
                if field_info.default is not None:
                    kwargs[field_name] = field_info.default
                elif field_info.default_factory is not None:
                    # Call the factory function
                    factory = field_info.default_factory
                    kwargs[field_name] = factory()  # type: ignore

        return cls(**kwargs)

    @classmethod
    def get_space_info(cls) -> dict[str, dict[str, Any]]:
        """
        Get structured information about all search spaces in this Config.

        Returns a dictionary mapping field names to their space metadata,
        including ranges, distributions, choices, probabilities, and
        conditional dependencies.

        Returns:
            Dictionary with field names as keys and space info dicts as values.

        Example:
            >>> info = TrainingConfig.get_space_info()
            >>> print(info["learning_rate"])
            {
                'type': 'FloatSpace',
                'low': 1e-05,
                'high': 0.1,
                'distribution': 'LogDistribution',
                'bounds': 'both'
            }
        """
        info: dict[str, dict[str, Any]] = {}

        for field_name, space in cls._spaces.items():
            space_info: dict[str, Any]
            if isinstance(space, (FloatSpace, IntSpace)):
                space_info = {
                    "type": space.__class__.__name__,
                    "low": space.low,
                    "high": space.high,
                    "low_inclusive": space.low_inclusive,
                    "high_inclusive": space.high_inclusive,
                    "distribution": space.distribution.__class__.__name__,
                }
            elif isinstance(space, CategoricalSpace):
                space_info = {
                    "type": "CategoricalSpace",
                    "choices": space.choices,
                    "weights": space.weights,
                    "probs": space.probs,
                }
            elif isinstance(space, ConditionalSpace):
                space_info = {
                    "type": "ConditionalSpace",
                    "condition": repr(space.condition),
                    "true_branch": space.true_branch.__class__.__name__
                    if isinstance(space.true_branch, Space)
                    else repr(space.true_branch),
                    "false_branch": space.false_branch.__class__.__name__
                    if isinstance(space.false_branch, Space)
                    else repr(space.false_branch),
                }

                if cls._dependency_graph:
                    deps = cls._dependency_graph.get_dependencies(field_name)
                    space_info["depends_on"] = list(deps)
            else:
                # Generic fallback for custom Space types
                space_info = {
                    "type": space.__class__.__name__,
                    "details": str(space),
                }

            info[field_name] = space_info

        return info

    @classmethod
    def get_dependency_info(cls) -> dict[str, Any]:
        """
        Get dependency graph information for visualization.

        Returns:
            Dictionary with dependency graph data including:
            - nodes: List of field names
            - edges: List of dependency edges
            - order: Safe initialization order
            - dependencies: Map of field -> dependencies

        Example:
            >>> info = TrainingConfig.get_dependency_info()
            >>> print(info["order"])  # Safe initialization order
            ['optimizer', 'learning_rate', 'lr_multiplier']
        """
        if cls._dependency_graph is None:
            return {
                "nodes": list(cls._spaces.keys()),
                "edges": [],
                "order": list(cls._spaces.keys()),
                "dependencies": {},
            }

        return cls._dependency_graph.get_graph_data()

    def __repr__(self) -> str:
        """Return a detailed string representation."""
        field_strs = []
        for field_name in self.__class__.model_fields:
            value = getattr(self, field_name, None)
            field_strs.append(f"{field_name}={value!r}")

        return f"{self.__class__.__name__}({', '.join(field_strs)})"
