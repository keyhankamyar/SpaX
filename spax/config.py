"""
Configuration base class with integrated search space support.

This module provides the Config class that combines Pydantic's
validation with searchable parameter spaces for HPO.
"""

from typing import Any, ClassVar, Self

from pydantic import BaseModel, model_validator

from .dependency_graph import DependencyGraph
from .spaces import CategoricalSpace, ConditionalSpace, FloatSpace, IntSpace, Space


class ConfigMeta(type(BaseModel)):
    """
    Metaclass that collects Space descriptors from Config classes.

    This metaclass scans class attributes during class creation and
    identifies all Space instances, storing them in a _spaces class
    variable for later introspection and sampling.
    """

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> type:
        """
        Create a new Config class with space collection.

        Args:
            name: Name of the class being created.
            bases: Base classes.
            namespace: Class namespace dictionary.
            **kwargs: Additional arguments passed to type.__new__.

        Returns:
            The newly created class with _spaces populated.
        """
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Collect all Space fields defined directly in this class
        spaces: dict[str, Space] = {}
        for key, value in namespace.items():
            if isinstance(value, Space):
                spaces[key] = value

        # Inherit spaces from parent classes
        for base in bases:
            if hasattr(base, "_spaces"):
                # Parent spaces are added first, so child can override
                for key, value in base._spaces.items():
                    if key not in spaces:  # Don't override child definitions
                        spaces[key] = value

        cls._spaces = spaces

        # Build dependency graph for conditional spaces
        if spaces:
            try:
                cls._dependency_graph = DependencyGraph(spaces)
            except ValueError as e:
                raise TypeError(f"Error in Config class '{name}': {e}") from e
        else:
            cls._dependency_graph = None

        return cls


class Config(BaseModel, metaclass=ConfigMeta):
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

        # First pass: validate non-conditional spaces and set temp values
        for field_name in ordered_fields:
            if field_name not in data:
                raise RuntimeError(f"{field_name} not provided in the data")

            value = data[field_name]
            space = cls._spaces.get(field_name)

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
    def random(cls) -> Self:
        """
        Generate a random configuration by sampling all search spaces.

        This method samples each Space field randomly according to its
        distribution, and uses default values for non-space fields.

        For conditional spaces, respects dependency ordering to ensure
        conditions can be properly evaluated.

        For nested Config types in Categorical spaces, recursively
        generates random instances.

        Returns:
            A randomly generated Config instance.

        Example:
            >>> config = TrainingConfig.random()
            >>> print(config.learning_rate)  # Random value in [1e-5, 1e-1]
        """
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
                value = value.random()

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
                    "distribution": space.distribution.__class__.__name__,
                    "bounds": space.bounds,
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
        for field_name in self.model_fields:
            value = getattr(self, field_name, None)
            field_strs.append(f"{field_name}={value!r}")

        return f"{self.__class__.__name__}({', '.join(field_strs)})"
