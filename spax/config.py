"""
Configuration base class with integrated search space support.

This module provides the Config class that combines Pydantic's
validation with searchable parameter spaces for HPO.
"""

from typing import Any, ClassVar, Self

from pydantic import BaseModel, model_validator
from pydantic_core import PydanticUndefined

from .graph_node import GraphNode
from .spaces import (
    UNSET,
    CategoricalSpace,
    ConditionalSpace,
    FloatSpace,
    IntSpace,
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
            cls._root_node = GraphNode(
                annotation=cls,
                space=None,
                has_default=False,
                fixed_value=None,
            )
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

        # Get ordered fields from root node
        assert cls._root_node
        assert cls._root_node.field_order
        ordered_fields = cls._root_node.field_order

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

                assert cls._root_node
                node = cls._root_node.get_child(field_name)
                if node:
                    space_info["depends_on"] = list(node.dependencies)
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
        """Get dependency information from the graph structure.

        Returns a dictionary with nodes, edges, order, and dependencies
        for visualization and analysis purposes.
        """
        if cls._root_node is None:
            return {
                "nodes": list(cls._spaces.keys()),
                "edges": [],
                "order": list(cls._spaces.keys()),
                "dependencies": {},
            }

        # Build edges from dependencies
        edges: list[dict[str, str]] = []
        dependencies: dict[str, list[str]] = {}

        for field_name, child_node in cls._root_node.children.items():
            deps = child_node.dependencies
            if deps:
                dependencies[field_name] = list(deps)
                for dep in deps:
                    edges.append({"from": dep, "to": field_name})

        return {
            "nodes": list(cls._root_node.children.keys()),
            "edges": edges,
            "order": cls._root_node.field_order,
            "dependencies": dependencies,
        }

    def __repr__(self) -> str:
        """Return a detailed string representation."""
        field_strs = []
        for field_name in self.__class__.model_fields:
            value = getattr(self, field_name, None)
            field_strs.append(f"{field_name}={value!r}")

        return f"{self.__class__.__name__}({', '.join(field_strs)})"
