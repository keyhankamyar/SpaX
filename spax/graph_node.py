"""Graph node representation for configuration space structure.

This module provides the GraphNode class which represents nodes in the
configuration space graph, enabling visualization, validation, and traversal
of complex nested configuration structures.
"""

from collections.abc import Generator
from types import UnionType
from typing import Any, Literal, Self, Union, get_origin

from pydantic_core import PydanticUndefined

from spax.spaces import (
    UNSET,
    CategoricalSpace,
    ConditionalSpace,
    Space,
)


class GraphNode:
    """Represents a node in the configuration space graph.

    Each node corresponds to a field in a Config and can have children when:
    - The field is a Config type (child_type="fields")
    - The field is a CategoricalSpace with Config choices (child_type="choices")
    - The field is a ConditionalSpace (child_type="branches")

    Attributes:
        annotation: The type annotation of the field
        space: The Space object if this field has one, None otherwise
        has_default: Whether this field has a default value
        fixed_value: The fixed value if this is not a Space field
        dependencies: Set of field names this node depends on (for ConditionalSpace)
        children: Dictionary of child nodes (key is the child's name/identifier)
        child_type: Type of children this node has, or None if no children
        field_order: Topological order of children (only for Config fields)
    """

    def __init__(
        self,
        annotation: type,
        space: Space | None,
        has_default: bool,
        fixed_value: Any | None = None,
        is_default_factory: bool | None = None,
    ) -> None:
        """Initialize a GraphNode.

        Args:
            annotation: The type annotation for this field
            space: The Space object for this field, or None for fixed values
            has_default: Whether this field has a default value
            fixed_value: The fixed value (for non-Space fields) or default factory

        Raises:
            TypeError: If annotation is not a type, space is not a Space or None,
                      or has_default is not a bool
            ValueError: If a field has no space, default, or default factory
        """
        # Validate inputs
        if (
            not isinstance(annotation, type)
            and get_origin(annotation) is not Union
            and not isinstance(annotation, UnionType)
        ):
            raise TypeError(
                f"annotation must be a type, got {type(annotation).__name__}"
            )
        if space is not None and not isinstance(space, Space):
            raise TypeError(
                f"space must be a Space instance or None, got {type(space).__name__}"
            )
        if not isinstance(has_default, bool):
            raise TypeError(
                f"has_default must be bool, got {type(has_default).__name__}"
            )

        self._annotation = annotation
        self._space = space
        self._has_default = has_default
        self._fixed_value = fixed_value
        self._is_default_factory = is_default_factory

        # To be populated during construction
        self._dependencies: set[str] = set()
        self._children: dict[str, GraphNode] = {}
        self._child_type: Literal["fields", "choices", "branches"] | None = None
        self._field_order: list[str] = []

        # Build the node structure
        self._populate()
        self._validate_and_order_children()

    @property
    def annotation(self) -> type | UnionType:
        """The type annotation of this field."""
        return self._annotation

    @property
    def space(self) -> Space | None:
        """The Space object for this field, or None."""
        return self._space

    @property
    def has_default(self) -> bool:
        """Whether this field has a default value."""
        return self._has_default

    @property
    def fixed_value(self) -> Any | None:
        """The fixed value for non-Space fields."""
        return self._fixed_value

    @property
    def is_default_factory(self) -> bool | None:
        """Is the fixed value a factory."""
        return self._is_default_factory

    @property
    def dependencies(self) -> set[str]:
        """Set of field names this node depends on (immutable view)."""
        return self._dependencies.copy()

    @property
    def children(self) -> dict[str, "GraphNode"]:
        """Dictionary of child nodes (immutable view)."""
        return self._children.copy()

    @property
    def child_type(self) -> Literal["fields", "choices", "branches"] | None:
        """The type of children this node has."""
        return self._child_type

    @property
    def field_order(self) -> list[str]:
        """Topological order of children (for Config fields only)."""
        return self._field_order.copy()

    def get_child(self, child_name: str) -> Self | None:
        return self._children.get(child_name)

    def ordered_children(self) -> Generator[tuple[str, Self], None, None]:
        for field_name in self._field_order:
            yield field_name, self._children[field_name]

    def _populate(self) -> None:
        """Build child nodes and extract dependencies based on node type.

        This method handles three cases:
        1. Config type fields - creates children from Config's fields
        2. CategoricalSpace with Config choices - creates children from choices
        3. ConditionalSpace - creates children from true/false branches
        """
        from spax.config import Config

        # Case 1: Belongs to a Config type. Build children from its fields
        if isinstance(self._annotation, type) and issubclass(self._annotation, Config):
            # Iterate over all fields in the Config
            for field_name, field_info in self._annotation.model_fields.items():
                annotation = field_info.annotation

                # Check if field is a nested Config
                if isinstance(annotation, type) and issubclass(annotation, Config):
                    self._children[field_name] = annotation._root_node
                else:
                    # Check if field has a Space defined
                    if field_name in self._annotation._spaces:
                        space = self._annotation._spaces[field_name]
                        has_default = space.default is not UNSET
                        fixed_value = None
                        is_default_factory = None
                    else:
                        space = None
                        has_default = True
                        if field_info.default is not PydanticUndefined:
                            fixed_value = field_info.default
                            is_default_factory = False
                        elif field_info.default_factory is not None:
                            fixed_value = field_info.default_factory
                            is_default_factory = True
                        else:
                            # Should not happen - Config validation prevents this
                            raise ValueError(
                                f"Field '{field_name}' has no space, default, or factory"
                            )

                    # Create child node
                    child_node = GraphNode(
                        annotation=annotation,
                        space=space,
                        has_default=has_default,
                        fixed_value=fixed_value,
                        is_default_factory=is_default_factory,
                    )
                    self._children[field_name] = child_node

            self._child_type = "fields"
            return

        # Case 2: CategoricalSpace - check if any choice is a Config
        if isinstance(self._space, CategoricalSpace) and any(  # has_config_choice
            isinstance(choice, type) and issubclass(choice, Config)
            for choice in self._space.choices
        ):
            # Build children for ALL choices
            for choice in self._space.choices:
                # Choices are either Spaces by now, or Configs, or fixed values.
                if isinstance(choice, type) and issubclass(choice, Config):
                    # Config choice - use its root node structure
                    child_node = choice._root_node
                    key = choice.__name__
                else:
                    # Fixed value choice
                    child_node = GraphNode(
                        annotation=type(choice),
                        space=None,
                        has_default=False,
                        fixed_value=choice,
                    )
                    key = str(choice)
                self._children[key] = child_node

            self._child_type = "choices"
            return

        # Case 3: ConditionalSpace
        if isinstance(self._space, ConditionalSpace):
            # Build true branch
            true_branch = self._space.true_branch
            if isinstance(true_branch, Space):
                true_node = GraphNode(
                    annotation=self._annotation,  # Same annotation as parent
                    space=true_branch,
                    has_default=False,
                    fixed_value=None,
                )
            elif isinstance(true_branch, type) and issubclass(true_branch, Config):
                true_node = true_branch._root_node
            else:
                # Fixed value
                true_node = GraphNode(
                    annotation=type(true_branch),
                    space=None,
                    has_default=False,
                    fixed_value=true_branch,
                )

            # Build false branch
            false_branch = self._space.false_branch
            if isinstance(false_branch, Space):
                false_node = GraphNode(
                    annotation=self._annotation,
                    space=false_branch,
                    has_default=False,
                    fixed_value=None,
                )
            elif isinstance(false_branch, type) and issubclass(false_branch, Config):
                false_node = false_branch._root_node
            else:
                # Fixed value
                false_node = GraphNode(
                    annotation=type(false_branch),
                    space=None,
                    has_default=False,
                    fixed_value=false_branch,
                )

            # ConditionalSpace - extract from condition
            self._dependencies = self._space.condition.get_required_fields()
            self._children = {"true": true_node, "false": false_node}
            self._child_type = "branches"

    def _validate_and_order_children(self) -> None:
        """Validate dependencies and compute topological order.

        Only performs ordering for Config-type nodes (child_type="fields").
        Validates that all dependencies exist and checks for circular dependencies.

        Raises:
            ValueError: If dependencies reference unknown fields or circular
                       dependencies are detected
        """
        if self._child_type != "fields" or not self._children:
            # We only want to build order for Config objects.
            self._field_order = []
            return

        # Check if any child has dependencies
        if not any(child._dependencies for child in self._children.values()):
            # No dependencies - use field order as-is
            self._field_order = list(self._children.keys())
            return

        # Build dependency graph and validate all dependencies exist
        for field_name, child_node in self._children.items():
            for dep in child_node._dependencies:
                if dep not in self._children:
                    raise ValueError(
                        f"Field '{field_name}' has dependency on unknown field '{dep}'"
                    )

        # Topological sort using Kahn's algorithm
        in_degree = dict.fromkeys(self._children, 0)

        # Calculate in-degrees
        for field_name, child_node in self._children.items():
            for _ in child_node._dependencies:
                in_degree[field_name] += 1

        # Queue of nodes with no dependencies
        queue: list[str] = [field for field, degree in in_degree.items() if degree == 0]
        ordered: list[str] = []

        while queue:
            queue.sort()  # Keep deterministic
            current = queue.pop(0)
            ordered.append(current)

            # Reduce in-degree for dependent fields
            for field_name, child_node in self._children.items():
                if current in child_node._dependencies:
                    in_degree[field_name] -= 1
                    if in_degree[field_name] == 0:
                        queue.append(field_name)

        # Check for circular dependencies
        if len(ordered) != len(self._children):
            remaining = set(self._children.keys()) - set(ordered)
            cycle_info = []
            for field in remaining:
                deps = self._children[field]._dependencies & remaining
                cycle_info.append(f"{field} -> {deps}")
            raise ValueError(
                f"Circular dependency detected. "
                f"Remaining fields: {remaining}. Dependencies: {', '.join(cycle_info)}"
            )

        self._field_order = ordered

    def __repr__(self) -> str:
        """Return a string representation of the node."""
        return (
            f"GraphNode(annotation={self._annotation.__name__}, "
            f"space={type(self._space).__name__ if self._space else None}, "
            f"children={len(self._children)})"
        )
