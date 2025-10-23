from collections.abc import Generator
from typing import Any

from pydantic_core import PydanticUndefined

from spax.spaces import (
    UNSET,
    CategoricalSpace,
    ConditionalSpace,
    NumberSpace,
    Space,
    infer_space_from_field_info,
)

from .base import Node
from .fixed import FixedNode
from .spaces import CategoricalNode, ConditionalNode, NumberNode, SpaceNode


class ConfigNode(Node):
    def __init__(self, config_class: type) -> None:
        self._config_class = config_class
        self._children: dict[str, Node] = {}
        self._field_order: list[str] = []
        self._populate()
        self._validate_and_order_children()

    def _populate(self) -> None:
        from spax.config import Config

        if not isinstance(self._config_class, type):
            raise TypeError(
                f"config_class must be a type, got {type(self._config_class).__name__}"
            )

        if not issubclass(self._config_class, Config):
            raise TypeError(
                f"config_class must be a Config class, got {type(self._config_class).__name__}"
            )

        def _get_space_node(space: Space) -> Node:
            if isinstance(space, NumberSpace):
                return NumberNode(space)
            elif isinstance(space, CategoricalSpace):
                # Simplify single-choice categorical spaces
                if len(space.choices) == 1:
                    single_choice = space.choices[0]
                    # If it's a Config type, remove the space entirely
                    if isinstance(single_choice, type) and issubclass(
                        single_choice, Config
                    ):
                        return single_choice._node
                    else:
                        return FixedNode(default=single_choice)
                else:
                    return CategoricalNode(space)
            else:
                return ConditionalNode(space)

        parent_node: ConfigNode | None = None
        if self._config_class.__mro__[1:2]:
            parent_class = self._config_class.__mro__[1]
            if (
                isinstance(parent_class, type)
                and issubclass(parent_class, Config)
                and parent_class is not Config
                and hasattr(parent_class, "_node")
                and parent_class._node is not None
            ):
                parent_node = parent_class._node

        for field_name, field_info in self._config_class.model_fields.items():
            # If the field has declared space:
            if isinstance(field_info.default, Space):
                space = field_info.default
                node = _get_space_node(space)
                self._children[field_name] = node
                if space.default is not UNSET:
                    if isinstance(node, FixedNode):
                        assert node.default == space.default
                    field_info.default = space.default
                else:
                    if isinstance(node, FixedNode):
                        field_info.default = node.default
                    else:
                        field_info.default = PydanticUndefined
            else:
                inferred_space = infer_space_from_field_info(field_info)
                if inferred_space is not None:
                    inferred_space.field_name = field_name
                    node = _get_space_node(inferred_space)
                    self._children[field_name] = node
                    if isinstance(node, FixedNode):
                        field_info.default = node.default
                # Check if value is fixed:
                elif field_info.default is not PydanticUndefined:
                    self._children[field_name] = FixedNode(default=field_info.default)
                elif field_info.default_factory is not None:
                    self._children[field_name] = FixedNode(
                        default_factory=field_info.default_factory
                    )
                # Nested Config case
                elif isinstance(field_info.annotation, type) and issubclass(
                    field_info.annotation, Config
                ):
                    self._children[field_name] = field_info.annotation._node
                elif parent_node is not None and field_name in parent_node._children:
                    self._children[field_name] = parent_node._children[field_name]
                else:
                    raise ValueError(
                        f"Unsupported field '{field_name}'. A field either has to:"
                        " a) be a config, b) have declared space, c) have an inferable"
                        " space annotation, d) have a default value or factory."
                    )

    def _validate_and_order_children(self) -> None:
        if not self._children:
            return

        # Check if any child has dependencies
        if not any(
            isinstance(child, ConditionalNode) for child in self._children.values()
        ):
            # No dependencies - use field order as-is
            self._field_order = list(self._children.keys())
            return

        # Build dependency graph and validate all dependencies exist
        for field_name, child_node in self._children.items():
            if isinstance(child_node, ConditionalNode):
                for dep in child_node.dependencies:
                    if dep not in self._children:
                        raise ValueError(
                            f"Field '{field_name}' has dependency on an unknown field '{dep}'"
                        )

        # Topological sort using Kahn's algorithm
        in_degree = dict.fromkeys(self._children, 0)

        # Calculate in-degrees
        for field_name, child_node in self._children.items():
            if isinstance(child_node, ConditionalNode):
                for _ in child_node.dependencies:
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
                if (
                    isinstance(child_node, ConditionalNode)
                    and current in child_node.dependencies
                ):
                    in_degree[field_name] -= 1
                    if in_degree[field_name] == 0:
                        queue.append(field_name)

        # Check for circular dependencies
        if len(ordered) != len(self._children):
            remaining = set(self._children.keys()) - set(ordered)
            cycle_info = []
            for field in remaining:
                if isinstance(self._children[field], ConditionalNode):
                    deps = self._children[field].dependencies & remaining
                    cycle_info.append(f"{field} -> {deps}")
            raise ValueError(
                f"Circular dependency detected. "
                f"Remaining fields: {remaining}. "
                f"Dependencies: {', '.join(cycle_info)}"
            )

        self._field_order = ordered

    def validate_spaces(self, data: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise ValueError(f"Got {data} which is {type(data).__name__}")

        validated: dict[str, Any] = {}

        # Create a temporary object to hold values for condition evaluation
        temp_obj = type("TempConfig", (), {})()

        # Get ordered children and validate in dependency order
        for field_name, field_node in self._config_class._node.ordered_children():
            # Find the value
            if field_name in data:
                value = data[field_name]
            else:
                # If field not in data, try to use default
                if isinstance(field_node, FixedNode):
                    value = field_node.get_default()
                elif (
                    isinstance(field_node, SpaceNode)
                    and field_node.space.default is not UNSET
                ):
                    value = field_node.space.default
                elif isinstance(field_node, ConditionalNode) and isinstance(
                    (active_node := field_node.get_active_node(temp_obj)), FixedNode
                ):
                    value = active_node.get_default()
                elif (
                    isinstance(field_node, ConditionalNode)
                    and isinstance(
                        (active_node := field_node.get_active_node(temp_obj)), SpaceNode
                    )
                    and active_node.space.default is not UNSET
                ):
                    value = active_node.space.default
                else:
                    raise RuntimeError(
                        f"Field '{field_name}' not provided "
                        "in the data and has no default value"
                    )

            # Validate it
            if isinstance(field_node, SpaceNode):
                space = field_node.space
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

        # Add any extra fields and leave validation to pydantic
        for name, value in data.items():
            if name not in validated:
                validated[name] = value

        return validated

    def get_child(self, child_name: str) -> Node | None:
        return self._children.get(child_name)

    def ordered_children(self) -> Generator[tuple[str, Node], None, None]:
        for field_name in self._field_order:
            yield field_name, self._children[field_name]
