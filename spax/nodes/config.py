from collections.abc import Generator
import hashlib
from typing import Any, Self

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
    def __init__(self, config_class: type, *, init_empty: bool = False) -> None:
        self._config_class = config_class
        self._children: dict[str, Node] = {}
        self._field_order: list[str] = []
        if not init_empty:
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
                if space.description:
                    field_info.description = space.description
                node = _get_space_node(space)
                self._children[field_name] = node
            else:
                inferred_space = infer_space_from_field_info(field_info)
                if inferred_space is not None:
                    inferred_space.field_name = field_name
                    node = _get_space_node(inferred_space)
                    self._children[field_name] = node
                # Check if value is fixed:
                elif field_info.default is not PydanticUndefined:
                    self._children[field_name] = FixedNode(default=field_info.default)
                elif field_info.default_factory is not None:
                    self._children[field_name] = FixedNode(
                        default_factory=field_info.default_factory
                    )
                    field_info.default_factory = None
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
            field_info.default = PydanticUndefined
            field_info.default_factory = None

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

    def apply_override(self, override: Any) -> Self:
        """Apply overrides to config children.

        Override must be a dict where keys are field names and values are
        the overrides to apply to those fields.
        """
        if not isinstance(override, dict):
            raise TypeError(
                f"ConfigNode override must be a dict, got {type(override).__name__}"
            )
        if not override:
            raise ValueError("Override dict cannot be empty")

        # Validate all override keys exist as children
        invalid_keys = set(override.keys()) - set(self._children.keys())
        if invalid_keys:
            raise ValueError(
                f"Override contains unknown fields: {invalid_keys}. "
                f"Valid fields are: {list(self._children.keys())}"
            )

        # Create new ConfigNode with same config class
        new_node = ConfigNode(self._config_class, init_empty=True)

        for child_name, child_node in self._children.items():
            if child_name in override:
                try:
                    new_node._children[child_name] = child_node.apply_override(
                        override[child_name]
                    )
                except Exception as e:
                    raise ValueError(
                        f"Exception overriding {child_name} child: {e}"
                    ) from e
            else:
                new_node._children[child_name] = child_node

        new_node._validate_and_order_children()
        return new_node

    def get_parameter_names(self, prefix: str = "") -> list[str]:
        """Return all parameter names from this config's children.

        Args:
            prefix: The hierarchical path prefix (e.g., "ModelConfig.encoder_config")

        Returns:
            List of all tunable parameter names in dependency order.
        """
        if prefix:
            prefix = f"{prefix}::{self._config_class.__name__}"
        else:
            prefix = self._config_class.__name__

        names = []

        for field_name, child_node in self.ordered_children():
            # Build the prefix for this child
            child_prefix = f"{prefix}.{field_name}"

            # Get parameter names from child
            names.extend(child_node.get_parameter_names(child_prefix))

        return names

    def sample(self, sampler: Any, prefix: str = "") -> Any:
        """Sample values for all fields in dependency order.

        Args:
            sampler: Sampler instance
            prefix: Parameter name prefix (usually the config class name)

        Returns:
            Dictionary of sampled field values
        """
        from .spaces import ConditionalNode

        if prefix:
            prefix += f"::{self._config_class.__name__}"
        else:
            prefix = self._config_class.__name__

        sampled_values = {}

        # Create a temporary object to hold values for condition evaluation
        temp_obj = type("TempConfig", (), {})()

        # Sample fields in dependency order
        for field_name, child_node in self.ordered_children():
            # Build the parameter name
            child_prefix = f"{prefix}.{field_name}"

            # Sample the value
            if isinstance(child_node, ConditionalNode):
                value = child_node.sample_with_config(sampler, child_prefix, temp_obj)
            else:
                value = child_node.sample(sampler, child_prefix)

            # Store the value
            sampled_values[field_name] = value
            setattr(temp_obj, field_name, value)

        return self._config_class.model_validate(sampled_values)

    def get_signature(self) -> str:
        """Get signature for config node."""
        # Build signatures for all fields in order
        field_sigs = []
        for field_name, child_node in self.ordered_children():
            child_sig = child_node.get_signature()
            field_sigs.append(f"{field_name}={child_sig}")

        return f"{self._config_class.__name__}({', '.join(field_sigs)})"

    def get_space_hash(self) -> str:
        signature = self.get_signature()
        return hashlib.sha256(signature.encode()).hexdigest()

    def get_override_template(self) -> dict[str, Any]:
        """Get override template for config node.

        Returns a dict with all field names as keys and their override templates as values.
        """
        template = {}

        for field_name, child_node in self.ordered_children():
            child_template = child_node.get_override_template()
            if child_template is not None:
                template[field_name] = child_template

        return template
