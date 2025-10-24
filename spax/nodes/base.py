"""Base classes for search space node representation.

This module defines the abstract base class for nodes in the internal tree
representation of search spaces. Nodes represent the structure of configuration
spaces and handle operations like override application, parameter enumeration,
and sampling.
"""

from abc import ABC, abstractmethod
from typing import Any, Self


class Node(ABC):
    """Abstract base class for all search space nodes.

    Nodes form an internal tree representation of the search space structure.
    Each node corresponds to a field in a Config class and can be:
    - FixedNode: A field with a fixed value
    - SpaceNode: A field with a sampling space (NumberNode, CategoricalNode, ConditionalNode)
    - ConfigNode: A nested Config class

    Nodes handle:
    - Applying overrides to narrow or fix search spaces
    - Enumerating parameter names in a hierarchical structure
    - Sampling values according to their space definitions
    - Generating signatures for space hashing
    - Creating override templates for users

    This is an internal API used by Config classes. Users typically don't
    interact with nodes directly.
    """

    @abstractmethod
    def apply_override(self, override: Any) -> Self:
        """Apply an override to this node, returning a new modified node.

        Overrides allow narrowing search spaces or fixing values. The exact
        behavior depends on the node type:
        - FixedNode: Raises error (cannot override fixed values)
        - NumberNode: Can narrow bounds or fix to a value
        - CategoricalNode: Can reduce choices or fix to a choice
        - ConditionalNode: Can override individual branches
        - ConfigNode: Can override nested fields

        Args:
            override: Override specification (type depends on node type).

        Returns:
            A new node with the override applied.

        Raises:
            ValueError: If override is invalid or incompatible.
            TypeError: If override has wrong type.
        """
        pass

    @abstractmethod
    def get_parameter_names(self, prefix: str) -> list[str]:
        """Get all parameter names in this node's subtree.

        Parameter names are hierarchical, using "::" to separate levels:
        - "Config.field_name" for top-level fields
        - "Config.nested::NestedConfig.field_name" for nested fields

        Args:
            prefix: The prefix to prepend to parameter names (typically the
                path from root to this node).

        Returns:
            List of fully-qualified parameter names.
        """
        pass

    @abstractmethod
    def sample(self, sampler: Any, prefix: str) -> Any:
        """Sample a value from this node using the given sampler.

        Args:
            sampler: A Sampler instance that generates parameter values.
            prefix: The parameter name prefix (for tracking in sampler).

        Returns:
            A sampled value (type depends on the node type).

        Raises:
            NotImplementedError: For nodes that require config context
                (e.g., ConditionalNode).
        """
        pass

    @abstractmethod
    def get_signature(self) -> str:
        """Get a deterministic signature representing this node's structure.

        Signatures are used for hashing search spaces to detect when the
        space structure changes. The signature includes all aspects that
        affect the possible parameter values (bounds, choices, conditions, etc.).

        Returns:
            A string signature that uniquely identifies this node's structure.
        """
        pass

    @abstractmethod
    def get_override_template(self) -> dict[str, Any] | None:
        """Get a template showing the structure for overriding this node.

        Override templates help users understand what overrides are possible
        and what structure they should follow. The template shows the nested
        dictionary/list structure expected for overrides.

        Returns:
            A template dict/list showing override structure, or None if
            this node cannot be overridden (e.g., FixedNode).
        """
        pass
