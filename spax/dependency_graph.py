"""
Dependency graph management for conditional configuration spaces.

Builds and validates dependency relationships between configuration fields,
ensuring ConditionalSpaces can be properly ordered for sampling and validation.
"""

from typing import Any

from .spaces import ConditionalSpace, Space


class DependencyGraph:
    """
    Builds and validates dependency ordering for conditional spaces.

    This class analyzes FieldCondition dependencies to:
    - Detect circular dependencies
    - Determine initialization order
    - Provide visualization data
    """

    def __init__(self, spaces: dict[str, Space]) -> None:
        """
        Initialize dependency graph from spaces.

        Args:
            spaces: Dictionary of field names to Space objects.

        Raises:
            ValueError: If circular dependencies are detected or if a dependency
                references an unknown field
        """
        self.spaces = spaces
        self.dependencies: dict[str, set[str]] = {}
        self.order: list[str] = []
        self._build_dependencies()
        self._validate_and_order()

    def _build_dependencies(self) -> None:
        """Build the dependency map for all spaces."""
        for field_name, space in self.spaces.items():
            if isinstance(space, ConditionalSpace):
                self.dependencies[field_name] = space.condition.get_required_fields()
            else:
                self.dependencies[field_name] = set()

    def _validate_and_order(self) -> None:
        """
        Validate no circular dependencies and compute initialization order.

        Uses topological sort (Kahn's algorithm) to determine safe ordering.

        Raises:
            ValueError: If circular dependencies are detected.
        """
        # Calculate in-degrees
        in_degree: dict[str, int] = dict.fromkeys(self.spaces, 0)

        # Validate all dependencies refer to known fields
        for field_name, deps in self.dependencies.items():
            for dep in deps:
                if dep not in self.spaces:
                    raise ValueError(
                        f"Field '{field_name}' has conditional dependency on "
                        f"unknown field '{dep}'"
                    )
                in_degree[field_name] += 1

        # Topological sort using Kahn's algorithm
        queue: list[str] = [field for field, degree in in_degree.items() if degree == 0]
        ordered: list[str] = []

        while queue:
            # Sort for deterministic ordering
            queue.sort()
            current = queue.pop(0)
            ordered.append(current)

            # Reduce in-degree for fields that depend on current
            for field_name, deps in self.dependencies.items():
                if current in deps:
                    in_degree[field_name] -= 1
                    if in_degree[field_name] == 0:
                        queue.append(field_name)

        # Check for circular dependencies
        if len(ordered) != len(self.spaces):
            # Find the cycle
            remaining = set(self.spaces.keys()) - set(ordered)
            cycle_info = []
            for field in remaining:
                deps = self.dependencies[field]
                cycle_info.append(f"{field} -> {deps & remaining}")

            raise ValueError(
                f"Circular dependency detected in conditional spaces. "
                f"Remaining fields: {remaining}. Dependencies: {', '.join(cycle_info)}"
            )

        self.order = ordered

    def get_ordered_fields(self) -> list[str]:
        """
        Get fields in safe initialization order.

        Returns:
            List of field names in dependency-safe order.
        """
        return self.order.copy()

    def get_dependencies(self, field_name: str) -> set[str]:
        """
        Get direct dependencies of a field.

        Args:
            field_name: The field to query.

        Returns:
            Set of field names that this field depends on.
        """
        return self.dependencies.get(field_name, set()).copy()

    def get_dependents(self, field_name: str) -> set[str]:
        """
        Get fields that depend on this field.

        Args:
            field_name: The field to query.

        Returns:
            Set of field names that depend on this field.
        """
        dependents: set[str] = set()
        for field, deps in self.dependencies.items():
            if field_name in deps:
                dependents.add(field)
        return dependents

    def get_graph_data(self) -> dict[str, Any]:
        """
        Get graph data for visualization.

        Returns:
            Dictionary with nodes, edges, and ordering information.
        """
        edges: list[dict[str, str]] = []

        for field_name, deps in self.dependencies.items():
            for dep in deps:
                edges.append({"from": dep, "to": field_name})

        return {
            "nodes": list(self.spaces.keys()),
            "edges": edges,
            "order": self.order,
            "dependencies": {k: list(v) for k, v in self.dependencies.items()},
        }
