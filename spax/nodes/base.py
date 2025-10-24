from abc import ABC, abstractmethod
from typing import Any, Self


class Node(ABC):
    """Base class for all nodes in the configuration tree."""

    @abstractmethod
    def apply_override(self, override: Any) -> Self:
        """Apply an override to this node and return a new modified node.

        This method never modifies the node in-place. It always returns a new
        instance with the override applied. This enables safe cloning of the
        entire configuration tree with modified search spaces.

        Args:
            override: The override specification.

        Returns:
            A new node instance with the override applied.
        """
        pass

    @abstractmethod
    def get_parameter_names(self, prefix: str) -> list[str]:
        """Get all tunable parameter names under this node.

        Args:
            prefix: The hierarchical path prefix for this node (e.g., "ModelConfig.encoder_config")

        Returns:
            List of fully qualified parameter names that can be sampled/tuned.
            Fixed values return empty list, while tunable spaces return their names.
        """
        pass

    @abstractmethod
    def sample(self, sampler: Any, prefix: str) -> Any:
        """Sample a value from this node using the provided sampler.

        Args:
            sampler: A Sampler instance that provides suggest_* methods
            prefix: The hierarchical path prefix for this node

        Returns:
            A sampled value appropriate for this node type.
        """
        pass

    @abstractmethod
    def get_signature(self) -> str:
        """Get a string representation of this node's structure.

        This signature captures all aspects of the search space that affect
        sampling, enabling detection of space changes between experiments.

        Returns:
            A deterministic string representation of the space structure.
        """
        pass

    @abstractmethod
    def get_override_template(self) -> dict[str, Any] | None:
        """Get a template dict structure for overrides.

        Returns a dict that shows the structure for overriding this node,
        with empty/null values. Users can fill this in to create overrides.

        Returns:
            Template dict for overrides, or None if not applicable
        """
        pass
