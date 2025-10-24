"""Fixed value nodes for immutable configuration fields.

This module provides FixedNode, which represents configuration fields with
fixed (non-searchable) values. These correspond to fields with explicit
defaults or default_factory in Config classes.
"""

import json
from typing import Any, Self

from spax.spaces.base import UNSET, _Unset

from .base import Node


class FixedNode(Node):
    """Node representing a field with a fixed value.

    FixedNode is used for configuration fields that have explicit default
    values or default factories. These fields are not part of the search
    space and cannot be sampled or overridden.

    Fixed nodes can have either:
    - A static default value (stored directly)
    - A default factory (callable that produces the value)

    Examples:
        >>> import spax as sp
        >>>
        >>> class MyConfig(sp.Config):
        ...     # Fixed value
        ...     model_name: str = "resnet50"
        ...
        ...     # Fixed with factory
        ...     timestamp: float = field(default_factory=time.time)
        ...
        ...     # Searchable parameter
        ...     learning_rate: float = sp.Float(ge=1e-5, le=1e-1)
    """

    def __init__(
        self, default: Any | _Unset = UNSET, default_factory: Any = None
    ) -> None:
        """Initialize a FixedNode.

        Args:
            default: The fixed value, or UNSET if using default_factory.
            default_factory: Callable that produces the default value, or None
                if using static default.

        Raises:
            AssertionError: If both or neither default and default_factory are provided.
        """
        assert sum([default is not UNSET, default_factory is not None]) == 1, (
            "Exactly one of default or default_factory must be provided"
        )

        self._default = default
        self._default_factory = default_factory

    @property
    def default(self) -> Any:
        """The static default value, or UNSET if using factory."""
        return self._default

    @property
    def default_factory(self) -> Any:
        """The default factory callable, or None if using static default."""
        return self._default_factory

    @property
    def is_factory(self) -> bool:
        """Whether this node uses a default factory (vs. static value)."""
        return self._default_factory is not None

    def get_default(self) -> Any:
        """Get the default value, calling factory if necessary.

        Returns:
            The default value. If using a factory, calls it to produce the value.
        """
        if self.is_factory:
            return self._default_factory()
        else:
            return self._default

    def apply_override(self, override: Any) -> Self:
        """Raise error - fixed nodes cannot be overridden.

        Args:
            override: The attempted override value.

        Raises:
            ValueError: Always raised, as fixed nodes are immutable.
        """
        raise ValueError(
            "Cannot override fixed value. Fixed nodes represent immutable values "
            f"that were declared with explicit defaults in the Config. Got {override}"
        )

    def get_parameter_names(self, prefix: str) -> list[str]:  # noqa: ARG002
        """Get parameter names (returns empty list for fixed nodes).

        Fixed nodes are not part of the search space, so they don't contribute
        any parameter names.

        Args:
            prefix: The parameter name prefix (unused).

        Returns:
            Empty list, as fixed nodes have no searchable parameters.
        """
        return []

    def sample(self, sampler: Any, prefix: str) -> Any:  # noqa: ARG002
        """Return the fixed default value (no sampling needed).

        Args:
            sampler: The sampler (unused).
            prefix: The parameter name prefix (unused).

        Returns:
            The fixed default value.
        """
        return self.get_default()

    def get_signature(self) -> str:
        """Get a signature representing the fixed value.

        The signature includes the value itself (serialized deterministically
        if possible) to detect when the fixed value changes.

        Returns:
            A string signature of the form "Fixed(<value>)".
        """
        value = self.get_default()

        # Use JSON serialization for deterministic representation
        try:
            value_str = json.dumps(value, sort_keys=True)
        except (TypeError, ValueError):
            # For non-serializable objects, use repr
            value_str = repr(value)

        return f"Fixed({value_str})"

    def get_override_template(self) -> None:
        """Return None - fixed nodes cannot be overridden.

        Returns:
            None, indicating that overrides are not supported.
        """
        return None
