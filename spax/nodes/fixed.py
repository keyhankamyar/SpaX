import json
from typing import Any, Self

from spax.spaces.base import UNSET, _Unset

from .base import Node


class FixedNode(Node):
    def __init__(
        self, default: Any | _Unset = UNSET, default_factory: Any = None
    ) -> None:
        assert sum([default is not UNSET, default_factory is not None]) == 1
        self._default = default
        self._default_factory = default_factory

    @property
    def default(self) -> Any:
        return self._default

    @property
    def default_factory(self) -> Any:
        return self._default_factory

    @property
    def is_factory(self) -> bool:
        return self._default_factory is not None

    def get_default(self) -> Any:
        if self.is_factory:
            return self._default_factory()
        else:
            return self._default

    def apply_override(self, override: Any) -> Self:  # noqa: ARG002
        """Cannot override fixed values."""
        raise ValueError(
            "Cannot override fixed value. Fixed nodes represent immutable values "
            f"that were declared with explicit defaults in the Config. Got {override}"
        )

    def get_parameter_names(self, prefix: str) -> list[str]:  # noqa: ARG002
        """Fixed nodes are not tunable, so return empty list."""
        return []

    def sample(self, sampler: Any, prefix: str) -> Any:  # noqa: ARG002
        """Return the fixed value (no sampling needed)."""
        return self.get_default()

    def get_signature(self) -> str:
        """Get signature for fixed node."""
        value = self.get_default()
        # Use JSON serialization for deterministic representation
        try:
            value_str = json.dumps(value, sort_keys=True)
        except (TypeError, ValueError):
            # For non-serializable objects, use repr
            value_str = repr(value)

        return f"Fixed({value_str})"

    def get_override_template(self) -> None:
        """Fixed nodes cannot be overridden, so no template."""
        return None
