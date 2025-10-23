from typing import Any

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
