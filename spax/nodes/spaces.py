from typing import Any

from spax.spaces import (
    CategoricalSpace,
    ConditionalSpace,
    NumberSpace,
    Space,
)

from .base import Node
from .fixed import FixedNode


class SpaceNode(Node):
    def __init__(self, space: Space) -> None:
        if not isinstance(space, Space):
            raise TypeError(f"space must be a Space, got {type(space).__name__}")

        self._space = space

    @property
    def space(self) -> Space:
        return self._space


class NumberNode(SpaceNode):
    def __init__(self, space: NumberSpace) -> None:
        super().__init__(space)
        if not isinstance(self._space, NumberSpace):
            raise TypeError(
                f"space must be a NumberSpace, got {type(self._space).__name__}"
            )


class CategoricalNode(SpaceNode):
    def __init__(self, space: CategoricalSpace) -> None:
        super().__init__(space)
        if not isinstance(self._space, CategoricalSpace):
            raise TypeError(
                f"space must be a CategoricalSpace, got {type(self._space).__name__}"
            )

        self._children: dict[int, Node] = {}

        from spax.config import Config

        for i, choice in enumerate(space.choices):
            if isinstance(choice, type) and issubclass(choice, Config):
                self._children[i] = choice._node
            else:
                self._children[i] = FixedNode(default=choice)


class ConditionalNode(SpaceNode):
    def __init__(self, space: ConditionalSpace) -> None:
        super().__init__(space)
        if not isinstance(self._space, ConditionalSpace):
            raise TypeError(
                f"space must be a ConditionalSpace, got {type(self._space).__name__}"
            )

        from spax.config import Config

        def _get_node(branch: Any) -> Node:
            if isinstance(branch, NumberSpace):
                return NumberNode(branch)
            elif isinstance(branch, CategoricalSpace):
                return CategoricalNode(branch)
            elif isinstance(branch, ConditionalSpace):
                return ConditionalNode(branch)
            elif isinstance(branch, type) and issubclass(branch, Config):
                return branch._node
            else:
                return FixedNode(default=branch)

        self._true_node = _get_node(space.true_branch)
        self._false_node = _get_node(space.false_branch)

    def get_active_node(self, config: Any) -> Node:
        if self._space.condition(config):
            return self._true_node
        else:
            return self._false_node

    @property
    def dependencies(self) -> set[str]:
        return self._space.condition.get_required_fields()
