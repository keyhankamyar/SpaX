"""Internal node representation of search space structures.

This module provides the node tree representation used internally by Config
classes. Nodes form a tree structure that mirrors the configuration hierarchy
and handles operations like validation, sampling, and override application.

Node Types:
-----------
- Node: Abstract base class for all nodes
- FixedNode: Represents fields with fixed values (not searchable)
- SpaceNode: Base class for nodes wrapping Space objects
  - NumberNode: Numeric ranges (int/float)
  - CategoricalNode: Discrete choices
  - ConditionalNode: Conditional spaces based on other parameters
- ConfigNode: Represents nested Config classes

This is an internal API. Users typically interact with Config classes and
Space objects rather than nodes directly.

Examples:
    >>> import spax as sp
    >>>
    >>> class MyConfig(sp.Config):
    ...     learning_rate: float = sp.Float(ge=1e-5, le=1e-1)
    ...     num_layers: int = sp.Int(ge=1, le=10)
    >>>
    >>> # Node tree is built automatically
    >>> node = MyConfig._node
    >>>
    >>> # Get all parameter names
    >>> print(node.get_parameter_names())
    >>>
    >>> # Get space signature/hash
    >>> print(node.get_space_hash())
"""

from .base import Node
from .config import ConfigNode
from .fixed import FixedNode
from .spaces import CategoricalNode, ConditionalNode, NumberNode, SpaceNode

__all__ = [
    # Base class
    "Node",
    # Node implementations
    "ConfigNode",
    "FixedNode",
    "SpaceNode",
    # SpaceNode subclasses
    "NumberNode",
    "CategoricalNode",
    "ConditionalNode",
]
