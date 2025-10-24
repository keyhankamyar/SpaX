"""Space nodes for searchable configuration fields.

This module provides node types for fields with search spaces:
- NumberNode: Numeric ranges (int/float)
- CategoricalNode: Discrete choices
- ConditionalNode: Conditional spaces based on other parameters
"""

import json
from typing import Any

from spax.spaces import (
    CategoricalSpace,
    ConditionalSpace,
    FloatSpace,
    IntSpace,
    NumberSpace,
    Space,
)
from spax.spaces.base import UNSET

from .base import Node
from .fixed import FixedNode


class SpaceNode(Node):
    """Base class for nodes that wrap a Space object.

    SpaceNode provides common functionality for nodes that contain a Space
    (NumberNode, CategoricalNode, ConditionalNode). It stores the space and
    provides a property to access it.
    """

    def __init__(self, space: Space) -> None:
        """Initialize a SpaceNode.

        Args:
            space: The Space object this node wraps.

        Raises:
            TypeError: If space is not a Space instance.
        """
        if not isinstance(space, Space):
            raise TypeError(f"space must be a Space, got {type(space).__name__}")
        self._space = space

    @property
    def space(self) -> Space:
        """The Space object this node wraps."""
        return self._space


class NumberNode(SpaceNode):
    """Node for numeric search spaces (int/float ranges).

    NumberNode handles both integer and float spaces with various bound
    configurations and distributions (uniform/log).

    Supports overrides:
    - Single numeric value: Fixes the field to that value (becomes FixedNode)
    - Dict with bounds (gt/ge/lt/le): Narrows the range

    Examples:
        >>> import spax as sp
        >>>
        >>> class MyConfig(sp.Config):
        ...     # Integer space
        ...     num_layers: int = sp.Int(ge=1, le=10)
        ...
        ...     # Float space with log distribution
        ...     learning_rate: float = sp.Float(ge=1e-5, le=1e-1, distribution='log')
    """

    def __init__(self, space: NumberSpace) -> None:
        """Initialize a NumberNode.

        Args:
            space: A NumberSpace (IntSpace or FloatSpace).

        Raises:
            TypeError: If space is not a NumberSpace.
        """
        super().__init__(space)
        if not isinstance(self._space, NumberSpace):
            raise TypeError(
                f"space must be a NumberSpace, got {type(self._space).__name__}"
            )

    def apply_override(self, override: Any) -> Node:
        """Apply an override to narrow the range or fix the value.

        Args:
            override: Either a numeric value (fixes to that value) or a dict
                with bound keys (gt/ge/lt/le) to narrow the range.

        Returns:
            FixedNode if override is a single value, or NumberNode with
            narrowed range if override is a dict.

        Raises:
            ValueError: If override is invalid or outside original bounds.
            TypeError: If override has wrong type.
        """
        # Case 1: Single value -> FixedNode
        if isinstance(override, (int, float)):
            # Validate the value is within original bounds
            try:
                override = self._space.validate(override)
            except ValueError as e:
                raise ValueError(
                    f"Override value {override} is not valid for this space: {e}"
                ) from e
            return FixedNode(default=override)

        # Case 2: Dict with bounds
        if not isinstance(override, dict):
            raise TypeError(
                f"Override must be a numeric value or dict with bounds, "
                f"got {type(override).__name__}"
            )

        # Validate dict keys
        valid_keys = {"gt", "ge", "lt", "le"}
        invalid_keys = set(override.keys()) - valid_keys
        if invalid_keys:
            raise ValueError(
                f"Invalid override keys: {invalid_keys}. Valid keys are: {valid_keys}"
            )

        # Create new space with override bounds
        kwargs = {
            **override,
            "distribution": self._space.distribution,
            "description": self._space.description,
        }

        if isinstance(self._space, IntSpace):
            new_space = IntSpace(**kwargs)
        else:
            new_space = FloatSpace(**kwargs)

        # Validate new space fits within original
        if not self._space.contains(new_space):
            raise ValueError(
                f"Override space {new_space} does not fit within original space {self._space}"
            )

        # Preserve default if it's still valid
        if self._space.default is not UNSET:
            try:
                new_space.validate(self._space.default)
                new_space.default = self._space.default
            except Exception:
                pass

        return NumberNode(new_space)

    def get_parameter_names(self, prefix: str) -> list[str]:
        """Get parameter name for this numeric field.

        Args:
            prefix: The parameter name prefix.

        Returns:
            List containing the single parameter name.
        """
        return [prefix]

    def sample(self, sampler: Any, prefix: str) -> int | float:
        """Sample a numeric value using the sampler.

        Args:
            sampler: A Sampler instance.
            prefix: The parameter name for tracking.

        Returns:
            Sampled int or float value.
        """
        space = self._space

        if isinstance(space, IntSpace):
            return sampler.suggest_int(
                name=prefix,
                low=space.low,
                high=space.high,
                low_inclusive=space.low_inclusive,
                high_inclusive=space.high_inclusive,
                distribution=space.distribution,
            )
        else:  # FloatSpace
            return sampler.suggest_float(
                name=prefix,
                low=space.low,
                high=space.high,
                low_inclusive=space.low_inclusive,
                high_inclusive=space.high_inclusive,
                distribution=space.distribution,
            )

    def get_signature(self) -> str:
        """Get signature including type, bounds, and distribution.

        Returns:
            Signature string like "IntSpace(low=1, high=10, ...)".
        """
        space = self._space
        return (
            f"{space.__class__.__name__}("
            f"low={space.low}, "
            f"high={space.high}, "
            f"low_inclusive={space.low_inclusive}, "
            f"high_inclusive={space.high_inclusive}, "
            f"distribution={space.distribution}"
            f")"
        )

    def get_override_template(self) -> dict[str, Any]:
        """Get template showing bound override structure.

        Returns:
            Dict with bound keys (gt/ge/lt/le) showing current values.
        """
        space = self._space
        template = {}

        # Add bound keys based on original inclusivity
        if space.low_inclusive:
            template["ge"] = space.low
        else:
            template["gt"] = space.low

        if space.high_inclusive:
            template["le"] = space.high
        else:
            template["lt"] = space.high

        return template


class CategoricalNode(SpaceNode):
    """Node for categorical (discrete choice) search spaces.

    CategoricalNode handles choices that can be simple values or nested
    Config classes. Each choice is represented as a child node.

    Supports overrides:
    - Single choice value: Fixes to that choice (returns corresponding child node)
    - List of choices: Narrows to subset of choices
    - Dict mapping choices to nested overrides: Narrows choices and applies
      nested overrides to Config children

    Examples:
        >>> import spax as sp
        >>>
        >>> class MyConfig(sp.Config):
        ...     # Simple categorical
        ...     optimizer: str = sp.Categorical(["adam", "sgd", "rmsprop"])
        ...
        ...     # Categorical with Config choices
        ...     model: ModelConfig = sp.Categorical([ResNet, VGG, Transformer])
    """

    def __init__(self, space: CategoricalSpace) -> None:
        """Initialize a CategoricalNode.

        Args:
            space: A CategoricalSpace.

        Raises:
            TypeError: If space is not a CategoricalSpace.
        """
        super().__init__(space)
        if not isinstance(self._space, CategoricalSpace):
            raise TypeError(
                f"space must be a CategoricalSpace, got {type(self._space).__name__}"
            )

        # Create child nodes for each choice
        self._children: dict[str, Node] = {}

        from spax.config import Config

        for choice in space.choices:
            if isinstance(choice, type) and issubclass(choice, Config):
                # Choice is a Config class - use its node
                self._children[choice.__name__] = choice._node
            else:
                # Choice is a simple value - create FixedNode
                self._children[str(choice)] = FixedNode(default=choice)

    @property
    def children(self) -> dict[str, Node]:
        """Dict mapping choice keys to their child nodes."""
        return self._children

    def apply_override(self, override: Any) -> Node:
        """Apply an override to narrow choices or fix to a single choice.

        Args:
            override: Can be:
                - Single value: Fixes to that choice
                - List of values: Narrows to those choices
                - Dict: Narrows choices and applies nested overrides

        Returns:
            Appropriate node based on override type.

        Raises:
            ValueError: If override references invalid choices.
            TypeError: If override has wrong type.
        """
        # Case 1: Single value (check before list/dict since they might match str(override))
        if str(override) in self._children:
            return self._children[str(override)]

        # Case 2: List of choices -> subset
        if isinstance(override, list):
            if not override:
                raise ValueError("Override list cannot be empty")

            # Validate all choices exist
            for choice in override:
                if str(choice) not in self._children:
                    raise ValueError(
                        f"Override choice {choice!r} not in original choices: "
                        f"{list(self._children.keys())}"
                    )

            # If single choice, return its node
            if len(override) == 1:
                return self._children[str(override[0])]

            # Multiple choices -> new CategoricalNode
            new_choices = []
            contains_default = False

            for choice in override:
                choice_key = str(choice)
                if self._space.default is not UNSET and choice_key == str(
                    self._space.default
                ):
                    contains_default = True

                child_node = self._children[choice_key]
                if isinstance(child_node, FixedNode):
                    new_choices.append(child_node.default)
                else:
                    # ConfigNode - get the config class
                    new_choices.append(child_node._config_class)

            kwargs = {"description": self._space.description}
            if contains_default:
                kwargs["default"] = self._space.default

            return CategoricalNode(CategoricalSpace(choices=new_choices, **kwargs))

        # Case 3: Dict -> fix to choice(s) and apply nested overrides
        if isinstance(override, dict):
            if not override:
                raise ValueError("Override dict cannot be empty")

            # Validate all keys exist
            for choice_key in override:
                if choice_key not in self._children:
                    raise ValueError(
                        f"Override choice {choice_key!r} not in original choices: "
                        f"{list(self._children.keys())}"
                    )

            # If single choice with overrides -> apply and return node
            if len(override) == 1:
                choice_key = list(override.keys())[0]
                nested_override = override[choice_key]
                child_node = self._children[choice_key]

                # Apply nested override if provided
                if nested_override:
                    try:
                        child_node = child_node.apply_override(nested_override)
                    except Exception as e:
                        raise ValueError(
                            f"Exception overriding {choice_key} child: {e}"
                        ) from e

                return child_node

            # Multiple choices with overrides -> new CategoricalNode
            new_choices = []
            contains_default = False

            for choice_key in override:
                if self._space.default is not UNSET and choice_key == str(
                    self._space.default
                ):
                    contains_default = True

                child_node = self._children[choice_key]
                if isinstance(child_node, FixedNode):
                    new_choices.append(child_node.default)
                else:
                    new_choices.append(child_node._config_class)

            kwargs = {"description": self._space.description}
            if contains_default:
                kwargs["default"] = self._space.default

            new_node = CategoricalNode(CategoricalSpace(choices=new_choices, **kwargs))

            # Apply nested overrides
            for choice_key, nested_override in override.items():
                if nested_override:
                    try:
                        new_node._children[choice_key] = new_node._children[
                            choice_key
                        ].apply_override(nested_override)
                    except Exception as e:
                        raise ValueError(
                            f"Exception overriding {choice_key} child: {e}"
                        ) from e

            return new_node

        raise TypeError(
            f"Override must be a value, list, or dict, got {type(override).__name__}"
        )

    def get_parameter_names(self, prefix: str) -> list[str]:
        """Get parameter names including the choice itself and nested parameters.

        Args:
            prefix: The parameter name prefix.

        Returns:
            List of parameter names.
        """
        # The categorical choice itself is a parameter
        names = [prefix]

        # Add nested parameters from Config choices
        for child_node in self._children.values():
            names.extend(child_node.get_parameter_names(prefix))

        return names

    def sample(self, sampler: Any, prefix: str) -> Any:
        """Sample a categorical choice and then sample from that choice's node.

        Args:
            sampler: A Sampler instance.
            prefix: The parameter name for tracking.

        Returns:
            Sampled value from the chosen branch.
        """
        # Get the choice keys and weights
        choice_keys = list(self._children.keys())
        weights = self._space.probs

        # Sample the choice
        chosen_key = sampler.suggest_categorical(
            name=prefix,
            choices=choice_keys,
            weights=weights,
        )

        # Get the corresponding child node and sample from it
        child_node = self._children[chosen_key]
        return child_node.sample(sampler, prefix)

    def get_signature(self) -> str:
        """Get signature including all choices and their signatures.

        Returns:
            Signature string with all choice signatures.
        """
        # Build signatures for all choices
        choice_sigs = []
        for choice_key in sorted(self._children.keys()):  # Sort for determinism
            child_node = self._children[choice_key]
            child_sig = child_node.get_signature()
            choice_sigs.append(f"{choice_key}:{child_sig}")

        # Include weights/probabilities
        weights_str = json.dumps(self._space.probs, sort_keys=True)
        return f"Categorical([{', '.join(choice_sigs)}], weights={weights_str})"

    def get_override_template(self) -> dict[str, Any] | list[str]:
        """Get override template showing choice structure.

        Returns:
            Dict if any choices are Configs (showing nested structure), or
            list of choice keys if all choices are simple values.
        """
        from .config import ConfigNode

        # Check if any choices are ConfigNodes
        has_config_choices = any(
            isinstance(child, ConfigNode) for child in self._children.values()
        )

        if has_config_choices:
            # Return dict structure for nested overrides
            template = {}
            for choice_key, child_node in self._children.items():
                child_template = child_node.get_override_template()
                if child_template is not None:
                    template[choice_key] = child_template
                else:
                    template[choice_key] = {}
            return template
        else:
            # Return list of simple choices
            return list(self._children.keys())


class ConditionalNode(SpaceNode):
    """Node for conditional spaces that depend on other parameters.

    ConditionalNode represents parameters whose possible values depend on
    the values of other parameters. It has two branches (true/false) and
    evaluates a condition on the config to determine which branch is active.

    Supports overrides:
    - Dict with 'true'/'false' keys: Applies overrides to respective branches

    Examples:
        >>> import spax as sp
        >>>
        >>> class MyConfig(sp.Config):
        ...     use_dropout: bool
        ...     dropout_rate: float = sp.Conditional(
        ...         sp.FieldCondition("use_dropout", sp.EqualsTo(True)),
        ...         true=sp.Float(gt=0.0, lt=0.5),
        ...         false=0.0
        ...     )
    """

    def __init__(self, space: ConditionalSpace) -> None:
        """Initialize a ConditionalNode.

        Args:
            space: A ConditionalSpace.

        Raises:
            TypeError: If space is not a ConditionalSpace.
        """
        super().__init__(space)
        if not isinstance(self._space, ConditionalSpace):
            raise TypeError(
                f"space must be a ConditionalSpace, got {type(self._space).__name__}"
            )

        from spax.config import Config

        def _get_node(branch: Any) -> Node:
            """Convert a branch (Space or value) to a Node."""
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

    @property
    def true_node(self) -> Node:
        """The node for the true branch."""
        return self._true_node

    @property
    def false_node(self) -> Node:
        """The node for the false branch."""
        return self._false_node

    def get_active_node(self, config: Any) -> Node:
        """Evaluate condition and return the active node.

        Args:
            config: Config object to evaluate condition on.

        Returns:
            The active node (true_node or false_node).
        """
        if self._space.condition(config):
            return self._true_node
        else:
            return self._false_node

    @property
    def dependencies(self) -> set[str]:
        """Set of field names this condition depends on."""
        return self._space.condition.get_required_fields()

    def apply_override(self, override: Any) -> Node:
        """Apply overrides to the true and/or false branches.

        Args:
            override: Dict with 'true' and/or 'false' keys containing overrides
                for respective branches.

        Returns:
            New ConditionalNode with overrides applied.

        Raises:
            ValueError: If override has invalid keys or is empty.
            TypeError: If override is not a dict.
        """
        if not isinstance(override, dict):
            raise TypeError(
                f"ConditionalNode override must be a dict with 'true'/'false' keys, "
                f"got {type(override).__name__}"
            )

        if not override:
            raise ValueError("Override dict cannot be empty")

        valid_keys = {"true", "false"}
        invalid_keys = set(override.keys()) - valid_keys
        if invalid_keys:
            raise ValueError(
                f"Invalid override keys: {invalid_keys}. Valid keys are: {valid_keys}"
            )

        new_true_node = self._true_node
        new_false_node = self._false_node

        # Extract the actual space/value from nodes for creating new ConditionalSpace
        def node_to_branch(node: Node) -> Any:
            """Convert a Node back to a branch (Space or value)."""
            if isinstance(node, FixedNode):
                return node.default
            elif isinstance(node, SpaceNode):
                return node._space
            else:
                # ConfigNode
                return node._config_class

        new_true_branch = node_to_branch(new_true_node)
        new_false_branch = node_to_branch(new_false_node)

        # Create new ConditionalNode with same condition
        new_node = ConditionalNode(
            ConditionalSpace(
                condition=self._space.condition,
                true=new_true_branch,
                false=new_false_branch,
                description=self._space.description,
            )
        )

        # Apply overrides to branches
        if "true" in override:
            try:
                new_node._true_node = new_node._true_node.apply_override(
                    override["true"]
                )
            except Exception as e:
                raise ValueError(f"Exception overriding true branch: {e}") from e

        if "false" in override:
            try:
                new_node._false_node = new_node._false_node.apply_override(
                    override["false"]
                )
            except Exception as e:
                raise ValueError(f"Exception overriding false branch: {e}") from e

        return new_node

    def get_parameter_names(self, prefix: str) -> list[str]:
        """Get parameter names from both branches.

        Args:
            prefix: The parameter name prefix.

        Returns:
            List of parameter names from both branches.
        """
        names = []

        # Get names from true branch
        names.extend(self._true_node.get_parameter_names(prefix + "::true_branch"))

        # Get names from false branch
        names.extend(self._false_node.get_parameter_names(prefix + "::false_branch"))

        return names

    def sample(self, sampler: Any, prefix: str) -> Any:
        """Raise error - ConditionalNode requires config context for sampling.

        Args:
            sampler: A Sampler instance (unused).
            prefix: The parameter name prefix (unused).

        Raises:
            NotImplementedError: Always raised.
        """
        raise NotImplementedError(
            f"ConditionalNode '{prefix}' cannot be sampled without config context. "
            "Use sample_with_config(sampler, prefix, config) instead."
        )

    def sample_with_config(self, sampler: Any, prefix: str, config: Any) -> Any:
        """Sample from the active branch based on config.

        Args:
            sampler: A Sampler instance.
            prefix: The parameter name prefix.
            config: Config object to evaluate condition on.

        Returns:
            Sampled value from the active branch.
        """
        if self._space.condition(config):
            if isinstance(self._true_node, ConditionalNode):
                return self._true_node.sample_with_config(
                    sampler, prefix + "::true_branch", config
                )
            else:
                return self._true_node.sample(sampler, prefix + "::true_branch")
        else:
            if isinstance(self._false_node, ConditionalNode):
                return self._false_node.sample_with_config(
                    sampler, prefix + "::false_branch", config
                )
            else:
                return self._false_node.sample(sampler, prefix + "::false_branch")

    def get_signature(self) -> str:
        """Get signature including condition and both branch signatures.

        Returns:
            Signature string with condition and branch signatures.
        """
        condition_str = repr(self._space.condition)
        true_sig = self._true_node.get_signature()
        false_sig = self._false_node.get_signature()

        return f"Conditional(condition={condition_str}, true={true_sig}, false={false_sig})"

    def get_override_template(self) -> dict[str, Any] | None:
        """Get override template showing true/false branch structure.

        Returns:
            Dict with 'true' and/or 'false' keys, or None if both branches
            cannot be overridden.
        """
        template = {}

        true_template = self._true_node.get_override_template()
        if true_template is not None:
            template["true"] = true_template

        false_template = self._false_node.get_override_template()
        if false_template is not None:
            template["false"] = false_template

        return template if template else None
