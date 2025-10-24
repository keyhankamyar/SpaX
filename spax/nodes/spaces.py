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

    def apply_override(self, override: Any) -> Node:
        """Apply override to numeric space.

        Override can be:
        - A single numeric value -> converts to FixedNode
        - A dict with bounds (ge/gt/le/lt) -> returns new NumberNode with narrowed range
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

        if self._space.default is not UNSET:
            try:
                new_space.validate(self._space.default)
                new_space.default = self._space.default
            except Exception:
                pass

        return NumberNode(new_space)

    def get_parameter_names(self, prefix: str) -> list[str]:
        """Return the parameter name for this numeric space."""
        return [prefix]

    def sample(self, sampler: Any, prefix: str) -> int | float:
        """Sample a numeric value using the sampler."""

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
        """Get signature for numeric space."""
        space = self._space

        # Include type, bounds, inclusivity, and distribution
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
        """Get override template for numeric space.

        Returns a dict showing the available override options.
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
    def __init__(self, space: CategoricalSpace) -> None:
        super().__init__(space)
        if not isinstance(self._space, CategoricalSpace):
            raise TypeError(
                f"space must be a CategoricalSpace, got {type(self._space).__name__}"
            )

        self._children: dict[str, Node] = {}

        from spax.config import Config

        for choice in space.choices:
            if isinstance(choice, type) and issubclass(choice, Config):
                self._children[choice.__name__] = choice._node
            else:
                self._children[str(choice)] = FixedNode(default=choice)

    def apply_override(self, override: Any) -> Node:
        """Apply override to categorical space.

        Override can be:
        - A single value -> converts to FixedNode
        - A list of choices (subset) -> returns new CategoricalNode or FixedNode
        - A dict -> fixes to choice(s) and recursively applies overrides to Config choices
        """

        # Case 1: Single value. Each choice can be a list or dict, so better to check first.
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

            # If single choice, return FixedNode
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

            # If single choice with overrides -> apply and return FixedNode or ConfigNode
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

            # Multiple choices with overrides -> new CategoricalNode with overridden children
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
        """Return parameter names for this categorical space and all its choices.

        Returns the choice parameter itself plus all nested parameters from
        Config choices.
        """

        # The categorical choice itself is a parameter
        names = [prefix]

        # Add nested parameters from Config choices
        for child_node in self._children.values():
            names.extend(child_node.get_parameter_names(prefix))

        return names

    def sample(self, sampler: Any, prefix: str) -> Any:
        """Sample a categorical choice using the sampler.

        First samples which choice to use, then recursively samples
        Config choices if needed.
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

        # Get the corresponding child node
        child_node = self._children[chosen_key]

        return child_node.sample(sampler, prefix)

    def get_signature(self) -> str:
        """Get signature for categorical space."""
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
        """Get override template for categorical space.

        Returns a list of available choices, or a dict for nested config overrides.
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

    def apply_override(self, override: Any) -> Node:
        """Apply override to conditional space branches.

        Override must be a dict with "true" and/or "false" keys to override
        the respective branches.
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

        # We need to extract the actual space/value from the nodes
        def node_to_branch(node: Node) -> Any:
            if isinstance(node, FixedNode):
                return node.default
            elif isinstance(node, SpaceNode):
                return node._space
            else:
                return node._config_class

        new_true_branch = node_to_branch(new_true_node)
        new_false_branch = node_to_branch(new_false_node)

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
        """Return parameter names from all possible branches.

        Since the active branch depends on runtime conditions, we return
        all possible parameter names from both branches.
        """
        names = []

        # Get names from true branch
        names.extend(self._true_node.get_parameter_names(prefix + "::true_branch"))

        # Get names from false branch
        names.extend(self._false_node.get_parameter_names(prefix + "::false_branch"))

        return names

    def sample(self, sampler: Any, prefix: str) -> Any:
        """ConditionalNode cannot be sampled without config context.

        Use sample_with_config() instead, which evaluates the condition
        against a partially-built config object.
        """
        raise NotImplementedError(
            f"ConditionalNode '{prefix}' cannot be sampled without config context. "
            "Use sample_with_config(sampler, prefix, config) instead."
        )

    def sample_with_config(self, sampler: Any, prefix: str, config: Any) -> Any:
        """Sample from the active branch based on config state.

        Args:
            sampler: Sampler instance
            prefix: Parameter name prefix
            config: Partially-built config object to evaluate condition

        Returns:
            Sampled value from the active branch
        """
        if self._space.condition(config):
            return self._true_node.sample(sampler, prefix + "::true_branch")
        else:
            return self._false_node.sample(sampler, prefix + "::false_branch")

    def get_signature(self) -> str:
        """Get signature for conditional space."""
        # Include condition representation and both branch signatures
        condition_str = repr(self._space.condition)
        true_sig = self._true_node.get_signature()
        false_sig = self._false_node.get_signature()

        return f"Conditional(condition={condition_str}, true={true_sig}, false={false_sig})"

    def get_override_template(self) -> dict[str, Any]:
        """Get override template for conditional space.

        Returns a dict with 'true' and 'false' keys for overriding branches.
        """
        template = {}

        true_template = self._true_node.get_override_template()
        if true_template is not None:
            template["true"] = true_template

        false_template = self._false_node.get_override_template()
        if false_template is not None:
            template["false"] = false_template

        return template if template else None
