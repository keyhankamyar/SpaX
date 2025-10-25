"""Attribute conditions for evaluating config object fields.

This module provides conditions that evaluate attributes of configuration
objects, enabling conditional parameters that depend on other parameter values.

These "attribute" conditions are used at the top level of ConditionalSpace to:
- express dependencies ("this parameter depends on that other parameter")
- support ordered sampling (dependent fields must be sampled after their parents)
- validate conditional branches

Key ideas:
- FieldCondition targets a single field (possibly nested via dot notation).
- MultiFieldLambdaCondition targets multiple fields at once.
- Both report which top-level fields they depend on so that ConfigNode can
  compute a safe evaluation order.
"""

from abc import abstractmethod
from collections.abc import Callable, Iterable
import inspect
from typing import Any, Self

from .base import Condition


class ParsedFieldPath:
    """Parsed representation of a (possibly dotted) field path.

    A field path is something like:
        "model.optimizer.name"

    We treat this as ["model", "optimizer", "name"] internally and provide:
    - .raw   -> original string form ("model.optimizer.name")
    - .parts -> list of path segments (["model", "optimizer", "name"])
    - .root  -> first segment ("model")
    - .resolve(config) -> walk the config using attribute access to get
                          the final value, with nice error messages.

    This allows us to:
    - support nested attribute access in FieldCondition and
      MultiFieldLambdaCondition
    - validate existence of intermediate objects early, not just at runtime
    """

    def __init__(self, path: str) -> None:
        """Create a parsed view of a (possibly dotted) attribute path.

        The input is a string like "model.optimizer.name". We:
        - strip whitespace,
        - split on ".",
        - validate there are no empty segments,
        - store both the raw string and the individual parts.

        This object is later used to:
        - report the root dependency (first segment),
        - resolve the path step-by-step on a config instance,
        - produce helpful error messages if something is missing.

        Args:
            path: The attribute path to parse. May include dots for nested
                  access (e.g. "trainer.optimizer.lr"). Must be a non-empty
                  string; segments cannot be empty.

        Raises:
            TypeError: If `path` is not a string.
            ValueError: If `path` is empty after stripping, or contains
                        invalid structure like "model..optimizer".
        """
        if not isinstance(path, str):
            raise TypeError(f"field_name must be str, got {type(path).__name__}")

        path = path.strip()
        if path == "":
            raise ValueError("field_name cannot be empty")

        # Support dotted access. e.g. "model.optimizer.name"
        # -> ["model", "optimizer", "name"]
        parts = [part.strip() for part in path.split(".")]

        # Validate that we didn't get something malformed like "model..optimizer"
        # or " model. .x "
        for i, part in enumerate(parts):
            if part == "":
                raise ValueError(
                    f"Invalid field path {path!r}: empty segment at position {i}"
                )

        self._raw = path
        self._parts = parts

    @property
    def raw(self) -> str:
        """Return the original dotted path string."""
        return self._raw

    @property
    def parts(self) -> list[str]:
        """Return a copy of individual path segments."""
        return self._parts.copy()

    @property
    def root(self) -> str:
        """Return the first segment of the path.

        Example:
            ParsedFieldPath("model.optimizer.name").root == "model"
        """
        return self._parts[0]

    @property
    def sub_path(self) -> Self:
        """Returns a ParsedFieldPath from the sub parts."""
        if len(self._parts) <= 1:
            raise ValueError("No sub path available for single-segment path")
        return ParsedFieldPath(".".join(self._parts[1:]))

    def resolve(self, config: Any) -> Any:
        """Resolve this path against a config object.

        We walk attribute-by-attribute:
            current = config
            current = getattr(current, "model")
            current = getattr(current, "optimizer")
            current = getattr(current, "name")
        etc.

        Raises:
            AttributeError: if any hop in the chain is missing.
                The error message includes how far we got, e.g.
                "Configuration object is missing attribute 'model.optimizer'"
        """
        current = config
        walked: list[str] = []

        for segment in self._parts:
            walked.append(segment)

            if not hasattr(current, segment):
                raise AttributeError(
                    "Configuration object is missing attribute "
                    f"{'.'.join(walked)!r} while resolving {self._raw}"
                )

            current = getattr(current, segment)

        return current

    def __repr__(self) -> str:
        return f"ParsedFieldPath({self._raw!r})"


class AttributeCondition(Condition):
    """Base class for conditions that evaluate one or more config attributes.

    AttributeConditions are used by ConditionalSpace to decide which branch
    (true/false) is active. Unlike ObjectConditions (EqualsTo, LargerThan, ...)
    which operate on a single *value*, AttributeConditions operate on the
    *configuration object* itself, pulling out one or more fields and checking
    relationships between them.

    They also expose dependency information to allow:
    - topological sorting of fields,
    - validation in correct order,
    - and deterministic sampling of dependent values later.

    Subclasses must implement:
    - get_required_fields(): which top-level fields must exist before
      this condition can be evaluated.
    - get_required_paths(): the full dotted paths referenced by this
      condition, as ParsedFieldPath objects.
    - __call__(config): the boolean evaluation itself.
    - __repr__(): The representation of the condition.
    """

    @abstractmethod
    def get_required_fields(self) -> set[str]:
        """Return the set of top-level field names this condition depends on.

        Why only top-level names?
        -------------------------
        The sampling / validation logic in ConfigNode orders fields such that
        dependencies are resolved first. If we say a condition depends on
        "model.optimizer.name", the only thing that must already exist at that
        stage is `model` (the root object). After `model` is constructed,
        nested resolution can continue.

        Returns:
            A set of field names that must be available on the config object
            before this condition can safely run. Example:
            - For "model.optimizer.name", we return {"model"}.
            - For ["trainer.batch_size", "model.hidden_dim"], we return
              {"trainer", "model"}.
        """
        pass

    @abstractmethod
    def get_required_paths(self) -> list[ParsedFieldPath]:
        """Return the full attribute paths this condition references.

        This is a stricter view than get_required_fields().

        Example:
            FieldCondition("model.optimizer.name", ...) ->
                [ParsedFieldPath("model.optimizer.name")]

            MultiFieldLambdaCondition(
                ["trainer.batch_size", "model.hidden_dim"],
                ...
            ) ->
                [
                    ParsedFieldPath("trainer.batch_size"),
                    ParsedFieldPath("model.hidden_dim"),
                ]

        This list will be used by ConfigNode to *deep-validate* that each hop
        in each dotted path actually exists in the declared config structure.
        """
        pass


class FieldCondition(AttributeCondition):
    """Condition on a single (possibly nested) field of a config object.

    This wraps an ObjectCondition (like EqualsTo, LargerThan, In, etc.) and
    applies it to the resolved value of the specified field path.

    Dependency tracking:
    get_required_fields() will return only the root segment:
        "model.optimizer.name" -> {"model"}

    This lets ConditionalSpace know that "batch_size" depends on "model"
    (and everything inside it), so "model" must be validated/sampled first.

    Examples:
        >>> # Basic (top-level):
        >>> import spax as sp
        >>>
        >>> # Make learning_rate conditional on optimizer choice
        >>> sp.FieldCondition("optimizer", EqualsTo("adam"))

        >>> # Make dropout conditional on model size
        >>> sp.FieldCondition("num_layers", LargerThan(5, or_equals=True))


        >>> # Nested with dot notation:
        >>> sp.FieldCondition("model.optimizer.name", sp.EqualsTo("adam"))
    """

    def __init__(self, field_name: str, condition: Condition) -> None:
        """Create a condition that applies to a single config field.

        This binds an ObjectCondition (EqualsTo, LargerThan, In, etc.) to a
        specific field on the config. The field may be nested, expressed with
        dot notation:

            FieldCondition("model.optimizer.name", EqualsTo("adam"))

        At runtime, we:
        1. Resolve config.model.optimizer.name.
        2. Pass that value into the provided `condition`.
        3. Use the boolean result.

        The FieldCondition is also responsible for telling the system which
        *top-level* field(s) it depends on so that conditional parameters can
        be evaluated in the right order.

        Args:
            field_name:
                The field path to inspect. Can be a simple field like "optimizer",
                or a dotted path like "model.optimizer.name".
            condition:
                An ObjectCondition that will be called on the resolved value,
                e.g. EqualsTo("adam"), LargerThan(5), In({...}), etc.

        Raises:
            TypeError: If `field_name` is not a string or `condition` is not
                       a Condition instance.
            ValueError: If `field_name` is syntactically invalid (empty segment,
                        empty string, etc.).
        """
        if not isinstance(field_name, str):
            raise TypeError(f"field_name must be str, got {type(field_name).__name__}")
        if not isinstance(condition, Condition):
            raise TypeError(
                f"condition must be a Condition instance, got {type(condition).__name__}"
            )

        # Parse and store path. This also validates the syntax of the path
        # (no empty segments, no empty string, etc.).
        self._field_path = ParsedFieldPath(field_name)

        # Store the wrapped condition (EqualsTo, In, LargerThan, etc.).
        self._condition = condition

        # Building internal paths to the leafs
        self._required_paths: list[ParsedFieldPath] = [self._field_path]
        if isinstance(condition, AttributeCondition):
            for path in condition.get_required_paths():
                self._required_paths.append(
                    ParsedFieldPath(f"{self._field_path.raw}.{path.raw}")
                )

    @property
    def field_name(self) -> str:
        """Return the original (raw) field path string."""
        return self._field_path.raw

    @property
    def condition(self) -> Condition:
        """Return the wrapped Condition that evaluates the field value."""
        return self._condition

    def get_required_fields(self) -> set[str]:
        """Return the top-level dependency set for this condition.

        Example:
            FieldCondition("model.optimizer.name", ...) -> {"model"}
        """
        return {self._field_path.root}

    def get_required_paths(self) -> list[ParsedFieldPath]:
        """Return the single ParsedFieldPath that this condition inspects."""
        return self._required_paths.copy()

    def __call__(self, config: Any) -> bool:
        """Evaluate the condition on the given config object.

        Steps:
        1. Resolve the field path on the config (with detailed error messages
           if something is missing).
        2. Apply the wrapped ObjectCondition to that value.
        3. Return the boolean result.

        Raises:
            AttributeError: if any hop in the dotted path does not exist.
            TypeError / ValueError: if the underlying ObjectCondition raises.
        """
        value = self._field_path.resolve(config)
        return self._condition(value)

    def __repr__(self) -> str:
        return (
            f"FieldCondition(field='{self.field_name}', condition={self._condition!r})"
        )


class MultiFieldLambdaCondition(AttributeCondition):
    """Condition on multiple (possibly nested) fields using a custom function.

    This is for expressing relationships between *several* parameters.

    You pass:
        - field_names: an iterable of field paths (each may be dotted)
        - func: a callable that takes ONE positional argument: a dict

    At runtime we:
        1. Resolve each field path against the config.
        2. Build a dict that maps the *raw* field path string to its value.
           Example:
               {
                   "model.optimizer.name": "adam",
                   "model.hidden_dim": 256,
               }
        3. Call:  func(data_dict)
        4. Expect a boolean result.

    This is different from the previous API, where we tried to map field names
    to kwargs using parameter names. This new version:
    - is simpler
    - is more explicit
    - avoids ambiguity when two paths end with the same leaf name
      (e.g. "model.size" and "dataset.size")

    Dependency tracking
    -------------------
    get_required_fields() returns the set of ROOT segments for all provided
    paths. For example:
        field_names = ["trainer.batch_size", "model.hidden_dim"]
        -> get_required_fields() == {"trainer", "model"}

    That tells ConditionalSpace that any parameter depending on this condition
    can only be evaluated *after* both trainer and model are available.

    Examples:
        >>> import spax as sp
        >>>
        >>> # Condition based on two fields
        >>> sp.MultiFieldLambdaCondition(
        ...     ["batch_size", "num_layers"],
        ...     lambda data: data["batch_size"] * data["num_layers"] < 1000
        ... )

        >>> # Condition with nested fields
        >>> sp.MultiFieldLambdaCondition(
        ...     ["optimizer.name", "learning_rate"],
        ...     lambda data: data["optimizer.name"] == "adam" and data["learning_rate"] > 0.001
        ... )
    """

    def __init__(self, field_names: Iterable[str], func: Callable[..., bool]) -> None:
        """Create a condition over multiple config fields at once.

        This lets you express relationships between several parameters.

        You provide:
            field_names = [
                "model.optimizer.name",
                "model.hidden_dim",
                "trainer.batch_size",
            ]

            func = lambda data: (
                data["model.optimizer.name"] == "adam"
                and data["model.hidden_dim"] >= 256
                and data["trainer.batch_size"] < 512
            )

        During evaluation:
        1. Each field path is resolved on the config (with nested dot access).
        2. We build a dict mapping each raw path string to its resolved value.
           For example:
               {
                   "model.optimizer.name": "adam",
                   "model.hidden_dim": 256,
                   "trainer.batch_size": 128,
               }
        3. We call `func(data_dict)` and expect a bool.

        This design:
        - avoids ambiguity when two different paths end in the same leaf name
          (e.g. "model.size" and "dataset.size"),
        - makes the user function signature simple: just one positional arg,
          a dict of resolved values by path.

        The condition also exposes dependency information to the system:
        it reports that it depends on each *root* segment of the paths
        ("model", "trainer", ...), which is used for correct sampling/validation
        ordering in ConditionalSpace.

        Args:
            field_names:
                Iterable of field paths (each may be dotted, e.g.
                "model.optimizer.name"). Must not be empty.
            func:
                A callable that will be invoked as `func(data_dict)` where
                `data_dict` is {raw_path: resolved_value, ...}.
                Must return a bool.

        Raises:
            TypeError: If `field_names` is not an iterable of strings, or if
                       `func` is not callable.
            ValueError: If `field_names` is empty, or any provided path is
                        syntactically invalid (empty segments, etc.).
        """
        # Validate field_names is iterable but not a string
        if not isinstance(field_names, Iterable) or isinstance(field_names, str):
            raise TypeError(
                f"Expected an iterable for field_names, got {type(field_names).__name__}"
            )

        # We'll keep both the original strings (to expose via .field_names and to
        # present to the callback), and their parsed versions for resolution.
        parsed_paths: dict[str, ParsedFieldPath] = {}
        for raw in field_names:
            parsed_paths[raw] = ParsedFieldPath(raw)

        if not parsed_paths:
            raise ValueError("field_names cannot be empty")

        if not callable(func):
            raise TypeError(f"func must be callable, got {type(func).__name__}")

        # Store:
        # - _paths: map raw field path -> ParsedFieldPath
        # - _field_names: set[str] of raw field paths
        # - _func: the user-supplied callable
        #
        # We are intentionally NOT validating the callable's signature anymore.
        # Instead, we treat it as func(data_dict) and let runtime errors surface
        # naturally if the user gives an incompatible function.
        self._paths = parsed_paths
        self._field_names = set(parsed_paths.keys())

        if len(self._field_names) != len(field_names):
            raise ValueError("field_names cannot contain duplicates")

        self._func = func

    @property
    def field_names(self) -> set[str]:
        """Return the set of raw field path strings this condition depends on."""
        return self._field_names.copy()

    @property
    def func(self) -> Callable[..., bool]:
        """Return the user-provided callable."""
        return self._func

    def get_required_fields(self) -> set[str]:
        """Return the set of top-level root fields required by this condition.

        Example:
            field_names = {"model.optimizer.name", "trainer.batch_size"}
            -> returns {"model", "trainer"}
        """
        return {p.root for p in self._paths.values()}

    def get_required_paths(self) -> list[ParsedFieldPath]:
        """Return all ParsedFieldPath objects this condition inspects.

        The order is deterministic (sorted by raw path) so that callers that
        care about stability (like hashing or error messages) get consistent
        behavior.
        """
        # sort by raw path for determinism
        return [self._paths[raw] for raw in sorted(self._paths.keys())]

    def __call__(self, config: Any) -> bool:
        """Evaluate this condition against a config object.

        Steps:
        1. Resolve each requested field path against the config.
        2. Build a dict mapping the raw path -> resolved value.
        3. Call the provided function with that dict as a single positional arg.
        4. Ensure the result is boolean.

        Raises:
            AttributeError: if any referenced path isn't resolvable.
            TypeError: if the callback doesn't return a bool.
        """
        data: dict[str, Any] = {}
        for raw_name, path in self._paths.items():
            data[raw_name] = path.resolve(config)

        # User function now must accept exactly one positional argument
        # (or be compatible with that), and return bool.
        result = self._func(data)

        if not isinstance(result, bool):
            raise TypeError(
                f"Lambda function must return bool, got {type(result).__name__}"
            )
        return result

    def __repr__(self) -> str:
        # We try to include the callable's signature where possible, but
        # this is just for debug / readability.
        try:
            sig = inspect.signature(self._func)
            func_name = getattr(self._func, "__name__", "<lambda>")
            return (
                "MultiFieldLambdaCondition("
                f"fields={self._field_names}, func={func_name}{sig})"
            )
        except (ValueError, TypeError):
            return f"MultiFieldLambdaCondition(fields={self._field_names})"
