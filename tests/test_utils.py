"""Tests for utility functions."""

from typing import Union

from spax.utils import is_comparable, type_from_annotation


class TestIsComparable:
    """Test is_comparable function."""

    def test_primitives_are_comparable(self):
        """Test that primitive types are comparable."""
        assert is_comparable(5) is True
        assert is_comparable(3.14) is True
        assert is_comparable("hello") is True
        assert is_comparable(True) is True
        assert is_comparable(False) is True
        assert is_comparable(None) is True

    def test_types_are_comparable(self):
        """Test that type objects are comparable."""
        assert is_comparable(int) is True
        assert is_comparable(str) is True
        assert is_comparable(list) is True

        class CustomClass:
            pass

        assert is_comparable(CustomClass) is True

    def test_custom_eq_is_comparable(self):
        """Test that objects with custom __eq__ are comparable."""

        class WithCustomEq:
            def __init__(self, value):
                self.value = value

            def __eq__(self, other):
                if not isinstance(other, WithCustomEq):
                    return False
                return self.value == other.value

        obj = WithCustomEq(42)
        assert is_comparable(obj) is True

    def test_default_eq_not_comparable(self):
        """Test that objects with default __eq__ are not comparable."""

        class WithDefaultEq:
            pass

        obj = WithDefaultEq()
        assert is_comparable(obj) is False

    def test_lists_and_dicts_are_comparable(self):
        """Test that built-in containers are comparable."""
        assert is_comparable([1, 2, 3]) is True
        assert is_comparable({"a": 1}) is True
        assert is_comparable((1, 2)) is True
        assert is_comparable({1, 2, 3}) is True

    def test_dataclass_is_comparable(self):
        """Test that dataclasses are comparable."""
        from dataclasses import dataclass

        @dataclass
        class Point:
            x: int
            y: int

        p = Point(1, 2)
        assert is_comparable(p) is True

    def test_config_is_comparable(self):
        """Test that Config instances are comparable."""
        from spax import Config

        class MyConfig(Config):
            pass

        # Type is comparable
        assert is_comparable(MyConfig) is True

        # Instance has custom __eq__ from pydantic
        instance = MyConfig()
        assert is_comparable(instance) is True

    def test_objects_without_eq_not_comparable(self):
        """Test that objects without __eq__ are not comparable."""

        class NoEq:
            __eq__ = None

        obj = NoEq()
        assert is_comparable(obj) is False

    def test_non_callable_eq_not_comparable(self):
        """Test that objects with non-callable __eq__ are not comparable."""

        class NonCallableEq:
            __eq__ = "not callable"

        obj = NonCallableEq()
        assert is_comparable(obj) is False

    def test_nan_like_values_not_comparable(self):
        """Test that NaN-like values (not equal to themselves) are not comparable."""

        class NaNLike:
            def __eq__(self, other):
                # Always returns False, even for self-comparison
                return False

        obj = NaNLike()
        assert is_comparable(obj) is False

    def test_always_true_eq_not_comparable(self):
        """Test that objects with broken __eq__ (always True) are not comparable."""

        class AlwaysEqual:
            def __eq__(self, other):
                # Broken implementation that always returns True
                return True

        obj = AlwaysEqual()
        assert is_comparable(obj) is False

    def test_zero_and_negative_zero(self):
        """Test that numeric values like 0 and -0 are comparable."""
        assert is_comparable(0) is True
        assert is_comparable(-0) is True
        assert is_comparable(0.0) is True
        assert is_comparable(-0.0) is True

    def test_empty_containers(self):
        """Test that empty containers are comparable."""
        assert is_comparable([]) is True
        assert is_comparable({}) is True
        assert is_comparable(()) is True
        assert is_comparable(set()) is True
        assert is_comparable("") is True

    def test_nested_containers(self):
        """Test that nested containers are comparable."""
        assert is_comparable([[1, 2], [3, 4]]) is True
        assert is_comparable({"a": {"b": 1}}) is True
        assert is_comparable([(1, 2), (3, 4)]) is True


class TestTypeFromAnnotation:
    """Test type_from_annotation function."""

    def test_direct_type_match(self):
        """Test finding a type when annotation is directly that type."""
        result = type_from_annotation(int, "int")
        assert result is int

        result = type_from_annotation(str, "str")
        assert result is str

    def test_direct_type_no_match(self):
        """Test that None is returned when type doesn't match."""
        result = type_from_annotation(int, "str")
        assert result is None

        result = type_from_annotation(str, "int")
        assert result is None

    def test_union_type_match(self):
        """Test finding a type in a Union."""
        result = type_from_annotation(Union[int, str], "int")  # noqa: UP007
        assert result is int

        result = type_from_annotation(Union[int, str], "str")  # noqa: UP007
        assert result is str

    def test_union_type_no_match(self):
        """Test that None is returned when type not in Union."""
        result = type_from_annotation(Union[int, str], "float")  # noqa: UP007
        assert result is None

    def test_pipe_union_type_match(self):
        """Test finding a type in a pipe union (|)."""
        result = type_from_annotation(int | str, "int")
        assert result is int

        result = type_from_annotation(int | str, "str")
        assert result is str

    def test_pipe_union_type_no_match(self):
        """Test that None is returned when type not in pipe union."""
        result = type_from_annotation(int | str, "float")
        assert result is None

    def test_custom_class_direct(self):
        """Test finding custom classes directly."""

        class MyClass:
            pass

        result = type_from_annotation(MyClass, "MyClass")
        assert result is MyClass

    def test_custom_class_in_union(self):
        """Test finding custom classes in Union."""

        class ClassA:
            pass

        class ClassB:
            pass

        result = type_from_annotation(Union[ClassA, ClassB], "ClassA")  # noqa: UP007
        assert result is ClassA

        result = type_from_annotation(Union[ClassA, ClassB], "ClassB")  # noqa: UP007
        assert result is ClassB

    def test_custom_class_in_pipe_union(self):
        """Test finding custom classes in pipe union."""

        class ClassA:
            pass

        class ClassB:
            pass

        result = type_from_annotation(ClassA | ClassB, "ClassA")
        assert result is ClassA

        result = type_from_annotation(ClassA | ClassB, "ClassB")
        assert result is ClassB

    def test_config_classes(self):
        """Test finding Config classes."""
        from spax import Config

        class ModelA(Config):
            pass

        class ModelB(Config):
            pass

        result = type_from_annotation(Union[ModelA, ModelB], "ModelA")  # noqa: UP007
        assert result is ModelA

        result = type_from_annotation(Union[ModelA, ModelB], "ModelB")  # noqa: UP007
        assert result is ModelB

    def test_multiple_types_in_union(self):
        """Test finding types in a union with multiple types."""

        class ClassA:
            pass

        class ClassB:
            pass

        class ClassC:
            pass

        annotation = Union[ClassA, ClassB, ClassC, int, str]  # noqa: UP007

        assert type_from_annotation(annotation, "ClassA") is ClassA
        assert type_from_annotation(annotation, "ClassB") is ClassB
        assert type_from_annotation(annotation, "ClassC") is ClassC
        assert type_from_annotation(annotation, "int") is int
        assert type_from_annotation(annotation, "str") is str
        assert type_from_annotation(annotation, "float") is None

    def test_non_type_annotation(self):
        """Test behavior with non-type annotations."""
        # Should return None for non-type objects
        result = type_from_annotation("not a type", "anything")
        assert result is None

        result = type_from_annotation(42, "int")
        assert result is None

    def test_none_annotation(self):
        """Test behavior with None annotation."""
        result = type_from_annotation(None, "NoneType")
        assert result is None

    def test_case_sensitive_matching(self):
        """Test that type name matching is case-sensitive."""

        class MyClass:
            pass

        result = type_from_annotation(MyClass, "myclass")
        assert result is None

        result = type_from_annotation(MyClass, "MYCLASS")
        assert result is None

        result = type_from_annotation(MyClass, "MyClass")
        assert result is MyClass

    def test_similar_named_classes(self):
        """Test distinguishing between similarly named classes."""

        class Model:
            pass

        class ModelConfig:
            pass

        result = type_from_annotation(Union[Model, ModelConfig], "Model")  # noqa: UP007
        assert result is Model

        result = type_from_annotation(Union[Model, ModelConfig], "ModelConfig")  # noqa: UP007
        assert result is ModelConfig

    def test_empty_type_name(self):
        """Test behavior with empty type name."""
        result = type_from_annotation(int, "")
        assert result is None

        result = type_from_annotation(Union[int, str], "")  # noqa: UP007
        assert result is None
