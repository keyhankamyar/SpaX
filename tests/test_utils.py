"""Tests for utility functions."""

from spax.utils import is_comparable


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
