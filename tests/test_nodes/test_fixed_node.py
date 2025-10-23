"""Tests for FixedNode."""

import pytest

from spax.nodes import FixedNode
from spax.spaces.base import UNSET


class TestFixedNode:
    """Test FixedNode."""

    def test_creation_with_default_value(self):
        """Test FixedNode creation with default value."""
        node = FixedNode(default=42)

        assert node.default == 42
        assert node.default_factory is None
        assert node.is_factory is False

    def test_creation_with_default_factory(self):
        """Test FixedNode creation with default_factory."""

        def factory():
            return [1, 2, 3]

        node = FixedNode(default_factory=factory)

        assert node.default is UNSET
        assert node.default_factory is factory
        assert node.is_factory is True

    def test_get_default_with_value(self):
        """Test get_default() with default value."""
        node = FixedNode(default=42)

        assert node.get_default() == 42
        # Should return same value each time
        assert node.get_default() == 42

    def test_get_default_with_factory(self):
        """Test get_default() with default_factory."""
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return {"count": call_count}

        node = FixedNode(default_factory=factory)

        # Should call factory each time
        result1 = node.get_default()
        assert result1 == {"count": 1}

        result2 = node.get_default()
        assert result2 == {"count": 2}

    def test_get_default_factory_returns_different_instances(self):
        """Test that factory creates new instances each time."""
        node = FixedNode(default_factory=list)

        list1 = node.get_default()
        list2 = node.get_default()

        # Should be different instances
        assert list1 is not list2

        # Modifying one shouldn't affect the other
        list1.append(1)
        assert len(list1) == 1
        assert len(list2) == 0

    def test_various_default_types(self):
        """Test FixedNode with various default value types."""
        # Integer
        node = FixedNode(default=42)
        assert node.get_default() == 42

        # String
        node = FixedNode(default="hello")
        assert node.get_default() == "hello"

        # None
        node = FixedNode(default=None)
        assert node.get_default() is None

        # List (will be same instance)
        my_list = [1, 2, 3]
        node = FixedNode(default=my_list)
        assert node.get_default() is my_list

    def test_error_when_both_provided(self):
        """Test error when both default and default_factory provided."""
        with pytest.raises(AssertionError):
            FixedNode(default=42, default_factory=list)

    def test_error_when_neither_provided(self):
        """Test error when neither default nor default_factory provided."""
        with pytest.raises(AssertionError):
            FixedNode()
