"""Tests for dependency graph module."""

import pytest

from spax.dependency_graph import DependencyGraph
from spax.spaces import And, Conditional, EqualsTo, FieldCondition, Float


class TestDependencyGraphBasic:
    """Tests for basic dependency graph functionality."""

    def test_no_dependencies(self):
        """Test graph with no dependencies."""
        spaces = {
            "a": Float(ge=0.0, le=1.0),
            "b": Float(ge=0.0, le=1.0),
            "c": Float(ge=0.0, le=1.0),
        }

        graph = DependencyGraph(spaces)

        # No edges
        assert len(graph.dependencies["a"]) == 0
        assert len(graph.dependencies["b"]) == 0
        assert len(graph.dependencies["c"]) == 0

        # Order should just be alphabetical or any valid order
        assert len(graph.order) == 3

    def test_simple_dependency(self):
        """Test simple A -> B dependency."""
        spaces = {
            "a": Float(ge=0.0, le=1.0),
            "b": Conditional(
                condition=FieldCondition("a", EqualsTo(0.5)),
                true=Float(ge=1.0, le=2.0),
                false=Float(ge=0.0, le=1.0),
            ),
        }

        graph = DependencyGraph(spaces)

        # b depends on a
        assert "a" in graph.dependencies["b"]
        assert len(graph.dependencies["a"]) == 0

        # a should come before b in order
        assert graph.order.index("a") < graph.order.index("b")

    def test_chain_dependency(self):
        """Test chain A -> B -> C."""
        spaces = {
            "a": Float(ge=0.0, le=1.0),
            "b": Conditional(
                condition=FieldCondition("a", EqualsTo(0.5)),
                true=Float(ge=1.0, le=2.0),
                false=Float(ge=0.0, le=1.0),
            ),
            "c": Conditional(
                condition=FieldCondition("b", EqualsTo(1.5)),
                true=Float(ge=2.0, le=3.0),
                false=Float(ge=0.0, le=2.0),
            ),
        }

        graph = DependencyGraph(spaces)

        # Check dependencies
        assert "a" in graph.dependencies["b"]
        assert "b" in graph.dependencies["c"]

        # Check ordering
        order = graph.order
        assert order.index("a") < order.index("b")
        assert order.index("b") < order.index("c")

    def test_multiple_dependencies_on_one(self):
        """Test multiple fields depending on one: A -> B, A -> C."""
        spaces = {
            "a": Float(ge=0.0, le=1.0),
            "b": Conditional(
                condition=FieldCondition("a", EqualsTo(0.5)),
                true=Float(ge=1.0, le=2.0),
                false=Float(ge=0.0, le=1.0),
            ),
            "c": Conditional(
                condition=FieldCondition("a", EqualsTo(0.5)),
                true=Float(ge=2.0, le=3.0),
                false=Float(ge=0.0, le=2.0),
            ),
        }

        graph = DependencyGraph(spaces)

        # Both b and c depend on a
        assert "a" in graph.dependencies["b"]
        assert "a" in graph.dependencies["c"]

        # a must come before both b and c
        order = graph.order
        a_idx = order.index("a")
        assert a_idx < order.index("b")
        assert a_idx < order.index("c")


class TestDependencyGraphMethods:
    """Tests for dependency graph query methods."""

    def test_get_ordered_fields(self):
        """Test getting ordered fields."""
        spaces = {
            "a": Float(ge=0.0, le=1.0),
            "b": Conditional(
                condition=FieldCondition("a", EqualsTo(0.5)),
                true=Float(ge=1.0, le=2.0),
                false=Float(ge=0.0, le=1.0),
            ),
        }

        graph = DependencyGraph(spaces)
        ordered = graph.get_ordered_fields()

        assert len(ordered) == 2
        assert ordered.index("a") < ordered.index("b")

    def test_get_dependencies(self):
        """Test getting direct dependencies of a field."""
        spaces = {
            "a": Float(ge=0.0, le=1.0),
            "b": Float(ge=0.0, le=1.0),
            "c": Conditional(
                condition=FieldCondition("a", EqualsTo(0.5)),
                true=Float(ge=1.0, le=2.0),
                false=Float(ge=0.0, le=1.0),
            ),
        }

        graph = DependencyGraph(spaces)

        # c depends on a only
        c_deps = graph.get_dependencies("c")
        assert c_deps == {"a"}

        # a has no dependencies
        a_deps = graph.get_dependencies("a")
        assert len(a_deps) == 0

    def test_get_dependents(self):
        """Test getting fields that depend on a given field."""
        spaces = {
            "a": Float(ge=0.0, le=1.0),
            "b": Conditional(
                condition=FieldCondition("a", EqualsTo(0.5)),
                true=Float(ge=1.0, le=2.0),
                false=Float(ge=0.0, le=1.0),
            ),
            "c": Conditional(
                condition=FieldCondition("a", EqualsTo(0.5)),
                true=Float(ge=2.0, le=3.0),
                false=Float(ge=0.0, le=2.0),
            ),
        }

        graph = DependencyGraph(spaces)

        # Both b and c depend on a
        a_dependents = graph.get_dependents("a")
        assert a_dependents == {"b", "c"}

        # Nothing depends on b
        b_dependents = graph.get_dependents("b")
        assert len(b_dependents) == 0

    def test_get_graph_data(self):
        """Test getting graph data for visualization."""
        spaces = {
            "a": Float(ge=0.0, le=1.0),
            "b": Conditional(
                condition=FieldCondition("a", EqualsTo(0.5)),
                true=Float(ge=1.0, le=2.0),
                false=Float(ge=0.0, le=1.0),
            ),
        }

        graph = DependencyGraph(spaces)
        data = graph.get_graph_data()

        # Check structure
        assert "nodes" in data
        assert "edges" in data
        assert "order" in data
        assert "dependencies" in data

        # Check content
        assert set(data["nodes"]) == {"a", "b"}
        assert len(data["edges"]) == 1
        assert data["edges"][0] == {"from": "a", "to": "b"}


class TestDependencyGraphErrors:
    """Tests for error handling in dependency graph."""

    def test_circular_dependency_raises_error(self):
        """Test that circular dependencies are detected."""
        spaces = {
            "a": Conditional(
                condition=FieldCondition("b", EqualsTo(0.5)),
                true=Float(ge=1.0, le=2.0),
                false=Float(ge=0.0, le=1.0),
            ),
            "b": Conditional(
                condition=FieldCondition("a", EqualsTo(0.5)),
                true=Float(ge=1.0, le=2.0),
                false=Float(ge=0.0, le=1.0),
            ),
        }

        with pytest.raises(ValueError, match="Circular dependency"):
            DependencyGraph(spaces)

    def test_self_dependency_raises_error(self):
        """Test that self-dependencies are detected."""
        spaces = {
            "a": Conditional(
                condition=FieldCondition("a", EqualsTo(0.5)),
                true=Float(ge=1.0, le=2.0),
                false=Float(ge=0.0, le=1.0),
            )
        }

        with pytest.raises(ValueError, match="Circular dependency"):
            DependencyGraph(spaces)

    def test_missing_dependency_field_raises_error(self):
        """Test that referencing non-existent fields raises error."""
        spaces = {
            "a": Float(ge=0.0, le=1.0),
            "b": Conditional(
                condition=FieldCondition("nonexistent", EqualsTo(0.5)),
                true=Float(ge=1.0, le=2.0),
                false=Float(ge=0.0, le=1.0),
            ),
        }

        with pytest.raises(ValueError, match="unknown field"):
            DependencyGraph(spaces)


class TestComplexDependencies:
    """Tests for complex dependency patterns."""

    def test_nested_conditional_dependencies(self):
        """Test nested conditionals with proper dependency extraction."""
        spaces = {
            "a": Float(ge=0.0, le=1.0),
            "b": Float(ge=0.0, le=1.0),
            "c": Conditional(
                condition=FieldCondition("a", EqualsTo(0.5)),
                true=Conditional(
                    condition=FieldCondition("b", EqualsTo(0.5)),
                    true=Float(ge=2.0, le=3.0),
                    false=Float(ge=1.0, le=2.0),
                ),
                false=Float(ge=0.0, le=1.0),
            ),
        }

        graph = DependencyGraph(spaces)

        # c depends on both a and b (b is in nested conditional)
        c_deps = graph.get_dependencies("c")
        assert "a" in c_deps
        assert "b" in c_deps

    def test_and_condition_dependencies(self):
        """Test dependencies with And conditions."""
        spaces = {
            "a": Float(ge=0.0, le=1.0),
            "b": Float(ge=0.0, le=1.0),
            "c": Conditional(
                condition=And(
                    [
                        FieldCondition("a", EqualsTo(0.5)),
                        FieldCondition("b", EqualsTo(0.5)),
                    ]
                ),
                true=Float(ge=1.0, le=2.0),
                false=Float(ge=0.0, le=1.0),
            ),
        }

        graph = DependencyGraph(spaces)

        # c depends on both a and b
        c_deps = graph.get_dependencies("c")
        assert c_deps == {"a", "b"}

        # Both a and b must come before c
        order = graph.order
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("c")

    def test_diamond_dependency(self):
        """Test diamond pattern: A -> B, A -> C, B -> D, C -> D."""
        spaces = {
            "a": Float(ge=0.0, le=1.0),
            "b": Conditional(
                condition=FieldCondition("a", EqualsTo(0.5)),
                true=Float(ge=1.0, le=2.0),
                false=Float(ge=0.0, le=1.0),
            ),
            "c": Conditional(
                condition=FieldCondition("a", EqualsTo(0.5)),
                true=Float(ge=1.0, le=2.0),
                false=Float(ge=0.0, le=1.0),
            ),
            "d": Conditional(
                condition=And(
                    [
                        FieldCondition("b", EqualsTo(1.5)),
                        FieldCondition("c", EqualsTo(1.5)),
                    ]
                ),
                true=Float(ge=2.0, le=3.0),
                false=Float(ge=0.0, le=2.0),
            ),
        }

        graph = DependencyGraph(spaces)

        # Check dependencies
        assert graph.get_dependencies("b") == {"a"}
        assert graph.get_dependencies("c") == {"a"}
        assert graph.get_dependencies("d") == {"b", "c"}

        # Check valid ordering (a before b,c; b,c before d)
        order = graph.order
        a_idx = order.index("a")
        b_idx = order.index("b")
        c_idx = order.index("c")
        d_idx = order.index("d")

        assert a_idx < b_idx
        assert a_idx < c_idx
        assert b_idx < d_idx
        assert c_idx < d_idx
