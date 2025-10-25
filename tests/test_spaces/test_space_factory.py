"""Tests for space factory (type inference from annotations)."""

from typing import Literal

from pydantic import Field
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from spax import Config
from spax.spaces import (
    CategoricalSpace,
    FloatSpace,
    IntSpace,
    infer_space_from_field_info,
)


class TestNumericInference:
    """Test numeric type inference."""

    def test_int_with_ge_le_constraints(self):
        """Test int with ge and le constraints."""
        field_info = FieldInfo(annotation=int, default=PydanticUndefined, ge=0, le=10)

        space = infer_space_from_field_info(field_info)

        assert isinstance(space, IntSpace)
        assert space.low == 0
        assert space.high == 10
        assert space.low_inclusive is True
        assert space.high_inclusive is True

    def test_int_with_gt_lt_constraints(self):
        """Test int with gt and lt constraints."""
        field_info = FieldInfo(annotation=int, default=PydanticUndefined, gt=0, lt=10)

        space = infer_space_from_field_info(field_info)

        assert isinstance(space, IntSpace)
        assert space.low == 0
        assert space.high == 10
        assert space.low_inclusive is False
        assert space.high_inclusive is False

    def test_int_with_ge_lt_constraints(self):
        """Test int with ge and lt constraints."""
        field_info = FieldInfo(annotation=int, default=PydanticUndefined, ge=0, lt=10)

        space = infer_space_from_field_info(field_info)

        assert isinstance(space, IntSpace)
        assert space.low_inclusive is True
        assert space.high_inclusive is False

    def test_int_with_gt_le_constraints(self):
        """Test int with gt and le constraints."""
        field_info = FieldInfo(annotation=int, default=PydanticUndefined, gt=0, le=10)

        space = infer_space_from_field_info(field_info)

        assert isinstance(space, IntSpace)
        assert space.low_inclusive is False
        assert space.high_inclusive is True

    def test_int_with_default_value(self):
        """Test int inference with default value."""
        field_info = FieldInfo(annotation=int, default=5, ge=0, le=10)

        space = infer_space_from_field_info(field_info)

        assert isinstance(space, IntSpace)
        assert space.default == 5

    def test_int_with_description(self):
        """Test int inference with description."""
        field_info = FieldInfo(
            annotation=int,
            default=PydanticUndefined,
            description="Test integer field",
            ge=0,
            le=10,
        )

        space = infer_space_from_field_info(field_info)

        assert isinstance(space, IntSpace)
        assert space.description == "Test integer field"

    def test_float_with_ge_le_constraints(self):
        """Test float with ge and le constraints."""
        field_info = FieldInfo(
            annotation=float, default=PydanticUndefined, ge=0.0, le=10.0
        )

        space = infer_space_from_field_info(field_info)

        assert isinstance(space, FloatSpace)
        assert space.low == 0.0
        assert space.high == 10.0

    def test_float_with_gt_lt_constraints(self):
        """Test float with gt and lt constraints."""
        field_info = FieldInfo(
            annotation=float, default=PydanticUndefined, gt=0.0, lt=10.0
        )

        space = infer_space_from_field_info(field_info)

        assert isinstance(space, FloatSpace)
        assert space.low_inclusive is False
        assert space.high_inclusive is False

    def test_int_without_constraints_returns_none(self):
        """Test that int without constraints returns None."""
        field_info = FieldInfo(annotation=int, default=PydanticUndefined)

        space = infer_space_from_field_info(field_info)
        assert space is None

    def test_int_with_only_lower_bound_returns_none(self):
        """Test that int with only lower bound returns None."""
        field_info = FieldInfo(annotation=int, default=PydanticUndefined, ge=0)

        space = infer_space_from_field_info(field_info)
        assert space is None

    def test_int_with_only_upper_bound_returns_none(self):
        """Test that int with only upper bound returns None."""
        field_info = FieldInfo(annotation=int, default=PydanticUndefined, le=10)

        space = infer_space_from_field_info(field_info)
        assert space is None


class TestBooleanInference:
    """Test boolean type inference."""

    def test_bool_infers_to_categorical(self):
        """Test that bool infers to Categorical([True, False])."""
        field_info = FieldInfo(annotation=bool, default=PydanticUndefined)

        space = infer_space_from_field_info(field_info)

        assert isinstance(space, CategoricalSpace)
        assert set(space.choices) == {True, False}

    def test_bool_with_default(self):
        """Test bool inference with default value."""
        field_info = FieldInfo(annotation=bool, default=True)

        space = infer_space_from_field_info(field_info)

        assert isinstance(space, CategoricalSpace)
        assert space.default is True

    def test_bool_with_description(self):
        """Test bool inference with description."""
        field_info = FieldInfo(
            annotation=bool, default=PydanticUndefined, description="Test boolean"
        )

        space = infer_space_from_field_info(field_info)

        assert isinstance(space, CategoricalSpace)
        assert space.description == "Test boolean"


class TestLiteralInference:
    """Test Literal type inference."""

    def test_literal_strings(self):
        """Test Literal with string values."""
        field_info = FieldInfo(
            annotation=Literal["a", "b", "c"], default=PydanticUndefined
        )

        space = infer_space_from_field_info(field_info)

        assert isinstance(space, CategoricalSpace)
        assert space.choices == ["a", "b", "c"]

    def test_literal_integers(self):
        """Test Literal with integer values."""
        field_info = FieldInfo(annotation=Literal[1, 2, 3], default=PydanticUndefined)

        space = infer_space_from_field_info(field_info)

        assert isinstance(space, CategoricalSpace)
        assert space.choices == [1, 2, 3]

    def test_literal_mixed_types(self):
        """Test Literal with mixed types."""
        field_info = FieldInfo(
            annotation=Literal[1, "two", 3.0], default=PydanticUndefined
        )

        space = infer_space_from_field_info(field_info)

        assert isinstance(space, CategoricalSpace)
        assert space.choices == [1, "two", 3.0]

    def test_literal_with_default(self):
        """Test Literal with default value."""
        field_info = FieldInfo(annotation=Literal["a", "b", "c"], default="b")

        space = infer_space_from_field_info(field_info)

        assert isinstance(space, CategoricalSpace)
        assert space.default == "b"

    def test_literal_single_value(self):
        """Test Literal with single value."""
        field_info = FieldInfo(annotation=Literal["only"], default=PydanticUndefined)

        space = infer_space_from_field_info(field_info)

        assert isinstance(space, CategoricalSpace)
        assert space.choices == ["only"]


class TestUnionInference:
    """Test Union type inference."""

    def test_union_simple_types(self):
        """Test Union with simple types."""
        field_info = FieldInfo(annotation=int | str | None, default=PydanticUndefined)

        space = infer_space_from_field_info(field_info)

        # Cannot infer from bare types like int, str
        # Only Literal, bool, None, and Config types are supported
        assert space is None

    def test_union_with_literals(self):
        """Test Union with Literal types."""
        field_info = FieldInfo(
            annotation=Literal["a", "b"] | Literal["c", "d"], default=PydanticUndefined
        )

        space = infer_space_from_field_info(field_info)

        assert isinstance(space, CategoricalSpace)
        # Should flatten all literal values
        assert set(space.choices) == {"a", "b", "c", "d"}

    def test_union_with_none(self):
        """Test Union with None."""
        field_info = FieldInfo(
            annotation=Literal["a", "b"] | None, default=PydanticUndefined
        )

        space = infer_space_from_field_info(field_info)

        assert isinstance(space, CategoricalSpace)
        assert set(space.choices) == {"a", "b", None}

    def test_union_with_bool(self):
        """Test Union with bool."""
        field_info = FieldInfo(
            annotation=bool | Literal["maybe"], default=PydanticUndefined
        )

        space = infer_space_from_field_info(field_info)

        assert isinstance(space, CategoricalSpace)
        assert set(space.choices) == {True, False, "maybe"}

    def test_union_with_config_types(self):
        """Test Union with Config types."""

        class ConfigA(Config):
            x: int = 1

        class ConfigB(Config):
            y: int = 2

        field_info = FieldInfo(annotation=ConfigA | ConfigB, default=PydanticUndefined)

        space = infer_space_from_field_info(field_info)

        assert isinstance(space, CategoricalSpace)
        assert set(space.choices) == {ConfigA, ConfigB}

    def test_union_mixed_supported_types(self):
        """Test Union with mix of supported types."""

        class MyConfig(Config):
            x: int = 1

        field_info = FieldInfo(
            annotation=MyConfig | Literal["none"] | None, default=PydanticUndefined
        )

        space = infer_space_from_field_info(field_info)

        assert isinstance(space, CategoricalSpace)
        assert set(space.choices) == {MyConfig, "none", None}

    def test_union_with_default(self):
        """Test Union with default value."""
        field_info = FieldInfo(annotation=Literal["a", "b"] | None, default=None)

        space = infer_space_from_field_info(field_info)

        assert isinstance(space, CategoricalSpace)
        assert space.default is None

    def test_union_with_unsupported_type_returns_none(self):
        """Test Union with unsupported types returns None."""
        field_info = FieldInfo(
            annotation=int | str,  # Bare int and str not supported
            default=PydanticUndefined,
        )

        space = infer_space_from_field_info(field_info)
        assert space is None


class TestEdgeCases:
    """Test edge cases in type inference."""

    def test_unsupported_type_returns_none(self):
        """Test that unsupported types return None."""
        field_info = FieldInfo(annotation=dict, default=PydanticUndefined)

        space = infer_space_from_field_info(field_info)
        assert space is None

    def test_complex_unsupported_type_returns_none(self):
        """Test that complex unsupported types return None."""
        field_info = FieldInfo(annotation=list[int], default=PydanticUndefined)

        space = infer_space_from_field_info(field_info)
        assert space is None

    def test_none_annotation_returns_none(self):
        """Test that None annotation returns None."""
        field_info = FieldInfo(annotation=None, default=PydanticUndefined)

        space = infer_space_from_field_info(field_info)
        assert space is None


class TestIntegrationWithPydanticField:
    """Test integration with actual Pydantic Field usage."""

    def test_with_pydantic_field_int(self):
        """Test with actual Pydantic Field for int."""
        field_info = Field(ge=0, le=100, description="Test field")
        field_info.annotation = int

        space = infer_space_from_field_info(field_info)

        assert isinstance(space, IntSpace)
        assert space.low == 0
        assert space.high == 100
        assert space.description == "Test field"

    def test_with_pydantic_field_float(self):
        """Test with actual Pydantic Field for float."""
        field_info = Field(gt=0.0, lt=1.0, default=0.5)
        field_info.annotation = float

        space = infer_space_from_field_info(field_info)

        assert isinstance(space, FloatSpace)
        assert space.low == 0.0
        assert space.high == 1.0
        assert space.default == 0.5
