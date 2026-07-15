import pytest

from crosstl.translator.arithmetic_conversions import (
    ArithmeticScalarKind,
    ArithmeticType,
    resolve_arithmetic_conversion,
)


def _bool_type(lanes=1):
    target_type = "bool" if lanes == 1 else f"bvec{lanes}"
    return ArithmeticType(
        source_type=target_type,
        target_type=target_type,
        kind=ArithmeticScalarKind.BOOLEAN,
        bits=1,
        lanes=lanes,
    )


@pytest.mark.parametrize("operator", ["*", ">>"])
def test_boolean_scalar_arithmetic_uses_promoted_integer_intermediates(operator):
    plan = resolve_arithmetic_conversion(
        _bool_type(),
        _bool_type(),
        operator,
        supported_integer_widths=frozenset({32, 64}),
        supported_floating_widths=frozenset({32, 64}),
    )

    assert plan is not None
    assert plan.left_target_type == "int"
    assert plan.right_target_type == "int"
    assert plan.common_type == "int"
    assert plan.result_type == "int"


@pytest.mark.parametrize("operator", ["*", ">>"])
def test_boolean_vector_arithmetic_preserves_lanes_after_promotion(operator):
    plan = resolve_arithmetic_conversion(
        _bool_type(3),
        _bool_type(),
        operator,
        supported_integer_widths=frozenset({32}),
        supported_floating_widths=frozenset({32}),
    )

    assert plan is not None
    assert plan.left_target_type == "ivec3"
    assert plan.right_target_type == "int"
    assert plan.common_type == "ivec3"
    assert plan.result_type == "ivec3"
