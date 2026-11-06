"""Builtins applied to members of dependent values in uninstantiated Metal templates.

``y.x`` for a parameter ``const U y`` has a type only per instantiation, so a builtin over it, such as
``::metal::abs(y.x)``, is dependent and is kept in the CrossGL generic body instead of refusing the program.
"""

import pytest

import crosstl
from crosstl.backend.common_ast import (
    ArrayAccessNode,
    BinaryOpNode,
    CastNode,
    FunctionCallNode,
    MemberAccessNode,
    TernaryOpNode,
    VariableNode,
)
from crosstl.backend.Metal.dependent_builtins import (
    NOT_DEPENDENT,
    dependent_access,
    dependent_builtin_result_type,
)

KERNEL = """
kernel void k(device float* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  out[i] = 1.0f;
}
"""


def _to_crossgl(tmp_path, body):
    path = tmp_path / "unit.metal"
    path.write_text(
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "template <typename T, typename U>\n"
        "inline T ratio(const T x, const U y) {\n" + body + "\n}\n" + KERNEL
    )
    return crosstl.translate(
        str(path), backend="cgl", format_output=False, source_backend="metal"
    )


def test_builtin_over_dependent_members_stays_generic(tmp_path):
    code = _to_crossgl(
        tmp_path,
        "  if (::metal::abs(y.x) >= ::metal::abs(y.y)) {\n"
        "    const auto rat = y.y / y.x;\n"
        "    return T(x.x + x.y * rat, x.y);\n"
        "  }\n"
        "  return x;",
    )

    assert "generic<T, U> T ratio(T x, U y)" in code
    assert "if (abs(y.x) >= abs(y.y)) {" in code


def test_builtin_over_dependent_arithmetic_stays_generic(tmp_path):
    code = _to_crossgl(
        tmp_path,
        "  const auto r = ::metal::sqrt(y.x * y.x + y.y * y.y);\n"
        "  return r > 1.0f ? x : T(0);",
    )

    assert "const auto r = sqrt(y.x * y.x + y.y * y.y);" in code


def test_reduction_over_dependent_element_is_bool(tmp_path):
    code = _to_crossgl(
        tmp_path,
        "  bool nan_lanes = ::metal::all(::metal::isnan(y.lanes[0]));\n"
        "  return nan_lanes ? x : T(0);",
    )

    assert "bool nan_lanes = all(isnan(y.lanes[0]));" in code


def _types(**known):
    return lambda node: known.get(getattr(node, "name", None))


DEPENDENT = MemberAccessNode(VariableNode("", "y"), "x")
CONCRETE = MemberAccessNode(VariableNode("", "v"), "x")


@pytest.mark.parametrize(
    "expression, expected",
    [
        (DEPENDENT, True),
        (CONCRETE, False),
        (ArrayAccessNode(MemberAccessNode(VariableNode("", "y"), "lanes"), 0), True),
        (BinaryOpNode(CONCRETE, "*", DEPENDENT), True),
        (FunctionCallNode("f", [CONCRETE, DEPENDENT]), True),
        (TernaryOpNode(DEPENDENT, CONCRETE, CONCRETE), False),
        (CastNode("float", DEPENDENT), False),
        (CastNode("U", CONCRETE), True),
        (VariableNode("", "y"), False),
    ],
)
def test_dependent_access(expression, expected):
    type_of = _types(y="U", v="float2")

    assert dependent_access(expression, type_of, frozenset({"U"})) is expected


def test_dependent_access_without_unbound_names():
    assert not dependent_access(DEPENDENT, _types(y="U"), frozenset())


@pytest.mark.parametrize(
    "result_kind, expected", [("same", None), ("bool_reduction", "bool")]
)
def test_dependent_unknown_result(result_kind, expected):
    assert (
        dependent_builtin_result_type(
            result_kind, [None], frozenset({"U"}), dependent_unknown=True
        )
        == expected
    )


def test_unknown_argument_without_dependence_is_not_dependent():
    assert (
        dependent_builtin_result_type("same", [None], frozenset({"U"})) is NOT_DEPENDENT
    )
