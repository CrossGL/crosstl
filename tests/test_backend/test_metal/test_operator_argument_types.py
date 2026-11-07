"""Metal import types conditional and unary-sign call arguments for template deduction.

A constrained or deduced call such as ``cast_to<float>(m ? x * scale : 0)`` or ``max(0, -pad)`` used to be refused
because neither shape had an argument-type rule. The conditional takes the common type of its arms (C++17
[expr.cond]/7) and a unary sign the promoted type of its operand (C++17 [expr.unary.op]/7-8).
"""

import re

import pytest

from crosstl.backend.Metal.operator_argument_types import _has_top_level_assignment
from crosstl.backend.Metal.preprocessor import (
    MetalPreprocessor,
    MetalTemplateSpecializationError,
)

LOCALS = (
    "bool c = i > 2u; float f = 1.0f; half h = 1.0h; short s = 1; short t = 2; "
    "char ch = 3; uint u = 4u; int n = 5; float4 v = float4(1.0f); "
    "float4 w = float4(2.0f); bfloat b = 1.0bf; Pair p = {1.0f, 2.0f};"
)

PROGRAM = """
#include <metal_stdlib>
using namespace metal;
struct Pair {{ float a; float b; }};
Pair operator-(Pair x) {{ return Pair{{-x.a, -x.b}}; }}
template <typename U, enable_if_t<is_floating_point_v<U>, bool> = true>
float pick(U x) {{ return float(x); }}
template <typename U, enable_if_t<!is_floating_point_v<U>, bool> = true>
float pick(U x) {{ return float(x) * 2.0f; }}
kernel void k(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) {{
    {locals}
    o[i] = pick({argument});
}}
"""


def deduced(argument):
    """The argument type the call ``pick(argument)`` is specialized for."""
    output = MetalPreprocessor().preprocess(
        PROGRAM.format(locals=LOCALS, argument=argument)
    )
    names = set(re.findall(r"\bpick_(\w+)\(", output))
    assert len(names) == 1, names
    return names.pop()


@pytest.mark.parametrize(
    ("argument", "expected"),
    [
        ("c ? f : 0", "float"),
        ("c ? 1u : 0u", "uint"),
        ("c ? h : 1.0f", "float"),
        ("c ? h : n", "half"),
        ("c ? s : t", "short"),
        ("c ? s : ch", "int"),
        ("c ? u : 1", "uint"),
        ("c ? f : n", "float"),
        ("c ? v : w", "float4"),
        ("(c ? (c ? 1 : 2) : 3)", "int"),
        ("u < 3u ? f * 2.0f : 0", "float"),
    ],
)
def test_conditional_argument_has_the_common_arm_type(argument, expected):
    assert deduced(argument) == expected


@pytest.mark.parametrize(
    ("argument", "expected"),
    [
        ("-n", "int"),
        ("-s", "int"),
        ("-c", "int"),
        ("-u", "uint"),
        ("+h", "half"),
        ("-v", "float4"),
        ("- -f", "float"),
        ("-n * 2", "int"),
    ],
)
def test_unary_sign_argument_has_the_promoted_operand_type(argument, expected):
    assert deduced(argument) == expected


@pytest.mark.parametrize(
    ("argument", "floating"),
    [("c ? f : 0", True), ("c ? 1u : 0u", False), ("-n", False), ("-h", True)],
)
def test_operator_arguments_select_the_constrained_overload(argument, floating):
    output = MetalPreprocessor().preprocess(
        PROGRAM.format(locals=LOCALS, argument=argument)
    )
    assert f"({argument});" in output
    assert ("return float(x) * 2.0f;" in output) is not floating


@pytest.mark.parametrize(
    "argument",
    [
        "c ? h : b",  # half and bfloat have equal rank and no common type
        "c ? v : 0.0f",  # a vector arm and a scalar arm
        "c ? p : f",  # a struct arm and a scalar arm
        "-p",  # a user-defined operator- chooses its own result type
        "c ? f : undeclared",
    ],
)
def test_untyped_operator_arguments_are_refused(argument):
    with pytest.raises(MetalTemplateSpecializationError, match="inferred"):
        deduced(argument)


@pytest.mark.parametrize(
    ("text", "assigns"),
    [
        ("x = c", True),
        ("x += 1", True),
        ("x <<= 1", True),
        ("x >>= 1", True),
        ("x == c", False),
        ("x != c", False),
        ("x <= c", False),
        ("x >= c", False),
        ("f(x = 1)", False),
        ("= x", True),
    ],
)
def test_top_level_assignment_detection(text, assigns):
    assert _has_top_level_assignment(text) is assigns
