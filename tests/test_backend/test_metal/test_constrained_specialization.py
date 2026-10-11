"""Constrained primaries and explicit bodies retain their distinct identities."""

import pytest

from crosstl.backend.Metal.preprocessor import (
    MetalPreprocessor,
    MetalTemplateSpecializationError,
)

CASES = (
    "integer",
    "float",
    "alias",
    "deduced",
    "empty",
    "explicit-integer",
    "explicit-float",
    "default",
    "nondefault",
    "explicit-default",
    "default-deduced",
    "nondefault-deduced",
    "comparison-template",
    "shift-template",
)


def source(case="integer"):
    kind = "int" if case in {"integer", "explicit-integer"} else "float"
    specialization_type = "Scalar" if case == "alias" else "float"
    arguments = {"deduced": "", "empty": "<>"}.get(case, f"<{specialization_type}>")
    if case.endswith("-deduced"):
        arguments = ""
    call_arguments = f"<{kind}>" if case.startswith("explicit-") else ""
    default = ", int Bias = 0" if "default" in case else ""
    if case.startswith("nondefault"):
        call_arguments = "<float, 1>"
    elif case == "explicit-default":
        call_arguments = "<float, 0>"
    operator = {"comparison-template": "<", "shift-template": "<<"}.get(case)
    dependent_operator = ""
    if operator is not None:
        dependent_operator = f"""
template <typename T, T Value>
struct StaticValue {{ static constexpr constant T value = Value; }};
template <typename T, T Left, typename U, U Right>
constexpr auto operator{operator}(StaticValue<T, Left>, StaticValue<U, Right>) {{
    constexpr auto result = Left {operator} Right;
    return StaticValue<decltype(result), result>{{}};
}}
"""
    return f"""#include <metal_stdlib>
using namespace metal;
using Scalar = float;
{dependent_operator}
template <typename T{default},
          enable_if_t<is_same_v<T, int> || is_same_v<T, float>, bool> = true>
T update_value(device T* values, T value, uint index) {{
    return values[index] + value;
}}
template <>
{specialization_type} update_value{arguments}(
    device {specialization_type}* values, {specialization_type} value, uint index) {{
    return values[index] - value;
}}
kernel void select_body(device {kind}* results [[buffer(0)]],
                        device {kind}* values [[buffer(1)]],
                        uint index [[thread_position_in_grid]]) {{
    results[index + 1u] = update_value{call_arguments}(values, {kind}(7), index);
}}
"""


@pytest.mark.parametrize("case", CASES)
def test_constrained_call_selects_explicit_body_only_for_matching_type(case):
    output = MetalPreprocessor().preprocess(source(case))
    kind = "int" if case in {"integer", "explicit-integer"} else "float"
    helper = f"update_value_{kind}"
    if "default" in case:
        helper += "_1" if case.startswith("nondefault") else "_0"
    assert f"= {helper}(values, {kind}(7), index);" in output
    assert f"{kind} {helper}(" in output
    body = output[output.index(f"{kind} {helper}(") :]
    operator = "+" if kind == "int" or case.startswith("nondefault") else "-"
    assert f"return values[index] {operator} value;" in body
    assert helper in MetalPreprocessor().preprocess(source(case))


@pytest.mark.parametrize(
    "competitor",
    (
        "int update_value(device int* values, int value, uint index);",
        "template <typename T> T update_value(device T* values, T value, uint index);",
        "template <> int update_value<int>(device int* values, int value, uint index);",
    ),
)
def test_explicit_body_does_not_suppress_unproven_competitor(competitor):
    text = source().replace(
        "kernel void select_body", competitor + "\nkernel void select_body"
    )
    with pytest.raises(
        MetalTemplateSpecializationError,
        match="competing free-function overload has unproven precedence",
    ):
        MetalPreprocessor().preprocess(text)


def test_global_explicit_body_does_not_replace_namespaced_primary():
    text = (
        source("float")
        .replace(
            "kernel void select_body",
            """namespace nested {
template <typename T, enable_if_t<is_same_v<T, float>, bool> = true>
T update_value(device T* values, T value, uint index) {
    return values[index] * value;
}
}
kernel void select_body""",
        )
        .replace("= update_value(values", "= nested::update_value(values")
    )
    output = MetalPreprocessor().preprocess(text)
    assert "return values[index] * value;" in output
    assert "= update_value_float(values" in output


def test_equivalent_explicit_specializations_are_diagnosed():
    text = source("float").replace(
        "kernel void select_body",
        """template <>
Scalar update_value<Scalar>(device Scalar* values, Scalar value, uint index) {
    return values[index] * value;
}
kernel void select_body""",
    )
    with pytest.raises(
        MetalTemplateSpecializationError, match="more than one explicit body"
    ):
        MetalPreprocessor().preprocess(text)


def test_constrained_specialization_uses_concrete_parameter_signature():
    text = """
template <typename T, metal::enable_if_t<metal::is_integral_v<T>, bool> = true>
T choose(T value) { return value + T(1); }
template <typename T, metal::enable_if_t<metal::is_integral_v<T>, bool> = true>
T choose(device T* value, int index) { return value[index] + T(9); }
template <> int choose<int>(device int* value, int index) { return value[index] + 99; }
kernel void run(device int* out [[buffer(0)]]) { out[0] = choose(7); }
"""
    output = MetalPreprocessor().preprocess(text)
    assert "int choose_int(int value)" in output
    assert "return value + int(1);" in output
