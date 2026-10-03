"""Concrete function-template bodies retain ordinary and call-operator members."""

import pytest

from crosstl.backend.Metal.preprocessor import MetalPreprocessor

CASES = ("member", "both", "transitive", "stateful", "effects")


def source(case="member"):
    fields = "T bias;" if case == "stateful" else ""
    adjustment = " + bias" if case == "stateful" else ""
    initialization = "op.bias = T(5);" if case == "stateful" else ""
    value = "op(value, T(3))" if case in {"both", "stateful"} else "value"
    after = ""
    if case == "effects":
        value = "op(value++, T(3))"
        after = "results[index] += value;"
    helper = ""
    call = "invoke<int, Update<int>>(results, values[index], index + 1);"
    if case == "transitive":
        helper = """
template <typename T, typename Op>
void relay(device T* results, T value, uint index) {
    invoke<T, Op>(results, value, index);
}
"""
        call = "relay<int, Update<int>>(results, values[index], index + 1);"
    return f"""#include <metal_stdlib>
using namespace metal;
template <typename T> struct Update {{
    {fields}
    T operator()(T a, T b) {{ return a + b{adjustment}; }}
    void apply(device T* results, T value, uint index) {{
        results[index] += value{adjustment};
    }}
}};
template <typename T, typename Op>
void invoke(device T* results, T value, uint index) {{
    Op op;
    {initialization}
    op.apply(results, {value}, index);
    {after}
}}
{helper}
kernel void forward(const device int* values [[buffer(0)]],
                    device int* results [[buffer(1)]],
                    uint index [[thread_position_in_grid]]) {{
    {call}
}}
"""


@pytest.mark.parametrize("case", CASES)
def test_preprocessor_rewrites_members_in_concrete_helpers(case):
    result = MetalPreprocessor().preprocess(source(case))
    definition = result[result.index("inline void invoke_") :]
    assert "op.apply(" not in definition
    assert "Update_int__apply(op," in definition
    assert "results[index] += value" in result
    if case in {"both", "stateful", "effects"}:
        assert "Update_int__operator_call(op," in definition
    if case == "effects":
        assert definition.count("value++") == 1
    if case == "stateful":
        assert "self.bias" in result
