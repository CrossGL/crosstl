import pytest

from tests.test_translator.metal_contract import operator_implementations


@pytest.mark.parametrize("qualifiers", ["", "inline ", "static ", "static inline "])
@pytest.mark.parametrize("result_type", ["float", "bfloat", "complex_t_float", "bool"])
def test_operator_implementations_retain_private_helpers(qualifiers, result_type):
    source = (
        "__attribute__((unused))\n"
        f"{qualifiers}{result_type} Add__operator_call__float(thread Add& self, float x, float y) {{\n"
        "    return x + y;\n"
        "}\n\n"
        "__attribute__((unused))\n"
        f"{qualifiers}{result_type} Add__operator_call__float__temporary(float x, float y) {{\n"
        "    Add self;\n"
        "    return Add__operator_call__float(self, x, y);\n"
        "}\n"
    )
    names = operator_implementations(source, "Add")
    assert names == (
        "Add__operator_call__float",
        "Add__operator_call__float__temporary",
    )
    assert len([name for name in names if not name.endswith("__temporary")]) == 1
    assert operator_implementations(source, "Multiply") == ()


def test_operator_implementations_detect_extra_bodies():
    source = (
        "static inline float Add__operator_call(float x, float y) { return x + y; }\n"
        "inline float Multiply__operator_call(float x, float y) { return x * y; }\n"
        "static float Multiply__operator_call__temporary(float x, float y) { return x * y; }\n"
        "static inline float ExtraMultiply__operator_call(float x, float y) { return x * y; }\n"
        "float Multiply__operator_call_extra(float x, float y) { return x * y; }\n"
        "kernel void main() {\n"
        "    Multiply__operator_call(1.0f, 2.0f);\n"
        "}\n"
    )
    assert operator_implementations(source, "Multiply") == (
        "Multiply__operator_call",
        "Multiply__operator_call__temporary",
    )
