"""GLSL import writes the narrow result of GLSL's 8- and 16-bit scalar arithmetic explicitly.

GL_EXT_shader_explicit_arithmetic_types computes ``int8_t`` and ``int16_t`` arithmetic at the operands' width, while
CrossGL promotes narrow scalars to ``int`` first. ``int8_t v = 13; float(v * v)`` is ``-87.0`` in GLSL, so the
imported CrossGL must convert the product back to ``int8_t`` or it computes ``169.0``.
"""

import pytest

import crosstl
from crosstl.backend.GLSL.narrow_arithmetic import GlslNarrowArithmeticError

HEADER = """#version 460
#extension GL_EXT_shader_explicit_arithmetic_types : require
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer B0 { int iin[]; };
layout(std430, binding = 1) buffer B1 { float outp[]; };
"""


def _import(tmp_path, body, helpers=""):
    path = tmp_path / "shader.comp"
    path.write_text(
        HEADER
        + helpers
        + "void main() {\n    uint i = gl_GlobalInvocationID.x;\n"
        + body
        + "\n}\n"
    )
    return crosstl.translate(str(path), backend="cgl", format_output=False)


def _lines(code):
    """Stripped lines without the instance prefixes of the imported B0 and B1 blocks."""
    return [
        line.strip().replace("b0.", "").replace("b1.", "") for line in code.splitlines()
    ]


@pytest.mark.parametrize(
    ("declaration", "expression", "expected"),
    [
        ("int8_t", "x * y", "int8_t((uint(x) * uint(y)))"),
        ("uint8_t", "x + y", "uint8_t((uint(x) + uint(y)))"),
        ("int16_t", "x - y", "int16_t((uint(x) - uint(y)))"),
        ("uint16_t", "x << y", "uint16_t((uint(x) << y))"),
        ("int16_t", "x / y", "int16_t((x / y))"),
        ("int8_t", "-x", "int8_t((-x))"),
        ("uint16_t", "~x", "uint16_t((~x))"),
    ],
)
def test_range_changing_operations_convert_back(
    tmp_path, declaration, expression, expected
):
    body = (
        f"    {declaration} x = {declaration}(iin[i]);\n"
        f"    {declaration} y = {declaration}(iin[i + 1u]);\n"
        f"    outp[i] = float({expression});"
    )
    code = _import(tmp_path, body)

    assert f"outp[i] = float({expected});" in _lines(code)


@pytest.mark.parametrize(
    ("declaration", "expression"),
    [
        ("uint16_t", "x / y"),
        ("int16_t", "x % y"),
        ("uint8_t", "x >> y"),
        ("int8_t", "x & y"),
        ("uint8_t", "x | y"),
        ("int16_t", "x ^ y"),
        ("int16_t", "~x"),
    ],
)
def test_in_range_operations_keep_their_form(tmp_path, declaration, expression):
    body = (
        f"    {declaration} x = {declaration}(iin[i]);\n"
        f"    {declaration} y = {declaration}(iin[i + 1u]);\n"
        f"    outp[i] = float({expression});"
    )
    code = _import(tmp_path, body)

    assert f"outp[i] = float(({expression}));" in _lines(code)


def test_nested_operations_convert_each_wrapping_result(tmp_path):
    body = "    int8_t v = int8_t(iin[i]);\n" "    outp[i] = float((v & v) * v + v);"
    code = _import(tmp_path, body)

    assert (
        "outp[i] = float(int8_t((uint(int8_t((uint((v & v)) * uint(v)))) + uint(v))));"
        in _lines(code)
    )


def test_stores_of_the_same_type_are_left_to_the_store(tmp_path):
    helpers = "int16_t square(int16_t s) { return s * s; }\n"
    body = (
        "    int8_t v = int8_t(iin[i]);\n"
        "    int8_t p = v * v;\n"
        "    p = p + v;\n"
        "    outp[i] = float(int8_t(v * v)) + float(p) + float(square(int16_t(v)));"
    )
    lines = _lines(_import(tmp_path, body, helpers))

    assert "int8_t p = (v * v);" in lines
    assert "p = (p + v);" in lines
    assert "return (s * s);" in lines
    assert any("float(int8_t((v * v)))" in line for line in lines)


def test_a_store_into_another_type_still_converts(tmp_path):
    body = (
        "    int8_t v = int8_t(iin[i]);\n"
        "    int wide = v * v;\n"
        "    int16_t middle = int16_t(v * v);\n"
        "    outp[i] = float(wide + int(middle));"
    )
    lines = _lines(_import(tmp_path, body))

    assert "int wide = int8_t((uint(v) * uint(v)));" in lines
    assert "int16_t middle = int16_t(int8_t((uint(v) * uint(v))));" in lines


def test_a_32_bit_operand_promotes_without_conversion(tmp_path):
    body = (
        "    int8_t v = int8_t(iin[i]);\n"
        "    int k = iin[i + 1u];\n"
        "    outp[i] = float(v * k) + float(v * 2);"
    )
    lines = _lines(_import(tmp_path, body))

    assert "outp[i] = (float((v * k)) + float((v * 2)));" in lines


def test_short_literals_and_specialization_constants_are_narrow(tmp_path):
    helpers = "layout(constant_id = 0) const int16_t SCALE = 300s;\n"
    body = (
        "    int16_t v = int16_t(iin[i]);\n"
        "    outp[i] = float(v * 2s) + float(v * SCALE);"
    )
    lines = _lines(_import(tmp_path, body, helpers))

    assert (
        "outp[i] = (float(int16_t((uint(v) * uint(2)))) + "
        "float(int16_t((uint(v) * uint(SCALE)))));"
    ) in lines


def test_narrow_vectors_keep_their_form(tmp_path):
    body = (
        "    u16vec2 v = u16vec2(uint16_t(iin[i]), uint16_t(iin[i + 1u]));\n"
        "    outp[i] = float((v * v).x);"
    )
    lines = _lines(_import(tmp_path, body))

    assert "outp[i] = float((v * v).x);" in lines


def test_mixed_narrow_types_are_refused(tmp_path):
    body = (
        "    int8_t a = int8_t(iin[i]);\n"
        "    int16_t b = int16_t(iin[i + 1u]);\n"
        "    outp[i] = float(a * b);"
    )
    with pytest.raises(GlslNarrowArithmeticError, match="mixes int8_t and int16_t"):
        _import(tmp_path, body)


def test_direct_and_round_trip_translations_wrap_the_product(tmp_path):
    body = "    int8_t v = int8_t(iin[i]);\n    outp[i] = float(v * v);"
    crossgl = _import(tmp_path, body)
    crossgl_path = tmp_path / "imported.cgl"
    crossgl_path.write_text(crossgl)

    round_trip = crosstl.translate(
        str(crossgl_path), backend="opengl", format_output=False
    )
    direct = crosstl.translate(
        str(tmp_path / "shader.comp"), backend="opengl", format_output=False
    )

    wrapped = "outp[i] = float(bitfieldExtract(int((uint(v) * uint(v))), 0, 8));"
    assert wrapped in _lines(round_trip)
    assert wrapped in _lines(direct)
