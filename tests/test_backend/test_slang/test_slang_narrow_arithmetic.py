"""Slang input keeps 8- and 16-bit integer arithmetic at the width Slang computes it.

Slang does not promote narrow integer operands (slangc 2026.20 writes ``int8_t * int8_t`` as ``OpIMul %char``), and
CrossGL does, so the importer writes the narrow result explicitly: ``int8_t v = 13; float(v * v)`` is -87 in Slang
and must stay -87 after the round trip. This was the Slang input coverage row ``int8``, which computed 169 on
lavapipe. Each round trip is compiled with slangc when it is installed.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl


def to_crossgl(tmp_path, body, helpers=""):
    source = (
        "RWStructuredBuffer<int> iin;\n"
        "RWStructuredBuffer<float> outp;\n"
        f"{helpers}"
        '[shader("compute")]\n'
        "[numthreads(1, 1, 1)]\n"
        "void main(uint3 id : SV_DispatchThreadID) {\n"
        "    uint i = id.x;\n"
        "    int8_t a = int8_t(iin[i]);\n"
        "    uint8_t b = uint8_t(iin[i + 1]);\n"
        "    int16_t c = int16_t(iin[i + 2]);\n"
        "    uint16_t u = uint16_t(iin[i + 3]);\n"
        "    int k = iin[i + 4];\n"
        f"{body}"
        "}\n"
    )
    path = tmp_path / "probe.slang"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="cgl")


def to_slang(tmp_path, crossgl):
    path = tmp_path / "round.cgl"
    path.write_text(crossgl, encoding="utf-8")
    return crosstl.translate(str(path), backend="slang")


def compile_slang(tmp_path, code):
    slangc = shutil.which("slangc")
    if slangc is None:
        pytest.skip("slangc is not installed")
    source, binary = tmp_path / "out.slang", tmp_path / "out.spv"
    source.write_text(code, encoding="utf-8")
    result = subprocess.run(
        [slangc, str(source), "-target", "spirv", "-o", str(binary)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def squash(code):
    return re.sub(r"\s+", " ", code)


@pytest.mark.parametrize(
    "expression, expected",
    [
        # Both operands narrow: Slang wraps modulo 2^N of the common type.
        ("a * a", "int8_t(a * a)"),
        ("a + a", "int8_t(a + a)"),
        ("a - a", "int8_t(a - a)"),
        ("u * u", "uint16_t(u * u)"),
        ("a * c", "int16_t(a * c)"),
        # Equal width, different signedness: the unsigned type (C++ [expr.arith.conv]).
        ("a * b", "uint8_t(uint8_t(a) * b)"),
        ("a * u", "uint16_t(uint16_t(a) * u)"),
        # A shift keeps its left operand's type.
        ("a << 3", "int8_t(a << 3)"),
        ("a << k", "int8_t(a << k)"),
        # Unary minus wraps; ~ wraps only an unsigned operand.
        ("-a", "int8_t(-a)"),
        ("~b", "uint8_t(~b)"),
        ("~a", "~a"),
        # A nested narrow result is narrow again.
        ("(a * a) + a", "int8_t(int8_t(a * a) + a)"),
        ("a * (a + c)", "int16_t(a * int16_t(a + c))"),
    ],
)
def test_narrow_scalar_results_keep_their_slang_width(tmp_path, expression, expected):
    crossgl = to_crossgl(tmp_path, f"    outp[i] = float({expression});\n")
    assert f"outp[i] = float({expected});" in squash(crossgl)
    compile_slang(tmp_path, to_slang(tmp_path, crossgl))


@pytest.mark.parametrize(
    "expression",
    [
        # An int operand, an integer literal, a uint literal or a float converts the narrow operand instead,
        # which CrossGL's promotion computes the same way (slangc writes OpIMul %int for a * 2).
        "a * 2",
        "a + 1",
        "a * k",
        "a * 2u",
        "a * 2.0",
        "k << a",
        # These stay in the range of the common type.
        "a / a",
        "a % a",
        "a & a",
        "a >> 1",
    ],
)
def test_operations_that_cannot_wrap_keep_their_form(tmp_path, expression):
    crossgl = to_crossgl(tmp_path, f"    outp[i] = float({expression});\n")
    assert f"outp[i] = float({expression});" in squash(crossgl)


@pytest.mark.parametrize(
    "expression, expected",
    [
        # -1 compares, divides and ors as 255 when Slang reads it as uint8_t.
        ("a < b", "uint8_t(a) < b"),
        ("a / b", "uint8_t(a) / b"),
        ("a % b", "uint8_t(a) % b"),
        ("a | b", "uint8_t(a) | b"),
        ("a == u", "uint16_t(a) == u"),
        # int16_t represents every uint8_t value: same result either way.
        ("b < c", "int16_t(b) < c"),
    ],
)
def test_mixed_signedness_operands_convert_to_the_common_type(
    tmp_path, expression, expected
):
    crossgl = to_crossgl(tmp_path, f"    outp[i] = float({expression});\n")
    assert f"outp[i] = float({expected});" in squash(crossgl)
    compile_slang(tmp_path, to_slang(tmp_path, crossgl))


def test_conditional_values_convert_to_their_common_type(tmp_path):
    crossgl = to_crossgl(tmp_path, "    outp[i] = float(k > 0 ? a : b);\n")
    assert "float((k > 0 ? uint8_t(a) : b))" in squash(crossgl)
    compile_slang(tmp_path, to_slang(tmp_path, crossgl))


def test_division_assignment_converts_its_right_operand(tmp_path):
    crossgl = to_crossgl(
        tmp_path,
        "    int8_t d = a;\n"
        "    d /= k;\n"
        "    d %= b;\n"
        "    d /= 2;\n"
        "    d /= 300;\n"
        "    d *= k;\n"
        "    outp[i] = float(d);\n",
    )
    code = squash(crossgl)
    assert "d /= int8_t(k);" in code
    assert "d %= int8_t(b);" in code
    # A literal int8_t represents is the same value either way; 300 is 44 as an int8_t.
    assert "d /= 2;" in code
    assert "d /= int8_t(300);" in code
    # Multiplication is exact modulo 2^8 after the store.
    assert "d *= k;" in code
    compile_slang(tmp_path, to_slang(tmp_path, crossgl))


def test_narrow_values_of_calls_buffers_and_vectors(tmp_path):
    crossgl = to_crossgl(
        tmp_path,
        "    int8_t2 pair = int8_t2(a, a);\n"
        "    outp[i] = float(twice(a) * a);\n"
        "    outp[i + 1] = float(bytes[i] * bytes[i]);\n"
        "    outp[i + 2] = float(pair.x * pair[1]);\n"
        "    outp[i + 3] = float(abs(a) * a);\n"
        "    outp[i + 4] = float(int8_t(k) * a);\n",
        helpers="RWStructuredBuffer<int8_t> bytes;\nint8_t twice(int8_t x) { return x; }\n",
    )
    code = squash(crossgl)
    assert "float(int8_t(twice(a) * a))" in code
    assert "float(int8_t(bytes[i] * bytes[i]))" in code
    assert "float(int8_t(pair.x * pair[1]))" in code
    assert "float(int8_t(abs(a) * a))" in code
    assert "float(int8_t(int8_t(k) * a))" in code


def test_vector_arithmetic_is_not_rewritten(tmp_path):
    crossgl = to_crossgl(
        tmp_path,
        "    int8_t2 pair = int8_t2(a, a);\n"
        "    int8_t2 square = pair * pair;\n"
        "    outp[i] = float(square.x);\n",
    )
    assert "square = pair * pair;" in squash(crossgl)


def test_round_trip_computes_the_slang_product(tmp_path):
    """The Slang output of the imported program promotes and then narrows, which is Slang's -87 for 13 * 13."""
    crossgl = to_crossgl(tmp_path, "    outp[i] = float(a * a);\n")
    slang = to_slang(tmp_path, crossgl)
    assert "float(int8_t(int(a) * int(a)))" in squash(slang)
    compile_slang(tmp_path, slang)
