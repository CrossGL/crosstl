"""Bfloat narrowing retains value precision with widened OpenGL storage."""

import json
import math
import os
import random
import re
import struct
import sys
from pathlib import Path

import pytest

from crosstl import translate
from crosstl.translator.ast import FunctionCallNode
from crosstl.translator.codegen.directx_codegen import (
    DirectXBFloat16UnsupportedError,
    HLSLCodeGen,
)
from crosstl.translator.codegen.GLSL_codegen import (
    GLSLCodeGen,
    OpenGLCompoundAssignmentError,
    OpenGLScalarConversionError,
)
from tests.test_translator.test_boolean_buffer_runtime import _bound_values, _request
from tests.test_translator.test_half_buffer_runtime import _validate_half
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_opengl_half_conversion import _compile, _shader
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_BFLOAT_CONVERSION_RUNTIME"
TARGET = {"darwin": "metal", "linux": "opengl", "win32": "directx"}.get(sys.platform)
CASES = {
    "constructor": "results[i + 1u] = float(bfloat(values[i]));",
    "static-cast": "results[i + 1u] = float(static_cast<bfloat>(values[i]));",
    "c-cast": "results[i + 1u] = float((bfloat)values[i]);",
    "assignment": (
        "bfloat value = bfloat(0.0); value = bfloat(values[i]); results[i + 1u] = float(value);"
    ),
    "return": "results[i + 1u] = float(narrow(values[i]));",
    "argument": "results[i + 1u] = widen(bfloat(values[i]));",
    "struct": (
        "Cell cell; cell.value = bfloat(values[i]); results[i + 1u] = float(cell.value);"
    ),
    "vector2": (
        "bfloat2 value = bfloat2(bfloat(values[i])); results[i + 1u] = float(value.x);"
    ),
    "vector4": (
        "bfloat4 value = bfloat4(bfloat(values[i])); results[i + 1u] = float(value.w);"
    ),
    "component": (
        "bfloat2 value = bfloat2(bfloat(0.0)); value[1] = bfloat(values[i]); results[i + 1u] = float(value[1]);"
    ),
    "swizzle": (
        "bfloat2 value = bfloat2(bfloat(0.0)); value.yx = bfloat2(bfloat(values[i])); results[i + 1u] = float(value.x);"
    ),
    "side-effect": (
        "uint index = i; bfloat value = bfloat(values[index++]); results[i + 1u] = index == i + 1u ? float(value) : 42.0;"
    ),
    "nested": "results[i + 1u] = float(bfloat(bfloat(values[i])));",
}
SCALAR_CASES = tuple(
    name
    for name in CASES
    if name not in {"struct", "vector2", "vector4", "component", "swizzle"}
)
WORDS = (
    0,
    0x80000000,
    1,
    0x80000001,
    0x8000,
    0x18000,
    0x7FFFFF,
    0x800000,
    0x3F800000,
    0x3F807FFF,
    0x3F808000,
    0x3F808001,
    0x3F818000,
    0xBF808000,
    0xBF818000,
    0x7F7FFFFF,
    0xFF7FFFFF,
    0x7F800000,
    0xFF800000,
    0x3EAAAAAB,
)


def _round(word):
    if word & 0x7FFFFFFF > 0x7F800000:
        return (word & 0xFFFF0000) | 0x00400000
    return (word + 0x7FFF + ((word >> 16) & 1)) & 0xFFFF0000


def _source(body, declarations=""):
    return f"""#include <metal_stdlib>
using namespace metal;
{declarations}
struct Cell {{ bfloat value; }};
bfloat narrow(float value) {{ return bfloat(value); }}
float widen(bfloat value) {{ return float(value); }}
kernel void bfloat_conversion(const device float* values [[buffer(0)]],
                              device float* results [[buffer(1)]],
                              uint i [[thread_position_in_grid]]) {{
    {body}
}}
"""


def _case(
    root, target, name, words=WORDS, *, body=None, expected_words=None, declarations=""
):
    root.mkdir(parents=True, exist_ok=True)
    source = _source(CASES[name] if body is None else body, declarations)
    _, descriptor, package = _package(
        root, target, "uint", (1, 1, 1), source=source, software_subgroups=False
    )

    def value(words):
        return {
            "dtype": "float32",
            "encoding": "ieee754-binary32",
            "values": list(words),
            "shape": [len(words)],
        }

    guard = 0x422A0000
    inputs = {"values": value(words), "results": value([guard] * (len(words) + 2))}
    result = [_round(w) for w in words] if expected_words is None else expected_words
    outputs = {"results": value([guard, *result, guard])}
    request = _request(descriptor, package, inputs, outputs, len(words))
    return source, request, _bound_values(descriptor, outputs)


@pytest.mark.parametrize("case", tuple(CASES))
def test_bfloat_narrowing_forms_compile(tmp_path, case):
    source, request, _ = _case(tmp_path, "opengl", case)
    generated = request.artifact_path.read_text()
    assert "crossgl_round_bfloat1" in generated
    assert re.search(r"\bbfloat16vec[234]\b", generated) is None
    _compile(generated, tmp_path)


@pytest.mark.parametrize(
    "dtype", ("bfloat", "bfloat16", "bfloat16_t", "bfloat2", "bfloat16vec3", "bfloat4")
)
def test_bfloat_constructors_and_initializers(tmp_path, dtype):
    width = int(dtype[-1]) if dtype[-1] in "234" else 1
    mapped = "float" if width == 1 else f"vec{width}"
    generated = GLSLCodeGen().generate(
        _shader(
            f"{dtype} x = {mapped}(output[0]); output[1] = float(x{'[0]' if width > 1 else ''});"
        )
    )
    assert f"crossgl_round_bfloat{width}(" in generated
    _compile(generated, tmp_path)


def test_bfloat_helper_collision_and_reuse(tmp_path):
    generator = GLSLCodeGen()
    first = generator.generate(
        _shader(
            "float crossgl_round_bfloat1 = output[0]; bfloat x = crossgl_round_bfloat1;"
        )
    )
    assert "float crossgl_round_bfloat1_2(float value)" in first
    _compile(first, tmp_path)
    assert "crossgl_round_bfloat" not in generator.generate(_shader("output[0] = 1.0;"))


def test_bfloat_integer_helper_collision_single_evaluation_and_reuse(tmp_path):
    generator = GLSLCodeGen()
    generated = generator.generate(
        _shader(
            "uint value = uint(output[0]); float crossgl_integer_to_bfloat = 0.0; "
            "bfloat result = bfloat(value++); output[0] = float(result);"
        )
    )
    assert "float crossgl_integer_to_bfloat_2(uint value)" in generated
    assert generated.count("value++") == 1
    _compile(generated, tmp_path)
    assert "crossgl_integer_to_bfloat" not in generator.generate(
        _shader("output[0] = 1.0;")
    )


def test_bfloat_integer_shadowed_builtin_is_rejected():
    with pytest.raises(OpenGLScalarConversionError) as error:
        GLSLCodeGen().generate(
            _shader(
                "bfloat value = int(output[0]);", "int findMSB(int x) { return x; }"
            )
        )
    assert error.value.reason == "bfloat-target-builtin-shadowed"


def test_canonical_bfloat_assignment_argument_and_return_conversion(tmp_path):
    generated = GLSLCodeGen().generate(
        _shader(
            "bfloat value = output[0]; value = output[1]; output[0] = widen(output[0]); output[1] = float(narrow(output[1]));",
            "bfloat narrow(float x) { return x; } float widen(bfloat x) { return float(x); }",
        )
    )
    assert "return crossgl_round_bfloat1(x);" in generated
    assert "widen(crossgl_round_bfloat1(output_[0]))" in generated
    _compile(generated, tmp_path)


@pytest.mark.parametrize("builtin", ("floatBitsToUint", "uintBitsToFloat"))
def test_bfloat_shadowed_target_builtins_are_rejected(builtin):
    with pytest.raises(OpenGLScalarConversionError) as error:
        GLSLCodeGen().generate(
            _shader(
                "bfloat value = output[0];", f"float {builtin}(float x) {{ return x; }}"
            )
        )
    assert error.value.reason == "bfloat-target-builtin-shadowed"


def test_bfloat_arithmetic_rounds_before_widening(tmp_path):
    generated = GLSLCodeGen().generate(_shader("""
        bfloat x = bfloat(output[0]);
        bfloat y = bfloat(output[1]);
        output[0] = float(x + y);
        x += y;
        ++x;
        output[1] = float(x);
    """))
    assert "crossgl_round_bfloat1((x + y))" in generated
    assert "x = crossgl_round_bfloat1(" in generated
    _compile(generated, tmp_path)


@pytest.mark.parametrize("left", ("bfloat", "bfloat16", "bfloat16_t"))
@pytest.mark.parametrize("right", ("bfloat", "bfloat16", "bfloat16_t"))
def test_bfloat_aliases_retain_arithmetic_precision(tmp_path, left, right):
    generated = GLSLCodeGen().generate(
        _shader(
            f"{left} x = {left}(output[0]); {right} y = {right}(output[1]); "
            f"output[0] = float((x + {right}(0.00390625)) * y);"
        )
    )
    assert (
        "crossgl_round_bfloat1((crossgl_round_bfloat1((x + 0.00390625)) * y))"
        in generated
    )
    _compile(generated, tmp_path)


@pytest.mark.parametrize(
    "left,right,expected",
    (
        ("bfloat16vec2", "bfloat2", "bfloat16vec2"),
        ("bfloat16", "bfloat4", "bfloat4"),
        ("bfloat4", "bfloat16_t", "bfloat4"),
        ("bfloat", "float", "float"),
        ("float", "bfloat16_t", "float"),
    ),
)
def test_bfloat_common_arithmetic_type(left, right, expected):
    assert GLSLCodeGen().glsl_common_arithmetic_type(left, right, "+") == expected


@pytest.mark.parametrize("target", ("opengl", "directx"))
@pytest.mark.parametrize("dtype", ("int", "uint"))
@pytest.mark.parametrize("operator", ("+", "-", "*", "/"))
@pytest.mark.parametrize("operand", ("x", "bfloat(x)"))
def test_integer_bfloat_compound_rounds_operands_and_result(
    tmp_path, target, dtype, operator, operand
):
    generator = GLSLCodeGen() if target == "opengl" else HLSLCodeGen()
    generated = generator.generate(
        _shader(
            f"bfloat16_t x = bfloat16_t(output[0]); {dtype} y = {dtype}(257); "
            f"y {operator}= {operand}; output[0] = float(y);"
        )
    )
    assignment = next(
        line for line in generated.splitlines() if line.strip().startswith("y = ")
    )
    if target == "opengl":
        assert "crossgl_integer_to_bfloat" in assignment
        assert "crossgl_round_bfloat1" in assignment
        _compile(generated, tmp_path)
    else:
        assert f"__crossgl_bfloat16_from_{dtype}" in assignment
        assert "__crossgl_bfloat16_from_float" in assignment
        assert "__crossgl_bfloat16_to_float" in assignment


@pytest.mark.parametrize("target", ("opengl", "directx"))
@pytest.mark.parametrize("side_effect", ("index", "value"))
def test_integer_bfloat_compound_diagnoses_unsafe_evaluation(target, side_effect):
    generator = GLSLCodeGen() if target == "opengl" else HLSLCodeGen()
    error_type = (
        OpenGLCompoundAssignmentError
        if target == "opengl"
        else DirectXBFloat16UnsupportedError
    )
    body = (
        "int cells[2]; cells[0] = 257; cells[1] = 259; uint index = 0u; "
        "bfloat x = bfloat(output[0]); "
    )
    body += (
        "cells[index++] += x;"
        if side_effect == "index"
        else "cells[index] += change(index, x);"
    )
    with pytest.raises(error_type) as error:
        generator.generate(
            _shader(
                body, "bfloat change(inout uint index, bfloat x) { index++; return x; }"
            )
        )
    assert error.value.reason == (
        "rhs-may-change-assignment-target"
        if side_effect == "value"
        else (
            "lvalue-side-effects"
            if target == "opengl"
            else "side-effecting-assignment-target"
        )
    )


@pytest.mark.parametrize("target", ("opengl", "directx"))
def test_integer_bfloat_compound_rejects_unproven_wide_conversion(target):
    generator = GLSLCodeGen() if target == "opengl" else HLSLCodeGen()
    error_type = (
        OpenGLScalarConversionError
        if target == "opengl"
        else DirectXBFloat16UnsupportedError
    )
    with pytest.raises(error_type) as error:
        generator.generate(
            _shader(
                "bfloat x = bfloat(output[0]); int64_t y = int64_t(257); y += x; output[0] = float(y);"
            )
        )
    assert error.value.reason == (
        "bfloat-double-rounding"
        if target == "opengl"
        else "unsupported-bfloat16-compound-integer-width"
    )


@pytest.mark.parametrize("dtype,initial", (("int", 257), ("uint", 257), ("int", -257)))
@pytest.mark.parametrize("operator", ("+", "-", "*", "/"))
@pytest.mark.parametrize("lvalue", ("local", "member", "array", "resource"))
def test_integer_bfloat_compound_executes_natively(
    tmp_path, dtype, initial, operator, lvalue
):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required bfloat compound assignments")
    values = [
        0.25,
        0.5,
        0.99609375,
        1.0,
        1.0078125,
        1.5,
        2.0,
        3.0,
        63.0,
        127.5,
        128.0,
        255.0,
    ]

    def bits(value):
        return struct.unpack("<I", struct.pack("<f", value))[0]

    def narrow(value):
        return struct.unpack("<f", struct.pack("<I", _round(bits(value))))[0]

    def reference(value):
        x = narrow(value)
        left = narrow(initial)
        computed = {
            "+": lambda: left + x,
            "-": lambda: left - x,
            "*": lambda: left * x,
            "/": lambda: left / x,
        }[operator]()
        return bits(float(int(narrow(computed))))

    declaration, target = {
        "local": (f"{dtype} y = {initial};", "y"),
        "member": (f"IntegerCell cell; cell.value = {initial};", "cell.value"),
        "array": (
            f"{dtype} cells[2]; cells[0] = {initial}; cells[1] = 259; uint index = 0u;",
            "cells[index]",
        ),
        "resource": ("", "cells[i + 1u]"),
    }[lvalue]
    body = f"narrow_t x = narrow_t(values[i]); {declaration} {target} {operator}= x; "
    body += f"results[i + 1u] = float({target});"
    declarations = f"using narrow_t = bfloat; struct IntegerCell {{ {dtype} value; }};"
    if lvalue == "resource":
        source = _source(body, declarations).replace(
            "uint i [[thread_position_in_grid]]",
            f"device {dtype}* cells [[buffer(2)]], uint i [[thread_position_in_grid]]",
        )
        _, descriptor, package = _package(
            tmp_path, TARGET, "uint", (1, 1, 1), source=source, software_subgroups=False
        )

        def floats(words):
            return {
                "dtype": "float32",
                "encoding": "ieee754-binary32",
                "values": words,
                "shape": [len(words)],
            }

        def integers(numbers):
            return {
                "dtype": "int32" if dtype == "int" else "uint32",
                "values": numbers,
                "shape": [len(numbers)],
            }

        expected_numbers = [
            int(struct.unpack("<f", struct.pack("<I", reference(v)))[0]) for v in values
        ]
        inputs = {
            "values": floats([bits(v) for v in values]),
            "results": floats([bits(42.0)] * (len(values) + 2)),
            "cells": integers([42, *([initial] * len(values)), 42]),
        }
        outputs = {
            "results": floats(
                [bits(42.0), *[reference(v) for v in values], bits(42.0)]
            ),
            "cells": integers([42, *expected_numbers, 42]),
        }
        request = _request(descriptor, package, inputs, outputs, len(values))
        expected = _bound_values(descriptor, outputs)
    else:
        source, request, expected = _case(
            tmp_path,
            TARGET,
            "integer-compound",
            [bits(v) for v in values],
            body=body,
            expected_words=[reference(v) for v in values],
            declarations=declarations,
        )
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="bfloat_conversion",
        validate=_validate_half,
    )


def test_bfloat_constants_are_rounded_before_global_initialization(tmp_path):
    generated = GLSLCodeGen().generate(
        _shader(
            "output[0] = float(value); bfloat zero = bfloat(0);",
            "const bfloat value = 1.00390625;",
        )
    )
    assert "const float value = 1.0;" in generated
    assert "float zero = 0.0;" in generated
    _compile(generated, tmp_path)


@pytest.mark.parametrize(
    "body",
    (
        "bfloat value = double(output[0]);",
        "output[0] = float(bfloat(double(output[0])));",
        "bfloat value = int64_t(int(output[0]));",
    ),
)
def test_bfloat_unproven_double_rounding_is_rejected(body):
    with pytest.raises(OpenGLScalarConversionError) as error:
        GLSLCodeGen().generate(_shader(body))
    assert error.value.reason == "bfloat-double-rounding"


@pytest.mark.parametrize("version", ("#version 330 core", "#version 300 es"))
def test_bfloat_unproven_profiles_are_rejected(version):
    generator = GLSLCodeGen()
    generator.current_glsl_version_line = version
    ast = _shader("output[0] = bfloat(1.0);")
    call = next(node for node in ast.walk() if isinstance(node, FunctionCallNode))
    with pytest.raises(OpenGLScalarConversionError) as error:
        generator.generate_expression(call)
    assert error.value.reason == "bfloat-unsupported-profile"


@pytest.mark.parametrize("case", SCALAR_CASES if TARGET == "directx" else tuple(CASES))
def test_bfloat_conversions_execute_natively(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required bfloat conversions")
    source, request, expected = _case(tmp_path, TARGET, case)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="bfloat_conversion",
        validate=_validate_half,
    )


@pytest.mark.parametrize("start", range(0, 65536, 4096))
def test_bfloat_boundary_rounding_executes_natively(tmp_path, start):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required bfloat conversions")
    words = [
        high << 16 | low
        for high in range(start, start + 4096)
        if high & 0x7F80 != 0x7F80
        for low in (0x7FFF, 0x8000, 0x8001)
    ]
    source, request, expected = _case(tmp_path, TARGET, "constructor", words)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="bfloat_conversion",
        validate=_validate_half,
    )


@pytest.mark.parametrize("operation", ("+", "-", "*", "/", "compound", "chain"))
def test_bfloat_arithmetic_executes_natively(tmp_path, operation):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required bfloat arithmetic")

    def bits(value):
        return struct.unpack("<I", struct.pack("<f", value))[0]

    def number(word):
        return struct.unpack("<f", struct.pack("<I", word))[0]

    values = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        1.0078125,
        -1.0078125,
        127.0,
        128.0,
        0.125,
        3.1415927,
        65535.0,
        1.0e-20,
    ]
    words = [bits(v) for v in values]
    right = (
        3.0
        if operation == "/"
        else 1.0078125 if operation in {"*", "chain"} else 0.00390625
    )
    prefix = f"bfloat x = bfloat(values[i]); bfloat y = bfloat({right}); "
    if operation == "compound":
        body = prefix + "x += y; results[i + 1u] = float(x);"
    elif operation == "chain":
        body = prefix + "results[i + 1u] = float((x + bfloat(0.00390625)) * y);"
    else:
        body = prefix + f"results[i + 1u] = float(x {operation} y);"
    expected = []
    for word in words:
        left = number(_round(word))
        if operation == "chain":
            value = number(_round(bits(left + 0.00390625))) * right
        else:
            value = {
                "+": lambda: left + right,
                "-": lambda: left - right,
                "*": lambda: left * right,
                "/": lambda: left / right,
                "compound": lambda: left + right,
            }[operation]()
        expected.append(_round(bits(value)))
    source, request, expected = _case(
        tmp_path, TARGET, "arithmetic", words, body=body, expected_words=expected
    )
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="bfloat_conversion",
        validate=_validate_half,
    )


INFERRED_ARITHMETIC = {
    "right-add": "x + 257",
    "left-add": "257 + x",
    "right-subtract": "x - 257",
    "left-subtract": "257 - x",
    "right-multiply": "x * 257",
    "left-multiply": "257 * x",
    "right-divide": "x / 3",
    "left-divide": "3 / x",
    "nested": "1 / (1 + x)",
    "cancellation": "(x + 1) - x",
    "comparison": "x == 257",
    "widened": "x + 257.0f",
    "boolean": "x + true",
    "compound": "x",
    "side-effect": "counter++ + x",
    "vector-right-add": "(bfloat2(x) + 257).x",
    "vector-left-add": "(257 + bfloat2(x)).x",
    "vector-widened": "(float2(257.0f) + x).x",
    "vector-widened-reverse": "(x + float2(257.0f)).x",
}
BFLOAT_VECTOR_ARITHMETIC = {"vector-right-add", "vector-left-add"}


def _inferred_arithmetic_reference(value, operation):
    def bits(number):
        return struct.unpack("<I", struct.pack("<f", number))[0]

    def narrow(number):
        return struct.unpack("<f", struct.pack("<I", _round(bits(number))))[0]

    def divide(left, right):
        if right == 0:
            return math.copysign(math.inf, left * math.copysign(1.0, right))
        return left / right

    x = narrow(value)
    rounded_integer = narrow(257)
    result = {
        "right-add": lambda: x + rounded_integer,
        "left-add": lambda: rounded_integer + x,
        "right-subtract": lambda: x - rounded_integer,
        "left-subtract": lambda: rounded_integer - x,
        "right-multiply": lambda: x * rounded_integer,
        "left-multiply": lambda: rounded_integer * x,
        "right-divide": lambda: x / 3,
        "left-divide": lambda: divide(3, x),
        "nested": lambda: divide(1, narrow(1 + x)),
        "cancellation": lambda: narrow(x + 1) - x,
        "comparison": lambda: float(x == rounded_integer),
        "widened": lambda: x + 257.0,
        "boolean": lambda: x + 1,
        "compound": lambda: narrow(x + rounded_integer) / 3,
        "side-effect": lambda: rounded_integer + x,
        "vector-right-add": lambda: x + rounded_integer,
        "vector-left-add": lambda: rounded_integer + x,
        "vector-widened": lambda: 257.0 + x,
        "vector-widened-reverse": lambda: x + 257.0,
    }[operation]()
    return bits(result) if "widened" in operation else _round(bits(result))


def test_inferred_bfloat_reference_requires_operand_and_intermediate_rounding():
    assert _inferred_arithmetic_reference(0.0, "right-add") == 0x43800000
    assert _inferred_arithmetic_reference(256.0, "cancellation") == 0
    assert _inferred_arithmetic_reference(256.0, "comparison") == 0x3F800000
    assert _inferred_arithmetic_reference(0.0, "widened") == 0x43808000


@pytest.mark.parametrize(
    "operation",
    tuple(
        operation
        for operation in INFERRED_ARITHMETIC
        if TARGET != "directx" or operation not in BFLOAT_VECTOR_ARITHMETIC
    ),
)
def test_inferred_bfloat_arithmetic_executes_natively(tmp_path, operation):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required inferred bfloat arithmetic")
    values = [0.0, -0.0, 0.25, -0.5, 1.0, -1.0, 2.0, 3.0, 128.0, 256.0, 257.0, 65536.0]
    words = [struct.unpack("<I", struct.pack("<f", value))[0] for value in values]
    body = "bfloat x = bfloat(values[i]); "
    if operation == "side-effect":
        body += "uint counter = 257u; "
    body += f"auto y = {INFERRED_ARITHMETIC[operation]}; "
    if operation == "compound":
        body += "y += 257; y /= 3; "
    if operation == "side-effect":
        body += "results[i + 1u] = counter == 258u ? float(y) : -99999.0f;"
    else:
        body += "results[i + 1u] = float(y);"
    source, request, expected = _case(
        tmp_path,
        TARGET,
        "inferred-arithmetic",
        words,
        body=body,
        expected_words=[
            _inferred_arithmetic_reference(value, operation) for value in values
        ],
    )
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="bfloat_conversion",
        validate=_validate_half,
    )


@pytest.mark.parametrize("namespace", ("metal", "metal::fast", "metal::precise"))
@pytest.mark.parametrize("operation", ("widened", "nested", "side-effect"))
def test_bfloat_math_wrapper_boundaries_execute_natively(
    tmp_path, namespace, operation
):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required bfloat math boundaries")
    mode = (
        "__METAL_FAST_MATH__"
        if namespace.endswith("fast")
        else "__METAL_PRECISE_MATH__"
    )
    open_namespace = " ".join(f"namespace {name} {{" for name in namespace.split("::"))
    declarations = f"""{open_namespace}
    METAL_FUNC bfloat exp(bfloat value) {{
        return bfloat(__metal_exp(float(value), {mode}));
    }}
    METAL_FUNC bfloat abs(bfloat value) {{
        return bfloat(__metal_fabs(float(value), {mode}));
    }}
{"}" * len(namespace.split("::"))}"""
    values = [-8.0, -4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0]

    def bits(value):
        return struct.unpack("<I", struct.pack("<f", value))[0]

    def narrow(value):
        return struct.unpack("<f", struct.pack("<I", _round(bits(value))))[0]

    expected_words = []
    for value in values:
        exponential = narrow(math.exp(abs(value) if operation == "nested" else value))
        result = (
            narrow(1.0 / narrow(1.0 + exponential))
            if operation == "nested"
            else exponential + 0.25
        )
        expected_words.append(bits(result))
    body = "bfloat x = bfloat(values[i]); "
    if operation == "nested":
        body += f"auto y = 1 / (1 + {namespace}::exp({namespace}::abs(x))); "
        body += "results[i + 1u] = float(y);"
    elif operation == "side-effect":
        body = "uint index = i; "
        body += f"float y = float({namespace}::exp(bfloat(values[index++]))) + 0.25f; "
        body += "results[i + 1u] = index == i + 1u ? y : -99999.0f;"
    else:
        body += f"results[i + 1u] = float({namespace}::exp(x)) + 0.25f;"
    source, request, expected = _case(
        tmp_path,
        TARGET,
        "math-wrapper",
        [bits(value) for value in values],
        body=body,
        expected_words=expected_words,
        declarations=declarations,
    )
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="bfloat_conversion",
        validate=_validate_half,
    )


@pytest.mark.parametrize("operation", sorted(BFLOAT_VECTOR_ARITHMETIC))
def test_inferred_bfloat_vector_arithmetic_keeps_directx_diagnostic(
    tmp_path, operation
):
    source = tmp_path / "vector.metal"
    source.write_text(
        _source(
            "bfloat x = bfloat(values[i]); "
            f"auto y = {INFERRED_ARITHMETIC[operation]}; results[i] = float(y);"
        )
    )
    with pytest.raises(DirectXBFloat16UnsupportedError) as error:
        translate(str(source), backend="directx", format_output=False)
    assert error.value.reason == "unsupported-bfloat16-builtin"
    assert error.value.operation == "bfloat2"


def test_bfloat_random_finite_rounding_executes_natively(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required bfloat conversions")
    rng = random.Random(6816)
    words = [rng.getrandbits(32) for _ in range(4096)]
    words = [word for word in words if word & 0x7F800000 != 0x7F800000]
    source, request, expected = _case(tmp_path, TARGET, "constructor", words)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="bfloat_conversion",
        validate=_validate_half,
    )


@pytest.mark.parametrize("signed", (False, True))
def test_bfloat_integer_rounding_executes_natively(tmp_path, signed):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required bfloat conversions")
    values = [0, 1, 127, 128, 255, 256]
    for exponent in range(24, 31 if signed else 32):
        shift = exponent - 7
        values.extend(
            (mantissa << shift) + (1 << (shift - 1)) + delta
            for mantissa in range(128, 256)
            for delta in (-1, 0, 1)
        )
    values.append(0x7FFFFFFF if signed else 0xFFFFFFFF)
    if signed:
        values.extend([-v for v in values[1:]])
        values.append(-0x80000000)
    expected = []
    for value in values:
        magnitude = abs(value)
        shift = max(0, magnitude.bit_length() - 8)
        upper, lower = divmod(magnitude, 1 << shift)
        if shift and (
            lower > (1 << (shift - 1)) or (lower == (1 << (shift - 1)) and upper % 2)
        ):
            upper += 1
        rounded = upper * (1 << shift) * (-1 if value < 0 else 1)
        expected.append(struct.unpack("<I", struct.pack("<f", rounded))[0])
    dtype = "int" if signed else "uint"
    source, request, outputs = _case(
        tmp_path,
        TARGET,
        "integer",
        [v & 0xFFFFFFFF for v in values],
        body=f"results[i + 1u] = float(bfloat(as_type<{dtype}>(values[i])));",
        expected_words=expected,
    )
    _execute(
        request,
        outputs,
        tmp_path,
        original_source=source,
        original_entry="bfloat_conversion",
        validate=_validate_half,
    )


def test_bfloat_nonfinite_conversion_executes_natively(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required bfloat conversions")
    words = [0x7F800000, 0xFF800000, 0x7FC10001, 0xFFC50001, 0x7F800001, 0xFF850001]
    source, request, expected = _case(tmp_path, TARGET, "constructor", words)
    # Metal may canonicalize NaN payloads during conversion. Round-trip results
    # must match original Metal; the portable helper's payload policy is separate.
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="bfloat_conversion",
        validate=_validate_half,
        compare_with_original=TARGET == "metal",
    )
    evidence = json.loads((tmp_path / "evidence.json").read_text())
    for record in evidence["records"].values():
        output = record["outputs"][next(iter(expected))]["values"]
        assert output[0] == output[-1] == 0x422A0000
        assert output[1:3] == words[:2]
        for actual, original in zip(output[3:-1], words[2:], strict=True):
            assert actual & 0xFFFF == 0
            assert actual & 0x7FC00000 == 0x7FC00000
            assert actual & 0x80000000 == original & 0x80000000


def test_bfloat_narrowing_does_not_requantize_identity_copies(tmp_path):
    source = tmp_path / "identity.metal"
    source.write_text("""#include <metal_stdlib>
using namespace metal;
bfloat identity(bfloat value) { return bfloat(value); }
kernel void copy(const device bfloat* values [[buffer(0)]], device bfloat* results [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    results[i] = identity(values[i]);
}
""")
    generated = translate(str(source), backend="opengl", format_output=False)
    assert "crossgl_round_bfloat" not in generated
    _compile(generated, tmp_path)


def test_bfloat_conversion_native_gate_is_required():
    from tools import ci_coverage

    workflow = Path(".github/workflows/demo-project-testing.yml").read_text()
    for name in (
        "Validate indexed OpenGL gather and resource aggregates",
        "Validate indexed DirectX gather and resource aggregates",
        "Validate Metal byte and vector storage",
    ):
        step = ci_coverage.workflow_step_section(workflow, name)
        assert f'{REQUIRE_ENV}: "1"' in step
        assert "tests/test_translator/test_opengl_bfloat_conversion.py" in step
        assert (
            "-n auto" in step and "continue-on-error" not in step and "if:" not in step
        )
