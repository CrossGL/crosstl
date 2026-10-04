"""Bfloat constants retain integer precision and explicit conversion boundaries."""

import os
import struct
from pathlib import Path

import pytest

from crosstl import translate
from crosstl.translator.ast import ConstructorNode, LiteralNode, PrimitiveType
from crosstl.translator.codegen.bfloat_constants import (
    bfloat16_constant_bits,
    scalar_constant_value,
)
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from tests.test_translator.test_boolean_buffer_runtime import _bound_values, _request
from tests.test_translator.test_half_buffer_runtime import _validate_half
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_opengl_bfloat_conversion import REQUIRE_ENV, TARGET
from tests.test_translator.test_opengl_half_conversion import _compile, _shader
from tests.test_translator.test_software_subgroup_product import _package

CASES = (
    ("bfloat(0)", 0x0000),
    ("-bfloat(0)", 0x8000),
    ("bfloat(-0.0f)", 0x8000),
    ("bfloat(1)", 0x3F80),
    ("bfloat(-1)", 0xBF80),
    ("bfloat(16842751u)", 0x4B80),
    ("bfloat(16842752u)", 0x4B80),
    ("bfloat(16842753u)", 0x4B81),
    ("bfloat(-16842753)", 0xCB81),
    ("bfloat(16973823u)", 0x4B81),
    ("bfloat(16973824u)", 0x4B82),
    ("bfloat(16973825u)", 0x4B82),
    ("bfloat(33488895u)", 0x4BFF),
    ("bfloat(33488896u)", 0x4C00),
    ("bfloat(33488897u)", 0x4C00),
    ("bfloat(2147483647)", 0x4F00),
    ("bfloat(-2147483647)", 0xCF00),
    ("bfloat(4294967295u)", 0x4F80),
    ("static_cast<bfloat>(16842753u)", 0x4B81),
    ("(bfloat)16842753u", 0x4B81),
    ("bfloat(bfloat(16842753u))", 0x4B81),
    ("bfloat(float(16842753u))", 0x4B80),
    ("bfloat(16842753.0f)", 0x4B80),
    ("bfloat(float(bfloat(16842753u)))", 0x4B81),
    ("bfloat(uint(16842753))", 0x4B81),
    ("bfloat(short(32767))", 0x4700),
    ("bfloat(ushort(65535))", 0x4780),
    ("BFloat(16842753u)", 0x4B81),
    ("bfloat(seed)", 0x4B81),
    ("16842753u", 0x4B81),
)

WIDE_CASES = (
    ("bfloat(9223372036854775807l)", 0x5F00),
    ("bfloat(-9223372036854775807l)", 0xDF00),
    ("bfloat(18446744073709551615ul)", 0x5F80),
    ("bfloat(9223372036854775809ul)", 0x5F00),
    ("bfloat(9259400833873739777ul)", 0x5F01),
    ("bfloat(float(9259400833873739777ul))", 0x5F00),
)


def _source(cases):
    seed = (
        "constant uint seed = 16842753u;"
        if any("seed" in expr for expr, _ in cases)
        else ""
    )
    constants = "\n".join(
        f"constant bfloat value_{index} = {expression};"
        for index, (expression, _bits) in enumerate(cases)
    )
    stores = "\n".join(
        f"results[{index + 1}] = float(value_{index});" for index in range(len(cases))
    )
    return f"""#include <metal_stdlib>
using namespace metal;
using BFloat = bfloat;
{seed}
{constants}
kernel void bfloat_constants(device float* results [[buffer(0)]]) {{
    {stores}
}}
"""


def test_integer_constants_round_at_every_bfloat_midpoint():
    for exponent in range(8, 64):
        spacing = 1 << (exponent - 7)
        for significand in range(128, 256):
            lower = significand * spacing
            for delta in (-1, 0, 1):
                value = lower + spacing // 2 + delta
                rounded = lower + spacing * (
                    delta > 0 or (delta == 0 and significand % 2)
                )
                for sign in (-1, 1):
                    expected = struct.unpack("<I", struct.pack("<f", sign * rounded))[0]
                    assert bfloat16_constant_bits(sign * value) << 16 == expected


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_metal_bfloat_constants_have_exact_target_initializers(tmp_path, target):
    cases = CASES
    source = tmp_path / "constants.metal"
    source.write_text(_source(cases))
    generated = translate(str(source), backend=target, format_output=False)
    for index, (_expression, bits) in enumerate(cases):
        if target == "directx":
            assert f"static const uint value_{index} = 0x{bits:04x}u;" in generated
        else:
            number = struct.unpack("<f", struct.pack("<I", bits << 16))[0]
            assert any(
                f"const float value_{index} = {initializer};" in generated
                for initializer in (repr(number), f"({number!r})")
            )
    if target == "opengl":
        _compile(generated, tmp_path)


@pytest.mark.parametrize("dtype", ("bfloat", "bfloat16", "bfloat16_t"))
@pytest.mark.parametrize("generator", (HLSLCodeGen, GLSLCodeGen))
def test_canonical_bfloat_constant_aliases(dtype, generator):
    generated = generator().generate(
        _shader(
            "output[0] = float(value);", f"const {dtype} value = {dtype}(16842753u);"
        )
    )
    assert "value = 0x4b81u;" in generated or "value = 16908288.0;" in generated


@pytest.mark.parametrize("target", ("directx", "opengl"))
@pytest.mark.parametrize(
    "expression",
    ("bfloat(-1u)", "bfloat(uint(4294967296ul))", *(expr for expr, _ in WIDE_CASES)),
)
def test_unproven_integer_constant_operations_are_diagnostic(
    tmp_path, target, expression
):
    source = tmp_path / "unproven.metal"
    source.write_text(_source(((expression, 0),)))
    with pytest.raises(ValueError) as error:
        translate(str(source), backend=target, format_output=False)
    assert error.value.project_diagnostic_code.startswith("project.translate.")


def test_constant_evaluator_preserves_constructor_nodes_and_rejects_narrow_casts():
    literal = LiteralNode(16842753, PrimitiveType("uint"))
    constructor = ConstructorNode(PrimitiveType("bfloat16_t"), [literal])
    assert scalar_constant_value(constructor) == 16908288.0
    narrowing = ConstructorNode(PrimitiveType("short"), [literal])
    assert scalar_constant_value(narrowing) is None


@pytest.mark.parametrize(
    "dtype,value,bits",
    (
        ("uint64_t", 9259400833873739777, 0x5F01),
        ("int64_t", -9223372036854775807, 0xDF00),
    ),
)
def test_typed_wide_integer_constants_do_not_round_through_float(dtype, value, bits):
    literal = LiteralNode(value, PrimitiveType(dtype))
    constructor = ConstructorNode(PrimitiveType("bfloat"), [literal])
    assert bfloat16_constant_bits(scalar_constant_value(constructor)) == bits


def test_global_integer_lookup_does_not_fold_a_shadowed_local(tmp_path):
    generated = GLSLCodeGen().generate(
        _shader(
            "uint seed = uint(output[0]); output[0] = float(bfloat(seed));",
            "const uint seed = 16842753u;",
        )
    )
    assert "crossgl_integer_to_bfloat(seed)" in generated
    _compile(generated, tmp_path)


def test_bfloat_constants_execute_natively(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required bfloat constants")
    cases = CASES
    source = _source(cases)
    _, descriptor, package = _package(
        tmp_path, TARGET, "uint", (1, 1, 1), source=source, software_subgroups=False
    )
    guard = 0x422A0000
    initial = {
        "results": {
            "dtype": "float32",
            "encoding": "ieee754-binary32",
            "shape": [len(cases) + 2],
            "values": [guard] * (len(cases) + 2),
        }
    }
    expected = {
        "results": {
            **initial["results"],
            "values": [guard, *(bits << 16 for _, bits in cases), guard],
        }
    }
    request = _request(descriptor, package, initial, expected, 1)
    _execute(
        request,
        _bound_values(descriptor, expected),
        tmp_path,
        original_source=source,
        original_entry="bfloat_constants",
        validate=_validate_half,
    )


def test_constant_native_gate_is_required():
    from tools import ci_coverage

    workflow = Path(".github/workflows/demo-project-testing.yml").read_text()
    for name in (
        "Validate indexed OpenGL gather and resource aggregates",
        "Validate indexed DirectX gather and resource aggregates",
        "Validate Metal byte and vector storage",
    ):
        step = ci_coverage.workflow_step_section(workflow, name)
        assert f'{REQUIRE_ENV}: "1"' in step
        assert "tests/test_translator/test_bfloat_constants.py" in step
