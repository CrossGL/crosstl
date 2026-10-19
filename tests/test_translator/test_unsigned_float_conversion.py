"""Preserve unsigned binary32 rounding across translated conversion boundaries."""

import json
import os
import random
import struct
import sys
from pathlib import Path

import pytest

from crosstl import translate
from crosstl.project import build_native_loader_dispatch_request
from crosstl.translator import parse
from crosstl.translator.codegen.directx_codegen import (
    DirectXContextualConversionError,
    HLSLCodeGen,
)
from tests.ci_helpers import assert_native_step_prerequisites, assert_paths_covered
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_UNSIGNED_FLOAT_CONVERSION"
GUARD = 0x6A15BEEF


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _values():
    values = {0, 1, 0xFFFFFFFF, 2189359745, 2207582591, 2220464258}
    for exponent in range(24, 32):
        shift = exponent - 23
        midpoint = 1 << (shift - 1)
        for mantissa in (0x800000, 0x800001, 0xAAAAAA, 0xBFFFFF, 0xFFFFFE, 0xFFFFFF):
            for remainder in (
                0,
                midpoint - 1,
                midpoint,
                midpoint + 1,
                (1 << shift) - 1,
            ):
                values.add((mantissa << shift) + remainder)
    for exponent in range(32):
        values.update((1 << exponent) + offset for offset in range(-2, 3))
    rng = random.Random(2247)
    values.update(rng.getrandbits(32) for _ in range(4096))
    return sorted(value for value in values if 0 <= value <= 0xFFFFFFFF)


def _source(mode):
    helpers = """
using Unsigned = uint;
struct Converted { float value; };
float promote(uint value) { return value; }
float accept(float value) { return value; }
uint record(uint value, thread uint& calls) { calls += 1u; return value; }
"""
    if mode == "scalar":
        body = """
        Unsigned value = values[i];
        float initialized = value;
        float assigned = 0.0f;
        assigned = value;
        Converted item;
        item.value = value;
        float pair[2] = {float(value), 0.0f};
        pair[1] = value;
        results[base + 0u] = as_type<uint>(float(value));
        results[base + 1u] = as_type<uint>((float)value);
        results[base + 2u] = as_type<uint>(static_cast<float>(value));
        results[base + 3u] = as_type<uint>(initialized);
        results[base + 4u] = as_type<uint>(assigned);
        results[base + 5u] = as_type<uint>(promote(value));
        results[base + 6u] = as_type<uint>(accept(value));
        results[base + 7u] = as_type<uint>(item.value);
        results[base + 8u] = as_type<uint>(pair[1]);
        results[base + 9u] = uint(value == float(value));
        results[base + 10u] = uint(float(value) == value);
        results[base + 11u] = as_type<uint>(value + 1.0f);
        results[base + 12u] = as_type<uint>(1.0f + value);
        """
        stride = 13
    else:
        width = int(mode[-1])
        components = ", ".join(
            ("value", "~value", "value ^ 0x80000000u", "value + 1u")[:width]
        )
        body = f"""
        uint value = values[i];
        uint calls = 0u;
        float{width} splat = float{width}(record(value, calls));
        uint{width} integers = uint{width}({components});
        float{width} converted = float{width}(integers);
        """
        for lane, component in enumerate("xyzw"[:width]):
            body += f"results[base + {lane}u] = as_type<uint>(splat.{component});\n"
            body += f"results[base + {width + lane}u] = as_type<uint>(converted.{component});\n"
        body += f"results[base + {2 * width}u] = calls;"
        stride = 2 * width + 1
    return (
        "#include <metal_stdlib>\nusing namespace metal;\n" + helpers + f"""
kernel void convert_unsigned(const device uint* values [[buffer(0)]],
                             device uint* results [[buffer(1)]],
                             uint i [[thread_position_in_grid]]) {{
    uint base = 32u + i * {stride}u;
    {body}
}}
""",
        stride,
    )


def _case(root, target, mode):
    source, stride = _source(mode)
    source, descriptor, package = _package(
        root, target, "uint", (1, 1, 1), source=source, software_subgroups=False
    )
    values = _values()
    wanted = [GUARD] * 32
    for value in values:
        if mode == "scalar":
            wanted.extend([_bits(value)] * 9)
            rounded = struct.unpack("<f", struct.pack("<I", _bits(value)))[0]
            wanted.extend([1, 1, _bits(rounded + 1.0), _bits(rounded + 1.0)])
        else:
            width = int(mode[-1])
            components = (
                value,
                value ^ 0xFFFFFFFF,
                value ^ 0x80000000,
                (value + 1) & 0xFFFFFFFF,
            )
            wanted.extend([_bits(value)] * width)
            wanted.extend(_bits(component) for component in components[:width])
            wanted.append(1)
    wanted.extend([GUARD] * 32)

    def packet(words):
        return {"dtype": "uint32", "shape": [len(words)], "values": words}

    inputs = _bound_values(
        descriptor, {"values": packet(values), "results": packet([GUARD] * len(wanted))}
    )
    outputs = _bound_values(descriptor, {"results": packet(wanted)})
    observed = next(
        binding["name"]
        for binding in descriptor["bindings"]
        if binding["scalarLayout"].get("memberName", binding["name"]) == "values"
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [len(values), 1, 1], "workgroupSize": [1, 1, 1]},
        observe_inputs=[observed],
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    (root / "values.json").write_text(
        json.dumps({"mode": mode, "values": values, "expected": wanted}),
        encoding="utf-8",
    )
    return source, request, outputs


def test_unsigned_reference_covers_ties_carries_and_signed_boundary():
    values = _values()
    assert len(values) > 4096 and len(values) < 65536
    assert {0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF} <= set(values)
    assert _bits(2189359745) == 1325563643
    assert _bits(2207582591) == 1325634825
    assert _bits(2220464258) == 1325685145
    assert _bits(0xFFFFFFFF) == 0x4F800000
    for exponent in range(24, 32):
        shift = exponent - 23
        for mantissa in (0x800000, 0x800001):
            value = (mantissa << shift) + (1 << (shift - 1))
            assert value in values
            rounded = (mantissa + (mantissa & 1)) << shift
            assert _bits(value) == _bits(rounded)


def _shader(body, helpers=""):
    return parse(f"""shader UnsignedConversion {{
        {helpers}
        RWStructuredBuffer<uint> values @ binding(0);
        RWStructuredBuffer<float> output @ binding(1);
        compute {{ @numthreads(1, 1, 1) void main() {{ {body} }} }}
    }}""")


@pytest.mark.parametrize(
    "expression", ["float(values[0])", "(float)values[0]", "values[0]"]
)
def test_directx_unsigned_scalar_rounds_once(tmp_path, expression):
    generated = HLSLCodeGen().generate(_shader(f"output[0] = {expression};"))
    assert "__crossgl_uint_to_float1(values[0])" in generated
    assert "return asfloat(((exponent + 127u) << 23u)" in generated
    assert "significand == 0x1000000u" in generated
    _compile(generated, "directx", tmp_path)


@pytest.mark.parametrize("width", [2, 3, 4])
def test_directx_unsigned_vector_single_evaluation(tmp_path, width):
    generated = HLSLCodeGen().generate(
        _shader(
            f"uint calls = 0u; float{width} converted = float{width}(record(calls)); "
            "output[0] = converted.x; output[1] = float(calls);",
            "uint record(inout uint calls) { calls += 1u; return 2189359745u; }",
        )
    )
    assert generated.count("record(calls)") == 1
    assert "__crossgl_uint_to_float1(record(calls))" in generated
    _compile(generated, "directx", tmp_path)


def test_directx_unsigned_mixed_vector_arguments(tmp_path):
    generated = HLSLCodeGen().generate(
        _shader(
            "uint2 pair = uint2(values[0], values[1]); "
            "float4 converted = float4(pair, values[2], 1.0); output[0] = converted.x;"
        )
    )
    assert (
        "float4(__crossgl_uint_to_float2(pair), __crossgl_uint_to_float1(values[2]), 1.0)"
        in generated
    )
    _compile(generated, "directx", tmp_path)


@pytest.mark.parametrize("name", ["asfloat", "firstbithigh"])
def test_directx_unsigned_rounding_rejects_shadowed_intrinsics(name):
    with pytest.raises(DirectXContextualConversionError) as error:
        HLSLCodeGen().generate(
            _shader("output[0] = float(values[0]);", f"uint {name} = 1u;")
        )
    assert error.value.reason == "unsigned-float-intrinsic-shadowed"


def test_directx_unsigned_helpers_are_collision_safe_and_reset(tmp_path):
    generator = HLSLCodeGen()
    generated = generator.generate(
        _shader(
            "uint __crossgl_uint_to_float1 = values[0]; output[0] = float(__crossgl_uint_to_float1);"
        )
    )
    assert "float __crossgl_uint_to_float1_(uint value)" in generated
    assert "__crossgl_uint_to_float1_(__crossgl_uint_to_float1)" in generated
    _compile(generated, "directx", tmp_path)
    assert "__crossgl_uint_to_float" not in generator.generate(
        _shader("output[0] = 1.0;")
    )


@pytest.mark.parametrize(
    "expression", ["asfloat(values[0])", "float(int(values[0]))", "double(values[0])"]
)
def test_directx_other_conversions_do_not_use_unsigned_helper(expression):
    generated = HLSLCodeGen().generate(_shader(f"output[0] = {expression};"))
    assert "__crossgl_uint_to_float" not in generated


def test_directx_unsigned_global_constants_compile(tmp_path):
    generated = HLSLCodeGen().generate(
        _shader("output[0] = rounded;", "const float rounded = float(4294967295u);")
    )
    _compile(generated, "directx", tmp_path)


@pytest.mark.parametrize(
    "operator", ["+", "-", "*", "/", "==", "!=", "<", "<=", ">", ">="]
)
@pytest.mark.parametrize("reverse", [False, True])
def test_directx_unsigned_mixed_arithmetic_rounds_before_operation(
    tmp_path, operator, reverse
):
    left, right = ("1.0", "values[0]") if reverse else ("values[0]", "1.0")
    generated = HLSLCodeGen().generate(
        _shader(f"output[0] = float({left} {operator} {right});")
    )
    assert "__crossgl_uint_to_float1(values[0])" in generated
    _compile(generated, "directx", tmp_path)


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize("mode", ["scalar", "vector2", "vector3", "vector4"])
def test_unsigned_conversions_translate_and_compile(tmp_path, target, mode):
    source, request, _ = _case(tmp_path, target, mode)
    intermediate = tmp_path / "saved.cgl"
    original = tmp_path / "original.metal"
    original.write_text(source, encoding="utf-8")
    intermediate.write_text(
        translate(str(original), backend="cgl", format_output=False), encoding="utf-8"
    )
    assert translate(
        str(intermediate), backend=target, format_output=False
    ) == translate(str(original), backend=target, format_output=False)
    _compile(request.artifact_path.read_text(), target, tmp_path)


@pytest.mark.parametrize("mode", ["scalar", "vector2", "vector3", "vector4"])
def test_unsigned_conversions_execute_natively(tmp_path, mode):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required unsigned-to-float execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source, request, expected = _case(tmp_path, target, mode)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="convert_unsigned",
    )


def test_directx_native_cast_control(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1" or sys.platform != "win32":
        pytest.skip("The standalone HLSL control requires Windows native validation")
    from tests.test_translator.test_directx_software_reductions import _execute_words

    values = _values()
    expected = [_bits(value) for value in values]
    source = """StructuredBuffer<uint> inputWords : register(t0);
RWStructuredBuffer<uint> outputWords : register(u1);
[numthreads(1, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    outputWords[tid.x] = asuint(float(inputWords[tid.x]));
}
"""
    actual = _execute_words(
        tmp_path,
        source,
        values,
        expected,
        workgroup_size=(1, 1, 1),
        workgroup_count=(len(values), 1, 1),
    )
    assert len(actual) == len(expected)
    mismatches = [
        {"index": index, "input": value, "expectedBits": want, "actualBits": got}
        for index, (value, want, got) in enumerate(zip(values, expected, actual))
        if want != got
    ]
    # This control isolates native HLSL behavior; translated parity is a separate gate.
    (tmp_path / "comparison.json").write_text(
        json.dumps(
            {
                "controlOnly": True,
                "replacesRequiredGate": False,
                "count": len(values),
                "mismatches": mismatches,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_unsigned_conversion_native_gate_is_required():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate mixed-width integer arithmetic"
    )
    assert "test_unsigned_float_conversion.py" in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "--timeout-seconds" in step and "--junitxml" in step and "-n auto" in step
    assert_native_step_prerequisites(step, job="portable-host")
    assert "continue-on-error" not in step
    for event in ("push", "pull_request"):
        assert_paths_covered(
            ci_coverage.workflow_event_path_filters(workflow, event),
            "tests/test_translator/test_unsigned_float_conversion.py",
        )
