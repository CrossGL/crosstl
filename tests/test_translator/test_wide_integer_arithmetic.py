"""Preserve source integer ranks through HLSL arithmetic and native dispatch."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_abi_descriptor,
    build_native_loader_dispatch_request,
    build_runtime_artifact_manifest,
    build_runtime_loader_manifest,
    build_runtime_package,
    translate_project,
)
from crosstl.translator import parse
from crosstl.translator.codegen.directx_codegen import (
    DirectXContextualConversionError,
    HLSLCodeGen,
)
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_metal_native_runtime import _native_request
from tests.test_translator.test_native_loader_dispatch_integration import _executor
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_WIDE_INTEGER_ARITHMETIC"
OPERATORS = ["+", "-", "*", "/", "%", "&", "|", "^", "<", "<=", ">", ">=", "==", "!="]


@pytest.mark.parametrize("operator", OPERATORS)
@pytest.mark.parametrize("reverse", [False, True])
def test_signed_wide_and_unsigned_narrow_operands_use_signed_arithmetic(
    operator, reverse
):
    left, right = ("b", "a") if reverse else ("a", "b")
    result = "bool" if operator in {"<", "<=", ">", ">=", "==", "!="} else "int64_t"
    source = f"shader Test {{ {result} combine(int64_t a, uint b) {{ return {left} {operator} {right}; }} }}"
    generated = HLSLCodeGen().generate(parse(source))
    left, right = ("int64_t(b)", "a") if reverse else ("a", "int64_t(b)")
    assert f"({left} {operator} {right})" in generated


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ("int", "uint64_t", "(uint64_t(a) / b)"),
        ("uint64_t", "int64_t", "(a / uint64_t(b))"),
        ("int64_t", "uint64_t", "(uint64_t(a) / b)"),
        ("uint", "int", "(a / b)"),
        ("int64_t2", "uint", "(a / int64_t(b))"),
        ("uint2", "int64_t", "(a / uint(b))"),
        ("uint", "int64_t2", "(int64_t(a) / b)"),
    ],
)
def test_integer_rank_and_vector_scalar_contracts(left, right, expected):
    source = f"shader Test {{ {left} combine({left} a, {right} b) {{ return {left}(a / b); }} }}"
    generated = HLSLCodeGen().generate(parse(source))
    assert expected in generated


@pytest.mark.parametrize("reverse", [False, True])
def test_conditional_result_type_is_preserved_in_nested_arithmetic(reverse):
    branches = "a : b" if reverse else "b : a"
    source = f"shader Test {{ int64_t combine(bool flag, int64_t a, uint b, uint c) {{ return (flag ? {branches}) / c; }} }}"
    generated = HLSLCodeGen().generate(parse(source))
    expected = "a : int64_t(b)" if reverse else "int64_t(b) : a"
    assert f"((flag ? {expected}) / int64_t(c))" in generated


def test_shifts_keep_left_operand_type_in_nested_arithmetic():
    source = "shader Test { uint combine(uint a, int64_t b, uint c) { return (a >> b) / c; } }"
    generated = HLSLCodeGen().generate(parse(source))
    assert "((a >> b) / c)" in generated


@pytest.mark.parametrize("operator", ["+=", "-=", "*=", "/=", "%=", "&=", "|=", "^="])
def test_compound_assignment_converts_rhs_without_repeating_lvalue(operator):
    source = f"shader Test {{ RWStructuredBuffer<int64_t> values; void combine(uint b) {{ uint index = 0u; values[index++] {operator} b; }} }}"
    generated = HLSLCodeGen().generate(parse(source))
    assert generated.count("index++") == 1
    assert f"{operator} int64_t(b)" in generated


@pytest.mark.parametrize("operator", ["+=", "-=", "*=", "/=", "%=", "&=", "|=", "^="])
def test_compound_assignment_widens_before_narrowing_and_evaluates_lvalue_once(
    operator,
):
    source = f"shader Test {{ RWStructuredBuffer<uint> values; void combine(int64_t b) {{ uint index = 0u; values[index++] {operator} b; }} }}"
    generated = HLSLCodeGen().generate(parse(source))
    assert generated.count("index++") == 1
    assert "inout uint target, int64_t value" in generated
    assert f"target = uint(int64_t(target) {operator[:-1]} value);" in generated
    assert "__crossgl_integer_compound(values[index++], b)" in generated


def test_compound_helper_names_do_not_shadow_source_identifiers_and_reset():
    source = """shader Test {
        uint combine(uint a, int64_t b) {
            uint __crossgl_integer_compound = 7u;
            a /= b;
            return a + __crossgl_integer_compound;
        }
    }"""
    codegen = HLSLCodeGen()
    generated = codegen.generate(parse(source))
    assert "__crossgl_integer_compound_(inout uint" in generated
    assert "__crossgl_integer_compound_(a, b)" in generated
    assert generated == codegen.generate(parse(source))
    assert "__crossgl_integer_compound" not in codegen.generate(
        parse("shader Empty {}")
    )


def test_compound_copy_in_rejects_potential_rhs_alias_modification():
    source = """shader Test {
        uint value;
        int64_t mutate() { value = 2u; return 3; }
        void combine() { value /= mutate(); }
    }"""
    with pytest.raises(DirectXContextualConversionError, match="right operand"):
        HLSLCodeGen().generate(parse(source))


@pytest.mark.parametrize("right", ["uint2", "int64_t3"])
def test_incompatible_source_vector_operations_are_diagnostic(right):
    source = (
        f"shader Test {{ int64_t2 combine(int64_t2 a, {right} b) {{ return a / b; }} }}"
    )
    with pytest.raises(DirectXContextualConversionError, match="vector operands"):
        HLSLCodeGen().generate(parse(source))


@pytest.mark.parametrize(
    "kind,mapped",
    [("int", "int"), ("int64_t", "int64_t"), ("int2", "ivec2"), ("i64vec4", "i64vec4")],
)
def test_glsl_signed_remainder_uses_truncating_division(kind, mapped):
    source = f"shader Test {{ {kind} combine({kind} a, {kind} b) {{ a %= b; return a % b; }} }}"
    generated = GLSLCodeGen().generate(parse(source))
    assert (
        f"{mapped} crossgl_signed_remainder_{mapped}({mapped} left, {mapped} right)"
        in generated
    )
    assert "return left - (left / right) * right;" in generated
    assert f"a = crossgl_signed_remainder_{mapped}(a, b)" in generated
    assert f"return crossgl_signed_remainder_{mapped}(a, b);" in generated


def test_glsl_signed_remainder_evaluates_operands_once_and_avoids_collisions():
    source = """shader Test { int combine(int a, int b) {
        int crossgl_signed_remainder_int = 7;
        return a++ % b++ + crossgl_signed_remainder_int;
    } }"""
    codegen = GLSLCodeGen()
    generated = codegen.generate(parse(source))
    assert generated.count("a++") == 1 and generated.count("b++") == 1
    assert "int crossgl_signed_remainder_int_" in generated
    assert generated == codegen.generate(parse(source))
    assert "crossgl_signed_remainder" not in codegen.generate(parse("shader Empty {}"))


def test_glsl_unsigned_remainder_and_constant_initializers_remain_legal():
    source = """shader Test {
        const int value = -94 % 128;
        uint combine(uint a, uint b) { a %= b; return a % b; }
    }"""
    generated = GLSLCodeGen().generate(parse(source))
    assert "crossgl_signed_remainder" not in generated
    assert "a %= b;" in generated and "return (a % b);" in generated
    assert " / 128) * 128)" in generated


def test_glsl_remainder_rejects_rhs_that_may_change_indexed_target():
    source = """shader Test { void combine(int a[2], int index) {
        a[index] %= index++;
    } }"""
    with pytest.raises(ValueError, match="right operand may change"):
        GLSLCodeGen().generate(parse(source))


def test_glsl_buffer_subscripts_retain_element_type_for_compound_conversion():
    source = """shader Test { RWStructuredBuffer<uint> values;
        void combine(int64_t a) { values[0] /= a; }
    }"""
    generated = GLSLCodeGen().generate(parse(source))
    assert "values[0] = uint((int64_t(values[0]) / a));" in generated


SOURCE = """#include <metal_stdlib>
using namespace metal;
kernel void products(device int64_t* numerators [[buffer(0)]],
                     device uint* denominators [[buffer(1)]],
                     device int64_t* outputs [[buffer(2)]],
                     uint tid [[thread_position_in_grid]]) {
    int64_t a = numerators[tid];
    uint b = denominators[tid];
    uint base = tid * 15u;
    outputs[base] = a / b;
    outputs[base + 1u] = a % b;
    outputs[base + 2u] = a < b ? 1 : 0;
    outputs[base + 3u] = b / a;
    outputs[base + 4u] = b % a;
    outputs[base + 5u] = b > a ? 1 : 0;
    outputs[base + 6u] = ((tid & 1u) != 0u ? a : b) / b;
    int64_t wide = a;
    wide /= b;
    outputs[base + 7u] = wide;
    uint narrow[2] = {b, b};
    uint index = 2u;
    narrow[0] /= a;
    narrow[1] %= a;
    outputs[base + 8u] = narrow[0];
    outputs[base + 9u] = narrow[1];
    outputs[base + 10u] = index;
    outputs[base + 11u] = a / (b++);
    outputs[base + 12u] = b;
    uint saved = denominators[tid];
    uint resourceIndex = tid;
    denominators[resourceIndex] /= a;
    outputs[base + 13u] = denominators[tid];
    outputs[base + 14u] = resourceIndex;
    denominators[tid] = saved;
}
"""

SIDE_EFFECT_SOURCE = SOURCE.replace(
    "uint index = 2u;\n    narrow[0] /= a;\n    narrow[1] %= a;",
    "uint index = 0u;\n    narrow[index++] /= a;\n    narrow[index++] %= a;",
).replace("denominators[resourceIndex] /= a", "denominators[resourceIndex++] /= a")


def test_optimized_dxc_keeps_signed_division_remainder_and_comparison(tmp_path):
    source, descriptor, package = _package(
        tmp_path,
        "directx",
        "int64_t",
        (1, 1, 1),
        source=SOURCE,
        software_subgroups=False,
    )
    artifacts = list(package.rglob("*.hlsl"))
    assert len(artifacts) == 1
    if not shutil.which("dxc"):
        if os.environ.get(REQUIRE_ENV) == "1" and sys.platform == "win32":
            pytest.fail("DXC is required for mixed-width arithmetic verification")
        pytest.skip("DXC is not installed")
    assembly = tmp_path / "optimized.ll"
    result = subprocess.run(
        [
            "dxc",
            "-T",
            "cs_6_6",
            "-E",
            "CSMain",
            "-O3",
            "-WX",
            str(artifacts[0]),
            "-Fc",
            str(assembly),
            "-Fo",
            str(tmp_path / "optimized.dxil"),
        ],
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    ir = assembly.read_text()
    assert "sdiv i64" in ir and "srem i64" in ir and "icmp slt i64" in ir
    assert "udiv i64" not in ir and "urem i64" not in ir and "icmp ult i64" not in ir


@pytest.mark.parametrize(
    "body,instruction",
    [
        ("outputs[tid.x] = a / b;", "udiv i64"),
        ("outputs[tid.x] = a % b;", "urem i64"),
        ("outputs[tid.x] = a / 3u;", "udiv i64"),
        ("outputs[tid.x] = a / 3ull;", "udiv i64"),
        ("outputs[tid.x] = a < b;", "icmp ult i64"),
        ("outputs[tid.x] = (a / b) / a;", "udiv i64"),
        ("outputs[tid.x] = (tid.x > 0 ? a : b) / a;", "udiv i64"),
        ("a /= b; outputs[tid.x] = a;", "udiv i64"),
        ("int64_t2 v = int64_t2(a, a); v /= b; outputs[tid.x] = v.x;", "udiv i64"),
        ("uint64_t2 v = uint2(b, b) / a; outputs[tid.x] = v.x;", "udiv i64"),
        ("uint64_t2 v = int64_t2(a, a) / b; outputs[tid.x] = v.x;", "udiv i64"),
        ("outputs[tid.x] = numerators[tid.x] / denominators[tid.x];", "udiv i64"),
    ],
)
def test_native_hlsl_integer_rank_survives_roundtrip(tmp_path, body, instruction):
    from crosstl import translate

    if not shutil.which("dxc"):
        if sys.platform == "win32" and os.environ.get(REQUIRE_ENV) == "1":
            pytest.fail("DXC is required for the HLSL compatibility gate")
        pytest.skip("DXC is required to inspect native HLSL integer conversions")
    source = f"""RWStructuredBuffer<int64_t> outputs : register(u0);
StructuredBuffer<int64_t> numerators : register(t0);
StructuredBuffer<uint> denominators : register(t1);
[numthreads(1, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {{
    int64_t a = numerators[tid.x];
    uint b = denominators[tid.x];
    {body}
}}
"""
    original = tmp_path / "original.hlsl"
    original.write_text(source)
    generated = tmp_path / "generated.hlsl"
    generated.write_text(
        translate(str(original), backend="directx", format_output=False)
    )
    intermediate = tmp_path / "intermediate.cgl"
    intermediate.write_text(
        translate(str(original), backend="cgl", format_output=False)
    )
    staged = tmp_path / "staged.hlsl"
    staged.write_text(
        translate(str(intermediate), backend="directx", format_output=False)
    )
    for artifact in (original, generated, staged):
        assembly = artifact.with_suffix(".ll")
        result = subprocess.run(
            [
                "dxc",
                "-T",
                "cs_6_6",
                "-E",
                "CSMain",
                "-O3",
                "-WX",
                str(artifact),
                "-Fc",
                str(assembly),
                "-Fo",
                str(artifact.with_suffix(".dxil")),
            ],
            text=True,
            capture_output=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert instruction in assembly.read_text(), artifact.name


def _quotient(left, right):
    return (abs(left) // abs(right)) * (-1 if (left < 0) != (right < 0) else 1)


@pytest.mark.parametrize("side_effects", [False, True])
def test_mixed_integer_arithmetic_executes_with_source_signedness(
    tmp_path, side_effects
):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native integer arithmetic")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source = SIDE_EFFECT_SOURCE if side_effects else SOURCE
    if target == "opengl" and side_effects:
        from crosstl import translate

        path = tmp_path / "side_effects.metal"
        path.write_text(source)
        with pytest.raises(ValueError, match="side-effecting lvalue"):
            translate(str(path), backend=target, format_output=False)
        return
    source, descriptor, package = _package(
        tmp_path, target, "int64_t", (1, 1, 1), source=source, software_subgroups=False
    )
    numerators = [-94, -128, -257, 94, 2**31, -(2**31), 2**32 + 1, -(2**40)]
    denominators = [128, 128, 128, 128, 2**32 - 1, 2**32 - 1, 2**32 - 1, 3]
    wanted = []
    for index, (a, b) in enumerate(zip(numerators, denominators)):
        quotient, inverse = _quotient(a, b), _quotient(b, a)
        remainder, inverse_remainder = a - quotient * b, b - inverse * a
        wanted.extend(
            [
                quotient,
                remainder,
                int(a < b),
                inverse,
                inverse_remainder,
                int(b > a),
                quotient if index & 1 else 1,
                quotient,
                inverse & 0xFFFFFFFF,
                inverse_remainder & 0xFFFFFFFF,
                2,
                quotient,
                (b + 1) & 0xFFFFFFFF,
                inverse & 0xFFFFFFFF,
                index + int(side_effects),
            ]
        )
    _execute_integer_case(
        tmp_path,
        source,
        descriptor,
        package,
        target,
        numerators,
        denominators,
        wanted,
        "int64",
        "uint32",
    )


def _execute_integer_case(
    tmp_path,
    source,
    descriptor,
    package,
    target,
    numerators,
    denominators,
    wanted,
    dtype,
    denominator_dtype,
    *,
    source_backend="metal",
):
    guards = [0x37000000 + index for index in range(8)]

    def payload(values, kind=dtype):
        return {"dtype": kind, "shape": [len(values)], "values": values}

    inputs = {
        "numerators": payload(numerators),
        "denominators": payload(denominators, denominator_dtype),
        "outputs": payload([91] * len(wanted) + guards),
    }
    expected = _bound_values(
        descriptor, {**inputs, "outputs": payload(wanted + guards)}
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        {
            name: {key: value for key, value in data.items() if key != "values"}
            for name, data in expected.items()
        },
        {"workgroupCount": [len(numerators), 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    compiled = tmp_path / "compiled"
    compiled.mkdir()
    _, module = _compile(request.artifact_path.read_text(), target, compiled)
    assert module.is_file(), "Native compiler is required"
    executor = _executor(target)
    records = {}
    try:
        assert executor.is_available(request).available
        result = executor.run(request)
        assert result.status == "ok", result
        records["generated"] = {"outputs": result.outputs, "details": result.details}
        if target == source_backend:
            original = tmp_path / "original"
            original.mkdir()
            artifact, original_module = _compile(source, target, original)
            state, native = _native_request(request)
            native = replace(
                native,
                artifact_path=artifact,
                module_path=original_module,
                entry_point="products" if target == "metal" else "CSMain",
            )
            actual = executor.runtime_adapter.runtime.dispatch(None, state, native)
            records["originalMetal" if target == "metal" else "originalHLSL"] = {
                "outputs": actual,
                "moduleSha256": (
                    hashlib.sha256(original_module.read_bytes()).hexdigest()
                ),
            }
        (tmp_path / "evidence.json").write_text(
            json.dumps(
                {
                    "target": target,
                    "inputs": inputs,
                    "expected": expected,
                    "descriptor": descriptor,
                    "records": records,
                    "sourceSha256": (
                        hashlib.sha256(request.artifact_path.read_bytes()).hexdigest()
                    ),
                    "validationModuleSha256": (
                        hashlib.sha256(module.read_bytes()).hexdigest()
                    ),
                },
                indent=2,
            )
        )
        for record in records.values():
            assert record["outputs"] == expected
    finally:
        close = getattr(executor.runtime_adapter.runtime, "close", None)
        if close:
            close()


def test_native_hlsl_arithmetic_preserves_source_results_on_each_target(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required source HLSL arithmetic")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source = """RWStructuredBuffer<int64_t> numerators : register(u0);
RWStructuredBuffer<uint> denominators : register(u1);
RWStructuredBuffer<int64_t> outputs : register(u2);
[numthreads(1, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    int64_t a = numerators[tid.x];
    uint b = denominators[tid.x];
    uint64_t2 wide = int64_t2(a, a) / b;
    int64_t2 compound = int64_t2(a, a);
    compound /= b;
    outputs[tid.x * 8u] = a / b;
    outputs[tid.x * 8u + 1u] = a % b;
    outputs[tid.x * 8u + 2u] = a < b;
    outputs[tid.x * 8u + 3u] = (tid.x > 0 ? a : b) / b;
    outputs[tid.x * 8u + 4u] = wide.x;
    outputs[tid.x * 8u + 5u] = compound.y;
    outputs[tid.x * 8u + 6u] = a / 3u;
    a /= b;
    outputs[tid.x * 8u + 7u] = a;
}
"""
    (tmp_path / "source.hlsl").write_text(source)
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            include_patterns=("source.hlsl",),
            targets=(target,),
            output_dir="out",
        ),
        format_output=False,
    )
    report.write_json(tmp_path / "report.json")
    assert report.to_json()["summary"]["failedCount"] == 0, report.to_json()
    manifest = build_runtime_artifact_manifest(tmp_path / "report.json")
    assert manifest["success"], manifest
    (tmp_path / "artifacts.json").write_text(json.dumps(manifest))
    package = tmp_path / "package"
    assert build_runtime_package(tmp_path / "artifacts.json", package)["success"]
    loader = build_runtime_loader_manifest(package / "runtime-package.json")
    assert loader["success"] and len(loader["loadUnits"]) == 1, loader
    descriptor = build_native_loader_abi_descriptor(
        loader, load_unit_id=loader["loadUnits"][0]["id"]
    )
    numerators = [-94, -128, -257, 94, 2**31, -(2**31), 2**32 + 1, -(2**40) + 1]
    denominators = [128, 128, 128, 128, 2**32 - 1, 2**32 - 1, 2**32 - 1, 3]
    wanted = []
    for index, (a, b) in enumerate(zip(numerators, denominators)):
        unsigned = a & (2**64 - 1)
        quotient = unsigned // b
        wanted.extend(
            [
                quotient,
                unsigned % b,
                int(unsigned < b),
                quotient if index else 1,
                quotient,
                quotient,
                unsigned // 3,
                quotient,
            ]
        )
    _execute_integer_case(
        tmp_path,
        source,
        descriptor,
        package,
        target,
        numerators,
        denominators,
        wanted,
        "int64",
        "uint32",
        source_backend="directx",
    )


@pytest.mark.parametrize("kind,dtype", [("int", "int32"), ("long", "int64")])
@pytest.mark.parametrize(
    "width,broadcast",
    [(1, False), (2, False), (3, False), (4, False), (2, True), (3, True), (4, True)],
)
def test_signed_remainder_executes_for_both_operand_signs(
    tmp_path, kind, dtype, width, broadcast
):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required signed remainder execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    vector = kind + (str(width) if width > 1 else "")

    def loads(name):
        return ", ".join(f"{name}[base + {i}u]" for i in range(width))

    right = (
        f"{kind} right = denominators[base];"
        if broadcast
        else f"{vector} right = {vector}({loads('denominators')});"
    )
    stores = "\n".join(
        f"outputs[tid * {2 * width}u + {i}u] = remainder{'.' + 'xyzw'[i] if width > 1 else ''};\n"
        f"outputs[tid * {2 * width}u + {width + i}u] = left{'.' + 'xyzw'[i] if width > 1 else ''};"
        for i in range(width)
    )
    source = f"""#include <metal_stdlib>
using namespace metal;
kernel void products(device {kind}* numerators [[buffer(0)]],
                     device {kind}* denominators [[buffer(1)]],
                     device {kind}* outputs [[buffer(2)]],
                     uint tid [[thread_position_in_grid]]) {{
    if (tid >= {12 // width}u) {{ return; }}
    uint base = tid * {width}u;
    {vector} left = {vector}({loads('numerators')});
    {right}
    {vector} remainder = left % right;
    left %= right;
    {stores}
}}
"""
    source, descriptor, package = _package(
        tmp_path, target, kind, (1, 1, 1), source=source, software_subgroups=False
    )
    minimum = -(2**31) if dtype == "int32" else -(2**63)
    numerators = [
        -94,
        128,
        -257,
        -94,
        0,
        94,
        minimum,
        minimum + 1,
        2**30 + 3,
        -1,
        1,
        127,
    ]
    denominators = [128, -94, 128, -128, -128, -128, 3, -3, -129, 2, -2, -127]
    remainders = []
    for index, a in enumerate(numerators):
        b = denominators[(index // width) * width if broadcast else index]
        remainders.append(a - _quotient(a, b) * b)
    wanted = []
    for start in range(0, len(remainders), width):
        wanted.extend(remainders[start : start + width] * 2)
    _execute_integer_case(
        tmp_path,
        source,
        descriptor,
        package,
        target,
        numerators,
        denominators,
        wanted,
        dtype,
        dtype,
    )


def test_hlsl_vector_scalar_arithmetic_matches_original_metal(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required vector arithmetic")
    if sys.platform not in {"darwin", "win32"}:
        pytest.skip("HLSL vector conversion is compared with its original Metal source")
    target = "metal" if sys.platform == "darwin" else "directx"
    source = """#include <metal_stdlib>
using namespace metal;
kernel void products(device long* numerators [[buffer(0)]],
                     device uint* denominators [[buffer(1)]],
                     device long* outputs [[buffer(2)]],
                     uint tid [[thread_position_in_grid]]) {
    long a = numerators[tid];
    uint b = denominators[tid];
    long2 wide = long2(a, a) / b;
    uint2 narrow = uint2(b, b) / a;
    outputs[tid * 6u] = wide.x;
    outputs[tid * 6u + 1u] = wide.y;
    outputs[tid * 6u + 2u] = narrow.x;
    outputs[tid * 6u + 3u] = narrow.y;
    outputs[tid * 6u + 4u] = a >> 1u;
    outputs[tid * 6u + 5u] = b >> int64_t(1);
}
"""
    source, descriptor, package = _package(
        tmp_path, target, "int64_t", (1, 1, 1), source=source, software_subgroups=False
    )
    numerators = [-94, -128, -257, 94, 2**31, -(2**31), 2**32 + 1, -(2**40) + 1]
    denominators = [128, 128, 128, 128, 2**32 - 1, 2**32 - 1, 2**32 - 1, 3]
    wanted = []
    for a, b in zip(numerators, denominators):
        wanted.extend(
            [_quotient(a, b)] * 2 + [b // (a & 0xFFFFFFFF)] * 2 + [a >> 1, b >> 1]
        )
    _execute_integer_case(
        tmp_path,
        source,
        descriptor,
        package,
        target,
        numerators,
        denominators,
        wanted,
        "int64",
        "uint32",
    )


def test_wide_integer_native_gate_is_required_on_every_target():
    from tools import ci_coverage

    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/mlx-portable-host.yml").read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate mixed-width integer arithmetic"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "test_wide_integer_arithmetic.py" in step
    assert "if:" not in step and "continue-on-error" not in step
    assert "--timeout-seconds 180" in step and "--junitxml" in step
    for event in ("pull_request", "push"):
        assert (
            "tests/test_translator/test_wide_integer_arithmetic.py"
            in ci_coverage.workflow_event_path_filters(workflow, event)
        )
