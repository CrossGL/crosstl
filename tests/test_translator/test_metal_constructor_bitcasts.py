"""Nested numeric constructors retain their native Metal bitcast widths."""

import os
import shutil
import sys
from pathlib import Path

import pytest

from crosstl import translate
from crosstl.project import translate_project
from crosstl.translator.ast import (
    ConstructorNode,
    FunctionCallNode,
    FunctionNode,
    IdentifierNode,
    LiteralNode,
    PrimitiveType,
    VectorType,
)
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from tests.test_translator.test_boolean_buffer_runtime import _bound_values, _request
from tests.test_translator.test_half_buffer_runtime import _validate_half
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_bfloat_vectors import REQUIRE_ENV
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_opengl_bfloat_conversion import _round
from tests.test_translator.test_software_subgroup_product import _package
from tests.test_translator.test_wide_integer_bfloat import (
    _payload,
    _round_integer,
    _values,
)


def _constructor(dtype, kind):
    argument = LiteralNode("1.0", "float")
    if kind == "constructor":
        return ConstructorNode(PrimitiveType(dtype), [argument])
    return FunctionCallNode(IdentifierNode(dtype), [argument])


@pytest.mark.parametrize("kind", ("constructor", "call"))
@pytest.mark.parametrize(
    "dtype,native",
    (
        ("bfloat", "bfloat"),
        ("bfloat16", "bfloat"),
        ("bfloat16_t", "bfloat"),
        ("Scalar", "bfloat"),
        ("bfloat2", "bfloat2"),
        ("bfloat16vec3", "bfloat3"),
        ("bfloat16vec4", "bfloat4"),
        ("Lanes", "bfloat2"),
        ("half", "half"),
        ("vec3<f16>", "half3"),
        ("float", "float"),
        ("mat3", "float3x3"),
    ),
)
def test_numeric_constructor_result_matches_emission(dtype, native, kind):
    generator = MetalCodeGen()
    generator.metal_type_aliases = {"Scalar": "bfloat16_t", "Lanes": "bfloat2"}
    result_type = generator.expression_result_type(_constructor(dtype, kind))
    assert result_type is not None
    assert generator.map_type(result_type) == native


@pytest.mark.parametrize("width", (2, 3, 4))
def test_structured_vector_constructor_retains_component_and_storage_width(width):
    generator = MetalCodeGen()
    expression = ConstructorNode(
        VectorType(PrimitiveType("bfloat16"), width), [LiteralNode("1.0", "float")]
    )
    result_type = generator.expression_result_type(expression)
    assert generator.map_type(result_type) == f"bfloat{width}"
    info = generator.metal_explicit_bitcast_type_info(result_type)
    assert info["componentWidth"] == 16
    assert info["lanes"] == width
    assert info["totalWidth"] == 16 * (4 if width == 3 else width)


@pytest.mark.parametrize("kind", ("constructor", "call"))
def test_nested_constructor_rejects_unequal_bitcast_width(kind):
    with pytest.raises(ValueError, match="total widths are 32 and 16 bits"):
        MetalCodeGen().generate_metal_explicit_as_type_call(
            "as_type<uint>", [_constructor("bfloat", kind)]
        )


@pytest.mark.parametrize("dtype", (None, "", "Unresolved", "bool", "MissingAlias"))
def test_unknown_bitcast_storage_is_not_inferred_as_float(dtype):
    generator = MetalCodeGen()
    generator.metal_type_aliases["MissingAlias"] = None
    assert generator.metal_explicit_bitcast_type_info(dtype) is None


@pytest.mark.parametrize("target", ("uint", "ushort"))
def test_unknown_call_remains_an_invalid_bitcast_operand(target):
    call = FunctionCallNode(IdentifierNode("unresolved"), [])
    with pytest.raises(ValueError, match="statically known numeric"):
        MetalCodeGen().generate_metal_explicit_as_type_call(
            f"as_type<{target}>", [call]
        )


def test_constructor_lookup_does_not_override_source_function_types():
    generator = MetalCodeGen()
    generator.metal_type_aliases["Value"] = "bfloat"
    call = FunctionCallNode(IdentifierNode("Value"), [IdentifierNode("unknown")])
    generator.function_return_types["Value"] = "float"
    assert generator.expression_result_type(call) == "float"
    generator.function_overloads_by_name["Value"] = [
        FunctionNode("Value", PrimitiveType("float"), [object()]),
        FunctionNode("Value", PrimitiveType("int"), [object()]),
    ]
    assert generator.expression_result_type(call) is None


@pytest.mark.parametrize(
    "expression,reason",
    (
        ("unresolved(float(output[0]))", "statically known numeric"),
        ("bfloat16(1.0)", "total widths are 32 and 16 bits"),
    ),
)
def test_project_retains_bitcast_diagnostics_without_an_artifact(
    tmp_path, expression, reason
):
    (tmp_path / "invalid.cgl").write_text(
        f"""shader InvalidBitcast {{
      compute {{
        void main(RWStructuredBuffer<uint> output @buffer(0)) {{
          output[0] = as_type<uint>({expression});
        }}
      }}
    }}""",
        encoding="utf-8",
    )
    payload = translate_project(tmp_path, targets=["metal"], output_dir="out").to_json()
    assert payload["summary"]["translatedCount"] == 0
    assert payload["summary"]["failedCount"] == 1
    (diagnostic,) = payload["diagnostics"]
    assert diagnostic["code"] == "project.translate.failed"
    assert diagnostic["severity"] == "error"
    assert diagnostic["target"] == "metal"
    assert diagnostic["location"]["file"] == "invalid.cgl"
    assert reason in diagnostic["message"]
    (artifact,) = payload["artifacts"]
    assert artifact["status"] == "failed"
    assert not (tmp_path / artifact["path"]).exists()


def _conversion_source(dtype):
    storage = "uint" if dtype == "float" else dtype
    operand = (
        "as_type<float>(values[{index}])" if dtype == "float" else "values[{index}]"
    )
    direct = operand.format(index="cursor++")
    value = operand.format(index="i")
    return f"""#include <metal_stdlib>
using namespace metal;
typedef bfloat Base;
using Value = Base;
kernel void constructor_bits(const device {storage}* values [[buffer(0)]],
                             device uint* results [[buffer(1)]],
                             uint i [[thread_position_in_grid]]) {{
    uint cursor = i;
    bfloat named = bfloat({value});
    results[6u * i + 1u] = uint(as_type<ushort>(bfloat({direct})));
    results[6u * i + 2u] = uint(as_type<ushort>(static_cast<bfloat>({value})));
    results[6u * i + 3u] = uint(as_type<ushort>(Value({value})));
    results[6u * i + 4u] = uint(as_type<ushort>(bfloat(float({value}))));
    results[6u * i + 5u] = uint(as_type<ushort>(named));
    results[6u * i + 6u] = cursor - i;
}}
"""


@pytest.mark.parametrize("dtype", ("float", "long", "ulong"))
def test_nested_bfloat_bitcasts_compile_from_source_and_saved_intermediate(
    tmp_path, dtype
):
    source = tmp_path / "source.metal"
    source.write_text(_conversion_source(dtype), encoding="utf-8")
    generated = translate(str(source), backend="metal", format_output=False)
    canonical = tmp_path / "source.cgl"
    canonical.write_text(
        translate(str(source), backend="crossgl", format_output=False), encoding="utf-8"
    )
    replay = translate(str(canonical), backend="metal", format_output=False)
    assert generated == replay
    assert generated.count("cursor++") == 1
    if not shutil.which("xcrun"):
        if os.environ.get(REQUIRE_ENV) == "1":
            pytest.fail("Metal compilation is required")
        pytest.skip("Metal compiler is unavailable")
    _compile(
        generated,
        "metal",
        tmp_path,
        metal_compile_flags=("-std=metal3.1", "-fno-fast-math"),
    )


def _identity_source(width):
    dtype = "Scalar" if width == 1 else f"vec<Scalar, {width}>"
    bits = "ushort" if width == 1 else f"ushort{4 if width == 3 else width}"
    stride = 4 * width + 1
    stores = []
    for index, name in enumerate(("direct", "cast_value", "named_bits", "returned")):
        for lane in range(width):
            expression = name if width == 1 else f"{name}[{lane}]"
            stores.append(
                f"results[{stride}u * i + {index * width + lane + 1}u] = uint({expression});"
            )
    return f"""#include <metal_stdlib>
using namespace metal;
typedef bfloat Scalar;
using Value = {dtype};
Value identity(Value value) {{ return value; }}
kernel void constructor_bits(const device uint* values [[buffer(0)]],
                             device uint* results [[buffer(1)]],
                             uint i [[thread_position_in_grid]]) {{
    uint cursor = i;
    Value named = Value(as_type<bfloat>(ushort(values[i])));
    {bits} direct = as_type<{bits}>(Value(as_type<bfloat>(ushort(values[cursor++]))));
    {bits} cast_value = as_type<{bits}>(static_cast<Value>(named));
    {bits} named_bits = as_type<{bits}>(named);
    {bits} returned = as_type<{bits}>(identity(Value(as_type<bfloat>(ushort(values[i])))));
    {chr(10).join(stores)}
    results[{stride}u * i + {stride}u] = cursor - i;
}}
"""


def _require_native():
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required constructor bitcasts")
    assert sys.platform == "darwin", "Metal constructor execution requires macOS"


def _execute_case(tmp_path, source, values, expected, dtype):
    guard = 0x50B17CA5
    expected = [guard, *expected, guard]
    _, descriptor, package = _package(
        tmp_path, "metal", "uint", (1, 1, 1), source=source, software_subgroups=False
    )
    inputs = {
        "values": {"dtype": dtype, "shape": [len(values)], "values": values},
        "results": {
            "dtype": "uint32",
            "shape": [len(expected)],
            "values": [guard] * len(expected),
        },
    }
    outputs = {"results": {**inputs["results"], "values": expected}}
    request = _request(descriptor, package, inputs, outputs, len(values))
    _execute(
        request,
        _bound_values(descriptor, outputs),
        tmp_path,
        original_source=source,
        original_entry="constructor_bits",
        metal_compile_flags=("-std=metal3.1",),
        validate=_validate_half,
    )


@pytest.mark.parametrize("width", (1, 2, 3, 4))
@pytest.mark.parametrize("start", (0, 32768))
def test_nested_constructor_bitcasts_preserve_every_bfloat_payload(
    tmp_path, width, start
):
    _require_native()
    values = list(range(start, start + 32768))
    expected = []
    for value in values:
        expected.extend([value] * (4 * width) + [1])
    _execute_case(tmp_path, _identity_source(width), values, expected, "uint32")


@pytest.mark.parametrize("dtype", ("float", "long", "ulong"))
def test_nested_conversions_preserve_direct_rounding_and_evaluation(tmp_path, dtype):
    _require_native()
    if dtype == "float":
        values = [
            sign | (exponent << 23) | (fraction << 16) | low
            for sign in (0, 0x80000000)
            for exponent in range(255)
            for fraction in (0, 1, 127)
            for low in (0, 0x7FFF, 0x8000, 0x8001, 0xFFFF)
        ] + [0x7F800000, 0xFF800000]
        direct = [_round(value) >> 16 for value in values]
        mediated = direct
    else:
        values = _values(dtype == "long")
        direct = [_payload(value) for value in values]
        mediated = [_payload(_round_integer(value, 24)) for value in values]
    expected = []
    for rounded, wide in zip(direct, mediated):
        expected.extend((rounded, rounded, rounded, wide, rounded, 1))
    _execute_case(
        tmp_path,
        _conversion_source(dtype),
        values,
        expected,
        {"float": "uint32", "long": "int64", "ulong": "uint64"}[dtype],
    )


def test_nested_constructor_native_gate_is_required():
    from tools import ci_coverage

    step = ci_coverage.workflow_step_section(
        Path(".github/workflows/demo-project-testing.yml").read_text(),
        "Validate bfloat vector round trips",
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "tests/test_translator/test_metal_constructor_bitcasts.py" in step
    assert "--timeout-seconds" in step and "--junitxml" in step
    assert "continue-on-error" not in step and "if:" not in step
