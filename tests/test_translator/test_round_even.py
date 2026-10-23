"""roundEven, the hub's ties-to-even rounding, maps to each target's ties-to-even function."""

import re

import pytest

import crosstl


@pytest.mark.parametrize(
    "backend, expected",
    [
        ("directx", "round(a)"),
        ("slang", "round(a)"),
        ("wgsl", "round(a)"),
        ("metal", "rint(a)"),
        ("cuda", "rintf(a)"),
        ("hip", "rintf(a)"),
        ("opengl", "roundEven(a)"),
    ],
)
def test_round_even_maps_to_each_targets_ties_to_even_rounding(
    tmp_path, backend, expected
):
    path = tmp_path / "round.cgl"
    path.write_text(
        "shader S {\n    compute {\n        void main() {\n            float a = 2.5;\n"
        "            float b = roundEven(a);\n        }\n    }\n}\n"
    )
    out = crosstl.translate(str(path), backend=backend, format_output=False)
    assert expected in out
    if backend != "opengl":
        assert "roundEven" not in out


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_round_even_keeps_double_precision_and_vector_operands(tmp_path, backend):
    path = tmp_path / "round.cgl"
    path.write_text(
        "shader S {\n    compute {\n        void main() {\n            double a = 2.5;\n"
        "            double b = roundEven(a);\n            vec4 c = vec4(2.5);\n"
        "            vec4 d = roundEven(c);\n        }\n    }\n}\n"
    )
    out = crosstl.translate(str(path), backend=backend, format_output=False)
    assert "rint(a)" in out
    assert "rintf(a)" not in out
    assert "rintf(c)" not in out
    assert "rintf(value.x)" in out


HLSL_ROUND = """
RWStructuredBuffer<float> output : register(u0);
[numthreads(4, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    output[id.x] = round(output[id.x]);
}
"""


@pytest.mark.parametrize(
    "backend, expected",
    [
        ("cgl", "roundEven("),
        ("directx", "round("),
        ("slang", "round("),
        ("wgsl", "round("),
        ("metal", "rint("),
        ("cuda", "rintf("),
        ("hip", "rintf("),
        ("opengl", "roundEven("),
    ],
)
def test_hlsl_round_imports_as_ties_to_even_rounding(tmp_path, backend, expected):
    # HLSL round(2.5) is 2 (DXC lowers it to Round_ne); Metal round and CUDA roundf give 3.
    path = tmp_path / "round.hlsl"
    path.write_text(HLSL_ROUND)
    out = crosstl.translate(str(path), backend=backend, format_output=False)
    assert expected in out
    if backend in {"metal", "cuda", "hip"}:
        assert "round" not in out


def test_hlsl_user_function_named_round_is_not_renamed(tmp_path):
    path = tmp_path / "round.hlsl"
    path.write_text(
        "RWStructuredBuffer<float> output : register(u0);\n"
        "float round(float x, float step) { return floor(x / step) * step; }\n"
        "[numthreads(4, 1, 1)]\n"
        "void CSMain(uint3 id : SV_DispatchThreadID) {\n"
        "    output[id.x] = round(output[id.x], 0.5);\n"
        "}\n"
    )
    out = crosstl.translate(str(path), backend="cgl", format_output=False)
    assert "round(output[id.x], 0.5)" in out
    assert "roundEven" not in out


def test_hlsl_round_calls_user_function_only_when_its_arity_fits(tmp_path):
    path = tmp_path / "round.hlsl"
    path.write_text(
        "float round(float x, float step) { return floor(x / step) * step; }\n"
        "float evaluate(float x) { return round(x) + round(x, 0.5); }\n"
    )
    out = crosstl.translate(str(path), backend="cgl", format_output=False)
    assert "roundEven(x) + round(x, 0.5)" in out


def test_hlsl_round_keeps_user_function_with_default_parameter(tmp_path):
    path = tmp_path / "round.hlsl"
    path.write_text(
        "float round(float x, float step = 1.0) { return floor(x / step) * step; }\n"
        "float evaluate(float x) { return round(x); }\n"
    )
    out = crosstl.translate(str(path), backend="cgl", format_output=False)
    assert "roundEven" not in out


@pytest.mark.parametrize(
    "hlsl_type, expected",
    [
        ("int", "output[id.x] = int(rint(float(output[id.x])));"),
        ("int2", "output[id.x] = int2(rint(float2(output[id.x])));"),
        ("uint", "rint(float(output[id.x]))"),
    ],
)
def test_hlsl_round_of_integer_converts_to_float_for_metal_rint(
    tmp_path, hlsl_type, expected
):
    # MSL rint has only float and half overloads.
    path = tmp_path / "round.hlsl"
    path.write_text(
        f"RWStructuredBuffer<{hlsl_type}> output : register(u0);\n"
        "[numthreads(4, 1, 1)]\n"
        "void CSMain(uint3 id : SV_DispatchThreadID) {\n"
        f"    output[id.x] = ({hlsl_type})round(output[id.x]);\n"
        "}\n"
    )
    out = crosstl.translate(str(path), backend="metal", format_output=False)
    assert expected in out


@pytest.mark.parametrize(
    "hlsl_type, overload_type, call",
    [
        ("float", "Float32", "round_even(v)"),
        ("float3", "SIMD[DType.float32, 4]", "round_even(v)"),
        ("half", "Float16", "round_even(v)"),
        ("half2", "SIMD[DType.float16, 2]", "round_even(v)"),
        ("min16float", "Float16", "round_even(v)"),
        ("double", "Float64", "round_even(v)"),
        ("double4", "SIMD[DType.float64, 4]", "round_even(v)"),
        ("int", "Float32", "round_even((v).cast[DType.float32]())"),
        ("uint", "Float32", "round_even((v).cast[DType.float32]())"),
        ("int2", "SIMD[DType.float32, 2]", "round_even((v).cast[DType.float32]())"),
    ],
)
def test_hlsl_round_calls_an_existing_mojo_round_even_overload(
    tmp_path, hlsl_type, overload_type, call
):
    path = tmp_path / "round.hlsl"
    path.write_text(
        "RWStructuredBuffer<float> output : register(u0);\n"
        "[numthreads(4, 1, 1)]\n"
        "void CSMain(uint3 id : SV_DispatchThreadID) {\n"
        f"    {hlsl_type} v = ({hlsl_type})output[id.x];\n"
        f"    {hlsl_type} r = round(v);\n"
        "}\n"
    )
    out = crosstl.translate(str(path), backend="mojo", format_output=False)
    assert call in out
    overloads = re.findall(r"^fn round_even\(\w+: (.+?)\) -> (.+?):$", out, re.M)
    assert (overload_type, overload_type) in overloads


def test_mojo_round_even_casts_integer_buffer_elements(tmp_path):
    path = tmp_path / "round.hlsl"
    path.write_text(HLSL_ROUND.replace("<float>", "<int>"))
    out = crosstl.translate(str(path), backend="mojo", format_output=False)
    assert "round_even((output[id[0]]).cast[DType.float32]())" in out


@pytest.mark.parametrize(
    "literal, call",
    [
        ("3", "round_even(Float32(3))"),
        ("2.5", "round_even(Float32(2.5))"),
        ("-2.5", "round_even(Float32((-2.5)))"),
    ],
)
def test_mojo_round_even_spells_literal_operands_as_float32(tmp_path, literal, call):
    # A float literal converts to every round_even overload and an integer literal has no cast method.
    path = tmp_path / "round.hlsl"
    path.write_text(
        "RWStructuredBuffer<float> output : register(u0);\n"
        "[numthreads(4, 1, 1)]\n"
        "void CSMain(uint3 id : SV_DispatchThreadID) {\n"
        f"    output[id.x] = round({literal});\n"
        "}\n"
    )
    out = crosstl.translate(str(path), backend="mojo", format_output=False)
    assert call in out
    assert ".cast[" not in out.split("fn CSMain")[-1]
