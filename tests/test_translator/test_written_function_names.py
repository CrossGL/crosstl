"""Built-in functions of another language that reach HLSL or Metal output as calls are refused.

The importers carry each source language's built-in functions into the IR as plain calls. A call the HLSL or Metal
generator had no lowering for was written unchanged (Metal ``lit(...)``, HLSL ``absdiff(...)``), which the target
compiler rejects as an undeclared identifier. The check runs on the rendered output, so a name the generator
lowers, a method call and a function the program declares are left alone.
"""

import json
import re
from pathlib import Path

import pytest

import crosstl
from crosstl.translator.codegen.hlsl_refusals import DirectXLoweringRefused
from crosstl.translator.codegen.metal_refusals import MetalLoweringRefused
from crosstl.translator.codegen.written_function_names import (
    HLSL_CALLABLE_FUNCTION_NAMES,
    HLSL_PAYLOAD_ACCESS_QUALIFIER,
    METAL_CALLABLE_FUNCTION_NAMES,
    written_foreign_call_names,
)
from crosstl.translator.semantics.builtin_function_names import (
    HLSL_INTRINSIC_FUNCTION_NAMES,
    METAL_LIBRARY_FUNCTION_NAMES,
)

INVENTORIES = Path(__file__).resolve().parents[1] / "conformance" / "inventory" / "data"

COMPUTE = """
shader S {
    compute {
        layout(local_size_x = 1) in;
        buffer float o[];
        %s
        void main() {
            %s
        }
    }
}
"""


def _translate(tmp_path, body, backend, declarations=""):
    path = tmp_path / "s.cgl"
    path.write_text(COMPUTE % (declarations, body))
    return crosstl.translate(str(path), backend=backend)


@pytest.mark.parametrize(
    ("language", "kind", "group"),
    [
        ("hlsl", "intrinsic", HLSL_INTRINSIC_FUNCTION_NAMES),
        ("metal", "library-function", METAL_LIBRARY_FUNCTION_NAMES),
    ],
)
def test_free_function_sets_are_the_inventory_free_functions(language, kind, group):
    items = json.loads((INVENTORIES / f"{language}.json").read_text())["items"]
    names = set()
    for item in items:
        if item.get("category") != "builtin-function" or item.get("kind") != kind:
            continue
        name = re.split(r"::|\.", item["name"].split("(")[0])[-1].split("<")[0].strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            names.add(name)
    assert group == frozenset(names)


@pytest.mark.parametrize(
    ("body", "name"),
    [
        ("vec4 l = lit(o[1], o[2], o[3]); o[0] = l.y;", "lit"),
        (
            "ivec4 c = D3DCOLORtoUBYTE4(vec4(o[1])); o[0] = float(c.x);",
            "D3DCOLORtoUBYTE4",
        ),
        ("o[0] = noise(o[1]);", "noise"),
        ("o[0] = acosf(o[1]);", "acosf"),
    ],
)
def test_metal_refuses_a_foreign_builtin_call(tmp_path, body, name):
    with pytest.raises(
        MetalLoweringRefused, match=rf"call to '{name}', a built-in function"
    ):
        _translate(tmp_path, body, "metal")


@pytest.mark.parametrize(
    ("body", "name"),
    [
        ("o[0] = absdiff(o[1], o[2]);", "absdiff"),
        ("o[0] = median3(o[1], o[2], o[3]);", "median3"),
        ("o[0] = simd_sum(o[1]);", "simd_sum"),
        ("o[0] = acosf(o[1]);", "acosf"),
        ("o[0] = float(alignof(o[1]));", "alignof"),
    ],
)
def test_hlsl_refuses_a_foreign_builtin_call(tmp_path, body, name):
    with pytest.raises(
        DirectXLoweringRefused, match=rf"call to '{name}', a built-in function"
    ):
        _translate(tmp_path, body, "directx")


@pytest.mark.parametrize("backend", ["metal", "directx"])
def test_a_function_the_program_declares_is_not_foreign(tmp_path, backend):
    declarations = "float noise(float x) { return x * 0.5; }"
    output = _translate(tmp_path, "o[0] = noise(o[1]);", backend, declarations)
    assert re.search(r"\bnoise\s*\(", output)


@pytest.mark.parametrize(
    ("backend", "body", "spelling"),
    [
        ("metal", "o[0] = fract(o[1]);", "fract("),
        ("metal", "o[0] = inversesqrt(o[1]);", "rsqrt("),
        ("directx", "o[0] = fract(o[1]);", "frac("),
        ("directx", "o[0] = mix(o[1], o[2], 0.5);", "lerp("),
    ],
)
def test_a_lowered_builtin_is_not_refused(tmp_path, backend, body, spelling):
    assert spelling in _translate(tmp_path, body, backend)


def test_metal_text_outside_code_and_qualified_calls_are_not_calls():
    code = """
// lit(a, b, c) in a comment
constant char *message = "msad4(x)";
struct Out { float4 color [[color(0)]]; };
kernel void main0(texture2d<float> t [[texture(0)]], uint i [[thread_position_in_grid]]) {
    float a = metal::fast::sin(1.0);
    float4 v = t.read(uint2(i, 0));
}
"""
    assert written_foreign_call_names(code, METAL_CALLABLE_FUNCTION_NAMES) == []


def test_hlsl_methods_and_attributes_are_not_free_calls():
    code = """
Texture2D<float4> t : register(t0);
SamplerState s : register(s0);
RWStructuredBuffer<float> o : register(u0);
[numthreads(1, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    uint w, h;
    t.GetDimensions(w, h);
    o[0] = t.SampleLevel(s, float2(0.5, 0.5), 0).x;
}
"""
    assert written_foreign_call_names(code, HLSL_CALLABLE_FUNCTION_NAMES) == []


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("float4 v = GetDimensions(t);", ["GetDimensions"]),
        ("float v = sample(t, s, uv).x;", ["sample"]),
        ("float v = Sample(t, s, uv).x;", ["Sample"]),
    ],
)
def test_a_method_called_as_a_free_function_is_foreign(code, expected):
    assert written_foreign_call_names(code, METAL_CALLABLE_FUNCTION_NAMES) == expected


@pytest.mark.parametrize(
    "declaration",
    [
        "static inline float lit(float a) { return a; }",
        "template <typename T>\nT lit(T a) { return a; }",
        "#define lit(a) (a)",
        "struct lit { float a; };",
        "[noinline] float lit(float a) { return a; }",
    ],
)
def test_a_name_the_output_declares_is_not_foreign(declaration):
    code = declaration + "\nfloat f(float x) { return lit(x); }\n"
    assert written_foreign_call_names(code, METAL_CALLABLE_FUNCTION_NAMES) == []


def test_a_call_after_return_is_not_a_declaration():
    code = "float f(float x) {\n    return lit(x);\n}\n"
    assert written_foreign_call_names(code, METAL_CALLABLE_FUNCTION_NAMES) == ["lit"]


def test_hlsl_payload_access_qualifiers_are_not_calls():
    code = (
        "struct [raypayload] Payload {\n"
        "    float3 color : read(caller, closesthit) : write(closesthit, miss);\n"
        "    float hit : read() : write(anyhit);\n"
        "};\n"
    )
    assert written_foreign_call_names(code, HLSL_CALLABLE_FUNCTION_NAMES) == [
        "read",
        "write",
    ]
    assert (
        written_foreign_call_names(
            code, HLSL_CALLABLE_FUNCTION_NAMES, (HLSL_PAYLOAD_ACCESS_QUALIFIER,)
        )
        == []
    )


def test_a_read_call_that_is_not_a_payload_qualifier_is_still_found():
    code = "float f(bool c) { return c ? 1.0 : read(x); }\n"
    assert written_foreign_call_names(
        code, HLSL_CALLABLE_FUNCTION_NAMES, (HLSL_PAYLOAD_ACCESS_QUALIFIER,)
    ) == ["read"]
