"""Arrays of arrays keep their shape through CrossGL.

CrossGL nests ``T a[2][3]`` as an array of 2 arrays of 3 ``T`` (``ArrayType(ArrayType(T, 3), 2)``), the reading of
C, GLSL 4.60 section 4.1.9, HLSL and MSL: the first suffix is the outermost dimension, ``a[i]`` is a ``T[3]`` and
``a.length()`` is 2. A declarator's suffixes enclose its type's own, so GLSL ``float[3] c[2]`` and MSL
``array<float, 3> c[2]`` are the same type. Every emitter declares that shape, SPIR-V per-vertex stage arrays of
arrays keep the per-vertex dimension outermost, and arrays of arrays of resources become one-dimensional Vulkan
descriptor arrays.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.ast import ArrayType
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics import DescriptorArrayUnsupportedError
from tests.spirv_friendly_names import friendly_spirv

GRID_CGL = """
shader main {
    compute {
        layout(local_size_x = 1) in;
        buffer int data[];
        void main() {
            float a[2][3];
            float[2][3] b;
            float[3] c[2];
            a[1][2] = 1.0;
            b[1][2] = 2.0;
            c[1][2] = 3.0;
            data[0] = a.length();
            data[1] = a[0].length();
            data[2] = int(a[1][2] + b[1][2] + c[1][2]);
        }
    }
}
"""


def _translate(tmp_path, source, name, backend, **options):
    path = tmp_path / name
    path.write_text(source)
    if backend == "vulkan":
        options["format_output"] = False
        return friendly_spirv(crosstl.translate(str(path), backend=backend, **options))
    return crosstl.translate(str(path), backend=backend, **options)


def _shape(type_node):
    sizes = []
    while isinstance(type_node, ArrayType):
        size = type_node.size
        sizes.append(getattr(size, "value", size))
        type_node = type_node.element_type
    return sizes


def _parse(code):
    return Parser(Lexer(code).tokens).parse()


def _spirv_val(tmp_path, code):
    if shutil.which("spirv-as") is None or shutil.which("spirv-val") is None:
        return
    asm = tmp_path / "out.spvasm"
    asm.write_text(code)
    binary = tmp_path / "out.spv"
    for command in (
        ["spirv-as", "--target-env", "vulkan1.1", str(asm), "-o", str(binary)],
        ["spirv-val", "--target-env", "vulkan1.1", str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr


def test_first_suffix_is_the_outermost_dimension():
    ast = _parse("""
    shader S {
        struct Cell { vec2 w[2][3]; float[3] k[2]; };
        float g[2][3];
        in float weights @location(1)[][4];
        compute {
            void main(float p[4][5]) { float[2][3] b; }
        }
    }
    """)
    cell = next(s for s in ast.structs if s.name == "Cell")
    assert [_shape(m.member_type) for m in cell.members] == [[2, 3], [2, 3]]
    by_name = {v.name: v for v in ast.global_variables}
    assert _shape(by_name["g"].var_type) == [2, 3]
    # The per-vertex dimension of a GLSL ``in float weights[][4]`` stays the outer, unsized one.
    assert _shape(by_name["weights"].var_type) == [None, 4]
    entry = next(iter(ast.stages.values())).entry_point
    assert _shape(entry.parameters[0].param_type) == [4, 5]
    local = entry.body.statements[0]
    assert _shape(local.var_type) == [2, 3]


@pytest.mark.parametrize(
    ("backend", "declaration"),
    [
        ("opengl", "float a[2][3];"),
        ("directx", "float a[2][3];"),
        ("metal", "float a[2][3];"),
        ("slang", "float a[2][3];"),
        ("cuda", "float a[2][3];"),
        ("hip", "float a[2][3];"),
        ("wgsl", "var a: array<array<f32, 3>, 2>;"),
        ("rust", "let mut a: [[f32; 3]; 2]"),
        ("mojo", "var a = InlineArray[InlineArray[Float32, 3], 2]"),
    ],
)
def test_every_emitter_declares_two_arrays_of_three(tmp_path, backend, declaration):
    # Unformatted output does not depend on whether clang-format is installed.
    code = _translate(tmp_path, GRID_CGL, "grid.cgl", backend, format_output=False)
    compact = re.sub(r"\s+", " ", code)
    assert declaration in compact
    for name in ("b", "c"):
        assert declaration.replace(" a", f" {name}", 1) in compact


def test_spirv_array_of_arrays_and_lengths(tmp_path):
    code = _translate(tmp_path, GRID_CGL, "grid.cgl", "vulkan")
    inner = re.search(r"%(\w+) = OpTypeArray %float %int_3\n", code).group(1)
    assert re.search(rf"= OpTypeArray %{inner} %int_2\n", code)
    stores = re.findall(r"OpStore %\w+ %int_(\d)\n", code)
    # a.length() is 2 and a[0].length() is 3.
    assert stores[:2] == ["2", "3"]
    _spirv_val(tmp_path, code)


@pytest.mark.parametrize("backend", ["directx", "metal"])
def test_array_lengths_count_the_dimension_they_name(tmp_path, backend):
    code = _translate(tmp_path, GRID_CGL, "grid.cgl", backend)
    assert re.search(r"\b2\)?\)?;\n", code.split("float c[2][3];")[1])
    body = code.split("float c[2][3];")[1]
    assert body.index("2") < body.index("3")


def test_glsl_type_suffix_inside_declarator_suffix(tmp_path):
    source = """#version 450
layout(local_size_x = 1) in;
struct S { float m[2][3]; float[3] n[2]; };
layout(std430, binding = 0) buffer B { float data[]; };
void main() {
    float[3] c[2];
    S s;
    c[1][2] = 4.0;
    s.n[1][2] = 5.0;
    data[0] = c[1][2] + s.n[1][2];
}
"""
    crossgl = _translate(tmp_path, source, "suffix.comp", "cgl")
    assert "float n[2][3];" in crossgl
    assert "float c[2][3];" in crossgl
    hlsl = _translate(tmp_path, source, "suffix.comp", "directx")
    assert "float c[2][3];" in hlsl


def test_metal_nested_array_templates_import_outer_first(tmp_path):
    source = """#include <metal_stdlib>
using namespace metal;
struct S { array<array<float, 3>, 2> m; array<float, 3> n[2]; };
kernel void k(device float *out [[buffer(0)]], uint id [[thread_position_in_grid]]) {
    array<array<float, 3>, 2> a;
    array<float, 3> c[2];
    S s;
    a[1][2] = 1.0; c[1][2] = 3.0; s.m[1][2] = 4.0; s.n[1][2] = 5.0;
    out[0] = a[1][2] + c[1][2] + s.m[1][2] + s.n[1][2];
}
"""
    crossgl = _translate(tmp_path, source, "nested.metal", "cgl")
    for name in ("m", "n", "a", "c"):
        assert f"float[2][3] {name};" in crossgl
    hlsl = _translate(tmp_path, source, "nested.metal", "directx")
    assert "float a[2][3];" in hlsl
    assert "float c[2][3];" in hlsl


TESC_GLSL = """#version 450
layout(vertices = 3) out;
layout(location = 0) in vec3 inNormal[];
layout(location = 1) in float inWeights[][4];
layout(location = 0) out vec3 outNormal[];
layout(location = 5) out float outPatch[][10];
void main() {
    outNormal[gl_InvocationID] = inNormal[gl_InvocationID];
    for (int i = 0; i < 10; i++)
        outPatch[gl_InvocationID][i] = inWeights[gl_InvocationID][i % 4] * float(i);
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
    if (gl_InvocationID == 0) {
        gl_TessLevelOuter[0] = 1.0; gl_TessLevelOuter[1] = 1.0; gl_TessLevelOuter[2] = 1.0;
        gl_TessLevelInner[0] = 1.0;
    }
}
"""

TESE_GLSL = """#version 450
layout(triangles, equal_spacing, ccw) in;
layout(location = 0) in vec3 inNormal[];
layout(location = 5) in float inPatch[][10];
layout(location = 0) out vec3 outNormal;
layout(location = 1) out float outValue;
void main() {
    outNormal = inNormal[0] * gl_TessCoord.x + inNormal[1] * gl_TessCoord.y + inNormal[2] * gl_TessCoord.z;
    outValue = inPatch[0][9] + inPatch[2][3];
    gl_Position = gl_in[0].gl_Position;
}
"""


def test_glsl_per_vertex_arrays_of_arrays_keep_the_vertex_dimension_outermost(
    tmp_path,
):
    control = _translate(tmp_path, TESC_GLSL, "aoa.tesc", "opengl")
    assert "in float inWeights[][4];" in control
    assert "out float outPatch[3][10];" in control
    evaluation = _translate(tmp_path, TESE_GLSL, "aoa.tese", "opengl")
    assert "in float inPatch[][10];" in evaluation


@pytest.mark.parametrize(
    ("source", "name", "variable", "inner"),
    [
        (TESC_GLSL, "aoa.tesc", "inWeights", 4),
        (TESE_GLSL, "aoa.tese", "inPatch", 10),
    ],
)
def test_spirv_per_vertex_arrays_of_arrays_are_sized_by_the_patch(
    tmp_path, source, name, variable, inner
):
    code = _translate(tmp_path, source, name, "vulkan")
    pointer = re.search(rf"%{variable} = OpVariable %(\w+) Input", code).group(1)
    pointee = re.search(rf"%{pointer} = OpTypePointer Input %(\w+)", code).group(1)
    element, length = re.search(
        rf"%{pointee} = OpTypeArray %(\w+) %(\w+)\n", code
    ).groups()
    # gl_MaxPatchVertices outer elements of float[inner], as glslang declares them.
    assert length == "int_32"
    assert re.search(rf"%{element} = OpTypeArray %float %int_{inner}\n", code)
    _spirv_val(tmp_path, code)


def test_hlsl_resource_arrays_of_arrays_take_every_register(tmp_path):
    source = """
Texture2D<float4> atlas[2][4] : register(t0);
Texture2D<float4> after;
SamplerState samp : register(s0);
float4 main(float2 uv : TEXCOORD0) : SV_Target0 {
    return atlas[1][3].Sample(samp, uv) + after.Sample(samp, uv);
}
"""
    hlsl = _translate(tmp_path, source, "atlas.ps.hlsl", "directx")
    assert "Texture2D atlas[2][4] : register(t0);" in hlsl
    # The 8 descriptors of atlas take t0..t7, so the next texture starts at t8.
    assert "Texture2D after : register(t8);" in hlsl


ATLAS_GLSL = """#version 450
#extension GL_EXT_nonuniform_qualifier : require
layout(set = 0, binding = 0) uniform sampler2D atlas[2][4];
layout(location = 0) in vec2 uv;
layout(location = 1) flat in int page;
layout(location = 0) out vec4 color;
void main() {
    color = texture(atlas[1][2], uv) + texture(atlas[page][nonuniformEXT(page + 1)], uv);
}
"""


def test_spirv_flattens_arrays_of_arrays_of_samplers_row_major(tmp_path):
    code = _translate(tmp_path, ATLAS_GLSL, "atlas.frag", "vulkan")
    pointer = re.search(r"%atlas = OpVariable %(\w+) UniformConstant", code).group(1)
    pointee = re.search(rf"%{pointer} = OpTypePointer UniformConstant %(\w+)", code)
    assert re.search(rf"%{pointee.group(1)} = OpTypeArray %\w+ %int_8\n", code)
    # atlas[page][nonuniformEXT(page + 1)] selects descriptor page * 4 + page + 1, still non-uniform.
    assert "OpIMul %uint" in code and "%uint_4" in code
    chains = re.findall(r"%(\w+) = OpAccessChain %\w+ %atlas %(\w+)\n", code)
    assert len(chains) == 2
    non_uniform = set(re.findall(r"OpDecorate %(\w+) NonUniform\n", code))
    assert chains[1][1] in non_uniform and chains[1][0] in non_uniform
    _spirv_val(tmp_path, code)
    crossgl = _translate(tmp_path, ATLAS_GLSL, "atlas.frag", "cgl")
    assert "sampler2D atlas[2][4]" in crossgl


def test_spirv_refuses_a_sub_array_of_a_descriptor_grid(tmp_path):
    source = """#version 450
layout(set = 0, binding = 0) uniform sampler2D atlas[2][4];
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
vec4 first(sampler2D row[4]) { return texture(row[0], uv); }
void main() { color = first(atlas[1]); }
"""
    with pytest.raises(DescriptorArrayUnsupportedError, match="every dimension"):
        _translate(tmp_path, source, "partial.frag", "vulkan")


def test_spirv_sample_level_converts_an_integer_lod(tmp_path):
    source = """
RWStructuredBuffer<float> results : register(u0);
Texture2D<float4> tex : register(t0);
SamplerState samp : register(s0);
[numthreads(1, 1, 1)]
void CSMain(uint3 tid : SV_GroupThreadID) {
    results[0] = tex.SampleLevel(samp, float2(0.5, 0.5), 0).x;
}
"""
    code = _translate(tmp_path, source, "lod.comp.hlsl", "vulkan")
    converted = re.search(r"%(\w+) = OpConvertSToF %float %int_0\n", code).group(1)
    assert re.search(
        rf"OpImageSampleExplicitLod %v4float %\w+ %\w+ Lod %{converted}\n", code
    )
    _spirv_val(tmp_path, code)
