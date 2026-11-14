"""Source identifiers spelled like Metal keywords reach Metal under other names.

GLSL and HLSL accept parameters, variables and members named ``kernel``,
``device``, ``thread`` or ``and``; MSL parses those as function qualifiers,
address spaces or C++ keywords (sascha-vulkan computeshader edgedetect,
emboss and sharpen pass a ``float[9] kernel`` parameter).
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.ast import MemberAccessNode
from crosstl.translator.codegen.metal_keyword_identifiers import (
    rename_metal_keyword_members,
    rename_metal_keyword_parameters,
)
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

GLSL_COMPUTE = """#version 450
layout (local_size_x = 16, local_size_y = 16) in;
layout (binding = 0, rgba8) uniform readonly image2D inputImage;
layout (binding = 1, rgba8) uniform image2D resultImage;
float conv(in float[9] kernel, in float[9] data, in float denom, in float offset)
{
    float res = 0.0;
    for (int i = 0; i < 9; ++i)
        res += kernel[i] * data[i];
    return clamp(res / denom + offset, 0.0, 1.0);
}
float twice(float vertex) { float vertex_ = vertex * 2.0; return vertex_; }
void main()
{
    float k[9];
    float d[9];
    for (int i = 0; i < 9; ++i) {
        k[i] = 1.0;
        d[i] = imageLoad(inputImage, ivec2(gl_GlobalInvocationID.xy)).r;
    }
    float r = twice(conv(k, d, 9.0, 0.0));
    imageStore(resultImage, ivec2(gl_GlobalInvocationID.xy), vec4(r));
}
"""

HLSL_COMPUTE = """RWStructuredBuffer<float> results : register(u0);
cbuffer Params : register(b0) { float thread; float delete; };
float scale(float and, float device) { float or = and * device; return or + thread; }
[numthreads(4, 1, 1)]
void main(uint3 kernel : SV_DispatchThreadID) {
    float new = scale((float)kernel.x, delete);
    float vertex = new + 1.0;
    results[kernel.x] = vertex;
}
"""

GLSL_FRAGMENT = """#version 450
layout(location = 0) in vec2 fragment;
layout(location = 0) out vec4 outColor;
layout(binding = 0) uniform UBO { float device; float and; } ubo;
void main() {
    float char = ubo.device * 0.5;
    float signed = char + ubo.and;
    outColor = vec4(fragment, signed, 1.0);
}
"""

CROSSGL = """shader main {
    struct Light { vec3 vertex; float kernel; };
    float shade(Light light, float thread) { return light.kernel * light.vertex.x * thread; }
    fragment {
        vec4 main() @location(0) {
            Light light;
            light.vertex = vec3(1.0);
            light.kernel = 2.0;
            return vec4(shade(light, 0.5));
        }
    }
}
"""


SHADOWED_COMPUTE = """#version 450
layout (local_size_x = 1, local_size_y = 1) in;
layout (binding = 0) buffer Values { float values[]; };
float conv(in float[9] kernel, in float[9] data, in float vertex)
{
    float res = 0.0;
    for (int i = 0; i < 9; ++i)
        res += kernel[i] * data[i];
    return res * vertex;
}
void main()
{
    float kernel[9];
    float data[9];
    for (int i = 0; i < 9; ++i) {
        kernel[i] = float(i);
        data[i] = values[i];
    }
    values[9] = conv(kernel, data, 2.0);
}
"""


def _translate(tmp_path, name, source):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend="metal", format_output=False)


def _compile(tmp_path, code):
    if shutil.which("xcrun") is None:
        pytest.skip("the Metal toolchain is not installed")
    found = subprocess.run(
        ["xcrun", "-sdk", "macosx", "--find", "metal"], capture_output=True
    )
    if found.returncode != 0:
        pytest.skip("the Metal toolchain is not installed")
    artifact = tmp_path / "out.metal"
    artifact.write_text(code)
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", str(artifact)]
        + ["-o", str(tmp_path / "out.air")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_glsl_helper_parameters_are_renamed(tmp_path):
    code = _translate(tmp_path, "shader.comp", GLSL_COMPUTE)
    assert "float kernel_[9]" in code
    assert "kernel_[i] * data[i]" in code
    # ``vertex_`` is taken by a local, so the parameter takes the next name.
    assert "float twice(float vertex__)" in code
    assert "vertex__ * 2.0" in code
    assert "float kernel[" not in code
    _compile(tmp_path, code)


def test_parameters_and_locals_sharing_a_keyword_name_are_renamed(tmp_path):
    code = _translate(tmp_path, "shader.comp", SHADOWED_COMPUTE)
    assert "float conv(float kernel_[9], float data[9], float vertex_)" in code
    assert "kernel_[i] * data[i]" in code
    assert "res * vertex_" in code
    assert "float kernel[" not in code
    _compile(tmp_path, code)


def test_hlsl_parameters_locals_and_cbuffer_members_are_renamed(tmp_path):
    code = _translate(tmp_path, "shader.hlsl", HLSL_COMPUTE)
    assert "float thread_;" in code
    assert "float delete_;" in code
    assert "params.thread_" in code
    assert "float scale(float and_, float device_" in code
    assert "float new_ = " in code
    assert "results[kernel_.x] = vertex_;" in code
    _compile(tmp_path, code)


def test_glsl_stage_inputs_and_block_members_are_renamed(tmp_path):
    code = _translate(tmp_path, "shader.frag", GLSL_FRAGMENT)
    assert "float2 fragment_ [[user(locn0)]]" in code
    assert ".device_ * 0.5" in code
    assert ".and_;" in code
    assert "float char_ = " in code
    assert "input.fragment_" in code
    _compile(tmp_path, code)


def test_struct_members_are_renamed_with_their_accesses(tmp_path):
    code = _translate(tmp_path, "shader.cgl", CROSSGL)
    assert "float3 vertex_;" in code
    assert "float kernel_;" in code
    assert "light.kernel_ * light.vertex_.x * thread_" in code
    assert "light.vertex_ = float3(1.0);" in code
    _compile(tmp_path, code)


def test_renames_leave_the_input_ast_unchanged():
    ast = Parser(Lexer(CROSSGL).get_tokens()).parse()
    members = rename_metal_keyword_members(ast)
    parameters = rename_metal_keyword_parameters(members)
    assert members is not ast
    assert parameters is not members
    assert [member.name for member in ast.structs[0].members] == ["vertex", "kernel"]
    assert [member.name for member in members.structs[0].members] == [
        "vertex_",
        "kernel_",
    ]
    function = next(f for f in parameters.functions if f.name == "shade")
    assert [parameter.name for parameter in function.parameters] == [
        "light",
        "thread_",
    ]


def test_programs_without_keyword_names_are_returned_as_is():
    ast = Parser(
        Lexer(
            "shader main { float f(float x) { return x; } "
            "fragment { vec4 main() @location(0) { return vec4(f(1.0)); } } }"
        ).get_tokens()
    ).parse()
    assert rename_metal_keyword_members(ast) is ast
    assert rename_metal_keyword_parameters(ast) is ast


def test_enumerator_accesses_are_not_renamed():
    ast = Parser(
        Lexer(
            "shader main { enum Mode { and, other }; struct S { float and; }; "
            "fragment { vec4 main() @location(0) { S s; s.and = 1.0; "
            "if (Mode.and == Mode.other) { s.and = 2.0; } return vec4(s.and); } } }"
        ).get_tokens()
    ).parse()
    renamed = rename_metal_keyword_members(ast)
    members = {}
    for node in _walk(renamed):
        if isinstance(node, MemberAccessNode):
            owner = getattr(node.object_expr, "name", None)
            members.setdefault(owner, set()).add(node.member)
    assert members == {"s": {"and_"}, "Mode": {"and", "other"}}


def _walk(node, seen=None):
    seen = set() if seen is None else seen
    if isinstance(node, dict):
        node = list(node.values())
    if isinstance(node, (list, tuple)):
        for item in node:
            yield from _walk(item, seen)
        return
    if not hasattr(node, "__dict__") or id(node) in seen:
        return
    seen.add(id(node))
    yield node
    for key, value in vars(node).items():
        if key != "parent":
            yield from _walk(value, seen)
