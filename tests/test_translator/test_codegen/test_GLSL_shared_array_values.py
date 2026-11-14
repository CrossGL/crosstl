"""OpenGL output names a shared array whole where GLSL copies it as an array value."""

import pytest

import crosstl
import crosstl.translator
from crosstl.translator.codegen.GLSL_codegen import (
    GLSLCodeGen,
    OpenGLWorkgroupPointerError,
)
from tests.test_translator.test_codegen.test_GLSL_codegen import (
    assert_glsl_compute_validates_if_available,
)

GLSL_ARGUMENT = """#version 450
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer Outp { float o[]; };
shared float s[64];
float bumps(float a[64], uint i) {
    a[(i + 1u) % 64u] += 100.0;
    return a[(i + 1u) % 64u];
}
void main() {
    uint i = gl_GlobalInvocationID.x;
    s[i] = float(i);
    barrier();
    o[i] = bumps(s, i) + s[(i + 1u) % 64u];
}
"""

GLSL_COPIES = """#version 450
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer Outp { float o[]; };
shared float s[64];
void keep(inout float a[64], uint i) { a[i] += 1.0; }
void main() {
    uint i = gl_GlobalInvocationID.x;
    s[i] = float(i);
    barrier();
    float t[64] = s;
    float u[64];
    u = s;
    keep(u, i);
    o[i] = t[(i + 1u) % 64u] + u[i];
}
"""

HLSL_ARGUMENT = """RWStructuredBuffer<float> o : register(u0);
groupshared float s[64];
float bumps(float a[64], uint i) { a[(i + 1) % 64] += 100.0; return a[(i + 1) % 64]; }
[numthreads(64, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    s[i] = float(i);
    GroupMemoryBarrierWithGroupSync();
    o[i] = bumps(s, i) + s[(i + 1) % 64];
}
"""


def _to_opengl(tmp_path, name, source):
    path = tmp_path / name
    path.write_text(source)
    cgl_path = tmp_path / f"{name}.cgl"
    cgl_path.write_text(crosstl.translate(str(path), backend="cgl"))
    return crosstl.translate(str(cgl_path), backend="opengl")


def test_shared_array_passes_to_by_value_array_parameter(tmp_path):
    generated = _to_opengl(tmp_path, "argument.comp", GLSL_ARGUMENT)

    assert "shared float s[64];" in generated
    assert "float bumps(float a[64], uint i)" in generated
    assert "bumps(s, i)" in generated
    assert_glsl_compute_validates_if_available(generated, tmp_path, "argument")


def test_groupshared_array_from_hlsl_passes_to_array_parameter(tmp_path):
    generated = _to_opengl(tmp_path, "argument.hlsl", HLSL_ARGUMENT)

    assert "shared float s[64];" in generated
    assert "bumps(s, i)" in generated
    assert_glsl_compute_validates_if_available(generated, tmp_path, "hlsl_argument")


def test_shared_array_initializes_and_assigns_array_values(tmp_path):
    generated = _to_opengl(tmp_path, "copies.comp", GLSL_COPIES)

    assert "float t[64] = s;" in generated
    assert "u = s;" in generated
    assert "keep(u, i);" in generated
    assert_glsl_compute_validates_if_available(generated, tmp_path, "copies")


def test_workgroup_pointer_alias_into_array_parameter_stays_refused():
    shader = """
    shader WorkgroupPointerAliasArgument {
        float first(float a[8]) { return a[0]; }
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

            void main() {
                threadgroup float values[8];
                threadgroup float* p = values;
                float x = first(p);
            }
        }
    }
    """

    with pytest.raises(OpenGLWorkgroupPointerError) as exc_info:
        GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert exc_info.value.parameter_name == "p"
    assert exc_info.value.reason == "bare-pointer-expression"


def test_shared_array_compared_as_pointer_stays_refused():
    shader = """
    shader SharedArrayPointerComparison {
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

            void main() {
                threadgroup float values[8];
                bool nonNull = values != nullptr;
            }
        }
    }
    """

    with pytest.raises(OpenGLWorkgroupPointerError) as exc_info:
        GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert exc_info.value.reason == "bare-pointer-expression"
