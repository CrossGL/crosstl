"""Unsized resource arrays indexed at run time stay runtime-sized in GLSL, and GLSL ES refuses them."""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.glsl_refusals import WebGLLoweringRefused

HLSL_RUNTIME = """Texture2D<float4> texs[] : register(t0);
SamplerState samp : register(s0);
cbuffer Params : register(b0) { uint material; };
float4 main(float2 uv : TEXCOORD1) : SV_Target {
    return texs[material].Sample(samp, uv);
}
"""

GLSL_NON_UNIFORM = """#version 450
#extension GL_EXT_nonuniform_qualifier : require
layout(location = 0) flat in uint idx;
layout(location = 1) in vec2 uv;
layout(binding = 0) uniform sampler2D texs[];
layout(location = 0) out vec4 o_color;
void main() { o_color = texture(texs[nonuniformEXT(idx)], uv); }
"""

GLSL_CONSTANT = """#version 450
layout(location = 1) in vec2 uv;
layout(binding = 0) uniform sampler2D texs[];
layout(location = 0) out vec4 o_color;
void main() { o_color = texture(texs[2], uv) + texture(texs[0], uv); }
"""

GLSL_IMAGES = """#version 450
#extension GL_EXT_nonuniform_qualifier : require
layout(local_size_x = 8) in;
layout(binding = 0, rgba8) uniform writeonly image2D imgs[];
layout(std430, binding = 1) buffer Index { uint which; };
void main() { imageStore(imgs[nonuniformEXT(which)], ivec2(gl_GlobalInvocationID.xy), vec4(1.0)); }
"""


def _translate(tmp_path, source, backend, name):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _validate(code, stage):
    if shutil.which("glslangValidator") is None:
        return
    result = subprocess.run(
        ["glslangValidator", "--stdin", "-S", stage],
        input=code,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "source, name, stage, declaration",
    [
        (HLSL_RUNTIME, "shader.frag.hlsl", "frag", r"uniform sampler2D texs\[\];"),
        (GLSL_NON_UNIFORM, "shader.frag", "frag", r"uniform sampler2D texs\[\];"),
        (GLSL_IMAGES, "shader.comp", "comp", r"uniform image2D imgs\[\];"),
    ],
)
def test_glsl_keeps_runtime_indexed_resource_arrays_runtime_sized(
    tmp_path, source, name, stage, declaration
):
    glsl = _translate(tmp_path, source, "opengl", name)

    # No fixed size: a run-time index has no bound the shader states.
    assert re.search(declaration, glsl), glsl
    assert "#extension GL_EXT_nonuniform_qualifier : require" in glsl
    _validate(glsl, stage)


@pytest.mark.parametrize("backend", ["opengl", "webgl"])
def test_constant_indices_size_an_unsized_resource_array(tmp_path, backend):
    glsl = _translate(tmp_path, GLSL_CONSTANT, backend, "shader.frag")

    assert re.search(r"uniform sampler2D texs\[3\];", glsl), glsl
    assert "GL_EXT_nonuniform_qualifier" not in glsl
    _validate(glsl, "frag")


@pytest.mark.parametrize(
    "source, name",
    [(HLSL_RUNTIME, "shader.frag.hlsl"), (GLSL_NON_UNIFORM, "shader.frag")],
)
def test_glsl_es_refuses_runtime_indexed_resource_arrays(tmp_path, source, name):
    with pytest.raises(
        WebGLLoweringRefused, match="runtime-sized resource array 'texs'"
    ):
        _translate(tmp_path, source, "webgl", name)
