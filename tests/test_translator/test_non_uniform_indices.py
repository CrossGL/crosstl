"""Non-uniform resource indices take each target's spelling, or none where the target has no qualifier."""

import shutil
import subprocess

import pytest

import crosstl

HLSL_TEXTURES = """Texture2D<float4> texs[3] : register(t0);
SamplerState samp : register(s0);
float4 main(nointerpolation uint idx : TEXCOORD0, float2 uv : TEXCOORD1) : SV_Target {
    return texs[NonUniformResourceIndex(idx)].Sample(samp, uv);
}
"""

GLSL_TEXTURES = """#version 450
#extension GL_EXT_nonuniform_qualifier : require
layout(location = 0) flat in uint idx;
layout(location = 1) in vec2 uv;
layout(binding = 0) uniform sampler2D texs[4];
layout(location = 0) out vec4 o_color;
void main() { o_color = texture(texs[nonuniformEXT(idx)], uv); }
"""


def _translate(tmp_path, source, backend, name):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    assert result.returncode == 0, result.stdout + result.stderr


def _metal_compiler():
    if shutil.which("xcrun") is None:
        return None
    found = subprocess.run(
        ["xcrun", "-sdk", "macosx", "--find", "metal"], capture_output=True
    )
    return found.returncode == 0


def _validate_metal(tmp_path, code):
    if not _metal_compiler():
        return
    source = tmp_path / "out.metal"
    source.write_text(code)
    _run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", str(source)]
        + ["-o", str(tmp_path / "out.air")]
    )


def test_glsl_spells_hlsl_non_uniform_indices_with_the_extension(tmp_path):
    glsl = _translate(tmp_path, HLSL_TEXTURES, "opengl", "shader.frag.hlsl")

    assert "NonUniformResourceIndex" not in glsl
    assert "texture(texs[nonuniformEXT(idx)], uv)" in glsl
    assert "#extension GL_EXT_nonuniform_qualifier : require" in glsl
    if shutil.which("glslangValidator"):
        _run(["glslangValidator", "--stdin", "-S", "frag"], input=glsl)


@pytest.mark.parametrize(
    "source, name, read",
    [
        (HLSL_TEXTURES, "shader.frag.hlsl", "texs[idx].sample("),
        (GLSL_TEXTURES, "shader.frag", "texs[input.idx].sample("),
    ],
)
def test_metal_indexes_resource_arrays_without_a_qualifier(
    tmp_path, source, name, read
):
    metal = _translate(tmp_path, source, "metal", name)

    assert "NonUniformResourceIndex" not in metal
    assert "nonuniformEXT" not in metal
    assert read in metal
    _validate_metal(tmp_path, metal)


def test_glsl_es_selects_resources_with_constant_indices_and_no_qualifier(tmp_path):
    essl = _translate(tmp_path, HLSL_TEXTURES, "webgl", "shader.frag.hlsl")

    assert "NonUniformResourceIndex" not in essl
    assert "nonuniformEXT" not in essl
    assert "GL_EXT_nonuniform_qualifier" not in essl
    for index in range(3):
        assert f"texture(texs[{index}], uv)" in essl


def test_hlsl_spells_glsl_non_uniform_indices_as_non_uniform_resource_index(
    tmp_path,
):
    hlsl = _translate(tmp_path, GLSL_TEXTURES, "directx", "shader.frag")

    assert "nonuniformEXT" not in hlsl
    assert "texs[NonUniformResourceIndex(input.idx)]" in hlsl
