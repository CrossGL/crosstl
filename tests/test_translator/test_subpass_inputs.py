"""Vulkan subpass inputs become texel fetches at the fragment's pixel where the target has no input attachments."""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.semantics import SubpassInputUnsupportedError

SUBPASS_INPUTS = """#version 450
layout (input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput inputColor;
layout (input_attachment_index = 1, set = 0, binding = 1) uniform usubpassInput inputIds;
layout (input_attachment_index = 2, set = 0, binding = 2) uniform subpassInputMS inputSamples;
layout(location = 0) out vec4 o_color;
void main() {
    uint id = subpassLoad(inputIds).r;
    o_color = subpassLoad(inputColor) * float(id) + subpassLoad(inputSamples, 1);
}
"""

HELPER = """#version 450
layout (input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput inputColor;
layout(location = 0) out vec4 o_color;
vec4 readTwice(subpassInput src) { return subpassLoad(src) * 2.0; }
void main() { o_color = readTwice(inputColor); }
"""

MULTIVIEW = """#version 450
#extension GL_EXT_multiview : require
layout (input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput inputColor;
layout(location = 0) out vec4 o_color;
void main() { o_color = subpassLoad(inputColor) * float(gl_ViewIndex); }
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.frag"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    assert result.returncode == 0, result.stdout + result.stderr


def _metal_available():
    if shutil.which("xcrun") is None:
        return False
    found = subprocess.run(
        ["xcrun", "-sdk", "macosx", "--find", "metal"], capture_output=True
    )
    return found.returncode == 0


def test_glsl_reads_input_attachments_as_textures_at_the_fragment(tmp_path):
    glsl = _translate(tmp_path, SUBPASS_INPUTS, "opengl")

    assert "layout(binding = 0) uniform sampler2D inputColor;" in glsl
    assert "layout(binding = 1) uniform usampler2D inputIds;" in glsl
    assert "layout(binding = 2) uniform sampler2DMS inputSamples;" in glsl
    assert "texelFetch(inputIds, ivec2(gl_FragCoord.xy), 0).r" in glsl
    assert "texelFetch(inputColor, ivec2(gl_FragCoord.xy), 0)" in glsl
    assert "texelFetch(inputSamples, ivec2(gl_FragCoord.xy), 1)" in glsl
    assert "subpass" not in glsl
    if shutil.which("glslangValidator"):
        _run(["glslangValidator", "--stdin", "-S", "frag"], input=glsl)


def test_glsl_reads_an_input_attachment_through_a_helper(tmp_path):
    glsl = _translate(tmp_path, HELPER, "opengl")

    assert "vec4 readTwice(sampler2D src)" in glsl
    assert "texelFetch(src, ivec2(gl_FragCoord.xy), 0)" in glsl
    if shutil.which("glslangValidator"):
        _run(["glslangValidator", "--stdin", "-S", "frag"], input=glsl)


def test_glsl_es_reads_an_input_attachment_as_a_texture(tmp_path):
    essl = _translate(tmp_path, HELPER, "webgl")

    assert "uniform sampler2D inputColor;" in essl
    assert "texelFetch(src, ivec2(gl_FragCoord.xy), 0)" in essl
    if shutil.which("glslangValidator"):
        _run(["glslangValidator", "--stdin", "-S", "frag"], input=essl)


def test_hlsl_reads_input_attachments_as_textures_at_the_fragment(tmp_path):
    hlsl = _translate(tmp_path, SUBPASS_INPUTS, "directx")

    assert "Texture2D inputColor : register(t0);" in hlsl
    assert "Texture2D<uint4> inputIds : register(t1);" in hlsl
    assert "Texture2DMS<float4> inputSamples : register(t2);" in hlsl
    position = re.search(r"float4 (\w+) : SV_Position", hlsl)
    assert position, hlsl
    coordinate = f"int2({position.group(1)}.xy)"
    assert f"inputIds.Load(int3({coordinate}, 0)).r" in hlsl
    assert f"inputSamples.Load({coordinate}, 1)" in hlsl
    assert "subpass" not in hlsl.lower()


def test_hlsl_threads_the_fragment_position_into_a_helper(tmp_path):
    hlsl = _translate(tmp_path, HELPER, "directx")

    assert re.search(r"float4 readTwice\(Texture2D src, float4 \w+\)", hlsl), hlsl


def test_metal_reads_input_attachments_as_textures_at_the_fragment(tmp_path):
    metal = _translate(tmp_path, SUBPASS_INPUTS, "metal")

    assert re.search(r"texture2d<float> inputColor \[\[texture\(0\)\]\]", metal)
    assert re.search(r"texture2d<uint> inputIds \[\[texture\(1\)\]\]", metal)
    assert re.search(r"texture2d_ms<float> inputSamples \[\[texture\(2\)\]\]", metal)
    position = re.search(r"float4 (\w+) \[\[position\]\]", metal)
    assert position, metal
    assert f"inputSamples.read(uint2(int2({position.group(1)}.xy)), uint(1))" in metal
    assert "subpass" not in metal.lower()
    if _metal_available():
        source = tmp_path / "out.metal"
        source.write_text(metal)
        _run(
            ["xcrun", "-sdk", "macosx", "metal", "-c", str(source)]
            + ["-o", str(tmp_path / "out.air")]
        )


def test_metal_refuses_an_input_attachment_read_in_a_helper(tmp_path):
    with pytest.raises(
        SubpassInputUnsupportedError, match="helper function 'readTwice'"
    ):
        _translate(tmp_path, HELPER, "metal")


@pytest.mark.parametrize("backend", ["opengl", "directx", "metal"])
def test_multiview_input_attachment_reads_are_refused(tmp_path, backend):
    with pytest.raises(SubpassInputUnsupportedError, match="multiview"):
        _translate(tmp_path, MULTIVIEW, backend)


def test_spirv_keeps_input_attachments(tmp_path):
    spirv = _translate(tmp_path, SUBPASS_INPUTS, "vulkan")

    assert "InputAttachmentIndex 0" in spirv
    assert "Dim SubpassData" in spirv or "SubpassData" in spirv
    assert "OpImageRead" in spirv
