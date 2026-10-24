"""Shading-rate built-ins and constant switch labels in generated SPIR-V.

HLSL ``SV_ShadingRate`` on a pixel-shader input and GLSL ``gl_ShadingRateEXT`` are the SPIR-V ``ShadingRateKHR``
built-in; ``SV_ShadingRate`` on a vertex output and ``gl_PrimitiveShadingRateEXT`` are ``PrimitiveShadingRateKHR``
(SPV_KHR_fragment_shading_rate). The pixel-shader input used to become an ordinary flat varying with a location,
which reads an unrelated value. Switch case labels that name integer constants (``case RATE_2X2:``) fold to the
literal ``OpSwitch`` needs instead of being refused.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl

PIXEL_SHADER = """
cbuffer Params : register(b0) { uint mode; };
float4 main(float4 color : COLOR0, uint rate : SV_ShadingRate) : SV_Target
{
    const uint RATE_1X1 = 0;
    const uint RATE_2X2 = 5;
    if (mode == 1) {
        switch (rate) {
            case RATE_1X1: return color;
            case RATE_2X2: return color * 0.5;
            case RATE_2X2 + 1: return color * 0.25;
            default: break;
        }
    }
    return color * 0.125;
}
"""

VERTEX_SHADER = """
struct VSOut { float4 pos : SV_Position; uint rate : SV_ShadingRate; };
VSOut main(float4 pos : POSITION) { VSOut o; o.pos = pos; o.rate = 0x5; return o; }
"""

GLSL_FRAGMENT = """#version 450
#extension GL_EXT_fragment_shading_rate : require
layout(location = 0) out vec4 outColor;
void main() {
    int rate = gl_ShadingRateEXT;
    outColor = (rate & gl_ShadingRateFlag2VerticalPixelsEXT) != 0 ? vec4(1.0) : vec4(0.0);
}
"""

GLSL_VERTEX = """#version 450
#extension GL_EXT_fragment_shading_rate : require
layout(location = 0) in vec4 pos;
void main() {
    gl_Position = pos;
    gl_PrimitiveShadingRateEXT = gl_ShadingRateFlag4HorizontalPixelsEXT;
}
"""


def _translate(tmp_path, name, source):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="vulkan", format_output=False)


def _validate(tmp_path, assembly):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools are not on PATH")
    source = tmp_path / "out.spvasm"
    binary = tmp_path / "out.spv"
    source.write_text(assembly, encoding="utf-8")
    for command in (
        ["spirv-as", "--target-env", "vulkan1.1", str(source), "-o", str(binary)],
        ["spirv-val", "--target-env", "vulkan1.1", str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stdout + result.stderr


def _builtin_variable(assembly, builtin):
    match = re.search(rf"OpDecorate %(\w+) BuiltIn {builtin}\b", assembly)
    assert match, assembly
    return match.group(1)


def test_hlsl_pixel_shading_rate_reads_the_shading_rate_builtin(tmp_path):
    assembly = _translate(tmp_path, "rate.hlsl", PIXEL_SHADER)
    variable = _builtin_variable(assembly, "ShadingRateKHR")
    assert f"OpDecorate %{variable} Location" not in assembly
    # Vulkan VUID-StandaloneSpirv-Flat-04744: an integer fragment input, built-ins included, is flat.
    assert f"OpDecorate %{variable} Flat" in assembly
    assert "OpCapability FragmentShadingRateKHR" in assembly
    assert 'OpExtension "SPV_KHR_fragment_shading_rate"' in assembly
    _validate(tmp_path, assembly)


def test_switch_labels_fold_named_integer_constants(tmp_path):
    assembly = _translate(tmp_path, "rate.hlsl", PIXEL_SHADER)
    switch = re.search(r"OpSwitch %\w+ %\w+((?: \d+ %\w+)*)", assembly)
    assert switch, assembly
    assert re.findall(r" (\d+) %", switch.group(1)) == ["0", "5", "6"]


def test_hlsl_vertex_shading_rate_writes_the_primitive_rate_builtin(tmp_path):
    assembly = _translate(tmp_path, "rate_vs.hlsl", VERTEX_SHADER)
    variable = _builtin_variable(assembly, "PrimitiveShadingRateKHR")
    assert f"OpDecorate %{variable} Location" not in assembly
    assert "OpCapability FragmentShadingRateKHR" in assembly
    _validate(tmp_path, assembly)


def test_glsl_shading_rate_builtins_and_flags(tmp_path):
    fragment = _translate(tmp_path, "rate.frag", GLSL_FRAGMENT)
    _builtin_variable(fragment, "ShadingRateKHR")
    # GL_EXT_fragment_shading_rate section 7.3: gl_ShadingRateFlag2VerticalPixelsEXT is 1.
    assert re.search(r"OpBitwiseAnd %\w+ %\w+ %\w+", fragment)
    _validate(tmp_path, fragment)

    vertex = _translate(tmp_path, "rate.vert", GLSL_VERTEX)
    variable = _builtin_variable(vertex, "PrimitiveShadingRateKHR")
    store = re.search(rf"OpStore %{variable} %(\w+)", vertex)
    assert store, vertex
    # gl_ShadingRateFlag4HorizontalPixelsEXT is 8.
    assert re.search(rf"%{store.group(1)} = OpConstant %\w+ 8\b", vertex), vertex
    _validate(tmp_path, vertex)
