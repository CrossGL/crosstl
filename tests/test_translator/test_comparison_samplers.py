"""Comparison samplers import as CrossGL ``comparison_sampler`` from every source language.

A comparison sampler is its own resource type in HLSL (``SamplerComparisonState``), Vulkan GLSL (``samplerShadow``,
GL_KHR_vulkan_glsl) and WGSL (``sampler_comparison``). MSL has one ``sampler`` type, and a sampler compares when
``sample_compare`` or ``gather_compare`` use it (MSL 3.2 section 6.12.4). CrossGL keeps the distinction, so an
unused or forwarded comparison sampler still declares a comparison sampler on every target, and SPIR-V, which has
one ``OpTypeSampler`` for both, declares that type once.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

from crosstl.backend.usage_slots import UsageSlots
from tests.test_translator.test_metal_stage_varyings import metal_compiler
from tests.test_translator.test_spirv_separate_textures import (
    _image_shape,
    _translate,
    _validate,
    _variable_pointee,
)

HLSL_COMPARE = """Texture2D<float> shadowMap : register(t0);
Texture2D<float4> colorMap : register(t3);
SamplerComparisonState cs : register(s1);
SamplerComparisonState unusedCs : register(s2);
SamplerState linearS : register(s0);
SamplerComparisonState csArr[2] : register(s4);
float pcf(Texture2D<float> m, SamplerComparisonState c, float2 uv, float z) {
    return m.SampleCmpLevelZero(c, uv, z);
}
float4 main(float4 pos : SV_Position, float2 uv : TEXCOORD0) : SV_Target {
    float lit = pcf(shadowMap, cs, uv, 0.5) + shadowMap.SampleCmp(csArr[1], uv, 0.25);
    return colorMap.Sample(linearS, uv) * lit;
}
"""

VULKAN_GLSL_COMPARE = """#version 450
layout(set = 0, binding = 0) uniform texture2D shadowTex;
layout(set = 0, binding = 1) uniform samplerShadow cmpS;
layout(set = 0, binding = 2) uniform samplerShadow unusedCmp;
layout(set = 0, binding = 3) uniform sampler linearS;
layout(set = 0, binding = 4) uniform texture2D colorTex;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 fragColor;
void main() {
    float lit = texture(sampler2DShadow(shadowTex, cmpS), vec3(uv, 0.5));
    fragColor = texture(sampler2D(colorTex, linearS), uv) * lit;
}
"""

VULKAN_GLSL_FORWARDED = """#version 450
layout(set = 0, binding = 0) uniform texture2D shadowTex;
layout(set = 0, binding = 1) uniform samplerShadow cmpS;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 fragColor;
float pcf(texture2D t, samplerShadow s, vec2 p) { return texture(sampler2DShadow(t, s), vec3(p, 0.5)); }
void main() {
    fragColor = vec4(pcf(shadowTex, cmpS, uv));
}
"""

MSL_COMPARE = """#include <metal_stdlib>
using namespace metal;
struct VOut { float4 pos [[position]]; float2 uv; };
float pcf(depth2d<float> m, sampler c, float2 uv) { return m.sample_compare(c, uv, 0.5); }
fragment float4 fmain(VOut in [[stage_in]], depth2d<float> shadowMap [[texture(0)]],
                      texture2d<float> colorMap [[texture(1)]],
                      depthcube<float> cubeShadow [[texture(2)]],
                      sampler cs [[sampler(0)]], sampler ls [[sampler(1)]], sampler mixed [[sampler(2)]],
                      array<sampler, 2> csArr [[sampler(3)]]) {
    float lit = pcf(shadowMap, cs, in.uv) + shadowMap.sample_compare(cs, in.uv, 0.25, level(0));
    lit += cubeShadow.sample_compare(csArr[1], float3(in.uv, 1.0), 0.5);
    float4 g = shadowMap.gather_compare(csArr[0], in.uv, 0.5);
    return colorMap.sample(ls, in.uv) * lit + g + colorMap.sample(mixed, in.uv);
}
"""

MSL_MIXED = """#include <metal_stdlib>
using namespace metal;
struct VOut { float4 pos [[position]]; float2 uv; };
float compare(depth2d<float> m, sampler c, float2 uv) { return m.sample_compare(c, uv, 0.5); }
fragment float4 fmain(VOut in [[stage_in]], depth2d<float> shadowMap [[texture(0)]],
                      texture2d<float> colorMap [[texture(1)]],
                      sampler both [[sampler(0)]], sampler compareOnly [[sampler(1)]]) {
    float lit = compare(shadowMap, both, in.uv) + compare(shadowMap, compareOnly, in.uv);
    return colorMap.sample(both, in.uv) * lit;
}
"""


def _uniform_samplers(code):
    """The ids of the sampler types every ``UniformConstant`` sampler variable points to."""
    return set(re.findall(r"(%\w+) = OpTypeSampler\b", code))


# Imports: each source's comparison sampler becomes a CrossGL comparison_sampler.


def test_hlsl_sampler_comparison_states_import_as_comparison_samplers(tmp_path):
    crossgl = _translate(tmp_path, HLSL_COMPARE, ".hlsl", "cgl")

    for declaration in (
        "comparison_sampler cs;",
        "comparison_sampler unusedCs;",
        "comparison_sampler csArr[2];",
        "sampler linearS;",
        "float pcf(texture2DShadow m, comparison_sampler c, vec2 uv, float z)",
    ):
        assert declaration in crossgl, declaration


def test_vulkan_glsl_shadow_samplers_import_as_comparison_samplers(tmp_path):
    crossgl = _translate(tmp_path, VULKAN_GLSL_COMPARE, ".frag", "cgl")

    # A texture2D combined only into sampler2DShadow holds depth, so it imports as texture2DShadow.
    assert "texture2DShadow shadowTex @set(0) @binding(0);" in crossgl
    assert "comparison_sampler cmpS @set(0) @binding(1);" in crossgl
    assert "comparison_sampler unusedCmp @set(0) @binding(2);" in crossgl
    assert "sampler linearS @set(0) @binding(3);" in crossgl
    assert "texture2D colorTex @set(0) @binding(4);" in crossgl
    assert "textureCompare(shadowTex, cmpS, input.uv, 0.5)" in crossgl


def test_vulkan_glsl_shadow_texture_is_found_through_function_parameters(tmp_path):
    crossgl = _translate(tmp_path, VULKAN_GLSL_FORWARDED, ".frag", "cgl")

    assert "texture2DShadow shadowTex @set(0) @binding(0);" in crossgl
    assert "comparison_sampler cmpS @set(0) @binding(1);" in crossgl
    assert "float pcf(texture2DShadow t, comparison_sampler s, vec2 p)" in crossgl


def test_msl_samplers_used_only_to_compare_import_as_comparison_samplers(tmp_path):
    crossgl = _translate(tmp_path, MSL_COMPARE, ".metal", "cgl")

    assert "float pcf(texture2DShadow m, comparison_sampler c, vec2 uv)" in crossgl
    for parameter in (
        "comparison_sampler cs @sampler(0)",
        "sampler ls @sampler(1)",
        "sampler mixed @sampler(2)",
        "comparison_sampler[2] csArr @sampler(3)",
    ):
        assert parameter in crossgl, parameter


def test_msl_sampler_that_also_samples_stays_a_plain_sampler(tmp_path):
    crossgl = _translate(tmp_path, MSL_MIXED, ".metal", "cgl")

    # ``compare`` binds both samplers to one parameter, so neither can become a comparison sampler alone.
    assert "sampler both @sampler(0)" in crossgl
    assert "sampler compareOnly @sampler(1)" in crossgl
    assert "comparison_sampler" not in crossgl


def test_usage_slots_share_one_kind_set_per_joined_group():
    slots = UsageSlots()
    slots.join("argument", "parameter")
    slots.use("parameter", "compare")
    slots.join("other", "unrelated")
    slots.use("other", "compare")
    slots.use("unrelated", "sample")

    assert slots.only("argument", "compare")
    assert not slots.only("other", "compare")
    assert not slots.only("never-used", "compare")


# Targets: a comparison sampler stays a comparison sampler, and the output validates.


def test_hlsl_round_trip_keeps_an_unused_sampler_comparison_state(tmp_path):
    hlsl = _translate(tmp_path, HLSL_COMPARE, ".hlsl", "directx")

    assert "SamplerComparisonState cs : register(s1);" in hlsl
    assert "SamplerComparisonState unusedCs : register(s2);" in hlsl
    assert "SamplerComparisonState csArr[2] : register(s4);" in hlsl
    assert "SamplerState linearS : register(s0);" in hlsl
    _compile_hlsl(tmp_path, hlsl)


@pytest.mark.parametrize(
    ("source", "suffix", "samplers"),
    [
        (HLSL_COMPARE, ".hlsl", ("cs", "unusedCs", "linearS")),
        (VULKAN_GLSL_COMPARE, ".frag", ("cmpS", "unusedCmp", "linearS")),
        (VULKAN_GLSL_FORWARDED, ".frag", ("cmpS",)),
        (MSL_COMPARE, ".metal", ("cs", "ls", "mixed")),
    ],
)
def test_spirv_declares_one_sampler_type_for_both_kinds(
    tmp_path, source, suffix, samplers
):
    code = _translate(tmp_path, source, suffix, "vulkan")

    sampler_types = _uniform_samplers(code)
    assert len(sampler_types) == 1, sampler_types
    for name in samplers:
        assert _variable_pointee(code, name)[0] == "OpTypeSampler", name
    assert "OpImageSampleDref" in code
    _validate(tmp_path, code)


def test_spirv_compares_a_vulkan_glsl_shadow_texture_as_a_depth_image(tmp_path):
    code = _translate(tmp_path, VULKAN_GLSL_COMPARE, ".frag", "vulkan")

    assert _image_shape(code, "shadowTex") == ("float", "2D", "1", "0", "0")
    assert _image_shape(code, "colorTex") == ("float", "2D", "0", "0", "0")
    _validate(tmp_path, code)


def test_hlsl_declares_vulkan_glsl_shadow_samplers_as_comparison_states(tmp_path):
    hlsl = _translate(tmp_path, VULKAN_GLSL_COMPARE, ".frag", "directx")
    forwarded = _translate(tmp_path, VULKAN_GLSL_FORWARDED, ".frag", "directx")

    assert "SamplerComparisonState cmpS : register(s1);" in hlsl
    assert "SamplerComparisonState unusedCmp : register(s2);" in hlsl
    assert "SamplerState linearS : register(s3);" in hlsl
    assert "SamplerComparisonState cmpS : register(s1);" in forwarded
    assert "SamplerComparisonState s" in forwarded
    _compile_hlsl(tmp_path, hlsl)
    _compile_hlsl(tmp_path, forwarded)


def test_hlsl_hoists_a_msl_comparison_sampler_array_from_the_entry(tmp_path):
    hlsl = _translate(tmp_path, MSL_COMPARE, ".metal", "directx")

    assert "SamplerComparisonState cs : register(s0);" in hlsl
    assert "SamplerState ls : register(s1);" in hlsl
    assert "SamplerState mixed : register(s2);" in hlsl
    # An entry parameter array is a global resource array in HLSL; DXC rejects it in the signature.
    assert "SamplerComparisonState csArr[2] : register(s3);" in hlsl
    entry = re.search(r"\w+ \w+\(([^)]*)\)\s*:\s*SV_TARGET", hlsl, re.I)
    assert entry is not None and "csArr" not in entry.group(1)
    _compile_hlsl(tmp_path, hlsl)


@pytest.mark.parametrize(
    ("source", "suffix"),
    [(HLSL_COMPARE, ".hlsl"), (VULKAN_GLSL_COMPARE, ".frag")],
)
def test_wgsl_and_slang_declare_comparison_samplers(tmp_path, source, suffix):
    unused = "unusedCs" if suffix == ".hlsl" else "unusedCmp"
    wgsl = _translate(tmp_path, source, suffix, "wgsl")
    slang = _translate(tmp_path, source, suffix, "slang")

    assert re.search(rf"var {unused}: sampler_comparison;", wgsl)
    assert re.search(rf"SamplerComparisonState {unused}\b", slang)
    assert not re.search(rf"SamplerState {unused}\b", slang)
    if "csArr" not in source:
        # Core WGSL has no sampler arrays, so only the array-free source can validate.
        _validate_wgsl(tmp_path, wgsl)


@pytest.mark.parametrize(
    ("source", "suffix"),
    [
        (HLSL_COMPARE, ".hlsl"),
        (VULKAN_GLSL_COMPARE, ".frag"),
        (VULKAN_GLSL_FORWARDED, ".frag"),
        (MSL_COMPARE, ".metal"),
    ],
)
def test_glsl_and_metal_compare_through_depth_textures(tmp_path, source, suffix):
    glsl = _translate(tmp_path, source, suffix, "opengl")
    metal = _translate(tmp_path, source, suffix, "metal")

    assert re.search(r"uniform sampler2DShadow \w+;", glsl)
    assert re.search(r"depth2d<float> \w+ \[\[texture\(\d+\)\]\]", metal)
    _validate_glsl(tmp_path, glsl)
    _compile_metal(tmp_path, metal)


def _compile_hlsl(tmp_path, hlsl):
    dxc = shutil.which("dxc")
    if dxc is None:
        return
    entry = re.search(r"(\w+)\([^)]*\)\s*:\s*SV_TARGET", hlsl, re.I).group(1)
    path = tmp_path / "shader.hlsl"
    path.write_text(hlsl, encoding="utf-8")
    result = subprocess.run(
        [dxc, "-T", "ps_6_0", "-E", entry, str(path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]


def _validate_glsl(tmp_path, glsl):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        return
    path = tmp_path / "shader.frag"
    path.write_text(glsl, encoding="utf-8")
    result = subprocess.run(
        [glslang, str(path)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _validate_wgsl(tmp_path, wgsl):
    naga = shutil.which("naga")
    if naga is None:
        return
    path = tmp_path / "shader.wgsl"
    path.write_text(wgsl, encoding="utf-8")
    result = subprocess.run(
        [naga, str(path)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _compile_metal(tmp_path, metal):
    if metal_compiler() is None:
        return
    path = tmp_path / "shader.metal"
    path.write_text(metal, encoding="utf-8")
    result = subprocess.run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(path),
            "-o",
            str(tmp_path / "shader.air"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
