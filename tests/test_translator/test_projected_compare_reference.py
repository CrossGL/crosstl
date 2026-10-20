"""Projected depth compares divide the reference by ``q`` (``textureCompareProj*``).

CrossGL's ``textureCompareProj*(texture, P, ref, ...)`` divides both the coordinate and the reference by ``P``'s last
component ``q``. That is the SPIR-V ``OpImageSampleProjDref*`` definition (Vulkan, "Projection Operation") and GLSL's
``textureProj`` on shadow samplers, whose reference is ``P.z / P.w``; glslang and SPIRV-Cross implement it so. The
GLSL, HLSL, Metal, Slang and SPIR-V emitters project by hand, so they must divide the reference as well.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.image_access_contracts import (
    projected_texture_compare_reference,
)

PROJECTED_COMPARES = """
shader ProjectedCompare {
    sampler2DShadow shadowMap;
    samplerCubeShadow shadowCube;
    sampler2DArrayShadow shadowArray;

    struct FSInput {
        vec3 uvq @ TEXCOORD0;
        vec4 uvzq @ TEXCOORD1;
        vec4 dirq @ TEXCOORD2;
        vec4 uvLayerQ @ TEXCOORD3;
        float depth @ TEXCOORD4;
    };

    fragment {
        vec4 main(FSInput input) @ gl_FragColor {
            float a = textureCompareProj(shadowMap, input.uvq, input.depth);
            float b = textureCompareProj(shadowMap, input.uvzq, input.depth + 0.25);
            float c = textureCompareProjLod(shadowMap, input.uvq, input.depth, 0.0);
            float d = textureCompareProjOffset(shadowMap, input.uvq, input.depth, ivec2(1, 0));
            float e = textureCompareProj(shadowCube, input.dirq, input.depth);
            float f = textureCompareProj(shadowArray, input.uvLayerQ, input.depth);
            return vec4(a + b + c, d, e, f);
        }
    }
}
"""

# The reference operand each target must test, per call above.
EXPECTED = {
    "opengl": [
        "texture(shadowMap, vec3(uvq.xy / uvq.z, depth / uvq.z))",
        "texture(shadowMap, vec3(uvzq.xy / uvzq.w, (depth + 0.25) / uvzq.w))",
        "textureLod(shadowMap, vec3(uvq.xy / uvq.z, depth / uvq.z), 0.0)",
        "textureOffset(shadowMap, vec3(uvq.xy / uvq.z, depth / uvq.z), ivec2(1, 0))",
        "texture(shadowCube, vec4(dirq.xyz / dirq.w, depth / dirq.w))",
        "texture(shadowArray, vec4(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z, depth / uvLayerQ.w))",
    ],
    "directx": [
        "shadowMap.SampleCmp(shadowMapSampler, input.uvq.xy / input.uvq.z, input.depth / input.uvq.z)",
        "shadowMap.SampleCmp(shadowMapSampler, input.uvzq.xy / input.uvzq.w, (input.depth + 0.25) / input.uvzq.w)",
        "shadowMap.SampleCmpLevelZero(shadowMapSampler, input.uvq.xy / input.uvq.z, input.depth / input.uvq.z)",
        "shadowMap.SampleCmp(shadowMapSampler, input.uvq.xy / input.uvq.z, input.depth / input.uvq.z, int2(1, 0))",
        "shadowCube.SampleCmp(shadowCubeSampler, input.dirq.xyz / input.dirq.w, input.depth / input.dirq.w)",
        "shadowArray.SampleCmp(shadowArraySampler, float3(input.uvLayerQ.xy / input.uvLayerQ.w, input.uvLayerQ.z), "
        "input.depth / input.uvLayerQ.w)",
    ],
    "metal": [
        "input.uvq.xy / input.uvq.z, input.depth / input.uvq.z)",
        "input.uvzq.xy / input.uvzq.w, (input.depth + 0.25) / input.uvzq.w)",
        "input.uvq.xy / input.uvq.z, input.depth / input.uvq.z, level(0.0))",
        "input.uvq.xy / input.uvq.z, input.depth / input.uvq.z, int2(1, 0))",
        "input.dirq.xyz / input.dirq.w, input.depth / input.dirq.w)",
        "input.uvLayerQ.xy / input.uvLayerQ.w, uint(rint(clamp(float(input.uvLayerQ.z), 0.0, 2047.0))), input.depth / input.uvLayerQ.w)",
    ],
    "slang": [
        "shadowMap.SampleCmp(input.uvq.xy / input.uvq.z, input.depth / input.uvq.z)",
        "shadowMap.SampleCmp(input.uvzq.xy / input.uvzq.w, (input.depth + 0.25) / input.uvzq.w)",
        "shadowMap.SampleCmpLevel(input.uvq.xy / input.uvq.z, input.depth / input.uvq.z, 0.0)",
        "shadowMap.SampleCmp(input.uvq.xy / input.uvq.z, input.depth / input.uvq.z, int2(1, 0))",
        "shadowCube.SampleCmp(input.dirq.xyz / input.dirq.w, input.depth / input.dirq.w)",
        "shadowArray.SampleCmp(float3(input.uvLayerQ.xy / input.uvLayerQ.w, input.uvLayerQ.z), "
        "input.depth / input.uvLayerQ.w)",
    ],
}

GLSL_PROJECTED_SHADOW = """#version 450
layout(binding = 0) uniform sampler2DShadow shadowMap;
layout(location = 0) in vec4 P;
layout(location = 0) out vec4 color;
void main() {
    color = vec4(textureProj(shadowMap, P), textureProjOffset(shadowMap, P, ivec2(1, 0)), 0.0, 1.0);
}
"""


def _translate(tmp_path, source, backend, name="shader.cgl"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _flat(code):
    return " ".join(code.split())


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "compare, divisor, expected",
    [
        ("depth", "uvq.z", "depth / uvq.z"),
        ("input.depth", "input.uvq.w", "input.depth / input.uvq.w"),
        ("0.5", "p.w", "0.5 / p.w"),
        ("refs[i]", "p.w", "refs[i] / p.w"),
        ("(depth + 0.25)", "p.w", "(depth + 0.25) / p.w"),
        ("depth + 0.25", "p.w", "(depth + 0.25) / p.w"),
        ("(a) + (b)", "p.w", "((a) + (b)) / p.w"),
        ("-depth", "p.w", "(-depth) / p.w"),
    ],
)
def test_reference_is_divided_by_q_with_minimal_parentheses(compare, divisor, expected):
    assert projected_texture_compare_reference(compare, divisor) == expected


@pytest.mark.parametrize("backend", sorted(EXPECTED))
def test_source_targets_divide_the_reference_by_q(tmp_path, backend):
    code = _flat(_translate(tmp_path, PROJECTED_COMPARES, backend))
    for snippet in EXPECTED[backend]:
        assert snippet in code, snippet
    assert "textureCompareProj" not in code


def test_spirv_divides_dref_by_q_before_sampling(tmp_path):
    code = _translate(tmp_path, PROJECTED_COMPARES, "vulkan")
    lines = [line.strip() for line in code.splitlines()]
    samples = [line for line in lines if "OpImageSampleDref" in line]
    assert len(samples) == 6
    divisions = {line.split(" = ")[0]: line for line in lines if " = OpFDiv " in line}
    extracts = {
        line.split(" = ")[0] for line in lines if " = OpCompositeExtract " in line
    }
    for sample in samples:
        # %result = OpImageSampleDref* %type %sampled_image %coordinate %dref ...
        dref = sample.split()[6]
        # Dref is a division whose divisor is a component extracted from the coordinate.
        assert dref in divisions, sample
        assert divisions[dref].split()[-1] in extracts, divisions[dref]
    if shutil.which("spirv-as") and shutil.which("spirv-val"):
        asm, binary = tmp_path / "out.spvasm", tmp_path / "out.spv"
        asm.write_text(code)
        _run(["spirv-as", "--target-env", "vulkan1.0", str(asm), "-o", str(binary)])
        _run(["spirv-val", "--target-env", "vulkan1.0", str(binary)])


def test_glsl_texture_proj_shadow_reference_survives_translation(tmp_path):
    """GLSL ``textureProj(sampler2DShadow, P)`` tests ``P.z / P.w``, in every target."""
    hlsl = _flat(_translate(tmp_path, GLSL_PROJECTED_SHADOW, "directx", "s.frag"))
    assert "input.P.xyw.xy / input.P.xyw.z, input.P.z / input.P.xyw.z)" in hlsl
    assert (
        "input.P.xyw.xy / input.P.xyw.z, input.P.z / input.P.xyw.z, int2(1, 0))" in hlsl
    )
    metal = _flat(_translate(tmp_path, GLSL_PROJECTED_SHADOW, "metal", "s.frag"))
    assert "input.P.xyw.xy / input.P.xyw.z, input.P.z / input.P.xyw.z)" in metal
    glsl = _flat(_translate(tmp_path, GLSL_PROJECTED_SHADOW, "opengl", "s.frag"))
    assert "texture(shadowMap, vec3(P.xyw.xy / P.xyw.z, P.z / P.xyw.z))" in glsl


def test_glsl_output_compiles_with_glslang(tmp_path):
    if not shutil.which("glslangValidator"):
        pytest.skip("glslangValidator is not installed")
    path = tmp_path / "out.frag"
    path.write_text(_translate(tmp_path, PROJECTED_COMPARES, "opengl"))
    _run(["glslangValidator", str(path)])


def test_metal_output_compiles_with_the_metal_compiler(tmp_path):
    if not shutil.which("xcrun"):
        pytest.skip("xcrun is not installed")
    probe = subprocess.run(["xcrun", "--find", "metal"], capture_output=True, text=True)
    if probe.returncode != 0:
        pytest.skip("the Metal compiler is not installed")
    path = tmp_path / "out.metal"
    path.write_text(_translate(tmp_path, PROJECTED_COMPARES, "metal"))
    _run(["xcrun", "metal", "-c", str(path), "-o", str(tmp_path / "out.air")])


def test_hlsl_output_compiles_with_dxc(tmp_path):
    if not shutil.which("dxc"):
        pytest.skip("dxc is not installed")
    path = tmp_path / "out.hlsl"
    path.write_text(_translate(tmp_path, PROJECTED_COMPARES, "directx"))
    _run(["dxc", "-T", "ps_6_0", "-E", "main", str(path), "-Fo", str(tmp_path / "o")])


def test_slang_output_compiles_with_slangc(tmp_path):
    if not shutil.which("slangc"):
        pytest.skip("slangc is not installed")
    path = tmp_path / "out.slang"
    path.write_text(_translate(tmp_path, PROJECTED_COMPARES, "slang"))
    _run(
        [
            "slangc",
            str(path),
            "-target",
            "spirv",
            "-entry",
            "main",
            "-stage",
            "fragment",
            "-o",
            str(tmp_path / "out.spv"),
        ]
    )


@pytest.mark.parametrize(
    "call",
    [
        "textureCompareProj(shadowMap, input.uvq.xy, input.depth)",
        "textureCompareProj(shadowCube, input.uvq, input.depth)",
    ],
)
def test_slang_rejects_coordinates_without_a_projection_component(tmp_path, call):
    source = PROJECTED_COMPARES.replace(
        "textureCompareProj(shadowMap, input.uvq, input.depth)", call, 1
    )
    code = _flat(_translate(tmp_path, source, "slang"))
    # The Slang emitter reports refusals as loss markers; it is never emitted verbatim.
    assert (
        "/* unsupported Slang shadow compare: textureCompareProj requires sampler2DShadow vec3/vec4"
        in code
    )
    assert f"= {call}" not in code
